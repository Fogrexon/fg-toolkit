import argparse
import json
import os
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from transformers.utils import get_json_schema
from optimum.exporters.onnx import main_export

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma for tool calling.")
    parser.add_argument("--model_id", type=str, default="google/functiongemma-270m-it", help="Model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, help="Path to local training dataset (JSON).")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--use_peft", action="store_true", help="Use LoRA for fine-tuning.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision.")
    parser.add_argument("--export_onnx", action="store_true", default=True, help="Export the model to ONNX format after training.")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token for gated models.")
    return parser.parse_args()

def load_local_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def main():
    args = parse_args()
    logging.set_verbosity_info()

    # Load Model and Tokenizer
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype="auto",
        attn_implementation="eager", # Use flash_attention_2 if available
        token=args.token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.token)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for PEFT if requested
    if args.use_peft:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    
    # Load and Prepare Dataset
    if args.dataset_path:
        dataset = load_local_data(args.dataset_path)
    else:
        # Default placeholder if no path provided (for testing/demo)
        print("No dataset path provided. Using synthetic demo data.")
        demo_data = [
            {"user_content": "What is the reimbursement limit for travel meals?", "tool_name": "search_knowledge_base", "tool_arguments": '{"query": "travel meal reimbursement limit policy"}'},
            {"user_content": "What is the current stock price of Google?", "tool_name": "search_google", "tool_arguments": '{"query": "current Google stock price"}'},
        ]
        dataset = Dataset.from_list(demo_data)

    # Tool definitions (Dummy functions to generate schema)
    def search_knowledge_base(query: str) -> str:
        """
        Search internal company documents, policies and project data.

        Args:
            query: The search query string.
        """
        return "Internal Result"
    def search_google(query: str) -> str:
        """
        Search public information.

        Args:
            query: The search query string.
        """
        return "Public Result"

    TOOLS = [get_json_schema(search_knowledge_base), get_json_schema(search_google)]
    DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"

    def create_conversation(sample):
        assistant_msg = {"role": "assistant"}
        if sample.get("tool_name"):
            assistant_msg["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": sample["tool_name"],
                        "arguments": json.loads(sample["tool_arguments"])
                    }
                }
            ]
        else:
            assistant_msg["content"] = sample.get("assistant_content", "I am not sure how to help with that.")

        return {
            "messages": [
                {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
                {"role": "user", "content": sample["user_content"]},
                assistant_msg,
            ],
            "tools": TOOLS
        }

    dataset = dataset.map(create_conversation, remove_columns=dataset.column_names, batched=False)
    
    # SFT Training Config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",  # Note: notebook uses 'epoch', but keeping 'no' for speed unless requested
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training completed.")

    if args.export_onnx:
        print("Starting ONNX export...")
        onnx_path = os.path.join(args.output_dir, "onnx")
        
        # If PEFT was used, we need to merge the weights before exporting
        if args.use_peft:
            print("Merging PEFT weights...")
            # If loaded in 4bit, we need to reload the model in float16/bfloat16 to merge
            if args.load_in_4bit:
                print("Reloading model in float16 for merging (4-bit merging is not supported)...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    token=args.token,
                )
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.output_dir, token=args.token)
                model = model.merge_and_unload()
            else:
                model = model.merge_and_unload()
            
            # Save the merged model temporarily or use it directly
            merged_model_path = os.path.join(args.output_dir, "merged")
            model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            export_source = merged_model_path
        else:
            export_source = args.output_dir

        try:
            main_export(
                export_source,
                output=onnx_path,
                task="text-generation",  # Changed from "causal-lm" to support KV cache for transformers.js
                device="cpu", # Export on CPU is more stable
            )
            print(f"ONNX export completed. Saved to {onnx_path}")

        except Exception as e:
            print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    main()
