import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned FunctionGemma model.")
    parser.add_argument("--base_model_id", type=str, default="google/functiongemma-270m-it", help="Base model ID from Hugging Face.")
    parser.add_argument("--adapter_path", type=str, help="Path to the fine-tuned LoRA adapter (e.g., ./output).")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision.")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Ensure HF Token is present
    if not args.token:
        raise ValueError("Hugging Face token is required. Please pass --token or set HF_TOKEN environment variable.")

    print(f"Loading base model: {args.base_model_id}...")
    
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=args.token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, token=args.token)

    if args.adapter_path:
        print(f"Loading adapter from: {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path, token=args.token)

    print("\n--- Model Loaded. Type 'exit' to quit. ---\n")

    messages = []
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            messages.append({"role": "user", "content": user_input})
            
            # Simple chat template application
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
            
            outputs = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
            
            # Decode only the new tokens
            generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            print(f"Assistant: {generated_text}\n")
            messages.append({"role": "assistant", "content": generated_text})
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
