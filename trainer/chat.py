import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned FunctionGemma model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory (e.g., ./output).")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX Runtime for inference (expects model in model_path/onnx).")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Token is optional for local fine-tuned models
    if not args.token and not os.path.exists(args.model_path):
         # If path doesn't exist, maybe it's a Hub ID? Then warn or let transformers handle it.
         pass

    print(f"Loading model from: {args.model_path}...")
    
    if args.use_onnx:
        onnx_path = os.path.join(args.model_path, "onnx")
        print(f"Using ONNX Runtime. Loading from: {onnx_path}")
        model = ORTModelForCausalLM.from_pretrained(
            onnx_path,
            device_map="auto", # optimum handles device
            token=args.token,
        )
        # Tokenizer is typically saved in the same ONNX dir or fallback to base
        tokenizer = AutoTokenizer.from_pretrained(onnx_path, token=args.token)
    else:
        # Load the full model directly
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            token=args.token,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.token)

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
