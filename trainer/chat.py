import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned FunctionGemma model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory (e.g., ./output).")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Ensure HF Token is present
    if not args.token:
        raise ValueError("Hugging Face token is required. Please pass --token or set HF_TOKEN environment variable.")

    print(f"Loading model from: {args.model_path}...")
    
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
