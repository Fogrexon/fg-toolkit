import os
import json
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

def main():
    model_path = os.path.join(os.path.dirname(__file__), "model")
    
    print(f"Loading ONNX model from {model_path}...")
    # Load model and tokenizer
    # We use provider="CPUExecutionProvider" for general compatibility. 
    # Use "CUDAExecutionProvider" if you have a GPU and onnxruntime-gpu installed.
    model = ORTModelForCausalLM.from_pretrained(model_path, provider="CPUExecutionProvider")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tool definitions (Must match what the model was trained with for consistency)
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search internal company documents, policies and project data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query string."}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_google",
                "description": "Search public information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query string."}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Sample user queries
    queries = [
        "What is the reimbursement limit for travel meals?",
        "What is the current stock price of Google?",
    ]

    for user_query in queries:
        print(f"\nUser: {user_query}")
        
        messages = [
            {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
            {"role": "user", "content": user_query}
        ]

        # Apply chat template with tools
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # Generate output
        outputs = model.generate(inputs, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
