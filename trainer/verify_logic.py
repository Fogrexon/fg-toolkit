import json
from datasets import Dataset

# Extracting the core logic from train.py to verify it without loading the full model
def test_data_processing():
    # Dummy tool schema generation (simplified)
    def dummy_get_json_schema(func):
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        }

    def search_knowledge_base(query: str) -> str:
        """Search internal company documents."""
        return "Internal Result"
    
    def search_google(query: str) -> str:
        """Search public information."""
        return "Public Result"

    TOOLS = [dummy_get_json_schema(search_knowledge_base), dummy_get_json_schema(search_google)]
    DEFAULT_SYSTEM_MSG = "You are a model that can do function calling..."

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

    # Test cases
    samples = [
        {
            "user_content": "What is the reimbursement limit?",
            "tool_name": "search_knowledge_base",
            "tool_arguments": '{"query": "reimbursement limit"}'
        },
        {
            "user_content": "Hello!",
            "assistant_content": "Hi there!"
        }
    ]

    print("Verifying data processing logic...")
    for i, sample in enumerate(samples):
        processed = create_conversation(sample)
        print(f"\nSample {i+1}:")
        print(json.dumps(processed, indent=2))
        
        # Basic assertions
        assert processed["messages"][0]["role"] == "developer"
        assert processed["messages"][1]["content"] == sample["user_content"]
        if "tool_name" in sample:
            assert "tool_calls" in processed["messages"][2]
            assert processed["messages"][2]["tool_calls"][0]["function"]["name"] == sample["tool_name"]
        else:
            assert "content" in processed["messages"][2]
            assert processed["messages"][2]["content"] == sample["assistant_content"]
    
    print("\n✅ Data processing logic verification passed!")

if __name__ == "__main__":
    test_data_processing()
