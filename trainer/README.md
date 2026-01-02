# FunctionGemma Personal Trainer

A simplified fine-tuning tool for FunctionGemma, designed to help you create custom tool-calling models.

## Overview

This directory contains a training script (`train.py`) and environment configuration for fine-tuning FunctionGemma models. It supports:
- **LoRA / PEFT**: Efficient fine-tuning using Low-Rank Adaptation.
- **Quantization**: 4-bit training for lower memory usage.
- **ONNX Export**: Automatic export of the fine-tuned model to ONNX format for web use (e.g., with `fg-toolkit-web`).
- **Custom Datasets**: Easy-to-use JSON dataset format.

## Installation

You can run the trainer locally or via Docker.

### Local Installation

1. **Prerequisites**: Python 3.10+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

The trainer expects a JSON array of conversation objects. Each object represents a single training example.

**File Format**: JSON

**Example (`dataset.json`)**:
```json
[
  {
    "user_content": "What is the weather in Tokyo?",
    "tool_name": "get_weather",
    "tool_arguments": "{\"location\": \"Tokyo\"}"
  },
  {
    "user_content": "Hello, how are you?",
    "assistant_content": "I'm doing well, thank you! How can I help you?"
  }
]
```

### Fields

- `user_content`: The message from the user.
- `tool_name` (optional): The name of the function tool the model should call.
- `tool_arguments` (optional): A JSON string representing the arguments for the tool. **Must be a stringified JSON object.**
- `assistant_content` (optional): The text response from the assistant (used when no tool call is needed).

> [!IMPORTANT]
> Either `tool_name` + `tool_arguments` OR `assistant_content` must be provided for the assistant's response.

## Usage

Run the training script with your dataset.

### Basic Command

```bash
python train.py --dataset_path ./my_dataset.json
```

### Arguments

> [!IMPORTANT]
> **FunctionGemma is a gated model.** You MUST provide a Hugging Face token with access to the model, either via the `--token` argument or the `HF_TOKEN` environment variable.

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model_id` | `google/functiongemma-270m-it` | Hugging Face Model ID to use as the base. |
| `--dataset_path` | (Required) | Path to your local JSON dataset. |
| `--output_dir` | `./output` | Directory to save the trained model and logs. |
| `--epochs` | `8` | Number of training epochs. |
| `--batch_size` | `4` | Training batch size per device. |
| `--learning_rate` | `5e-5` | Learning rate. |
| `--use_peft` | `False` | Enable LoRA (PEFT) fine-tuning. |
| `--load_in_4bit` | `False` | Load model in 4-bit precision (reduces VRAM). |
| `--export_onnx` | `True` | Export to ONNX format after training. |
| `--token` | (Required) | Hugging Face token for gated models (FunctionGemma). defaults to `HF_TOKEN` env var. |

### Efficient Training (LoRA + 4-bit)

For consumer GPUs, use PEFT and 4-bit quantization:

```bash
python train.py \
  --dataset_path my_dataset.json \
  --use_peft \
  --load_in_4bit \
  --batch_size 2
```

## Docker Usage

Use Docker to run training in an isolated environment.

1. **Build the Image**
   ```bash
   docker build -t fg-trainer .
   ```

2. **Run Training**
   Mount your data directory to `/app/data` and output directory to `/app/output`.

   ```bash
   docker run --rm \
     -v $(pwd)/my_data:/app/data \
     -v $(pwd)/my_output:/app/output \
     fg-trainer \
     --dataset_path /app/data/dataset.json \
     --output_dir /app/output
   ```

   **With GPU Support**:
   Ensure you have the NVIDIA Container Toolkit installed.
   ```bash
   docker run --gpus all ...
   ```

## ONNX Export

By default, the trainer automatically exports the model to ONNX format at the end of training.

- The ONNX model will be saved in `<output_dir>/onnx`.
- If using PEFT/LoRA, the weights are automatically merged before export.

To disable this behavior, pass `--export_onnx=False` (Note: `argparse` boolean flags might require `--no-export_onnx` depending on implementation, but currently the script defaults to True and acts as a flag. *Correction*: The script uses `action="store_true"` default `True`... wait, `action="store_true"` defaults to False usually, but the code says `default=True`. Actually `action="store_true"` means "if present, True". The code in `train.py` says:
`parser.add_argument("--export_onnx", action="store_true", default=True, help="...")`
Guidance: This `argparse` definition is slightly ambiguous (usually `store_true` implies default False). However, if you want to *disable* it, you might need to modify the code or just rely on the default. Since it defaults to True, just running it exports. The script doesn't seem to have a `--no-export_onnx` flag.

## Troubleshooting

- **Out of Memory (OOM)**: Try reducing `--batch_size` (e.g., to 1) or enabling `--load_in_4bit`.
- **Missing Token**: If you see a `ValueError` or 401/403 error, ensure you have provided a valid Hugging Face token and that your account has accepted the license agreement for FunctionGemma.
