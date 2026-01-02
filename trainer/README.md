# FunctionGemma Personal Trainer

A simplified fine-tuning tool for FunctionGemma, designed to help you create custom tool-calling models.

## Overview

This directory contains a training script (`train.py`) and environment configuration for fine-tuning FunctionGemma models. It supports:
- **Full Fine-Tuning**: Updates all model parameters for maximum performance (recommended for small models like FunctionGemma 270M).
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
    "tool_arguments": {"location": "Tokyo"}
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
- `tool_arguments` (optional): A JSON object (or stringified JSON) representing the arguments for the tool.
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
| `--dataset_path` | (Required) | Path to your local JSON dataset. |
| `--output_dir` | `./output` | Directory to save the fine-tuned model and logs. |
| `--epochs` | `8` | Number of training epochs. |
| `--batch_size` | `4` | Training batch size per device. |
| `--learning_rate` | `5e-5` | Learning rate. |
| `--max_length` | `512` | Max sequence length. |
| `--export_onnx` | `True` | Export to ONNX format after training. |
| `--token` | (Required) | Hugging Face token for gated models (FunctionGemma). defaults to `HF_TOKEN` env var. |

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

## Chatting with the Model

You can interactively chat with your fine-tuned model using the `chat.py` script. This is useful for quick verification.

### Basic Usage

```bash
# PyTorch (Standard)
python chat.py --model_path ./output

# ONNX Runtime (Verify exported model)
python chat.py --model_path ./output --use_onnx
```

### Arguments

| Argument | Description |
| :--- | :--- |
| `--model_path` | (Required) Path to the fine-tuned model directory (e.g., `./output`). |
| `--use_onnx` | Use ONNX Runtime for inference. Expects ONNX subfolder (e.g., `./output/onnx`). |

### Docker Usage for Chat

To run the chat script inside the Docker container, use `--entrypoint python`:

```bash
docker run --rm -it \
  --entrypoint python \
  -v $(pwd)/model_output:/app/output \
  -e HF_TOKEN=$HF_TOKEN \
  fg-trainer \
  chat.py --model_path /app/output
```

## Troubleshooting

- **Out of Memory (OOM)**: Try reducing `--batch_size` (e.g., to 1).
- **Missing Token**: If you see a `ValueError` or 401/403 error, ensure you have provided a valid Hugging Face token and that your account has accepted the license agreement for FunctionGemma.
