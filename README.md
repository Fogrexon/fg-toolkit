# FunctionGemma Toolkit

Welcome to the FunctionGemma Toolkit repository. This project provides a comprehensive suite of tools for fine-tuning the FunctionGemma model with custom data, converting it to ONNX format, and integrating it into web-based applications via a wrapper library.

## Overview

FunctionGemma is an LLM designed for function calling and tool use. This toolkit aims to make it easy to:
- **Fine-tune**: Use a dedicated web-based tool and Docker-based environment to train FunctionGemma on your own datasets.
- **Convert**: Automatically convert trained models to ONNX format for efficient browser-based inference.
- **Integrate**: Utilize a specialized wrapper library (Transformers.js-based) to embed FunctionGemma directly into web apps for dynamic interactions, tool use, or client-side logic.

## Key Features

- **Trainer Web Tool**: A user-friendly interface for managing datasets and training runs within a virtualized Docker environment.
- **Inference Library**: A robust wrapper for Transformers.js that simplifies model loading and function calling in the browser.
- **Pre-configured Environment**: Docker-based setup ensures consistent results and easy deployment.

## Project Structure

- `trainer/`: The web-based training dashboard and Docker configurations.
- `packages/lib-web-runtime/`: The source code for the wrapper library.
- `examples/`: Sample projects and implementation guides.
- `docs/`: Technical specifications and API documentation.

## Usage

### Training with Docker

This environment is designed to be **built and run locally**. You do not need to pull images from Docker Hub.

#### Step 1: Build the Image Locally
You **must** build the image once before you can run it:
```bash
docker build -t fg-trainer ./trainer
```
Wait for this command to finish completely before proceeding to the next step.

#### Step 2: Run Training
Place your training data in a JSON file (e.g., `trainer/dataset.json`).

> [!IMPORTANT]
> **Hugging Face Authentication**: FunctionGemma is a **gated model**. You must accept the license on the [Hugging Face model page](https://huggingface.co/google/functiongemma-270m-it) and pass your [HF Token](https://huggingface.co/settings/tokens) using the `HF_TOKEN` environment variable.

Run the container by mounting your local directories:

```bash
# Windows (PowerShell)
docker run --rm `
  -e HF_TOKEN="your_huggingface_token_here" `
  -v ${PWD}/trainer/dataset.json:/app/data/train.json `
  -v ${PWD}/trainer/output:/app/output `
  fg-trainer --dataset_path /app/data/train.json --output_dir /app/output

# Linux/macOS
docker run --rm \
  -e HF_TOKEN="your_huggingface_token_here" \
  -v $(pwd)/trainer/dataset.json:/app/data/train.json \
  -v $(pwd)/trainer/output:/app/output \
  fg-trainer --dataset_path /app/data/train.json --output_dir /app/output
```

#### Results
The fine-tuned model and its **ONNX version** will be saved in your local `trainer/output` folder.

## License

*(License information to be added.)*
