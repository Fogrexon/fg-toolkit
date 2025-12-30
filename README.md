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

To fine-tune FunctionGemma using the provided Docker environment:

1. **Build the Docker Image**:
   ```bash
   docker build -t fg-trainer ./trainer
   ```

2. **Prepare your Dataset**:
   Place your training data in a JSON file (e.g., `my_dataset.json`). See `trainer/dataset.json` for the expected format.

3. **Run Training**:
   Run the container and mount your local data and output directories:
   ```bash
   # Windows (PowerShell)
   docker run --rm `
     -v ${PWD}/trainer/dataset.json:/app/data/train.json `
     -v ${PWD}/trainer/output:/app/output `
     fg-trainer --dataset_path /app/data/train.json --output_dir /app/output

   # Linux/macOS
   docker run --rm \
     -v $(pwd)/trainer/dataset.json:/app/data/train.json \
     -v $(pwd)/trainer/output:/app/output \
     fg-trainer --dataset_path /app/data/train.json --output_dir /app/output
   ```

4. **Results**:
   The fine-tuned model and its ONNX version will be saved in your local `trainer/output` folder.

## License

*(License information to be added.)*
