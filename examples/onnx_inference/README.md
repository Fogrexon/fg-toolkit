# ONNX Inference Example

This example demonstrates how to perform tool-calling inference using a fine-tuned FunctionGemma model in ONNX format.

## Prerequisites

Ensure you have the necessary dependencies installed:

```bash
pip install transformers optimum[onnxruntime] torch
```

## How to Run

Before running the script, make sure you have copied the ONNX model files to the `model/` directory.

```bash
python inference.py
```

## Note on Model Files

The model files in the `model/` directory are excluded from Git to keep the repository size manageable. You can generate these files by running the trainer in the `trainer/` directory.
