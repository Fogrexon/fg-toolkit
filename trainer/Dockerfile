# Use PyTorch with dev dependencies for building extensions if needed
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for build-essential and git (for some HF libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . /app/

# Create directories for data and output (users should mount these)
RUN mkdir -p /app/data /app/output

# Default entrypoint to the training script
# This allows passing arguments like: docker run ... --dataset_path /app/data/train.json
ENTRYPOINT ["python", "train.py"]

# Default arguments if none are provided
CMD ["--dataset_path", "/app/data/train.json", "--output_dir", "/app/output"]
