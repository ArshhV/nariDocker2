FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    runpod \
    deepseek-vl2 \
    torch \
    transformers

# Set up working directory
WORKDIR /app

# Copy handler code
COPY runpod_handler.py .

# Start handler
CMD [ "python3", "-u", "runpod_handler.py" ]