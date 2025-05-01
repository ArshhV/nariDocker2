FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    runpod \
    torch \
    transformers \
    pydub \
    soundfile \
    huggingface_hub

# Install Dia from GitHub
RUN pip3 install --no-cache-dir git+https://github.com/nari-labs/dia.git

# Create model directory
RUN mkdir -p /root/.cache/huggingface/hub/models--nari-labs--Dia-1.6B/

# Pre-download the model
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='nari-labs/Dia-1.6B', filename='model.safetensors', local_dir='/root/.cache/huggingface/hub/models--nari-labs--Dia-1.6B/snapshots/main/')"

# Copy handler code
COPY runpod_handler.py .

# Start handler
CMD [ "python3", "-u", "runpod_handler.py" ]