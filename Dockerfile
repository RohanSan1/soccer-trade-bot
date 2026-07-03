# Soccer Trade Bot - Training Image
# For OVH AI Training (H100 GPUs)
# Build: docker build -t ghcr.io/rohan5commit/soccer-trade-bot:training .
# Run: docker run --gpus all ghcr.io/rohan5commit/soccer-trade-bot:training

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL maintainer="rohan5commit"
LABEL description="Soccer trade bot training environment"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker layer caching)
COPY infra/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make scripts executable
RUN chmod +x infra/*.sh 2>/dev/null || true

# Default command: run ensemble training
CMD ["python", "-m", "model.train"]
