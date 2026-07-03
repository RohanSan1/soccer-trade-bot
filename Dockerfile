FROM python:3.11-slim-bookworm

LABEL maintainer="rohan5commit"
LABEL description="Soccer trade bot training environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install training dependencies
COPY infra/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Copy project
COPY . .

RUN chmod +x infra/*.sh 2>/dev/null || true
RUN mkdir -p data model && chmod 777 data model
RUN chmod -R 777 /app

# Pre-download YOLOv10-X weights during build (~58MB)
RUN python -c "from ultralytics import YOLO; YOLO('yolov10x.pt')" || true

# Pre-download CLIP ViT-L/14 weights (~400MB) to writable cache
RUN mkdir -p /tmp/clip_cache && CLIP_HOME=/tmp/clip_cache python -c "import clip; clip.load('ViT-L/14', device='cpu', download_root='/tmp/clip_cache')" || true

ENV HOME=/tmp
ENV ULTRALYTICS_CACHE_DIR=/tmp/ultralytics
ENV CLIP_HOME=/tmp/clip_cache
ENV PYTHONPATH=/app

CMD ["sh", "-c", "python -c \"from data.build_dataset import build_dataset; build_dataset('data/train.parquet')\" && python -m model.train --data data/train.parquet --output /tmp/model && ls -la /tmp/model/"]
