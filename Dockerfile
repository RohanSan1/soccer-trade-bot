FROM python:3.11-slim-bookworm

LABEL maintainer="rohan5commit"
LABEL description="Soccer trade bot training environment (slim)"

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

# Install PyTorch CPU+CUDA runtime (smaller than devel)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
COPY infra/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x infra/*.sh 2>/dev/null || true

CMD ["python", "-m", "model.train"]
