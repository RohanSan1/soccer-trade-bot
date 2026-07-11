#!/bin/bash
# Lightning AI Studio startup script
# Run this in a Lightning AI Studio terminal after creating a GPU Studio

set -euo pipefail

echo "=== Soccer Trade Bot - Lightning AI Setup ==="

# Install requirements
echo "Installing training requirements..."
pip install -q -r infra/requirements-lightning-train.txt

# Create directories
mkdir -p /teamspace/data /teamspace/model /teamspace/checkpoints

# Verify GPU
echo "GPU check:"
nvidia-smi || echo "No GPU detected"

# Build dataset (if not already cached)
if [ ! -f /teamspace/data/train.parquet ]; then
    echo "Building dataset from StatsBomb..."
    python -c "
from data.build_dataset import build_dataset
build_dataset('/teamspace/data/train.parquet')
"
else
    echo "Dataset already exists at /teamspace/data/train.parquet"
fi

# Run training with Optuna (200 trials, CatBoost enabled)
echo "Starting training with Optuna (200 trials)..."
python -m model.train_lightning \
    --data /teamspace/data/train.parquet \
    --output /teamspace/model \
    --optuna \
    --optuna-trials 200 \
    --catboost

echo "=== Training complete! Model saved to /teamspace/model ==="
ls -la /teamspace/model/