#!/usr/bin/env python3
"""Lightning AI Training Entry Point.

Run this directly in a Lightning AI Studio with GPU machine:
    python -m model.train_lightning --optuna --optuna-trials 200 --catboost

Or via Lightning SDK:
    studio.run('python -m model.train_lightning --optuna --optuna-trials 200', machine='gpu-fast')
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure local package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LightGBM + CatBoost ensemble on Lightning AI"
    )
    parser.add_argument(
        "--data",
        default="/teamspace/data/train.parquet",
        help="Training data path (default: /teamspace/data/train.parquet)",
    )
    parser.add_argument(
        "--output",
        default="/teamspace/model",
        help="Output directory (default: /teamspace/model)",
    )
    parser.add_argument("--optuna", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--optuna-trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument("--no-catboost", action="store_true", help="Disable CatBoost")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Build dataset if not exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.info("Training data not found at %s — building from StatsBomb...", data_path)
        from data.build_dataset import build_dataset
        data_path.parent.mkdir(parents=True, exist_ok=True)
        build_dataset(output_path=str(data_path))
        logger.info("Dataset built and saved to %s", data_path)

    # Train
    train(
        data_path=str(data_path),
        output_dir=args.output,
        use_optuna=args.optuna,
        optuna_trials=args.optuna_trials,
        use_catboost=not args.no_catboost,
    )

    logger.info("Training complete. Model saved to %s", args.output)


if __name__ == "__main__":
    main()