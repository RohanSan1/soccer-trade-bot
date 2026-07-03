"""YOLOv10-X fine-tuning on SoccerNet Tracking dataset.

Fine-tunes YOLOv10-X (largest variant) for player detection on broadcast footage.
Trained on SoccerNet Tracking (200 clips × 30s with bounding boxes).

GPU CONSTRAINT: Only CUDA 0 and 1 allowed. CUDA 2 and 3 belong to another agent.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

# ENFORCE GPU CONSTRAINT BEFORE ANYTHING ELSE
from infra.gpu_constraint import enforce_gpu_constraint, get_allowed_device
enforce_gpu_constraint()

logger = logging.getLogger(__name__)


def prepare_soccernet_tracking(data_dir: str = "data/soccernet_tracking") -> str:
    """Prepare SoccerNet Tracking dataset in YOLO format.

    Converts SoccerNet tracking annotations to YOLO training format:
    - train/images/ and train/labels/
    - val/images/ and val/labels/
    - data.yaml with class names

    Args:
        data_dir: Root directory of SoccerNet Tracking data.

    Returns:
        Path to the data.yaml file.
    """
    data_path = Path(data_dir)
    output_path = Path("data/yolo_soccer")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    for split in ["train", "val"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Process SoccerNet tracking data
    # SoccerNet format: frame_{idx}.jpg + tracking_{idx}.txt
    # Each tracking line: class_id confidence x_center y_center width height

    video_dirs = sorted((data_path / "tracks").glob("*"))
    if not video_dirs:
        logger.warning(
            "No SoccerNet data found at %s. Creating synthetic dataset for testing.",
            data_dir,
        )
        return _create_synthetic_dataset(output_path)

    for i, video_dir in enumerate(video_dirs):
        split = "train" if i < int(len(video_dirs) * 0.8) else "val"
        frames = sorted(video_dir.glob("*.jpg"))
        labels = sorted(video_dir.glob("*.txt"))

        for frame_path, label_path in zip(frames, labels):
            # Copy frame
            out_frame = output_path / split / "images" / frame_path.name
            shutil.copy2(frame_path, out_frame)

            # Convert label to YOLO format
            out_label = output_path / split / "labels" / label_path.with_suffix(".txt").name
            _convert_soccernet_label(label_path, out_label)

    # Create data.yaml
    yaml_content = f"""train: train/images
val: val/images

nc: 3
names: ['player', 'referee', 'ball']
"""
    yaml_path = output_path / "data.yaml"
    yaml_path.write_text(yaml_content)

    logger.info("SoccerNet Tracking prepared at %s", output_path)
    return str(yaml_path)


def _convert_soccernet_label(
    soccernet_path: Path, output_path: Path
) -> None:
    """Convert SoccerNet tracking annotation to YOLO format."""
    lines = []
    with open(soccernet_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                # SoccerNet: class_id conf x_center y_center w h
                cls_id = int(parts[0])
                x, y, w, h = parts[2], parts[3], parts[4], parts[5]
                # YOLO format: class_id x_center y_center w h (normalized)
                lines.append(f"{cls_id} {x} {y} {w} {h}")

    output_path.write_text("\n".join(lines))


def _create_synthetic_dataset(output_path: Path) -> str:
    """Create a minimal synthetic dataset for testing pipeline."""
    import numpy as np
    from PIL import Image, ImageDraw

    for split, count in [("train", 100), ("val", 20)]:
        for i in range(count):
            # Create random frame
            img = Image.fromarray(
                np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            )
            img.save(output_path / split / "images" / f"frame_{i:04d}.jpg")

            # Create random labels (3 classes, 0-22 players)
            labels = []
            n_players = np.random.randint(10, 23)
            for _ in range(n_players):
                cls = np.random.choice([0, 1], p=[0.9, 0.1])  # mostly players
                x, y, w, h = np.random.uniform(0, 1, 4)
                w, h = max(w, 0.02), max(h, 0.04)
                x, y = np.clip(x, w / 2, 1 - w / 2), np.clip(y, h / 2, 1 - h / 2)
                labels.append(f"{cls} {x:.4f} {y:.4f} {w:.4f} {h:.4f}")

            # Add ball occasionally
            if np.random.random() < 0.3:
                x, y, w, h = np.random.uniform(0, 1, 4)
                labels.append(f"2 {x:.4f} {y:.4f} {max(w, 0.01):.4f} {max(h, 0.01):.4f}")

            (output_path / split / "labels" / f"frame_{i:04d}.txt").write_text(
                "\n".join(labels)
            )

    yaml_content = f"""train: train/images
val: val/images

nc: 3
names: ['player', 'referee', 'ball']
"""
    yaml_path = output_path / "data.yaml"
    yaml_path.write_text(yaml_content)

    logger.info("Synthetic dataset created at %s", output_path)
    return str(yaml_path)


def train(
    data_yaml: str,
    model_name: str = "yolov10x.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 16,
    project: str = "runs/train",
    name: str = "yolov10_soccer",
) -> None:
    """Train YOLOv10-X on soccer player detection.

    Args:
        data_yaml: Path to dataset YAML file.
        model_name: Pretrained model to start from.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        project: Output directory for runs.
        name: Run name.
    """
    from ultralytics import YOLO

    device = get_allowed_device()  # cuda:0
    logger.info("Training on device: %s", device)

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=20,
        save=True,
        save_period=10,
        device=device,
        workers=8,
        optimizer="auto",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    # Export best model
    best_path = Path(project) / name / "weights" / "best.pt"
    output_path = Path("model/yolov10_soccer.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_path.exists():
        import shutil
        shutil.copy2(best_path, output_path)
        logger.info("Best model saved to %s", output_path)

    logger.info("Training complete. Results: %s", results)


def main() -> None:
    """CLI entry point for YOLOv10 training."""
    parser = argparse.ArgumentParser(description="Train YOLOv10 for soccer player detection")
    parser.add_argument("--data", default="data/soccernet_tracking", help="SoccerNet data dir")
    parser.add_argument("--model", default="yolov10x.pt", help="Pretrained model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="runs/train", help="Output dir")
    parser.add_argument("--name", default="yolov10_soccer", help="Run name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Verify GPU constraint before training
    from infra.gpu_constraint import verify_device
    import torch
    device = torch.device(get_allowed_device())
    verify_device(device)

    data_yaml = prepare_soccernet_tracking(args.data)
    train(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
