"""CLIP ViT-L/14 fine-tuning on SoccerNet Action Spotting.

Fine-tunes CLIP for event classification on broadcast frames:
  goal, red_card, var, penalty, normal_play

GPU CONSTRAINT: Only CUDA 0 and 1 allowed. CUDA 2 and 3 belong to another agent.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ENFORCE GPU CONSTRAINT BEFORE TORCH IMPORT
from infra.gpu_constraint import enforce_gpu_constraint, get_allowed_device
enforce_gpu_constraint()

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

EVENT_CLASSES = [
    "goal scored",
    "red card shown",
    "VAR review active",
    "penalty awarded",
    "normal play",
]


class SoccerEventDataset(Dataset):
    """Dataset for soccer event classification from broadcast frames.

    Expects directory structure:
        data_dir/
            goal/        -> frame_*.jpg
            red_card/    -> frame_*.jpg
            var/         -> frame_*.jpg
            penalty/     -> frame_*.jpg
            normal/      -> frame_*.jpg
    """

    def __init__(self, data_dir: str, transform=None) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        class_to_idx = {
            "goal": 0,
            "red_card": 1,
            "var": 2,
            "penalty": 3,
            "normal": 4,
        }

        for class_name, idx in class_to_idx.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob("*.jpg")):
                    self.samples.append((img_path, idx))

        logger.info("Loaded %d samples from %s", len(self.samples), data_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_soccernet_actions(data_dir: str = "data/soccernet_actions") -> str:
    """Prepare SoccerNet Action Spotting data for CLIP fine-tuning.

    Args:
        data_dir: Root directory of SoccerNet action data.

    Returns:
        Path to prepared data directory.
    """
    data_path = Path(data_dir)
    output_path = Path("data/clip_soccer_data")

    if not data_path.exists():
        logger.warning(
            "SoccerNet action data not found at %s. Creating synthetic data.",
            data_dir,
        )
        return _create_synthetic_action_data(output_path)

    # Copy and organize data
    for class_name in ["goal", "red_card", "var", "penalty", "normal"]:
        src = data_path / class_name
        dst = output_path / class_name
        dst.mkdir(parents=True, exist_ok=True)

        if src.exists():
            for f in src.glob("*.jpg"):
                shutil.copy2(f, dst / f.name)

    return str(output_path)


def _create_synthetic_action_data(output_path: Path) -> str:
    """Create synthetic action data for pipeline testing."""
    output_path.mkdir(parents=True, exist_ok=True)

    class_names = ["goal", "red_card", "var", "penalty", "normal"]

    for class_name in class_names:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)

        for i in range(200):  # 200 images per class
            img = Image.fromarray(
                np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            )
            img.save(class_dir / f"frame_{i:04d}.jpg")

    logger.info("Synthetic action data created at %s", output_path)
    return str(output_path)


class CLIPEncoder(nn.Module):
    """CLIP encoder with classification head for event detection."""

    def __init__(self, clip_model, num_classes: int = 5) -> None:
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)
        self.freeze_clip()

    def freeze_clip(self) -> None:
        """Freeze CLIP backbone, only train classifier head."""
        for param in self.clip.parameters():
            param.requires_grad = False

    def unfreeze_clip(self, num_layers: int = 2) -> None:
        """Unfreeze last N layers of CLIP for fine-tuning."""
        # Unfreeze visual transformer last layers
        visual = self.clip.visual
        if hasattr(visual, "transformer"):
            layers = list(visual.transformer.resblocks)
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLIP encoder and classifier."""
        with torch.no_grad() if not self.clip.training else torch.enable_grad():
            features = self.clip.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

        return self.classifier(features.float())


def train_clip(
    data_dir: str,
    model_name: str = "ViT-L/14",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    unfreeze_after: int = 5,
    output_dir: str = "model/clip_soccer",
) -> None:
    """Train CLIP for soccer event classification.

    Args:
        data_dir: Directory with prepared action data.
        model_name: CLIP model variant.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        unfreeze_after: Epoch to unfreeze CLIP backbone.
        output_dir: Where to save the fine-tuned model.
    """
    import clip

    device = get_allowed_device()
    logger.info("Training CLIP on %s (GPU constraint: CUDA 0,1 only)", device)

    # Load CLIP
    clip_model, preprocess = clip.load(model_name, device=device)

    # Create dataset
    dataset = SoccerEventDataset(data_dir, transform=preprocess)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Create model
    model = CLIPEncoder(clip_model, num_classes=len(EVENT_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_loss = float("inf")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()

        # Unfreeze backbone after warmup
        if epoch == unfreeze_after:
            model.unfreeze_clip(num_layers=2)
            logger.info("Unfroze CLIP backbone at epoch %d", epoch)

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        logger.info(
            "Epoch %d/%d - Loss: %.4f - Accuracy: %.1f%%",
            epoch + 1, epochs, avg_loss, accuracy,
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save model state
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "clip_model_name": model_name,
                    "num_classes": len(EVENT_CLASSES),
                    "class_names": EVENT_CLASSES,
                },
                output_path / "clip_soccer.pt",
            )
            logger.info("Saved best model (loss=%.4f)", best_loss)

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "clip_model_name": model_name,
            "num_classes": len(EVENT_CLASSES),
            "class_names": EVENT_CLASSES,
        },
        output_path / "clip_soccer_final.pt",
    )

    logger.info("Training complete. Best loss: %.4f", best_loss)


def main() -> None:
    """CLI entry point for CLIP fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP for soccer event classification"
    )
    parser.add_argument("--data", default="data/soccernet_actions", help="Action data dir")
    parser.add_argument("--model", default="ViT-L/14", help="CLIP model variant")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--unfreeze-after", type=int, default=5, help="Epoch to unfreeze backbone")
    parser.add_argument("--output", default="model/clip_soccer", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Verify GPU constraint
    from infra.gpu_constraint import verify_device
    device = torch.device(get_allowed_device())
    verify_device(device)

    data_dir = prepare_soccernet_actions(args.data)
    train_clip(
        data_dir=data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_after=args.unfreeze_after,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
