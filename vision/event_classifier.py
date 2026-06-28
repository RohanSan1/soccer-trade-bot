"""Event classification using CLIP ViT-L/14 zero-shot.

Classifies broadcast frames into game events:
  goal, red_card, var, penalty, normal_play, celebration, substitution

High-confidence events (>0.75) trigger immediate re-evaluation of positions.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

EVENT_LABELS = [
    "goal scored",
    "red card shown",
    "VAR review active",
    "penalty awarded",
    "normal play",
    "celebration",
    "substitution",
]

CRITICAL_EVENTS = {"goal scored", "red card shown"}


@dataclass
class EventResult:
    """Result from event classification."""

    label: str
    confidence: float
    all_probs: Dict[str, float]
    inference_time_ms: float


class EventClassifier:
    """CLIP-based zero-shot event classifier for soccer broadcasts.

    Args:
        model_name: CLIP model variant (default: ViT-L/14).
        confidence_threshold: Minimum confidence to trigger re-evaluation.
        device: torch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        confidence_threshold: float = 0.75,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy-load CLIP model on first use."""
        if self._initialized:
            return

        try:
            import clip
            self._model, self._preprocess = clip.load(
                self.model_name, device=self.device
            )
            self._tokenizer = clip.tokenize
            logger.info("CLIP %s loaded on %s", self.model_name, self.device)
        except Exception as e:
            logger.error("Failed to load CLIP model: %s", e)
            raise

        self._initialized = True

    def classify(self, frame: Image.Image) -> EventResult:
        """Classify a single broadcast frame.

        Args:
            frame: PIL Image from the broadcast stream.

        Returns:
            EventResult with predicted label and confidence.
        """
        self.initialize()
        start = time.time()

        # Preprocess frame
        image_input = self._preprocess(frame).unsqueeze(0).to(self.device)

        # Tokenize text prompts
        text_inputs = self._tokenizer(EVENT_LABELS).to(self.device)

        # Run inference
        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(text_inputs)

            # Normalize and compute similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.T) * 100.0

            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        # Build result
        all_probs = {label: float(prob) for label, prob in zip(EVENT_LABELS, probs)}
        best_idx = int(np.argmax(probs))
        best_label = EVENT_LABELS[best_idx]
        best_conf = float(probs[best_idx])

        elapsed_ms = (time.time() - start) * 1000

        return EventResult(
            label=best_label,
            confidence=best_conf,
            all_probs=all_probs,
            inference_time_ms=elapsed_ms,
        )

    def is_critical(self, result: EventResult) -> bool:
        """Check if detected event requires immediate re-evaluation."""
        return (
            result.label in CRITICAL_EVENTS
            and result.confidence >= self.confidence_threshold
        )

    def classify_batch(self, frames: List[Image.Image]) -> List[EventResult]:
        """Classify multiple frames in a batch.

        Args:
            frames: List of PIL Images.

        Returns:
            List of EventResult for each frame.
        """
        self.initialize()
        if not frames:
            return []

        start = time.time()

        # Preprocess all frames
        image_inputs = torch.stack(
            [self._preprocess(f) for f in frames]
        ).to(self.device)

        text_inputs = self._tokenizer(EVENT_LABELS).to(self.device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_inputs)
            text_features = self._model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.T) * 100.0

            probs = logits.softmax(dim=-1).cpu().numpy()

        results = []
        for i in range(len(frames)):
            all_probs = {
                label: float(probs[i, j])
                for j, label in enumerate(EVENT_LABELS)
            }
            best_idx = int(np.argmax(probs[i]))
            results.append(
                EventResult(
                    label=EVENT_LABELS[best_idx],
                    confidence=float(probs[i, best_idx]),
                    all_probs=all_probs,
                    inference_time_ms=0,  # Will be set for batch
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        for r in results:
            r.inference_time_ms = elapsed_ms / len(frames)

        return results
