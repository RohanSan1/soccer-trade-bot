"""Player detection using YOLOv10-X fine-tuned on SoccerNet.

Counts players in attacking third to compute pressure zone signal.
Output: pressure_score (0.0-1.0) indicating attacking dominance.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PlayerDetection:
    """Single player detection result."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int  # 0=player, 1=referee, 2=ball
    zone: str  # 'attacking_third', 'middle_third', 'defending_third'


@dataclass
class PressureResult:
    """Result from pressure zone analysis."""

    total_players: int
    home_attacking: int
    away_attacking: int
    pressure_score: float  # 0.0-1.0, higher = home more dominant
    inference_time_ms: float


class PlayerDetector:
    """YOLOv10-based player detector for pressure zone analysis.

    Args:
        model_path: Path to fine-tuned YOLOv10 model weights.
        confidence_threshold: Minimum detection confidence.
        device: torch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_path: str = "model/yolov10_soccer.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy-load YOLO model on first use."""
        if self._initialized:
            return

        try:
            from ultralytics import YOLO

            if self.model_path.exists():
                self._model = YOLO(str(self.model_path))
                logger.info("Loaded fine-tuned YOLOv10 from %s", self.model_path)
            else:
                # Fall back to pretrained model
                self._model = YOLO("yolov10x.pt")
                logger.warning(
                    "Fine-tuned model not found at %s, using pretrained YOLOv10-X",
                    self.model_path,
                )
        except ImportError:
            logger.error("ultralytics not installed")
            raise

        self._initialized = True

    def detect(self, frame: Image.Image) -> PressureResult:
        """Detect players and compute pressure zone signal.

        Args:
            frame: PIL Image from the broadcast stream (full pitch view).

        Returns:
            PressureResult with player counts and pressure score.
        """
        self._model  # Ensure initialized
        self.initialize()
        start = time.time()

        frame_array = np.array(frame)
        h, w = frame_array.shape[:2]

        # Run detection
        results = self._model(frame_array, verbose=False)

        home_attacking = 0
        away_attacking = 0
        total_players = 0

        # Define zones (top 30% = away attacking, bottom 30% = home attacking)
        attacking_top = int(h * 0.3)
        defending_top = int(h * 0.7)

        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                if cls_id > 1:  # Skip non-player detections
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_y = (y1 + y2) // 2

                total_players += 1

                # Classify zone based on vertical position
                if center_y < attacking_top:
                    # Top of frame = away team attacking
                    away_attacking += 1
                elif center_y > defending_top:
                    # Bottom of frame = home team attacking
                    home_attacking += 1

        # Compute pressure score
        if total_players > 0:
            # Normalize: more home players in attacking third = higher pressure
            pressure = home_attacking / max(total_players, 1)
            # Scale to 0-1 range (max expected ~10 players in attacking third)
            pressure_score = min(pressure * 3, 1.0)
        else:
            pressure_score = 0.5  # Neutral if no detections

        elapsed_ms = (time.time() - start) * 1000

        return PressureResult(
            total_players=total_players,
            home_attacking=home_attacking,
            away_attacking=away_attacking,
            pressure_score=pressure_score,
            inference_time_ms=elapsed_ms,
        )

    def detect_players(self, frame: Image.Image) -> List[PlayerDetection]:
        """Get detailed player detections (for debugging/analysis).

        Args:
            frame: PIL Image from the broadcast stream.

        Returns:
            List of PlayerDetection with bbox, confidence, class, and zone.
        """
        self.initialize()
        frame_array = np.array(frame)
        h, w = frame_array.shape[:2]

        results = self._model(frame_array, verbose=False)
        detections = []

        attacking_top = int(h * 0.3)
        defending_top = int(h * 0.7)

        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_y = (y1 + y2) // 2

                if center_y < attacking_top:
                    zone = "attacking_third"
                elif center_y > defending_top:
                    zone = "defending_third"
                else:
                    zone = "middle_third"

                detections.append(
                    PlayerDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        class_id=cls_id,
                        zone=zone,
                    )
                )

        return detections
