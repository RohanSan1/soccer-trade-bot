"""Real-time inference for win probability prediction.

Loads trained ensemble model and predicts [p_home, p_draw, p_away]
for each GameState update.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from model.calibrate import ProbabilityCalibrator
from model.features import FEATURE_NAMES, state_to_array
from vision.game_state import GameState

logger = logging.getLogger(__name__)


class WinPredictor:
    """Real-time win probability predictor using trained ensemble.

    Args:
        model_dir: Directory containing model artifacts.
        device: Not used for tree models, kept for API consistency.
    """

    def __init__(
        self,
        model_dir: str = "model",
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self._ensemble = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy-load model on first use."""
        if self._initialized:
            return

        from model.train import SoccerEnsemble

        self._ensemble = SoccerEnsemble.load(str(self.model_dir))
        self._initialized = True
        logger.info("WinPredictor initialized from %s", self.model_dir)

    def predict(self, state: GameState) -> Tuple[float, float, float]:
        """Predict win probabilities for current game state.

        Args:
            state: Current GameState snapshot.

        Returns:
            Tuple of (p_home, p_draw, p_away) — calibrated probabilities.
        """
        self.initialize()
        start = time.time()

        # Convert state to feature array
        X = state_to_array(state)

        # Run ensemble prediction
        probs = self._ensemble.predict_single(X)

        elapsed_ms = (time.time() - start) * 1000
        logger.debug(
            "Prediction: home=%.3f draw=%.3f away=%.3f (%.1fms)",
            probs[0], probs[1], probs[2], elapsed_ms,
        )

        return float(probs[0]), float(probs[1]), float(probs[2])

    def predict_batch(self, states: list[GameState]) -> np.ndarray:
        """Predict for multiple game states.

        Args:
            states: List of GameState snapshots.

        Returns:
            Array of shape (N, 3) with probabilities.
        """
        self.initialize()

        if not states:
            return np.zeros((0, 3))

        X = np.vstack([state_to_array(s) for s in states])
        return self._ensemble.predict(X)

    def get_confidence(self, probs: Tuple[float, float, float]) -> float:
        """Get prediction confidence (max probability).

        Args:
            probs: (p_home, p_draw, p_away).

        Returns:
            Confidence score (0.0-1.0).
        """
        return max(probs)

    def is_tradable(self, state: GameState) -> bool:
        """Check if current state is suitable for trading.

        Returns False if:
        - OCR not reliable
        - Clock > 85 minutes (final 5 min skip)
        - Prediction confidence too low
        """
        if not state.ocr_reliable:
            return False

        if state.clock_minutes > 85:
            return False

        probs = self.predict(state)
        confidence = self.get_confidence(probs)

        return confidence >= 0.60  # Minimum confidence to trade
