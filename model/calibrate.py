"""Isotonic regression probability calibration.

Calibrates raw XGBoost/LightGBM probabilities to true probabilities.
Calibration is mandatory — raw tree model outputs are overconfident.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Per-class isotonic regression calibrator.

    Calibrates each outcome probability independently using
    isotonic regression on a held-out calibration set.
    """

    def __init__(self) -> None:
        self._calibrators: List[IsotonicRegression] = []
        self._is_fitted = False

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        """Fit calibrators on validation data.

        Args:
            y_true: True labels (0, 1, 2).
            y_prob: Raw model probabilities (N, 3).
        """
        n_classes = y_prob.shape[1]
        self._calibrators = []

        for c in range(n_classes):
            # Binary target: is this class the true outcome?
            binary_target = (y_true == c).astype(float)

            calibrator = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
            )
            calibrator.fit(y_prob[:, c], binary_target)
            self._calibrators.append(calibrator)

        self._is_fitted = True
        logger.info("Calibrator fitted on %d samples, %d classes", len(y_true), n_classes)

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities.

        Args:
            y_prob: Raw model probabilities (N, 3).

        Returns:
            Calibrated probabilities (N, 3), normalized to sum to 1.
        """
        if not self._is_fitted:
            logger.warning("Calibrator not fitted, returning raw probabilities")
            return y_prob

        calibrated = np.zeros_like(y_prob)
        for c, calibrator in enumerate(self._calibrators):
            calibrated[:, c] = calibrator.predict(y_prob[:, c])

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        calibrated = calibrated / row_sums

        return calibrated

    def predict_single(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate a single probability vector.

        Args:
            probs: Raw probabilities (3,).

        Returns:
            Calibrated probabilities (3,).
        """
        return self.predict(probs.reshape(1, -1))[0]

    def save(self, path: str = "model/calibrator.pkl") -> None:
        """Save calibrator to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._calibrators, path)
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: str = "model/calibrator.pkl") -> "ProbabilityCalibrator":
        """Load calibrator from disk."""
        calibrator = cls()
        calibrator._calibrators = joblib.load(path)
        calibrator._is_fitted = True
        logger.info("Calibrator loaded from %s", path)
        return calibrator
