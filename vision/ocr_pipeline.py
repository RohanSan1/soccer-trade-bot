"""OCR pipeline for scoreboard extraction.

Primary: PaddleOCR (99.6% accuracy on structured text)
Fallback: TrOCR (HuggingFace transformers)

Extracts: scores, clock, team names from configurable ROI regions.
Returns OCR confidence per field; blocks trading if confidence < threshold.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR extraction for a single field."""

    value: str
    confidence: float
    raw_texts: List[str]


@dataclass
class ScoreboardData:
    """Complete scoreboard extraction result."""

    home_score: OCRResult
    away_score: OCRResult
    clock: OCRResult
    home_team: OCRResult
    away_team: OCRResult
    overall_confidence: float
    extraction_time_ms: float


class OCRPipeline:
    """Extract scoreboard information from broadcast frames.

    Args:
        confidence_threshold: Minimum confidence to consider OCR reliable.
        score_roi: (x1, y1, x2, y2) bounding box for score region, or None for full frame.
        clock_roi: (x1, y1, x2, y2) bounding box for clock region, or None for full frame.
        use_gpu: Whether to use GPU acceleration for PaddleOCR.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        score_roi: Optional[Tuple[int, int, int, int]] = None,
        clock_roi: Optional[Tuple[int, int, int, int]] = None,
        use_gpu: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.score_roi = score_roi
        self.clock_roi = clock_roi
        self.use_gpu = use_gpu

        self._paddle_engine = None
        self._trocr_processor = None
        self._trocr_model = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy-load OCR engines on first use."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR
            self._paddle_engine = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=self.use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR initialized (GPU=%s)", self.use_gpu)
        except ImportError:
            logger.warning("PaddleOCR not available, falling back to TrOCR only")

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self._trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            logger.info("TrOCR fallback initialized")
        except ImportError:
            logger.warning("TrOCR not available")

        self._initialized = True

    def extract(self, frame: Image.Image) -> ScoreboardData:
        """Extract scoreboard data from a broadcast frame.

        Args:
            frame: PIL Image from the broadcast stream.

        Returns:
            ScoreboardData with all extracted fields and confidence scores.
        """
        self.initialize()
        start = time.time()

        # Crop ROIs if configured
        score_img = self._crop_roi(frame, self.score_roi)
        clock_img = self._crop_roi(frame, self.clock_roi)

        # Extract each field
        home_score = self._extract_field(score_img, "score")
        away_score = self._extract_field(score_img, "score")
        clock = self._extract_field(clock_img, "clock")
        home_team = self._extract_field(frame, "team_name")
        away_team = self._extract_field(frame, "team_name")

        # Calculate overall confidence
        scores = [home_score.confidence, away_score.confidence, clock.confidence]
        overall = sum(scores) / len(scores) if scores else 0.0

        elapsed_ms = (time.time() - start) * 1000

        return ScoreboardData(
            home_score=home_score,
            away_score=away_score,
            clock=clock,
            home_team=home_team,
            away_team=away_team,
            overall_confidence=overall,
            extraction_time_ms=elapsed_ms,
        )

    def _crop_roi(
        self, img: Image.Image, roi: Optional[Tuple[int, int, int, int]]
    ) -> Image.Image:
        """Crop image to ROI bounding box, or return full image."""
        if roi is None:
            return img
        return img.crop(roi)

    def _extract_field(self, img: Image.Image, field_type: str) -> OCRResult:
        """Extract a single field using PaddleOCR with TrOCR fallback."""
        # Try PaddleOCR first
        if self._paddle_engine is not None:
            result = self._paddle_ocr_extract(img, field_type)
            if result and result.confidence >= 0.5:
                return result

        # Fallback to TrOCR
        if self._trocr_model is not None:
            result = self._trocr_extract(img, field_type)
            if result:
                return result

        return OCRResult(value="", confidence=0.0, raw_texts=[])

    def _paddle_ocr_extract(
        self, img: Image.Image, field_type: str
    ) -> Optional[OCRResult]:
        """Run PaddleOCR on an image region."""
        try:
            img_array = np.array(img)
            result = self._paddle_engine.ocr(img_array, cls=True)

            if not result or not result[0]:
                return None

            texts = []
            confidences = []
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                texts.append(text)
                confidences.append(conf)

            if not texts:
                return None

            # Parse based on field type
            parsed = self._parse_field(texts, confidences, field_type)
            avg_conf = sum(confidences) / len(confidences)

            return OCRResult(
                value=parsed,
                confidence=avg_conf,
                raw_texts=texts,
            )

        except Exception as e:
            logger.debug("PaddleOCR extraction failed: %s", e)
            return None

    def _trocr_extract(
        self, img: Image.Image, field_type: str
    ) -> Optional[OCRResult]:
        """Run TrOCR on an image region."""
        try:
            import torch
            inputs = self._trocr_processor(images=img, return_tensors="pt")
            pixel_values = inputs.pixel_values

            with torch.no_grad():
                generated_ids = self._trocr_model.generate(pixel_values)

            text = self._trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            parsed = self._parse_field([text], [0.8], field_type)
            return OCRResult(
                value=parsed,
                confidence=0.8,  # TrOCR doesn't give per-char confidence
                raw_texts=[text],
            )

        except Exception as e:
            logger.debug("TrOCR extraction failed: %s", e)
            return None

    def _parse_field(
        self, texts: List[str], confidences: List[float], field_type: str
    ) -> str:
        """Parse OCR text based on expected field type."""
        combined = " ".join(texts).strip()

        if field_type == "score":
            return self._parse_score(combined)
        elif field_type == "clock":
            return self._parse_clock(combined)
        elif field_type == "team_name":
            return self._parse_team_name(combined)
        return combined

    def _parse_score(self, text: str) -> str:
        """Extract numeric score from OCR text."""
        # Look for digits, possibly with dash separator
        digits = re.findall(r"\d+", text)
        if digits:
            # Return the first standalone number found
            return digits[0]
        return ""

    def _parse_clock(self, text: str) -> str:
        """Extract match clock (minutes) from OCR text."""
        # Match patterns like "45:30", "45'", "45+2", "90:15"
        patterns = [
            r"(\d{1,3})\+?(\d{0,2})",  # 45+2, 90:15
            r"(\d{1,3})",              # Just a number
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                minutes = match.group(1)
                if int(minutes) <= 120:  # Sanity check
                    return minutes
        return ""

    def _parse_team_name(self, text: str) -> str:
        """Clean team name from OCR text."""
        # Remove common OCR artifacts
        cleaned = re.sub(r"[^a-zA-Z\s'-]", "", text)
        return cleaned.strip()

    def is_reliable(self, data: ScoreboardData) -> bool:
        """Check if OCR extraction meets confidence threshold."""
        return data.overall_confidence >= self.confidence_threshold

    def parse_clock_minutes(self, clock_text: str) -> int:
        """Parse clock text to integer minutes."""
        try:
            minutes = int(re.findall(r"\d+", clock_text)[0])
            return min(minutes, 120)  # Cap at 120
        except (IndexError, ValueError):
            return 0
