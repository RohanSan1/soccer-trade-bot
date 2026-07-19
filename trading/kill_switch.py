"""Kill switch for halting trading under adverse conditions.

Triggers:
1. Stream lag > 8 seconds
2. OCR confidence < 0.70 for 5 consecutive reads
3. API error count > 3 in last 60 seconds
4. Bankroll drawdown > 20% in single session
5. Any unhandled exception in signal loop
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchState:
    """Current state of kill switch monitoring."""

    is_halted: bool = False
    halt_reason: Optional[str] = None
    halt_timestamp: float = 0.0

    # Tracking
    consecutive_ocr_fails: int = 0
    api_error_count: int = 0
    api_error_window_start: float = 0.0
    peak_bankroll: float = 0.0
    current_bankroll: float = 0.0

    # Stream lag
    last_frame_timestamp: float = 0.0
    stream_lag_seconds: float = 0.0


class KillSwitch:
    """Monitors conditions and halts trading when triggered.

    Args:
        stream_lag_max: Maximum acceptable stream lag in seconds.
        ocr_fail_threshold: Consecutive OCR fails before halt.
        api_error_threshold: API errors in window before halt.
        api_error_window: Time window for API error counting.
        drawdown_threshold: Maximum drawdown percentage.
    """

    def __init__(
        self,
        stream_lag_max: int = 8,
        ocr_fail_threshold: int = 5,
        ocr_confidence_threshold: float = 0.70,
        api_error_threshold: int = 3,
        api_error_window: int = 60,
        drawdown_threshold: float = 0.20,
    ) -> None:
        self.stream_lag_max = stream_lag_max
        self.ocr_fail_threshold = ocr_fail_threshold
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.api_error_threshold = api_error_threshold
        self.api_error_window = api_error_window
        self.drawdown_threshold = drawdown_threshold
        self.state = KillSwitchState()

    def check_all(
        self,
        stream_lag: float = 0.0,
        ocr_confidence: float = 1.0,
        current_bankroll: Optional[float] = None,
    ) -> bool:
        """Check all kill switch conditions.

        Args:
            stream_lag: Current stream lag in seconds.
            ocr_confidence: Latest OCR confidence score.
            current_bankroll: Current bankroll (optional).

        Returns:
            True if any condition triggers halt.
        """
        if self.state.is_halted:
            return True

        # Check stream lag
        if self.check_stream_lag(stream_lag):
            return True

        # Check OCR confidence
        if self.check_ocr_confidence(ocr_confidence):
            return True

        # Check API errors
        if self.check_api_errors():
            return True

        # Check bankroll drawdown
        if current_bankroll is not None:
            if self.check_drawdown(current_bankroll):
                return True

        return False

    def check_stream_lag(self, lag_seconds: float) -> bool:
        """Check if stream lag exceeds threshold."""
        self.state.stream_lag_seconds = lag_seconds

        if lag_seconds > self.stream_lag_max:
            self._halt(f"Stream lag {lag_seconds:.1f}s > {self.stream_lag_max}s")
            return True
        return False

    def check_ocr_confidence(self, confidence: float) -> bool:
        """Check consecutive OCR failures."""
        if confidence < self.ocr_confidence_threshold:
            self.state.consecutive_ocr_fails += 1
        else:
            self.state.consecutive_ocr_fails = 0

        if self.state.consecutive_ocr_fails >= self.ocr_fail_threshold:
            self._halt(
                f"OCR confidence < {self.ocr_confidence_threshold} for "
                f"{self.state.consecutive_ocr_fails} consecutive reads"
            )
            return True
        return False

    def check_api_errors(self) -> bool:
        """Check API error rate."""
        now = time.time()

        # Reset window if expired
        if now - self.state.api_error_window_start > self.api_error_window:
            self.state.api_error_count = 0
            self.state.api_error_window_start = now

        if self.state.api_error_count >= self.api_error_threshold:
            self._halt(
                f"API error count {self.state.api_error_count} in last "
                f"{self.api_error_window}s"
            )
            return True
        return False

    def check_drawdown(self, current_bankroll: float) -> bool:
        """Check bankroll drawdown from peak."""
        self.state.current_bankroll = current_bankroll

        if current_bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = current_bankroll

        if self.state.peak_bankroll > 0:
            drawdown = (self.state.peak_bankroll - current_bankroll) / self.state.peak_bankroll
            if drawdown >= self.drawdown_threshold:
                self._halt(
                    f"Bankroll drawdown {drawdown:.1%} >= {self.drawdown_threshold:.1%}"
                )
                return True
        return False

    def record_api_error(self) -> None:
        """Record an API error for rate tracking."""
        now = time.time()
        if now - self.state.api_error_window_start > self.api_error_window:
            self.state.api_error_count = 0
            self.state.api_error_window_start = now
        self.state.api_error_count += 1

    def record_exception(self, exception: Exception) -> None:
        """Record an unhandled exception and halt."""
        self._halt(f"Unhandled exception: {type(exception).__name__}: {exception}")

    def _halt(self, reason: str) -> None:
        """Trigger halt."""
        if not self.state.is_halted:
            self.state.is_halted = True
            self.state.halt_reason = reason
            self.state.halt_timestamp = time.time()
            logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def reset(self) -> None:
        """Reset kill switch (use with extreme caution)."""
        logger.warning(
            "Kill switch reset (was halted: %s)", self.state.halt_reason
        )
        self.state = KillSwitchState()

    @property
    def is_halted(self) -> bool:
        """Whether trading is halted."""
        return self.state.is_halted

    @property
    def halt_reason(self) -> Optional[str]:
        """Reason for halt, if any."""
        return self.state.halt_reason
