"""Frame extraction from HLS/RTMP streams using FFmpeg.

Extracts 1 frame per second as JPEG for downstream vision processing.
Detects stream lag by comparing frame timestamp vs wall clock.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from an HLS or RTMP livestream via FFmpeg.

    Args:
        stream_url: HLS or RTMP URL to connect to.
        output_dir: Directory to write extracted frames.
        fps: Frames per second to extract (default 1).
        lag_threshold: Maximum acceptable lag in seconds before kill switch.
    """

    def __init__(
        self,
        stream_url: str,
        output_dir: str = "data/frames",
        fps: int = 1,
        lag_threshold: int = 8,
    ) -> None:
        self.stream_url = stream_url
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.lag_threshold = lag_threshold
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0
        self._lag_detected: bool = False

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start frame extraction in a background thread."""
        if self._running:
            logger.warning("FrameExtractor already running")
            return

        self._running = True
        self._lag_detected = False
        self._thread = threading.Thread(target=self._extract_loop, daemon=True)
        self._thread.start()
        logger.info("FrameExtractor started for %s", self.stream_url)

    def stop(self) -> None:
        """Stop frame extraction and kill FFmpeg process."""
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                self._process.kill()
            self._process = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("FrameExtractor stopped after %d frames", self._frame_count)

    def _extract_loop(self) -> None:
        """Main extraction loop: run FFmpeg and monitor output."""
        output_pattern = str(self.output_dir / "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", self.stream_url,
            "-vf", f"fps={self.fps}",
            "-q:v", "2",
            "-f", "image2",
            "-y",
            output_pattern,
        ]

        logger.info("Running FFmpeg: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            while self._running:
                # Check if FFmpeg is still alive
                if self._process.poll() is not None:
                    stderr = self._process.stderr
                    if stderr:
                        logger.error("FFmpeg exited with code %d: %s",
                                   self._process.returncode,
                                   stderr.read().decode(errors="replace")[:500])
                    break

                # Monitor for new frames
                self._check_lag()
                time.sleep(1)

        except Exception as e:
            logger.error("FrameExtractor error: %s", e)
        finally:
            self._running = False

    def _check_lag(self) -> None:
        """Check if stream lag exceeds threshold by comparing newest frame timestamp."""
        frames = sorted(self.output_dir.glob("frame_*.jpg"))
        if not frames:
            return

        newest = frames[-1]
        # Extract frame number from filename
        try:
            frame_num = int(newest.stem.split("_")[1])
        except (IndexError, ValueError):
            return

        # Estimate timestamp from frame number (1fps)
        frame_timestamp = frame_num / self.fps
        wall_clock = time.time()

        # Use file modification time as proxy
        file_mtime = newest.stat().st_mtime
        lag = wall_clock - file_mtime

        if lag > self.lag_threshold:
            if not self._lag_detected:
                logger.warning(
                    "Stream lag detected: %.1fs (threshold: %ds)",
                    lag, self.lag_threshold,
                )
                self._lag_detected = True
        else:
            self._lag_detected = False

        self._last_frame_time = file_mtime
        self._frame_count = frame_num

    def get_latest_frame(self) -> Optional[Path]:
        """Return path to the most recent extracted frame."""
        frames = sorted(self.output_dir.glob("frame_*.jpg"))
        return frames[-1] if frames else None

    @property
    def lag_detected(self) -> bool:
        """Whether stream lag exceeds threshold."""
        return self._lag_detected

    @property
    def is_running(self) -> bool:
        """Whether extraction is active."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Total frames extracted."""
        return self._frame_count

    def cleanup_frames(self, keep_last: int = 10) -> int:
        """Remove old frames, keeping the last N. Returns count removed."""
        frames = sorted(self.output_dir.glob("frame_*.jpg"))
        if len(frames) <= keep_last:
            return 0
        to_remove = frames[:-keep_last]
        for f in to_remove:
            f.unlink()
        return len(to_remove)
