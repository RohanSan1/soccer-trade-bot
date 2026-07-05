#!/usr/bin/env python3
"""OVH-side training wrapper with built-in hard stop.

Runs INSIDE the OVH container. Has its own timer that kills training
and saves whatever we have before the hard limit.

Usage (OVH command):
    python infra/ovh_train_with_timeout.py \
        --max-minutes 330 \
        --output /workspace/output/model \
        -- python -m model.train --data data/train.parquet --output /workspace/output/model --grid-search
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ovh-timeout] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_with_timeout(cmd: list, max_minutes: int, output_dir: str):
    """Run training command with a hard timeout. Saves partial results."""
    start = time.time()
    hard_stop = start + max_minutes * 60

    logger.info("=== OVH TRAIN WRAPPER ===")
    logger.info("Command: %s", " ".join(cmd))
    logger.info("Max minutes: %d", max_minutes)
    logger.info("Hard stop at: %s", time.strftime("%H:%M:%S", time.localtime(hard_stop)))
    logger.info("")

    # Start the training process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Monitor in background
    def _monitor():
        while proc.poll() is None:
            now = time.time()
            remaining = hard_stop - now
            if remaining <= 0:
                logger.info("")
                logger.info("*** HARD STOP REACHED — killing training (PID %d) ***", proc.pid)
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                return
            if remaining < 300:
                logger.info("[T+%.0fm] WARNING: %.0fs remaining — training will be killed",
                           (now - start) / 60, remaining)
            time.sleep(30)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    # Stream output
    try:
        for line in proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        logger.info("Interrupted — killing training")
        proc.kill()

    proc.wait()
    elapsed = time.time() - start

    logger.info("")
    logger.info("Training finished in %.0fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("Exit code: %d", proc.returncode)

    # Verify output
    output_path = Path(output_dir)
    if output_path.exists():
        files = list(output_path.iterdir())
        logger.info("Output files (%d):", len(files))
        for f in files:
            logger.info("  %s (%.1f KB)", f.name, f.stat().st_size / 1024)
    else:
        logger.warning("Output directory does not exist: %s", output_dir)

    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="OVH training wrapper with hard stop")
    parser.add_argument("--max-minutes", type=int, default=330,
                       help="Max training time in minutes (default: 5.5 hours)")
    parser.add_argument("--output", default="/workspace/output/model",
                       help="Output directory for model artifacts")
    parser.add_argument("command", nargs="*",
                       help="Command to run after '--'")
    args = parser.parse_args()

    if not args.command:
        # Default: build data then train
        cmd = [
            "sh", "-c",
            'python -c "from data.build_dataset import build_dataset; build_dataset(\'data/train.parquet\')"'
            " && python -m model.train --data data/train.parquet"
            f" --output {args.output} --grid-search"
        ]
    else:
        cmd = args.command

    rc = run_with_timeout(cmd, args.max_minutes, args.output)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
