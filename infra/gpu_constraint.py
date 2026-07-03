"""GPU isolation enforcement.

Hard stop: only CUDA devices 0 and 1 are allowed.
Devices 2 and 3 belong to another agent and must never be touched.
"""
from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)

ALLOWED_DEVICES = {0, 1}
FORBIDDEN_DEVICES = {2, 3}


def enforce_gpu_constraint() -> None:
    """Set CUDA_VISIBLE_DEVICES and verify compliance.

    Call this at the top of every training script BEFORE importing torch.
    Hard-stops the process if forbidden GPUs are detected.
    """
    # Force only CUDA 0 and 1 visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Verify after torch import
    try:
        import torch
        available = set(range(torch.cuda.device_count()))

        if available & FORBIDDEN_DEVICES:
            logger.critical(
                "HALT: Forbidden GPUs detected: %s. "
                "Only CUDA 0 and 1 are allowed.",
                available & FORBIDDEN_DEVICES,
            )
            sys.exit(1)

        if not (available & ALLOWED_DEVICES):
            logger.critical(
                "HALT: No allowed GPUs (0,1) found. Available: %s", available
            )
            sys.exit(1)

        logger.info(
            "GPU constraint enforced: using devices %s (forbidden: %s)",
            sorted(available & ALLOWED_DEVICES),
            sorted(FORBIDDEN_DEVICES),
        )

    except ImportError:
        logger.warning("torch not yet imported, constraint will be verified later")


def verify_device(device: "torch.device") -> bool:
    """Verify a torch device is on an allowed GPU.

    Args:
        device: torch.device to check.

    Returns:
        True if device is allowed.

    Raises:
        SystemExit: If device is on a forbidden GPU.
    """
    import torch

    if device.type != "cuda":
        return True

    gpu_id = device.index or 0

    if gpu_id in FORBIDDEN_DEVICES:
        logger.critical(
            "HALT: Attempted to use forbidden GPU %d. "
            "Only CUDA 0 and 1 are allowed.",
            gpu_id,
        )
        sys.exit(1)

    return True


def get_allowed_device() -> str:
    """Get a device string constrained to allowed GPUs.

    Returns:
        'cuda:0' or 'cpu' if no GPUs available.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"

    # CUDA_VISIBLE_DEVICES is already set by enforce_gpu_constraint()
    # So device 0 in torch = physical GPU 0
    return "cuda:0"


def get_allowed_devices() -> list[str]:
    """Get list of allowed device strings.

    Returns:
        ['cuda:0', 'cuda:1'] or ['cpu'] if no GPUs.
    """
    import torch

    if not torch.cuda.is_available():
        return ["cpu"]

    count = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(min(count, 2))]
