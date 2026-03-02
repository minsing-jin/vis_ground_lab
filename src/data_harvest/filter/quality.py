"""Quality filters: blur, dark overlay, and loading screen detection."""

from __future__ import annotations

import cv2
import numpy as np

from data_harvest.core.types import HarvestSample


def is_blurry(frame: np.ndarray, laplacian_threshold: float = 50.0) -> bool:
    """Return True if the frame is too blurry (low Laplacian variance)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(variance < laplacian_threshold)


def is_dark_overlay(frame: np.ndarray, threshold: float = 30.0) -> bool:
    """Return True if the frame is mostly dark (modal overlay, loading)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return bool(float(gray.mean()) < threshold)


def has_quality_issue(
    sample: HarvestSample,
    blur_threshold: float = 50.0,
    dark_threshold: float = 30.0,
) -> bool:
    """Return True if the sample has quality issues (blurry or dark)."""
    pre_path = sample.pre_frame_path
    if not pre_path.exists():
        return True

    frame = cv2.imread(str(pre_path))
    if frame is None:
        return True

    return is_blurry(frame, blur_threshold) or is_dark_overlay(frame, dark_threshold)
