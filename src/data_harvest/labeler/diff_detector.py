"""Diff detector: compare pre/post frames to find changed regions → bbox."""

from __future__ import annotations

import cv2
import numpy as np

from data_harvest.core.types import BBoxCandidate


def diff_bboxes(
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    threshold: float = 0.02,
    contour_min_area: int = 100,
) -> list[BBoxCandidate]:
    """Detect changed regions between pre and post frames.

    Returns a list of BBoxCandidate sorted by area (largest first).
    """
    if pre_frame.shape != post_frame.shape:
        return []

    gray_pre = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY) if len(pre_frame.shape) == 3 else pre_frame
    gray_post = cv2.cvtColor(post_frame, cv2.COLOR_BGR2GRAY) if len(post_frame.shape) == 3 else post_frame

    diff = cv2.absdiff(gray_pre, gray_post)
    thresh_val = int(threshold * 255)
    _, binary = cv2.threshold(diff, max(thresh_val, 1), 255, cv2.THRESH_BINARY)

    # Morphological close to merge nearby changes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[BBoxCandidate] = []
    total_pixels = float(gray_pre.shape[0] * gray_pre.shape[1])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < contour_min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Confidence based on relative area of change
        conf = min(1.0, area / total_pixels * 100)
        candidates.append(
            BBoxCandidate(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
                signal="diff",
                confidence=conf,
            )
        )

    # Sort by area descending
    candidates.sort(
        key=lambda c: (c.x_max - c.x_min) * (c.y_max - c.y_min), reverse=True
    )
    return candidates


def diff_ratio(pre_frame: np.ndarray, post_frame: np.ndarray) -> float:
    """Return the fraction of pixels that differ between pre and post frames."""
    if pre_frame.shape != post_frame.shape:
        return 1.0
    gray_pre = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY) if len(pre_frame.shape) == 3 else pre_frame
    gray_post = cv2.cvtColor(post_frame, cv2.COLOR_BGR2GRAY) if len(post_frame.shape) == 3 else post_frame
    diff = cv2.absdiff(gray_pre, gray_post)
    changed = np.count_nonzero(diff > 5)
    total = float(diff.shape[0] * diff.shape[1])
    return changed / total if total > 0 else 0.0
