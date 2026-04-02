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
    align: bool = False,
    signal_name: str = "diff",
) -> list[BBoxCandidate]:
    """Detect changed regions between pre and post frames.

    Returns a list of BBoxCandidate sorted by area (largest first).
    """
    if pre_frame.shape != post_frame.shape:
        return []

    gray_pre = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY) if len(pre_frame.shape) == 3 else pre_frame
    gray_post = cv2.cvtColor(post_frame, cv2.COLOR_BGR2GRAY) if len(post_frame.shape) == 3 else post_frame
    if align:
        gray_pre = _align_pre_to_post(gray_pre, gray_post)

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
                signal=signal_name,
                confidence=conf,
            )
        )

    # Sort by area descending
    candidates.sort(
        key=lambda c: (c.x_max - c.x_min) * (c.y_max - c.y_min), reverse=True
    )
    return candidates


def diff_ratio(pre_frame: np.ndarray, post_frame: np.ndarray, align: bool = False) -> float:
    """Return the fraction of pixels that differ between pre and post frames."""
    if pre_frame.shape != post_frame.shape:
        return 1.0
    gray_pre = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY) if len(pre_frame.shape) == 3 else pre_frame
    gray_post = cv2.cvtColor(post_frame, cv2.COLOR_BGR2GRAY) if len(post_frame.shape) == 3 else post_frame
    if align:
        gray_pre = _align_pre_to_post(gray_pre, gray_post)
    diff = cv2.absdiff(gray_pre, gray_post)
    changed = np.count_nonzero(diff > 5)
    total = float(diff.shape[0] * diff.shape[1])
    return changed / total if total > 0 else 0.0


def local_diff_bboxes(
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    click_x: float,
    click_y: float,
    radius: int = 180,
    threshold: float = 0.02,
    contour_min_area: int = 40,
    align: bool = True,
) -> list[BBoxCandidate]:
    """Diff around pointer neighborhood and map boxes to full-frame coordinates."""
    h, w = pre_frame.shape[:2]
    cx, cy = int(click_x), int(click_y)
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, w)
    y2 = min(cy + radius, h)
    if x2 <= x1 or y2 <= y1:
        return []

    pre_crop = pre_frame[y1:y2, x1:x2]
    post_crop = post_frame[y1:y2, x1:x2]
    cands = diff_bboxes(
        pre_crop,
        post_crop,
        threshold=threshold,
        contour_min_area=contour_min_area,
        align=align,
        signal_name="diff_local",
    )
    out: list[BBoxCandidate] = []
    for c in cands:
        out.append(
            BBoxCandidate(
                x_min=c.x_min + x1,
                y_min=c.y_min + y1,
                x_max=c.x_max + x1,
                y_max=c.y_max + y1,
                signal=c.signal,
                confidence=c.confidence,
                semantic_text=c.semantic_text,
            )
        )
    return out


def _align_pre_to_post(gray_pre: np.ndarray, gray_post: np.ndarray) -> np.ndarray:
    """Estimate translation and align pre frame to post frame."""
    pre_f = gray_pre.astype(np.float32)
    post_f = gray_post.astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(pre_f, post_f)
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        gray_pre,
        m,
        (gray_pre.shape[1], gray_pre.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned
