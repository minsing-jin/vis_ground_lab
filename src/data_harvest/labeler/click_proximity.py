"""Click proximity labeler: contour analysis around the click point → bbox."""

from __future__ import annotations

import cv2
import numpy as np

from data_harvest.core.types import BBoxCandidate


def click_proximity_bbox(
    frame: np.ndarray,
    click_x: float,
    click_y: float,
    crop_radius: int = 80,
    contour_min_area: int = 100,
) -> BBoxCandidate | None:
    """Find the most likely UI element bbox near the click point.

    1. Crop a region around (click_x, click_y).
    2. Edge-detect (Canny) and find contours.
    3. Return the contour closest to the click that meets the minimum area.
    """
    h, w = frame.shape[:2]
    cx, cy = int(click_x), int(click_y)

    # Define crop window
    x1 = max(cx - crop_radius, 0)
    y1 = max(cy - crop_radius, 0)
    x2 = min(cx + crop_radius, w)
    y2 = min(cy + crop_radius, h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: BBoxCandidate | None = None
    best_dist = float("inf")

    # Click position relative to crop
    rel_cx = cx - x1
    rel_cy = cy - y1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < contour_min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Center of bounding rect
        cnt_cx = bx + bw / 2
        cnt_cy = by + bh / 2
        dist = ((cnt_cx - rel_cx) ** 2 + (cnt_cy - rel_cy) ** 2) ** 0.5

        if dist < best_dist:
            best_dist = dist
            # Convert back to absolute coords
            abs_x1 = float(bx + x1)
            abs_y1 = float(by + y1)
            abs_x2 = float(bx + bw + x1)
            abs_y2 = float(by + bh + y1)
            conf = max(0.0, min(1.0, 1.0 - dist / crop_radius))
            best = BBoxCandidate(
                x_min=abs_x1,
                y_min=abs_y1,
                x_max=abs_x2,
                y_max=abs_y2,
                signal="click_proximity",
                confidence=conf,
            )

    return best
