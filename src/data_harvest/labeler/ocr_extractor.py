"""OCR extractor: EasyOCR → text bboxes + semantic text."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from data_harvest.core.types import BBoxCandidate

logger = logging.getLogger(__name__)

# Module-level reader cache (EasyOCR is heavy to init)
_reader_cache: dict[str, Any] = {}


def _get_reader(languages: list[str], gpu: bool = False) -> Any:
    key = ",".join(sorted(languages)) + f"_gpu={gpu}"
    if key not in _reader_cache:
        import easyocr  # lazy import

        _reader_cache[key] = easyocr.Reader(languages, gpu=gpu)
    return _reader_cache[key]


def ocr_bboxes(
    frame: np.ndarray,
    languages: list[str] | None = None,
    gpu: bool = False,
    min_confidence: float = 0.3,
) -> list[BBoxCandidate]:
    """Run EasyOCR on the frame and return text bounding boxes.

    Each result from EasyOCR is (bbox_polygon, text, confidence).
    We convert to axis-aligned BBoxCandidate.
    """
    if languages is None:
        languages = ["en"]

    reader = _get_reader(languages, gpu=gpu)
    results = reader.readtext(frame)

    candidates: list[BBoxCandidate] = []
    for bbox_pts, text, conf in results:
        if conf < min_confidence:
            continue
        # bbox_pts is a list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        candidates.append(
            BBoxCandidate(
                x_min=float(min(xs)),
                y_min=float(min(ys)),
                x_max=float(max(xs)),
                y_max=float(max(ys)),
                signal="ocr",
                confidence=float(conf),
                semantic_text=text.strip(),
            )
        )

    return candidates


def ocr_nearest_to_click(
    candidates: list[BBoxCandidate],
    click_x: float,
    click_y: float,
    max_distance: float = 200.0,
) -> BBoxCandidate | None:
    """Return the OCR candidate closest to the click point."""
    best: BBoxCandidate | None = None
    best_dist = float("inf")
    for c in candidates:
        cx = (c.x_min + c.x_max) / 2
        cy = (c.y_min + c.y_max) / 2
        dist = ((cx - click_x) ** 2 + (cy - click_y) ** 2) ** 0.5
        if dist < best_dist and dist <= max_distance:
            best_dist = dist
            best = c
    return best
