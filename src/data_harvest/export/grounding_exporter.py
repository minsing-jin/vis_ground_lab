"""Grounding JSONL exporter for visual grounding training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_harvest.core.types import HarvestSample, ReviewStatus

logger = logging.getLogger(__name__)


def export_grounding(
    samples: list[HarvestSample],
    out_path: str | Path,
    normalizing_range: int = 1000,
) -> Path:
    """Export samples to grounding JSONL format.

    Each line: {image_path, prompt, bbox, action, semantic_id, confidence, pre_path, post_path}
    Bbox is normalized to [0, normalizing_range].
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            if s.label is None or s.event is None:
                continue
            if s.review_status == ReviewStatus.rejected:
                continue
            if not s.pre_frame_path.exists():
                continue

            label = s.label
            event = s.event

            # Get image dimensions for normalization
            import cv2

            img = cv2.imread(str(s.pre_frame_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Use corrected bbox if edited
            if s.review_status == ReviewStatus.edited and s.review_corrections:
                bbox = s.review_corrections.get("bbox_xyxy", label.bbox_xyxy)
            else:
                bbox = label.bbox_xyxy

            # Normalize bbox
            x1, y1, x2, y2 = bbox
            norm_bbox = [
                int(x1 / w * normalizing_range),
                int(y1 / h * normalizing_range),
                int(x2 / w * normalizing_range),
                int(y2 / h * normalizing_range),
            ]

            # Build prompt
            prompt = label.semantic_text or event.action.value
            if label.semantic_id:
                prompt = f"{label.semantic_id}: {prompt}"

            record = {
                "image_path": str(s.pre_frame_path),
                "prompt": prompt,
                "bbox": norm_bbox,
                "action": event.action.value,
                "semantic_id": label.semantic_id,
                "confidence": label.confidence,
                "pre_path": str(s.pre_frame_path),
                "post_path": str(s.post_frame_path),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1

    logger.info("Exported %d samples to grounding JSONL: %s", exported, out_path)
    return out_path
