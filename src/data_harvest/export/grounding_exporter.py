"""Grounding JSONL exporter for actionable page elements."""

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
    """Export page elements to grounding JSONL.

    One line per actionable element:
    {image_path, prompt, bbox, semantic_id, function_id, primitive_id, situation_id, ...}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            if sample.review_status == ReviewStatus.rejected:
                continue
            label = sample.effective_label()
            if label is None or not sample.pre_frame_path.exists():
                continue

            import cv2

            img = cv2.imread(str(sample.pre_frame_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            primitive_id = label.route_label.primitive_id if label.route_label else None
            situation_id = label.page.situation_id if label.page else None
            screen_type = label.page.screen_type if label.page else None

            for element in label.elements:
                if element.enabled_state == "disabled":
                    continue
                x1, y1, x2, y2 = element.bbox_xyxy
                if x2 <= x1 or y2 <= y1:
                    continue
                norm_bbox = [
                    int(x1 / w * normalizing_range),
                    int(y1 / h * normalizing_range),
                    int(x2 / w * normalizing_range),
                    int(y2 / h * normalizing_range),
                ]
                prompt = element.semantic_text or element.semantic_id or element.function_id or primitive_id or "interactive_element"
                record = {
                    "image_path": str(sample.pre_frame_path),
                    "prompt": prompt,
                    "bbox": norm_bbox,
                    "semantic_id": element.semantic_id,
                    "semantic_text": element.semantic_text,
                    "function_id": element.function_id,
                    "primitive_id": primitive_id,
                    "available_actions": element.available_actions,
                    "hotkeys": element.hotkeys,
                    "enabled_state": element.enabled_state,
                    "is_route_target": element.is_route_target,
                    "situation_id": situation_id,
                    "screen_type": screen_type,
                    "confidence": element.confidence or label.confidence,
                    "pre_path": str(sample.pre_frame_path),
                    "post_path": str(sample.post_frame_path),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

    logger.info("Exported %d elements to grounding JSONL: %s", exported, out_path)
    return out_path
