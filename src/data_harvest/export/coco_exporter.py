"""COCO exporter for actionable grounding labels."""

from __future__ import annotations

import logging
from pathlib import Path

from data_harvest.core.types import HarvestSample, ReviewStatus

logger = logging.getLogger(__name__)


def export_coco(
    samples: list[HarvestSample],
    out_path: str | Path,
    category_names: list[str] | None = None,
) -> Path:
    """Export actionable page elements to COCO JSON."""
    from vis_ground_lab.data.coco import (
        add_annotation_entry,
        add_image_entry,
        empty_coco,
        register_categories,
        save_coco,
    )

    out_path = Path(out_path)
    category_names = category_names or ["interactive_element"]
    coco = empty_coco()
    cat_map = register_categories(coco, category_names)
    default_cat_id = cat_map[category_names[0]]

    ann_id = 1
    img_id = 1

    for sample in samples:
        if sample.review_status == ReviewStatus.rejected:
            continue
        label = sample.effective_label()
        if label is None or not sample.pre_frame_path.exists():
            continue

        add_image_entry(coco, sample.pre_frame_path, image_id=img_id)
        for element in label.elements:
            if element.enabled_state == "disabled":
                continue
            cat_id = default_cat_id
            if element.semantic_id and element.semantic_id in cat_map:
                cat_id = cat_map[element.semantic_id]
            add_annotation_entry(
                coco,
                annotation_id=ann_id,
                image_id=img_id,
                category_id=cat_id,
                bbox_xyxy=element.bbox_xyxy,
                score=element.confidence or label.confidence,
            )
            ann_id += 1
        img_id += 1

    save_coco(coco, out_path)
    logger.info("Exported COCO annotations to %s", out_path)
    return out_path
