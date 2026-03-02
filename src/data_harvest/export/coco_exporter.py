"""COCO JSON exporter — reuses vis_ground_lab.data.coco utilities."""

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
    """Export approved/labeled samples to COCO JSON format.

    Reuses empty_coco, register_categories, add_image_entry, add_annotation_entry, save_coco
    from vis_ground_lab.data.coco.
    """
    from vis_ground_lab.data.coco import (
        empty_coco,
        register_categories,
        add_image_entry,
        add_annotation_entry,
        save_coco,
    )

    out_path = Path(out_path)

    if category_names is None:
        category_names = ["ui_element"]

    coco = empty_coco()
    cat_map = register_categories(coco, category_names)
    default_cat_id = cat_map[category_names[0]]

    ann_id = 1
    img_id = 1
    exported = 0

    for s in samples:
        if s.label is None:
            continue
        if s.review_status == ReviewStatus.rejected:
            continue
        if not s.pre_frame_path.exists():
            continue

        add_image_entry(coco, s.pre_frame_path, image_id=img_id)

        # Use review corrections if edited
        label = s.label
        if s.review_status == ReviewStatus.edited and s.review_corrections:
            bbox_xyxy = s.review_corrections.get("bbox_xyxy", label.bbox_xyxy)
        else:
            bbox_xyxy = label.bbox_xyxy

        # Determine category
        cat_id = default_cat_id
        if label.semantic_id and label.semantic_id in cat_map:
            cat_id = cat_map[label.semantic_id]

        add_annotation_entry(
            coco,
            annotation_id=ann_id,
            image_id=img_id,
            category_id=cat_id,
            bbox_xyxy=bbox_xyxy,
            score=label.confidence,
        )

        ann_id += 1
        img_id += 1
        exported += 1

    save_coco(coco, out_path)
    logger.info("Exported %d samples to COCO: %s", exported, out_path)
    return out_path
