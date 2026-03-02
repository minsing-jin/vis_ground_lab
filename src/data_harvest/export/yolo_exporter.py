"""YOLO txt + data.yaml exporter."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml

from data_harvest.core.types import HarvestSample, ReviewStatus

logger = logging.getLogger(__name__)


def export_yolo(
    samples: list[HarvestSample],
    out_dir: str | Path,
    category_names: list[str] | None = None,
    train_ratio: float = 0.9,
) -> Path:
    """Export samples to YOLO format (images/ + labels/ + data.yaml).

    Each label file contains one line per bbox:
      class_id center_x center_y width height  (all normalized 0-1)
    """
    out_dir = Path(out_dir)
    if category_names is None:
        category_names = ["ui_element"]

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cat_map = {name: i for i, name in enumerate(category_names)}
    exported = 0

    valid_samples: list[HarvestSample] = []
    for s in samples:
        if s.label is None or s.review_status == ReviewStatus.rejected:
            continue
        if not s.pre_frame_path.exists():
            continue
        valid_samples.append(s)

    for s in valid_samples:
        label = s.label
        assert label is not None

        # Copy image
        dst_img = images_dir / f"{s.sample_id}.png"
        shutil.copy2(s.pre_frame_path, dst_img)

        # Get image dimensions
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

        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        class_id = 0
        if label.semantic_id and label.semantic_id in cat_map:
            class_id = cat_map[label.semantic_id]

        label_path = labels_dir / f"{s.sample_id}.txt"
        label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        exported += 1

    # Write data.yaml
    split_idx = int(len(valid_samples) * train_ratio)
    train_ids = [s.sample_id for s in valid_samples[:split_idx]]
    val_ids = [s.sample_id for s in valid_samples[split_idx:]]

    # Write train.txt and val.txt
    (out_dir / "train.txt").write_text(
        "\n".join(f"images/{sid}.png" for sid in train_ids) + "\n"
    )
    (out_dir / "val.txt").write_text(
        "\n".join(f"images/{sid}.png" for sid in val_ids) + "\n"
    )

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(category_names),
        "names": category_names,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info("Exported %d samples to YOLO: %s", exported, out_dir)
    return out_dir
