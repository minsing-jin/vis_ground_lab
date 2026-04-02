"""YOLO exporter for actionable grounding labels."""

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
    """Export page elements to YOLO format."""
    out_dir = Path(out_dir)
    category_names = category_names or ["interactive_element"]

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cat_map = {name: i for i, name in enumerate(category_names)}
    valid_samples: list[HarvestSample] = []

    for sample in samples:
        if sample.review_status == ReviewStatus.rejected:
            continue
        label = sample.effective_label()
        if label is None or not sample.pre_frame_path.exists():
            continue
        valid_samples.append(sample)

        shutil.copy2(sample.pre_frame_path, images_dir / f"{sample.sample_id}.png")

        import cv2

        img = cv2.imread(str(sample.pre_frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines: list[str] = []
        for element in label.elements:
            if element.enabled_state == "disabled":
                continue
            x1, y1, x2, y2 = element.bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            class_id = 0
            if element.semantic_id and element.semantic_id in cat_map:
                class_id = cat_map[element.semantic_id]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (labels_dir / f"{sample.sample_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    split_idx = int(len(valid_samples) * train_ratio)
    train_ids = [sample.sample_id for sample in valid_samples[:split_idx]]
    val_ids = [sample.sample_id for sample in valid_samples[split_idx:]]

    (out_dir / "train.txt").write_text("\n".join(f"images/{sid}.png" for sid in train_ids) + ("\n" if train_ids else ""))
    (out_dir / "val.txt").write_text("\n".join(f"images/{sid}.png" for sid in val_ids) + ("\n" if val_ids else ""))

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(category_names),
        "names": category_names,
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info("Exported %d samples to YOLO: %s", len(valid_samples), out_dir)
    return out_dir
