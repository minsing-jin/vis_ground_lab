"""Bottom-right ROI state dataset exporter."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2

from data_harvest.core.types import HarvestSample, ReviewStatus

logger = logging.getLogger(__name__)


def export_roi_state(
    samples: list[HarvestSample],
    out_dir: str | Path,
    roi_width_ratio: float = 0.28,
    roi_height_ratio: float = 0.22,
) -> Path:
    """Export bottom-right ROI crops + turn_state label CSV."""
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    exported = 0
    for s in samples:
        label = s.effective_label()
        if label is None or s.review_status == ReviewStatus.rejected:
            continue
        if not s.pre_frame_path.exists():
            continue

        img = cv2.imread(str(s.pre_frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        rw = max(1, int(w * roi_width_ratio))
        rh = max(1, int(h * roi_height_ratio))
        roi = img[h - rh : h, w - rw : w]

        out_path = images_dir / f"{s.sample_id}.png"
        cv2.imwrite(str(out_path), roi)
        exported += 1

        turn_state = "unknown"
        if s.review_corrections and s.review_corrections.get("turn_state"):
            turn_state = str(s.review_corrections["turn_state"])
        elif s.metadata and isinstance(s.metadata.get("turn_state"), str):
            turn_state = str(s.metadata["turn_state"])
        elif label.page and label.page.situation_id:
            turn_state = str(label.page.situation_id)
        elif label.route_label and label.route_label.primitive_id:
            turn_state = str(label.route_label.primitive_id)
        elif label.semantic_id:
            turn_state = label.semantic_id

        rows.append(
            {
                "sample_id": s.sample_id,
                "image_path": str(out_path),
                "turn_state": turn_state,
                "session_id": str(s.sample_dir.parent.parent.name),
            }
        )

    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "image_path", "turn_state", "session_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Session-level split files (single session still leak-safe for future merge)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    split_idx = int(len(rows) * 0.9)
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "turn_state", "session_id"])
        writer.writeheader()
        writer.writerows(rows[:split_idx])
    with open(val_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "turn_state", "session_id"])
        writer.writeheader()
        writer.writerows(rows[split_idx:])

    logger.info("Exported %d ROI state samples to %s", exported, out_dir)
    return out_dir
