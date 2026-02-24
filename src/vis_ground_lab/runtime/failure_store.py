"""Persist failure samples for review and retraining."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from vis_ground_lab.base import ActionableElement
from vis_ground_lab.data.coco import (
    add_annotation_entry,
    add_image_entry,
    empty_coco,
    register_categories,
    save_coco,
)


@dataclass(frozen=True)
class FailureSample:
    """A frame where the model underperformed or drift was detected."""

    frame_id: str
    image_path: str
    timestamp_ms: float | None
    elements: tuple[ActionableElement, ...]
    failure_reason: str  # "low_confidence"|"drift_detected"|"action_failed"|"user_report"
    observed_at: str


class FailureStore:
    """File-backed store for failure samples, with COCO export."""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.store_dir / "failures.jsonl"
        self._images_dir = self.store_dir / "images"
        self._images_dir.mkdir(exist_ok=True)

    def save_failure(self, sample: FailureSample, image: Image.Image | None = None) -> None:
        """Persist a failure sample and optionally save its image."""
        record = {
            "frame_id": sample.frame_id,
            "image_path": sample.image_path,
            "timestamp_ms": sample.timestamp_ms,
            "failure_reason": sample.failure_reason,
            "observed_at": sample.observed_at,
            "elements": [
                {
                    "class_name": e.class_name,
                    "bbox": [e.bbox.x_min, e.bbox.y_min, e.bbox.x_max, e.bbox.y_max],
                    "score": e.score,
                    "center": list(e.center),
                    "semantic_id": e.semantic_id,
                    "affordances": list(e.affordances),
                    "element_type": e.element_type,
                }
                for e in sample.elements
            ],
        }

        with self._index_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        if image is not None:
            img_path = self._images_dir / f"{sample.frame_id}.png"
            image.save(img_path)

    def load_failures(self, limit: int | None = None) -> list[FailureSample]:
        """Load failure samples from the index file."""
        if not self._index_file.exists():
            return []

        samples: list[FailureSample] = []
        with self._index_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                from vis_ground_lab.base import BoundingBox

                elements = tuple(
                    ActionableElement(
                        class_name=e["class_name"],
                        bbox=BoundingBox(*e["bbox"]),
                        score=e["score"],
                        center=tuple(e["center"]),
                        semantic_id=e["semantic_id"],
                        affordances=tuple(e["affordances"]),
                        element_type=e["element_type"],
                    )
                    for e in row.get("elements", [])
                )
                samples.append(
                    FailureSample(
                        frame_id=row["frame_id"],
                        image_path=row["image_path"],
                        timestamp_ms=row.get("timestamp_ms"),
                        elements=elements,
                        failure_reason=row["failure_reason"],
                        observed_at=row["observed_at"],
                    )
                )
                if limit and len(samples) >= limit:
                    break
        return samples

    def count(self) -> int:
        """Return total number of stored failures."""
        if not self._index_file.exists():
            return 0
        count = 0
        with self._index_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count

    def export_for_review(self, review_queue: Any) -> int:
        """Push failure samples to HITL review queue. Returns count exported."""
        from vis_ground_lab.hitl.review_queue import ReviewItem

        failures = self.load_failures()
        exported = 0
        for sample in failures:
            elem_dicts = [
                {
                    "class_name": e.class_name,
                    "bbox": [e.bbox.x_min, e.bbox.y_min, e.bbox.x_max, e.bbox.y_max],
                    "score": e.score,
                }
                for e in sample.elements
            ]
            item = ReviewItem(
                image_path=sample.image_path,
                frame_id=sample.frame_id,
                elements=elem_dicts,
                uncertainty_score=1.0 - (min(e.score for e in sample.elements) if sample.elements else 0.0),
                source="runtime_failure",
                timestamp=sample.observed_at,
            )
            review_queue.enqueue(item)
            exported += 1
        return exported

    def export_as_coco(self, out_path: str | Path, class_names: list[str] | None = None) -> Path:
        """Export failures as COCO annotations."""
        out_path = Path(out_path)
        failures = self.load_failures()

        coco = empty_coco()
        names = class_names or ["button"]
        cat_map = register_categories(coco, names)

        ann_id = 1
        for img_id, sample in enumerate(failures, start=1):
            img_path = self._images_dir / f"{sample.frame_id}.png"
            if not img_path.exists():
                img_path = Path(sample.image_path)
            if img_path.exists():
                add_image_entry(coco, img_path, image_id=img_id)

            for elem in sample.elements:
                label = elem.class_name
                cat_id = cat_map.get(label, next(iter(cat_map.values()), 1))
                add_annotation_entry(
                    coco,
                    annotation_id=ann_id,
                    image_id=img_id,
                    category_id=cat_id,
                    bbox_xyxy=[elem.bbox.x_min, elem.bbox.y_min, elem.bbox.x_max, elem.bbox.y_max],
                    score=elem.score,
                )
                ann_id += 1

        save_coco(coco, out_path)
        return out_path
