"""JSONL-backed review queue for human-in-the-loop corrections."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vis_ground_lab.base import UIElement
from vis_ground_lab.data.coco import (
    add_annotation_entry,
    add_image_entry,
    empty_coco,
    register_categories,
    save_coco,
)


@dataclass
class ReviewItem:
    """A single item in the HITL review queue."""

    image_path: str
    frame_id: str
    elements: list[dict[str, Any]]  # serialized UIElement dicts
    uncertainty_score: float
    source: str  # "auto_capture"|"runtime_failure"|"drift_detection"
    timestamp: str
    reviewed: bool = False
    corrections: dict[str, Any] | None = None

    @classmethod
    def from_elements(
        cls,
        image_path: str,
        frame_id: str,
        elements: list[UIElement],
        uncertainty_score: float,
        source: str,
        timestamp: str,
    ) -> ReviewItem:
        elem_dicts = [
            {
                "class_name": e.class_name,
                "bbox": [e.bbox.x_min, e.bbox.y_min, e.bbox.x_max, e.bbox.y_max],
                "score": e.score,
            }
            for e in elements
        ]
        return cls(
            image_path=image_path,
            frame_id=frame_id,
            elements=elem_dicts,
            uncertainty_score=uncertainty_score,
            source=source,
            timestamp=timestamp,
        )


class ReviewQueue:
    """JSONL-backed priority queue for human review of uncertain predictions."""

    def __init__(self, queue_dir: str | Path) -> None:
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue_file = self.queue_dir / "queue.jsonl"

    def enqueue(self, item: ReviewItem) -> None:
        """Append a review item to the queue."""
        with self._queue_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    def _load_all(self) -> list[dict[str, Any]]:
        if not self._queue_file.exists():
            return []
        items: list[dict[str, Any]] = []
        with self._queue_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _save_all(self, items: list[dict[str, Any]]) -> None:
        with self._queue_file.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def peek(self, n: int = 10) -> list[ReviewItem]:
        """Return top-N unreviewed items sorted by uncertainty (descending)."""
        items = self._load_all()
        unreviewed = [i for i in items if not i.get("reviewed", False)]
        unreviewed.sort(key=lambda x: x.get("uncertainty_score", 0), reverse=True)
        return [ReviewItem(**i) for i in unreviewed[:n]]

    def pending_count(self) -> int:
        """Return number of unreviewed items."""
        items = self._load_all()
        return sum(1 for i in items if not i.get("reviewed", False))

    def mark_reviewed(self, frame_id: str, corrections: dict[str, Any] | None = None) -> None:
        """Mark an item as reviewed with optional corrections."""
        items = self._load_all()
        for item in items:
            if item["frame_id"] == frame_id and not item.get("reviewed", False):
                item["reviewed"] = True
                item["corrections"] = corrections
                break
        self._save_all(items)

    def export_corrections_as_coco(
        self,
        out_path: str | Path,
        class_names: list[str] | None = None,
    ) -> Path:
        """Export reviewed corrections as COCO annotations."""
        out_path = Path(out_path)
        items = self._load_all()
        reviewed = [i for i in items if i.get("reviewed", False) and i.get("corrections")]

        coco = empty_coco()
        names = class_names or ["button"]
        cat_map = register_categories(coco, names)

        ann_id = 1
        for img_id, item in enumerate(reviewed, start=1):
            add_image_entry(coco, item["image_path"], image_id=img_id)

            corrections = item.get("corrections", {})
            for box_data in corrections.get("boxes", []):
                label = box_data.get("class_name", names[0])
                cat_id = cat_map.get(label, next(iter(cat_map.values()), 1))
                bbox = box_data.get("bbox", [0, 0, 0, 0])
                add_annotation_entry(
                    coco,
                    annotation_id=ann_id,
                    image_id=img_id,
                    category_id=cat_id,
                    bbox_xyxy=bbox,
                )
                ann_id += 1

        save_coco(coco, out_path)
        return out_path

    def stats(self) -> dict[str, int]:
        """Return queue statistics."""
        items = self._load_all()
        reviewed = sum(1 for i in items if i.get("reviewed", False))
        with_corrections = sum(
            1 for i in items if i.get("reviewed", False) and i.get("corrections")
        )
        return {
            "total": len(items),
            "reviewed": reviewed,
            "pending": len(items) - reviewed,
            "with_corrections": with_corrections,
        }
