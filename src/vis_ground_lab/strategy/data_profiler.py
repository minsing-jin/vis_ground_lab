"""Profile datasets to inform automatic strategy selection."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class DataProfile:
    """Statistical profile of a dataset."""

    num_images: int
    num_annotations: int
    num_classes: int
    avg_annotations_per_image: float
    avg_box_area_ratio: float
    class_balance_score: float  # 0.0=perfectly imbalanced, 1.0=perfectly balanced
    estimated_complexity: str  # "trivial"|"simple"|"moderate"|"complex"


class DataProfiler:
    """Analyze COCO or JSONL datasets to produce a DataProfile."""

    def profile_coco(self, coco_path: str | Path, image_dir: str | Path | None = None) -> DataProfile:
        """Profile a COCO-format annotation file."""
        data = json.loads(Path(coco_path).read_text(encoding="utf-8"))

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        num_images = len(images)
        num_annotations = len(annotations)
        num_classes = len(categories)

        avg_anns = num_annotations / max(1, num_images)

        # Compute average box area ratio
        image_areas: dict[int, float] = {}
        for img in images:
            image_areas[img["id"]] = float(img.get("width", 640) * img.get("height", 480))

        area_ratios: list[float] = []
        class_counts: Counter[int] = Counter()
        for ann in annotations:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            box_area = float(bbox[2]) * float(bbox[3])  # COCO format: [x,y,w,h]
            img_area = image_areas.get(ann.get("image_id", 0), 640 * 480)
            area_ratios.append(box_area / max(1.0, img_area))
            class_counts[ann.get("category_id", 0)] += 1

        avg_area_ratio = sum(area_ratios) / max(1, len(area_ratios))
        balance = self._class_balance(class_counts, num_classes)
        complexity = self._estimate_complexity(num_images, num_classes, avg_anns)

        return DataProfile(
            num_images=num_images,
            num_annotations=num_annotations,
            num_classes=num_classes,
            avg_annotations_per_image=round(avg_anns, 2),
            avg_box_area_ratio=round(avg_area_ratio, 4),
            class_balance_score=round(balance, 4),
            estimated_complexity=complexity,
        )

    def profile_jsonl(self, jsonl_path: str | Path, image_root: str | Path | None = None) -> DataProfile:
        """Profile a JSONL-format dataset."""
        path = Path(jsonl_path)
        lines: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))

        num_images = len(lines)
        class_counts: Counter[str] = Counter()
        area_ratios: list[float] = []

        for row in lines:
            label = row.get("text", row.get("label", "unknown"))
            class_counts[label] += 1

            bbox = row.get("bbox", {})
            if isinstance(bbox, dict):
                w = bbox.get("x_max", 0) - bbox.get("x_min", 0)
                h = bbox.get("y_max", 0) - bbox.get("y_min", 0)
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            else:
                w, h = 0.0, 0.0

            img_w = float(row.get("width", 1000))
            img_h = float(row.get("height", 1000))
            area_ratios.append((w * h) / max(1.0, img_w * img_h))

        num_classes = len(class_counts)
        avg_area_ratio = sum(area_ratios) / max(1, len(area_ratios))
        balance = self._class_balance(class_counts, num_classes)
        complexity = self._estimate_complexity(num_images, num_classes, 1.0)

        return DataProfile(
            num_images=num_images,
            num_annotations=num_images,  # 1 annotation per sample in JSONL
            num_classes=num_classes,
            avg_annotations_per_image=1.0,
            avg_box_area_ratio=round(avg_area_ratio, 4),
            class_balance_score=round(balance, 4),
            estimated_complexity=complexity,
        )

    @staticmethod
    def _class_balance(counts: Counter, num_classes: int) -> float:
        """Compute class balance score using normalized entropy."""
        if num_classes <= 1:
            return 1.0
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = -sum(
            (c / total) * math.log(c / total) for c in counts.values() if c > 0
        )
        max_entropy = math.log(num_classes)
        return entropy / max_entropy if max_entropy > 0 else 1.0

    @staticmethod
    def _estimate_complexity(num_images: int, num_classes: int, avg_anns: float) -> str:
        """Heuristic complexity estimation."""
        if num_images < 50 and num_classes <= 1:
            return "trivial"
        if num_images < 200 and num_classes <= 3:
            return "simple"
        if num_images < 1000 or num_classes <= 5:
            return "moderate"
        return "complex"
