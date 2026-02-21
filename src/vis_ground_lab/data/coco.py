"""COCO import/export helpers for detection workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def empty_coco() -> dict[str, Any]:
    return {
        "images": [],
        "annotations": [],
        "categories": [],
    }


def save_coco(coco: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")


def load_coco(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def register_categories(coco: dict[str, Any], class_names: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    start = len(coco["categories"]) + 1
    for idx, name in enumerate(class_names, start=start):
        coco["categories"].append({"id": idx, "name": name})
        mapping[name] = idx
    return mapping


def add_image_entry(coco: dict[str, Any], image_path: str | Path, image_id: int) -> dict[str, Any]:
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size

    item = {
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
    }
    coco["images"].append(item)
    return item


def add_annotation_entry(
    coco: dict[str, Any],
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox_xyxy: list[float],
    score: float | None = None,
    iscrowd: int = 0,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox_xyxy
    width = max(0.0, float(x2 - x1))
    height = max(0.0, float(y2 - y1))

    item: dict[str, Any] = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x1), float(y1), width, height],
        "area": float(width * height),
        "iscrowd": iscrowd,
    }
    if score is not None:
        item["score"] = float(score)
    coco["annotations"].append(item)
    return item


def coco_bbox_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]
