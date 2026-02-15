"""DataManager and PyTorch Dataset for JSONL visual grounding data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image
from torch.utils.data import Dataset

from vis_ground_lab.base import BaseDataset, BoundingBox, VGSample

NormalizeMode = Literal["none", "0-1", "0-1000"]


class JSONLVisualGroundingDataset(BaseDataset, Dataset[VGSample]):
    """Dataset for JSONL rows with image_path/prompt/bbox fields."""

    def __init__(
        self,
        source: str | Path,
        image_root: str | Path | None = None,
        normalize_mode: NormalizeMode = "none",
    ) -> None:
        self.image_root = Path(image_root) if image_root else None
        self.normalize_mode = normalize_mode
        self._records: list[dict[str, Any]] = []
        self._source = Path(source)
        self.load_data(self._source)

    def load_data(self, source: str | Path) -> None:
        self._records.clear()
        path = Path(source)

        with path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                record = json.loads(stripped)
                if not all(key in record for key in ("image_path", "prompt", "bbox")):
                    raise ValueError(f"Invalid JSONL record at line {line_idx}: missing required keys")

                bbox = record["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Invalid bbox at line {line_idx}: expected [x1, y1, x2, y2]")

                self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> VGSample:
        record = self._records[index]

        image_path = Path(record["image_path"])
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        x1, y1, x2, y2 = map(float, record["bbox"])
        bbox = self._normalize_bbox(x1, y1, x2, y2, width=width, height=height)

        return VGSample(
            image=image,
            text=str(record["prompt"]),
            bbox=bbox,
            image_id=str(record.get("image_id", image_path.name)),
            metadata={"image_path": str(image_path), "width": width, "height": height},
        )

    def samples(self) -> Sequence[VGSample]:
        return [self[i] for i in range(len(self))]

    def _normalize_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> BoundingBox:
        if self.normalize_mode == "none":
            return BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive for bbox normalization")

        if self.normalize_mode == "0-1":
            return BoundingBox(
                x_min=x1 / width,
                y_min=y1 / height,
                x_max=x2 / width,
                y_max=y2 / height,
            )

        if self.normalize_mode == "0-1000":
            return BoundingBox(
                x_min=(x1 / width) * 1000.0,
                y_min=(y1 / height) * 1000.0,
                x_max=(x2 / width) * 1000.0,
                y_max=(y2 / height) * 1000.0,
            )

        raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")
