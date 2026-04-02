"""DataManager and PyTorch Dataset for JSONL visual grounding data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from vis_ground_lab.base import BaseDataset, BoundingBox, VGSample

NormalizeMode = Literal["none", "0-1", "0-1000"]
IGNORE_INDEX = -100


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


class RouterClassificationDataset(Dataset[dict[str, Any]]):
    """CSV dataset for primitive routing classification."""

    def __init__(
        self,
        source: str | Path,
        image_root: str | Path | None = None,
        image_size: int = 224,
        label_column: str = "primitive_id",
        aux_label_columns: Sequence[str] | None = None,
        label_to_index: dict[str, int] | None = None,
        aux_label_to_index: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self._source = Path(source)
        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.label_column = label_column
        self.aux_label_columns = list(aux_label_columns or [])
        self.rows: list[dict[str, str]] = []
        self.label_to_index = dict(label_to_index or {})
        self.aux_label_to_index = {key: dict(value) for key, value in (aux_label_to_index or {}).items()}
        self.load_data(self._source)

    def load_data(self, source: str | Path) -> None:
        self.rows.clear()
        with Path(source).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_idx, row in enumerate(reader, start=2):
                label = str(row.get(self.label_column, "")).strip()
                if not label:
                    raise ValueError(
                        f"Invalid router CSV record at line {line_idx}: missing required label '{self.label_column}'"
                    )
                image_path = str(row.get("image_path", "")).strip()
                if not image_path:
                    raise ValueError(f"Invalid router CSV record at line {line_idx}: missing image_path")
                self.rows.append({key: str(value or "") for key, value in row.items()})

        if not self.label_to_index:
            labels = sorted({row[self.label_column].strip() for row in self.rows})
            self.label_to_index = {label: idx for idx, label in enumerate(labels)}

        for column in self.aux_label_columns:
            if column not in self.aux_label_to_index:
                labels = sorted({row.get(column, "").strip() for row in self.rows if row.get(column, "").strip()})
                self.aux_label_to_index[column] = {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        return {
            "pixel_values": self._image_to_tensor(image),
            "label": self.label_to_index[row[self.label_column].strip()],
            "aux_labels": self._encode_aux_labels(row),
            "sample_id": row.get("sample_id", f"sample_{index:06d}"),
            "image_path": str(image_path),
        }

    def _encode_aux_labels(self, row: dict[str, str]) -> dict[str, int]:
        encoded: dict[str, int] = {}
        for column in self.aux_label_columns:
            value = row.get(column, "").strip()
            mapping = self.aux_label_to_index.get(column, {})
            encoded[column] = mapping[value] if value in mapping else IGNORE_INDEX
        return encoded

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.image_size, self.image_size))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
