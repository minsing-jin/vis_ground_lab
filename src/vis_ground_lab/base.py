"""Core abstract interfaces for visual grounding models and datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class VGSample:
    """Single visual-grounding training/evaluation sample."""

    image: Any
    text: str
    bbox: BoundingBox
    image_id: str | None = None
    metadata: Mapping[str, Any] | None = None


class BaseVGModel(ABC):
    """Interface for model backends (e.g., Florence-2, Kosmos-2)."""

    @abstractmethod
    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Load tokenizer/processor/model weights and initialize runtime state."""

    @abstractmethod
    def preprocess(self, sample: VGSample) -> Mapping[str, Any]:
        """Convert a raw sample into model-ready features."""

    @abstractmethod
    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run one forward pass for training/evaluation."""

    @abstractmethod
    def predict(self, image: Any, text: str) -> BoundingBox:
        """Run inference and return the predicted bounding box."""


class BaseDataset(ABC):
    """Interface for datasets containing image-text pairs with box annotations."""

    @abstractmethod
    def load_data(self, source: str | Path) -> None:
        """Load and parse dataset entries from disk or a remote source."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> VGSample:
        """Return one sample containing image, text query, and bbox annotation."""

    @abstractmethod
    def samples(self) -> Sequence[VGSample]:
        """Return all loaded samples as an indexable sequence."""
