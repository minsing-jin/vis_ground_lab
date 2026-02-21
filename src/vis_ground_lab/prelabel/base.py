"""Interfaces for pseudo-label generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vis_ground_lab.base import BoundingBox


class Prelabeler(ABC):
    """Plugin interface for candidate box generation."""

    @abstractmethod
    def predict_boxes(self, image: Any) -> list[BoundingBox]:
        """Predict candidate boxes for an input image."""
