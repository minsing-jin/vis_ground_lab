"""GameProfile ABC: defines game-specific semantics, screen types, OCR config, ROI hints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class GameProfile(ABC):
    """Abstract base class for game-specific configuration profiles."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique profile identifier (e.g., 'civ6')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name (e.g., 'Civilization VI')."""

    @property
    @abstractmethod
    def semantic_dict(self) -> dict[str, str]:
        """Mapping of semantic IDs to human-readable descriptions.

        Example: {"btn_end_turn": "End Turn button", "btn_research": "Research button"}
        """

    @property
    @abstractmethod
    def screen_types(self) -> list[str]:
        """Known screen types for this game.

        Example: ["main_map", "city_view", "tech_tree", "diplomacy", "loading"]
        """

    @property
    def ocr_languages(self) -> list[str]:
        """OCR languages for this game (default: English)."""
        return ["en"]

    @property
    def roi_hints(self) -> dict[str, list[float]]:
        """Region-of-interest hints: named_region → [x_min, y_min, x_max, y_max] (normalized 0-1).

        Example: {"minimap": [0.0, 0.75, 0.2, 1.0], "top_bar": [0.0, 0.0, 1.0, 0.05]}
        """
        return {}

    def classify_screen(self, frame: Any) -> str | None:
        """Optionally classify the current screen type. Override for game-specific logic."""
        return None
