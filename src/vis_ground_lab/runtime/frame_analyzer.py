"""Core structured output for LLM/VLM agents — converts detections to FrameAnalysis."""

from __future__ import annotations

from typing import Any

from PIL import Image

from vis_ground_lab.base import ActionableElement, BoundingBox, FrameAnalysis, UIElement

DEFAULT_AFFORDANCES: dict[str, tuple[str, ...]] = {
    "button": ("click",),
    "text_field": ("type", "click"),
    "scroll_area": ("scroll",),
    "slider": ("drag",),
    "toggle": ("click", "toggle"),
    "menu_item": ("click",),
    "icon": ("click", "double_click"),
}

# Reverse map from affordances to element type
_TYPE_FROM_CLASS: dict[str, str] = {
    "button": "button",
    "text_field": "type",
    "scroll_area": "scroll",
    "slider": "drag",
    "toggle": "toggle",
    "menu_item": "button",
    "icon": "button",
}


def _spatial_bucket(cx: float, cy: float, width: int, height: int, grid: int = 3) -> str:
    """Assign a spatial bucket like 'r0c1' based on screen grid position."""
    col = min(grid - 1, int(cx / max(1, width) * grid))
    row = min(grid - 1, int(cy / max(1, height) * grid))
    return f"r{row}c{col}"


def _build_semantic_id(class_name: str, cx: float, cy: float, width: int, height: int) -> str:
    """Build a stable semantic ID from class name and spatial position."""
    bucket = _spatial_bucket(cx, cy, width, height)
    return f"{class_name}_{bucket}"


class FrameAnalyzer:
    """Analyze a frame image using a detector model and produce structured FrameAnalysis."""

    def __init__(
        self,
        model: Any,
        affordance_map: dict[str, tuple[str, ...]] | None = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model = model
        self.affordance_map = affordance_map or DEFAULT_AFFORDANCES
        self.confidence_threshold = confidence_threshold

    def analyze(
        self,
        image: str | Image.Image,
        frame_id: str = "",
        timestamp_ms: float | None = None,
    ) -> FrameAnalysis:
        """Run detection and convert to FrameAnalysis with affordances."""
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image

        width, height = pil_image.size

        # Run model prediction — expects list[UIElement]
        predictions: list[UIElement] = self.model.predict(pil_image)

        elements: list[ActionableElement] = []
        for pred in predictions:
            if pred.score < self.confidence_threshold:
                continue

            cx = (pred.bbox.x_min + pred.bbox.x_max) / 2.0
            cy = (pred.bbox.y_min + pred.bbox.y_max) / 2.0
            class_lower = pred.class_name.lower()

            affordances = self.affordance_map.get(class_lower, ("click",))
            element_type = _TYPE_FROM_CLASS.get(class_lower, "button")
            semantic_id = _build_semantic_id(class_lower, cx, cy, width, height)

            elements.append(
                ActionableElement(
                    class_name=pred.class_name,
                    bbox=pred.bbox,
                    score=pred.score,
                    center=(cx, cy),
                    semantic_id=semantic_id,
                    affordances=affordances,
                    element_type=element_type,
                )
            )

        return FrameAnalysis(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            elements=tuple(elements),
            resolution=(width, height),
            drift_score=0.0,
        )
