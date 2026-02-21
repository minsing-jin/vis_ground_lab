"""Florence2-based teacher prelabeler for UI button candidates."""

from __future__ import annotations

from typing import Any

from PIL import Image

from vis_ground_lab.base import BoundingBox
from vis_ground_lab.models.florence2 import Florence2Wrapper
from vis_ground_lab.prelabel.base import Prelabeler


class FlorenceTeacherPrelabeler(Prelabeler):
    """Use a Florence2 grounding model to propose candidate UI boxes."""

    DEFAULT_PROMPTS = [
        "detect the toolbar button",
        "detect the clickable icon",
        "find ui button",
        "find actionable menu item",
    ]

    def __init__(
        self,
        model_name: str,
        adapter_path_or_repo: str | None = None,
        prompts: list[str] | None = None,
        iou_dedup_threshold: float = 0.6,
    ) -> None:
        self.prompts = prompts or self.DEFAULT_PROMPTS
        self.iou_dedup_threshold = iou_dedup_threshold

        if adapter_path_or_repo:
            self.wrapper = Florence2Wrapper.from_pretrained_adapter(
                base_model_name=model_name,
                adapter_path_or_repo=adapter_path_or_repo,
            )
        else:
            self.wrapper = Florence2Wrapper(model_name=model_name, use_lora=False)
            self.wrapper.load_model()

    def predict_boxes(self, image: Any) -> list[BoundingBox]:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        width, height = image.size

        candidates: list[BoundingBox] = []
        for prompt in self.prompts:
            try:
                pred = self.wrapper.predict(image=image, text=prompt)
            except Exception:
                continue
            box = self._to_pixel_bbox(pred, width=width, height=height)
            if self._is_valid_box(box, width=width, height=height):
                candidates.append(box)

        return self._dedup_by_iou(candidates)

    @staticmethod
    def _to_pixel_bbox(bbox: BoundingBox, width: int, height: int) -> BoundingBox:
        values = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
        vmax = max(abs(v) for v in values)
        if vmax <= 1.5:
            return BoundingBox(
                x_min=bbox.x_min * width,
                y_min=bbox.y_min * height,
                x_max=bbox.x_max * width,
                y_max=bbox.y_max * height,
            )
        if vmax <= 1200.0:
            return BoundingBox(
                x_min=(bbox.x_min / 1000.0) * width,
                y_min=(bbox.y_min / 1000.0) * height,
                x_max=(bbox.x_max / 1000.0) * width,
                y_max=(bbox.y_max / 1000.0) * height,
            )
        return bbox

    @staticmethod
    def _is_valid_box(bbox: BoundingBox, width: int, height: int) -> bool:
        if bbox.x_max <= bbox.x_min or bbox.y_max <= bbox.y_min:
            return False
        if bbox.x_max < 0 or bbox.y_max < 0:
            return False
        if bbox.x_min > width or bbox.y_min > height:
            return False
        return True

    def _dedup_by_iou(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        kept: list[BoundingBox] = []
        for box in boxes:
            if any(self._iou(box, existing) >= self.iou_dedup_threshold for existing in kept):
                continue
            kept.append(box)
        return kept

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        x1 = max(a.x_min, b.x_min)
        y1 = max(a.y_min, b.y_min)
        x2 = min(a.x_max, b.x_max)
        y2 = min(a.y_max, b.y_max)
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih

        area_a = max(0.0, a.x_max - a.x_min) * max(0.0, a.y_max - a.y_min)
        area_b = max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union
