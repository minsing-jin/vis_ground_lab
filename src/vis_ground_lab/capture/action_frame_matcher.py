"""Correlate input events with extracted video frames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vis_ground_lab.base import BoundingBox, VGSample
from vis_ground_lab.capture.input_log import InputEvent
from vis_ground_lab.data.coco import (
    add_annotation_entry,
    add_image_entry,
    empty_coco,
    register_categories,
    save_coco,
)


@dataclass(frozen=True)
class ActionFramePair:
    """A matched input event paired with its closest video frame."""

    event: InputEvent
    frame_path: Path
    frame_index: int
    time_delta_ms: float
    auto_label: str | None = None


class ActionFrameMatcher:
    """Match input events to the nearest extracted video frame by timestamp."""

    def __init__(
        self,
        frame_dir: str | Path,
        fps: float,
        time_tolerance_ms: float = 200.0,
    ) -> None:
        self.frame_dir = Path(frame_dir)
        self.fps = fps
        self.time_tolerance_ms = time_tolerance_ms
        self._frame_paths = self._discover_frames()

    def _discover_frames(self) -> list[Path]:
        """Find and sort frame files by index."""
        frames: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            frames.extend(self.frame_dir.glob(ext))
        return sorted(frames)

    @staticmethod
    def _frame_index(path: Path) -> int:
        """Extract numeric index from frame filename like frame_000042.png."""
        match = re.search(r"(\d+)", path.stem)
        return int(match.group(1)) if match else 0

    def _frame_timestamp_ms(self, frame_path: Path) -> float:
        """Compute timestamp of a frame based on its index and fps."""
        idx = self._frame_index(frame_path)
        return (idx / self.fps) * 1000.0 if self.fps > 0 else 0.0

    def match(self, events: list[InputEvent]) -> list[ActionFramePair]:
        """Match each event to the closest frame within time_tolerance_ms."""
        if not self._frame_paths:
            return []

        pairs: list[ActionFramePair] = []
        frame_times = [(p, self._frame_timestamp_ms(p)) for p in self._frame_paths]

        for event in events:
            best_path: Path | None = None
            best_idx = 0
            best_delta = float("inf")

            for fpath, ftime in frame_times:
                delta = abs(event.timestamp_ms - ftime)
                if delta < best_delta:
                    best_delta = delta
                    best_path = fpath
                    best_idx = self._frame_index(fpath)

            if best_path is not None and best_delta <= self.time_tolerance_ms:
                auto_label = self._infer_label(event)
                pairs.append(
                    ActionFramePair(
                        event=event,
                        frame_path=best_path,
                        frame_index=best_idx,
                        time_delta_ms=best_delta,
                        auto_label=auto_label,
                    )
                )
        return pairs

    def to_coco(
        self,
        pairs: list[ActionFramePair],
        class_names: list[str],
        out_path: str | Path,
        crop_radius_px: int = 64,
    ) -> Path:
        """Export matched pairs as COCO annotations."""
        out_path = Path(out_path)
        coco = empty_coco()
        cat_map = register_categories(coco, class_names)

        ann_id = 1
        seen_images: dict[str, int] = {}

        for pair in pairs:
            fname = pair.frame_path.name
            if fname not in seen_images:
                img_id = len(seen_images) + 1
                add_image_entry(coco, pair.frame_path, image_id=img_id)
                seen_images[fname] = img_id
            img_id = seen_images[fname]

            label = pair.auto_label or (class_names[0] if class_names else "unknown")
            cat_id = cat_map.get(label, next(iter(cat_map.values()), 1))

            if pair.event.x is not None and pair.event.y is not None:
                cx, cy = pair.event.x, pair.event.y
                bbox_xyxy = [
                    max(0, cx - crop_radius_px),
                    max(0, cy - crop_radius_px),
                    cx + crop_radius_px,
                    cy + crop_radius_px,
                ]
                add_annotation_entry(
                    coco,
                    annotation_id=ann_id,
                    image_id=img_id,
                    category_id=cat_id,
                    bbox_xyxy=bbox_xyxy,
                    score=0.5,
                )
                ann_id += 1

        save_coco(coco, out_path)
        return out_path

    def to_vg_samples(
        self,
        pairs: list[ActionFramePair],
        crop_radius_px: int = 64,
    ) -> list[VGSample]:
        """Convert matched pairs to VGSample list for grounding training."""
        samples: list[VGSample] = []
        for pair in pairs:
            if pair.event.x is None or pair.event.y is None:
                continue
            cx, cy = pair.event.x, pair.event.y
            bbox = BoundingBox(
                x_min=max(0, cx - crop_radius_px),
                y_min=max(0, cy - crop_radius_px),
                x_max=cx + crop_radius_px,
                y_max=cy + crop_radius_px,
            )
            text = pair.auto_label or pair.event.event_type
            samples.append(
                VGSample(
                    image=str(pair.frame_path),
                    text=text,
                    bbox=bbox,
                    image_id=str(pair.frame_index),
                    metadata={"time_delta_ms": pair.time_delta_ms},
                )
            )
        return samples

    @staticmethod
    def _infer_label(event: InputEvent) -> str | None:
        """Infer a rough label from the event type."""
        mapping: dict[str, str] = {
            "click": "button",
            "keypress": "text_field",
            "scroll": "scroll_area",
            "drag_start": "slider",
            "drag_end": "slider",
        }
        return mapping.get(event.event_type)
