"""Recorder coordinator: event → pre-frame from ring → post-frame after delay → save."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import ActionType
from data_harvest.recorder.screen_ring import ScreenRingBuffer
from data_harvest.recorder.input_listener import InputListener

logger = logging.getLogger(__name__)

# Capture actionable events including hover/hold and keyboard events.
_RECORDABLE_ACTIONS = {
    ActionType.click,
    ActionType.right_click,
    ActionType.double_click,
    ActionType.hold,
    ActionType.hover,
    ActionType.drag,
    ActionType.scroll,
    ActionType.press,
    ActionType.type,
}


class HarvestRecorder:
    """Coordinates screen ring buffer + input listener to produce HarvestSamples."""

    def __init__(self, config: HarvestConfig) -> None:
        self.config = config
        self.session = HarvestSession(config)
        self.ring = ScreenRingBuffer(
            fps=config.recorder.capture_fps,
            buffer_seconds=config.recorder.buffer_seconds,
            monitor_index=config.recorder.monitor_index,
        )
        self.listener = InputListener(enable_hover=config.recorder.enable_hover)

    def start(self) -> None:
        """Initialize session and start capture daemons."""
        self.session.setup()
        self.ring.start()
        self.listener.start()
        logger.info("HarvestRecorder running — press Ctrl+C to stop.")

    def stop(self) -> None:
        self.listener.stop()
        self.ring.stop()
        logger.info(
            "HarvestRecorder stopped. Total samples: %d", self.session.sample_count
        )

    def run(self) -> None:
        """Main loop: listen for events and save samples until interrupted."""
        self.start()
        try:
            while True:
                event = self.listener.get_event(timeout=0.5)
                if event is None:
                    continue
                if event.action not in _RECORDABLE_ACTIONS:
                    continue
                if event.action == ActionType.hover and not self.config.recorder.enable_hover:
                    continue

                # 1. Grab pre-frame (nearest to event timestamp)
                pre = self.ring.nearest_frame(event.timestamp_ms)
                if pre is None:
                    logger.debug("No pre-frame available, skipping event.")
                    continue

                # Collect short pre clip window ([-300ms, event_ts])
                pre_clip = self.ring.frames_in_window(
                    event.timestamp_ms - 300.0,
                    event.timestamp_ms,
                )

                # 2. Wait for post-action delay
                delay_s = self.config.recorder.post_action_delay_ms / 1000.0
                time.sleep(delay_s)

                # 3. Grab post-frame
                post = self.ring.latest_frame()
                if post is None:
                    logger.debug("No post-frame available, skipping event.")
                    continue

                # Collect short post clip window ([event_ts, event_ts + 300ms])
                target_post_ms = event.timestamp_ms + 300.0
                while time.time() * 1000.0 < target_post_ms:
                    time.sleep(0.01)
                post_clip = self.ring.frames_in_window(
                    event.timestamp_ms,
                    target_post_ms,
                )

                # 4. Create sample and save
                sample = self.session.create_sample()
                sample.event = event
                sample.save_event()

                _save_frame(pre[1], sample.pre_frame_path)
                _save_frame(post[1], sample.post_frame_path)
                clip_info = _save_clip_frames(
                    sample_dir=sample.sample_dir,
                    pre_clip=pre_clip,
                    post_clip=post_clip,
                    pre_reference=pre[1],
                )
                sample.metadata = _build_sample_metadata(
                    event_x=event.x,
                    event_y=event.y,
                    monitor_region=self.ring.monitor_region,
                    pre_frame=pre[1],
                    post_frame=post[1],
                    ui_scale=self.config.recorder.ui_scale,
                )
                sample.metadata["clip"] = clip_info
                sample.save_metadata()

                logger.info(
                    "Saved %s  action=%s  xy=(%.0f, %.0f)",
                    sample.sample_id,
                    event.action.value,
                    event.x or 0,
                    event.y or 0,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def _save_frame(bgra: np.ndarray, path: str | Path) -> None:
    """Convert BGRA frame to BGR and save as PNG."""
    path = Path(path)
    if bgra.shape[2] == 4:
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    else:
        bgr = bgra
    cv2.imwrite(str(path), bgr)


def _build_sample_metadata(
    event_x: float | None,
    event_y: float | None,
    monitor_region: dict[str, int] | None,
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    ui_scale: float | None,
) -> dict[str, object]:
    pre_h, pre_w = int(pre_frame.shape[0]), int(pre_frame.shape[1])
    post_h, post_w = int(post_frame.shape[0]), int(post_frame.shape[1])

    event_global = {"x": event_x, "y": event_y}
    event_local = {"x": None, "y": None}
    event_norm = {"x": None, "y": None}

    if (
        monitor_region is not None
        and event_x is not None
        and event_y is not None
        and monitor_region.get("width", 0) > 0
        and monitor_region.get("height", 0) > 0
    ):
        left = float(monitor_region["left"])
        top = float(monitor_region["top"])
        width = float(monitor_region["width"])
        height = float(monitor_region["height"])
        local_x = float(event_x) - left
        local_y = float(event_y) - top
        event_local = {"x": local_x, "y": local_y}
        event_norm = {
            "x": local_x / width,
            "y": local_y / height,
        }

    return {
        "resolution": {
            "pre": {"width": pre_w, "height": pre_h},
            "post": {"width": post_w, "height": post_h},
        },
        "capture": {
            "ui_scale": ui_scale,
            "monitor_region": monitor_region,
        },
        "coordinates": {
            "event_global_xy": event_global,
            "event_local_xy": event_local,
            "event_normalized_xy": event_norm,
        },
    }


def _save_clip_frames(
    sample_dir: Path,
    pre_clip: list[tuple[float, np.ndarray]],
    post_clip: list[tuple[float, np.ndarray]],
    pre_reference: np.ndarray,
) -> dict[str, object]:
    clip_dir = sample_dir / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)

    pre_files: list[str] = []
    for idx, (_, frame) in enumerate(pre_clip):
        out = clip_dir / f"pre_{idx:03d}.png"
        _save_frame(frame, out)
        pre_files.append(str(out))

    post_files: list[str] = []
    for idx, (_, frame) in enumerate(post_clip):
        out = clip_dir / f"post_{idx:03d}.png"
        _save_frame(frame, out)
        post_files.append(str(out))

    post_effect_path = None
    post_stable_path = None
    if post_clip:
        from data_harvest.labeler.diff_detector import diff_ratio

        pre_bgr = cv2.cvtColor(pre_reference, cv2.COLOR_BGRA2BGR) if pre_reference.shape[2] == 4 else pre_reference
        # First post frame with visible change
        effect_idx = 0
        for i, (_, frame) in enumerate(post_clip):
            post_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
            if diff_ratio(pre_bgr, post_bgr) > 0.01:
                effect_idx = i
                break

        stable_idx = len(post_clip) - 1

        effect_frame = post_clip[effect_idx][1]
        stable_frame = post_clip[stable_idx][1]

        post_effect = clip_dir / "post_effect.png"
        post_stable = clip_dir / "post_stable.png"
        _save_frame(effect_frame, post_effect)
        _save_frame(stable_frame, post_stable)
        post_effect_path = str(post_effect)
        post_stable_path = str(post_stable)

    return {
        "pre_frames": pre_files,
        "post_frames": post_files,
        "post_effect": post_effect_path,
        "post_stable": post_stable_path,
    }
