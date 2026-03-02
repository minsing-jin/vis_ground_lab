"""Recorder coordinator: event → pre-frame from ring → post-frame after delay → save."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import ActionType
from data_harvest.recorder.screen_ring import ScreenRingBuffer
from data_harvest.recorder.input_listener import InputListener

logger = logging.getLogger(__name__)

# Only capture actionable events (skip individual keypresses/types for now)
_RECORDABLE_ACTIONS = {ActionType.click, ActionType.drag, ActionType.scroll}


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
        self.listener = InputListener()

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

                # 1. Grab pre-frame (nearest to event timestamp)
                pre = self.ring.nearest_frame(event.timestamp_ms)
                if pre is None:
                    logger.debug("No pre-frame available, skipping event.")
                    continue

                # 2. Wait for post-action delay
                delay_s = self.config.recorder.post_action_delay_ms / 1000.0
                time.sleep(delay_s)

                # 3. Grab post-frame
                post = self.ring.latest_frame()
                if post is None:
                    logger.debug("No post-frame available, skipping event.")
                    continue

                # 4. Create sample and save
                sample = self.session.create_sample()
                sample.event = event
                sample.save_event()

                _save_frame(pre[1], sample.pre_frame_path)
                _save_frame(post[1], sample.post_frame_path)

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
