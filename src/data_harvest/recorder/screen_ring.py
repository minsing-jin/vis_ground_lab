"""Ring buffer for screen capture using mss (daemon thread, configurable FPS)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ScreenRingBuffer:
    """Captures screenshots at a fixed FPS into a ring buffer (daemon thread).

    Each entry is ``(timestamp_ms, numpy_bgra_array)``.
    """

    def __init__(
        self,
        fps: int = 10,
        buffer_seconds: int = 5,
        monitor_index: int = 0,
    ) -> None:
        self.fps = fps
        self.buffer_size = fps * buffer_seconds
        self.monitor_index = monitor_index

        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=self.buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "ScreenRingBuffer started: %d FPS, buffer=%d frames, monitor=%d",
            self.fps,
            self.buffer_size,
            self.monitor_index,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("ScreenRingBuffer stopped.")

    def _capture_loop(self) -> None:
        import mss  # lazy import — only needed at runtime

        interval = 1.0 / self.fps
        with mss.mss() as sct:
            monitors = sct.monitors
            if self.monitor_index >= len(monitors):
                monitor: dict[str, Any] = monitors[0]
            else:
                monitor = monitors[self.monitor_index]

            while self._running:
                t0 = time.perf_counter()
                ts_ms = time.time() * 1000.0
                shot = sct.grab(monitor)
                frame = np.asarray(shot)  # BGRA
                with self._lock:
                    self._buffer.append((ts_ms, frame))
                elapsed = time.perf_counter() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def latest_frame(self) -> tuple[float, np.ndarray] | None:
        """Return the most recent frame or None if empty."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def nearest_frame(self, timestamp_ms: float) -> tuple[float, np.ndarray] | None:
        """Return the frame nearest to *timestamp_ms* from the ring buffer."""
        with self._lock:
            if not self._buffer:
                return None
            best: tuple[float, np.ndarray] | None = None
            best_delta = float("inf")
            for ts, frame in self._buffer:
                delta = abs(ts - timestamp_ms)
                if delta < best_delta:
                    best_delta = delta
                    best = (ts, frame)
            return best

    @property
    def is_running(self) -> bool:
        return self._running
