"""Passive input listener using pynput (mouse + keyboard)."""

from __future__ import annotations

import logging
import queue
import time
import threading

from data_harvest.core.types import ActionEvent, ActionType

logger = logging.getLogger(__name__)


class InputListener:
    """Passive listener that enqueues ActionEvents from mouse/keyboard input."""

    def __init__(self, maxsize: int = 1024, enable_hover: bool = False) -> None:
        self._queue: queue.Queue[ActionEvent] = queue.Queue(maxsize=maxsize)
        self._running = False
        self._enable_hover = enable_hover
        self._mouse_listener: threading.Thread | None = None
        self._keyboard_listener: threading.Thread | None = None
        self._drag_start: tuple[float, float] | None = None
        self._mouse_press_ts_ms: float | None = None
        self._mouse_button_name: str | None = None
        self._last_click_ts_ms: float | None = None
        self._last_click_xy: tuple[float, float] | None = None
        self._last_hover_ts_ms: float = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_mouse()
        self._start_keyboard()
        logger.info("InputListener started.")

    def stop(self) -> None:
        self._running = False
        if self._mouse_listener is not None:
            self._mouse_listener.stop()  # type: ignore[attr-defined]
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()  # type: ignore[attr-defined]
            self._keyboard_listener = None
        logger.info("InputListener stopped.")

    def _start_mouse(self) -> None:
        from pynput import mouse  # lazy import
        hold_threshold_ms = 400.0
        double_click_threshold_ms = 300.0
        hover_emit_interval_ms = 250.0

        def on_click(x: int, y: int, button: mouse.Button, pressed: bool) -> None:
            if not self._running:
                return
            ts = time.time() * 1000.0
            if pressed:
                self._drag_start = (float(x), float(y))
                self._mouse_press_ts_ms = ts
                self._mouse_button_name = button.name
            else:
                # Released — determine if click or drag
                if self._drag_start is not None:
                    sx, sy = self._drag_start
                    press_ts = self._mouse_press_ts_ms or ts
                    duration_ms = ts - press_ts
                    dist = ((float(x) - sx) ** 2 + (float(y) - sy) ** 2) ** 0.5
                    self._drag_start = None
                    self._mouse_press_ts_ms = None
                    button_name = self._mouse_button_name or button.name
                    self._mouse_button_name = None

                    if dist > 10:
                        evt = ActionEvent(
                            timestamp_ms=ts,
                            action=ActionType.drag,
                            x=sx,
                            y=sy,
                            end_x=float(x),
                            end_y=float(y),
                            button=button_name,
                            duration_ms=duration_ms,
                        )
                    elif button_name == "right":
                        evt = ActionEvent(
                            timestamp_ms=ts,
                            action=ActionType.right_click,
                            x=float(x),
                            y=float(y),
                            button=button_name,
                            duration_ms=duration_ms,
                        )
                    elif duration_ms >= hold_threshold_ms:
                        evt = ActionEvent(
                            timestamp_ms=ts,
                            action=ActionType.hold,
                            x=float(x),
                            y=float(y),
                            button=button_name,
                            duration_ms=duration_ms,
                        )
                    else:
                        is_double = False
                        if self._last_click_ts_ms is not None and self._last_click_xy is not None:
                            prev_x, prev_y = self._last_click_xy
                            delta_t = ts - self._last_click_ts_ms
                            delta_d = ((float(x) - prev_x) ** 2 + (float(y) - prev_y) ** 2) ** 0.5
                            is_double = delta_t <= double_click_threshold_ms and delta_d <= 12.0

                        evt = ActionEvent(
                            timestamp_ms=ts,
                            action=ActionType.double_click if is_double else ActionType.click,
                            x=float(x),
                            y=float(y),
                            button=button_name,
                            duration_ms=duration_ms,
                        )
                        self._last_click_ts_ms = ts
                        self._last_click_xy = (float(x), float(y))
                    self._put(evt)

        def on_move(x: int, y: int) -> None:
            if not self._running:
                return
            if not self._enable_hover:
                return
            ts = time.time() * 1000.0
            if ts - self._last_hover_ts_ms < hover_emit_interval_ms:
                return
            self._last_hover_ts_ms = ts
            evt = ActionEvent(
                timestamp_ms=ts,
                action=ActionType.hover,
                x=float(x),
                y=float(y),
            )
            self._put(evt)

        def on_scroll(x: int, y: int, dx: int, dy: int) -> None:
            if not self._running:
                return
            ts = time.time() * 1000.0
            evt = ActionEvent(
                timestamp_ms=ts,
                action=ActionType.scroll,
                x=float(x),
                y=float(y),
                text=f"dx={dx},dy={dy}",
            )
            self._put(evt)

        listener = mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll)
        listener.daemon = True
        listener.start()
        self._mouse_listener = listener  # type: ignore[assignment]

    def _start_keyboard(self) -> None:
        from pynput import keyboard  # lazy import

        def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if not self._running:
                return
            ts = time.time() * 1000.0
            key_str: str | None = None
            char: str | None = None
            if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                char = key.char  # type: ignore[union-attr]
                key_str = char
                action = ActionType.type
            else:
                key_str = str(key)
                action = ActionType.press
            evt = ActionEvent(
                timestamp_ms=ts,
                action=action,
                key=key_str,
                text=char,
            )
            self._put(evt)

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        self._keyboard_listener = listener  # type: ignore[assignment]

    def _put(self, event: ActionEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Input event queue full — dropping event.")

    def get_event(self, timeout: float = 1.0) -> ActionEvent | None:
        """Blocking get with timeout. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running
