"""Parse mouse/keyboard/controller input logs into structured events."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputEvent:
    """Single input event from a usage trace."""

    timestamp_ms: float
    event_type: str  # "click"|"keypress"|"scroll"|"drag_start"|"drag_end"
    x: float | None = None
    y: float | None = None
    key: str | None = None
    button: str | None = None
    metadata: dict[str, Any] | None = None


class InputLogParser:
    """Parse input logs from various formats into InputEvent lists."""

    @staticmethod
    def from_jsonl(path: str | Path) -> list[InputEvent]:
        """Parse JSONL input log where each line is a JSON object."""
        events: list[InputEvent] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                events.append(InputLogParser._row_to_event(row))
        return events

    @staticmethod
    def from_csv(
        path: str | Path,
        ts_col: str = "timestamp_ms",
        type_col: str = "event_type",
        x_col: str = "x",
        y_col: str = "y",
        key_col: str = "key",
        button_col: str = "button",
    ) -> list[InputEvent]:
        """Parse CSV input log with configurable column names."""
        events: list[InputEvent] = []
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                events.append(
                    InputEvent(
                        timestamp_ms=float(row[ts_col]),
                        event_type=row.get(type_col, "click"),
                        x=float(row[x_col]) if row.get(x_col) else None,
                        y=float(row[y_col]) if row.get(y_col) else None,
                        key=row.get(key_col) or None,
                        button=row.get(button_col) or None,
                    )
                )
        return events

    @staticmethod
    def _row_to_event(row: dict[str, Any]) -> InputEvent:
        return InputEvent(
            timestamp_ms=float(row["timestamp_ms"]),
            event_type=str(row.get("event_type", "click")),
            x=float(row["x"]) if row.get("x") is not None else None,
            y=float(row["y"]) if row.get("y") is not None else None,
            key=row.get("key"),
            button=row.get("button"),
            metadata={k: v for k, v in row.items() if k not in {"timestamp_ms", "event_type", "x", "y", "key", "button"}} or None,
        )
