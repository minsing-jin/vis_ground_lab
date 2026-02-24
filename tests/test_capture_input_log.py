"""Tests for capture.input_log module."""

from __future__ import annotations

from vis_ground_lab.capture.input_log import InputEvent, InputLogParser


def test_from_jsonl(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"timestamp_ms": 100.0, "event_type": "click", "x": 50, "y": 60}\n'
        '{"timestamp_ms": 200.0, "event_type": "keypress", "key": "a"}\n',
        encoding="utf-8",
    )
    events = InputLogParser.from_jsonl(log)
    assert len(events) == 2
    assert events[0].timestamp_ms == 100.0
    assert events[0].event_type == "click"
    assert events[0].x == 50.0
    assert events[0].y == 60.0
    assert events[1].event_type == "keypress"
    assert events[1].key == "a"
    assert events[1].x is None


def test_from_csv(tmp_path):
    log = tmp_path / "events.csv"
    log.write_text(
        "timestamp_ms,event_type,x,y,key,button\n"
        "100.0,click,50,60,,left\n"
        "200.0,scroll,30,40,,\n",
        encoding="utf-8",
    )
    events = InputLogParser.from_csv(log)
    assert len(events) == 2
    assert events[0].button == "left"
    assert events[1].event_type == "scroll"
    assert events[1].x == 30.0


def test_empty_jsonl(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    events = InputLogParser.from_jsonl(log)
    assert events == []


def test_metadata_passthrough(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"timestamp_ms": 100.0, "event_type": "click", "x": 1, "y": 2, "custom_field": "hello"}\n',
        encoding="utf-8",
    )
    events = InputLogParser.from_jsonl(log)
    assert events[0].metadata == {"custom_field": "hello"}
