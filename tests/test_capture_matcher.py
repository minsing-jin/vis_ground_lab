"""Tests for capture.action_frame_matcher module."""

from __future__ import annotations

from PIL import Image

from vis_ground_lab.capture.action_frame_matcher import ActionFrameMatcher
from vis_ground_lab.capture.input_log import InputEvent


def _create_frames(tmp_path, n=5, fps=2.0):
    """Create dummy frame images in tmp_path."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in range(n):
        img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
        img.save(frame_dir / f"frame_{i:06d}.png")
    return frame_dir


def test_match_basic(tmp_path):
    frame_dir = _create_frames(tmp_path, n=5, fps=2.0)
    # Frame 0 at 0ms, Frame 1 at 500ms, Frame 2 at 1000ms, etc.
    events = [
        InputEvent(timestamp_ms=0.0, event_type="click", x=50, y=50),
        InputEvent(timestamp_ms=480.0, event_type="click", x=30, y=30),
    ]
    matcher = ActionFrameMatcher(frame_dir=frame_dir, fps=2.0, time_tolerance_ms=200.0)
    pairs = matcher.match(events)
    assert len(pairs) == 2
    assert pairs[0].frame_index == 0
    assert pairs[1].frame_index == 1
    assert pairs[1].time_delta_ms < 200.0


def test_match_no_match_outside_tolerance(tmp_path):
    frame_dir = _create_frames(tmp_path, n=2, fps=1.0)
    # Frame 0 at 0ms, Frame 1 at 1000ms
    events = [InputEvent(timestamp_ms=600.0, event_type="click", x=10, y=10)]
    matcher = ActionFrameMatcher(frame_dir=frame_dir, fps=1.0, time_tolerance_ms=100.0)
    pairs = matcher.match(events)
    assert len(pairs) == 0


def test_to_coco(tmp_path):
    frame_dir = _create_frames(tmp_path, n=3, fps=2.0)
    events = [
        InputEvent(timestamp_ms=0.0, event_type="click", x=50, y=50),
    ]
    matcher = ActionFrameMatcher(frame_dir=frame_dir, fps=2.0)
    pairs = matcher.match(events)

    out = tmp_path / "out.coco.json"
    matcher.to_coco(pairs, class_names=["button"], out_path=out)
    assert out.exists()

    import json
    coco = json.loads(out.read_text(encoding="utf-8"))
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1


def test_to_vg_samples(tmp_path):
    frame_dir = _create_frames(tmp_path, n=3, fps=2.0)
    events = [
        InputEvent(timestamp_ms=0.0, event_type="click", x=50, y=50),
        InputEvent(timestamp_ms=500.0, event_type="scroll", x=20, y=20),
    ]
    matcher = ActionFrameMatcher(frame_dir=frame_dir, fps=2.0)
    pairs = matcher.match(events)
    samples = matcher.to_vg_samples(pairs)
    assert len(samples) == 2
    assert samples[0].text == "button"  # click → button
    assert samples[1].text == "scroll_area"  # scroll → scroll_area


def test_empty_frame_dir(tmp_path):
    frame_dir = tmp_path / "empty_frames"
    frame_dir.mkdir()
    events = [InputEvent(timestamp_ms=0.0, event_type="click", x=1, y=1)]
    matcher = ActionFrameMatcher(frame_dir=frame_dir, fps=2.0)
    pairs = matcher.match(events)
    assert pairs == []
