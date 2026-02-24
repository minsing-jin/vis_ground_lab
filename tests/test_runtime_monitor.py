"""Tests for runtime.monitor module."""

from __future__ import annotations

from PIL import Image

from vis_ground_lab.base import ActionableElement, BoundingBox, FrameAnalysis
from vis_ground_lab.runtime.monitor import RuntimeMonitor


def _make_analysis(scores: list[float], frame_id: str = "f1") -> FrameAnalysis:
    elements = tuple(
        ActionableElement(
            class_name="button",
            bbox=BoundingBox(0, 0, 10, 10),
            score=s,
            center=(5.0, 5.0),
            semantic_id=f"button_r0c0_{i}",
            affordances=("click",),
            element_type="button",
        )
        for i, s in enumerate(scores)
    )
    return FrameAnalysis(
        frame_id=frame_id,
        timestamp_ms=0.0,
        elements=elements,
        resolution=(640, 480),
        drift_score=0.0,
    )


def test_observe_high_confidence():
    monitor = RuntimeMonitor(low_confidence_threshold=0.3)
    analysis = _make_analysis([0.9, 0.8, 0.95])
    signals = monitor.observe(analysis)
    assert signals["mean_confidence"] > 0.8
    assert signals["uncertain_count"] == 0
    assert signals["should_collect_failure"] is False


def test_observe_low_confidence_triggers_collection():
    monitor = RuntimeMonitor(low_confidence_threshold=0.5)
    analysis = _make_analysis([0.1, 0.2, 0.1])
    signals = monitor.observe(analysis)
    assert signals["should_collect_failure"] is True
    assert signals["uncertain_count"] == 3


def test_detect_drift_no_reference():
    monitor = RuntimeMonitor()
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    drift = monitor.detect_drift(img)
    assert drift == 0.0


def test_detect_drift_same_image(tmp_path):
    ref_path = tmp_path / "ref.png"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(ref_path)

    monitor = RuntimeMonitor(reference_frames=[str(ref_path)])
    drift = monitor.detect_drift(img)
    assert drift < 0.1  # Same image, low drift


def test_detect_drift_different_image(tmp_path):
    ref_path = tmp_path / "ref.png"
    ref_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    ref_img.save(ref_path)

    monitor = RuntimeMonitor(reference_frames=[str(ref_path)], drift_hash_threshold=5)
    diff_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    drift = monitor.detect_drift(diff_img)
    assert drift > 0.0  # Different image, some drift


def test_save_load_state(tmp_path):
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (100, 100)).save(ref_path)

    monitor = RuntimeMonitor(reference_frames=[str(ref_path)])
    analysis = _make_analysis([0.5])
    monitor.observe(analysis)

    state_path = tmp_path / "state.json"
    monitor.save_state(state_path)

    monitor2 = RuntimeMonitor()
    monitor2.load_state(state_path)
    assert len(monitor2._reference_hashes) == 1
    assert len(monitor2._confidence_history) == 1


def test_empty_analysis():
    monitor = RuntimeMonitor()
    analysis = _make_analysis([])
    signals = monitor.observe(analysis)
    assert signals["mean_confidence"] == 0.0
