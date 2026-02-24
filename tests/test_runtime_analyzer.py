"""Tests for runtime.frame_analyzer module."""

from __future__ import annotations

from PIL import Image

from vis_ground_lab.base import BoundingBox, UIElement
from vis_ground_lab.runtime.frame_analyzer import FrameAnalyzer


class MockModel:
    """Mock detector that returns fixed UIElements."""

    def __init__(self, predictions: list[UIElement] | None = None):
        self._predictions = predictions or []

    def predict(self, image):
        return self._predictions


def test_analyze_basic():
    preds = [
        UIElement("button", BoundingBox(10, 20, 50, 60), 0.9),
        UIElement("text_field", BoundingBox(100, 100, 200, 120), 0.7),
    ]
    model = MockModel(preds)
    analyzer = FrameAnalyzer(model=model, confidence_threshold=0.25)

    image = Image.new("RGB", (640, 480))
    analysis = analyzer.analyze(image, frame_id="test_001", timestamp_ms=1000.0)

    assert analysis.frame_id == "test_001"
    assert analysis.timestamp_ms == 1000.0
    assert analysis.resolution == (640, 480)
    assert len(analysis.elements) == 2
    assert analysis.elements[0].class_name == "button"
    assert analysis.elements[0].affordances == ("click",)
    assert analysis.elements[1].element_type == "type"


def test_analyze_filters_low_confidence():
    preds = [
        UIElement("button", BoundingBox(10, 20, 50, 60), 0.1),  # below threshold
        UIElement("button", BoundingBox(100, 100, 200, 200), 0.8),
    ]
    model = MockModel(preds)
    analyzer = FrameAnalyzer(model=model, confidence_threshold=0.5)

    image = Image.new("RGB", (640, 480))
    analysis = analyzer.analyze(image)
    assert len(analysis.elements) == 1
    assert analysis.elements[0].score == 0.8


def test_semantic_id_spatial_bucket():
    preds = [
        UIElement("button", BoundingBox(0, 0, 20, 20), 0.9),  # top-left
        UIElement("button", BoundingBox(600, 440, 640, 480), 0.9),  # bottom-right
    ]
    model = MockModel(preds)
    analyzer = FrameAnalyzer(model=model)

    image = Image.new("RGB", (640, 480))
    analysis = analyzer.analyze(image)
    ids = {e.semantic_id for e in analysis.elements}
    assert len(ids) == 2  # Different spatial buckets


def test_center_computation():
    preds = [UIElement("button", BoundingBox(100, 200, 300, 400), 0.9)]
    model = MockModel(preds)
    analyzer = FrameAnalyzer(model=model)

    image = Image.new("RGB", (640, 480))
    analysis = analyzer.analyze(image)
    assert analysis.elements[0].center == (200.0, 300.0)


def test_to_json_roundtrip():
    preds = [UIElement("button", BoundingBox(10, 20, 50, 60), 0.9)]
    model = MockModel(preds)
    analyzer = FrameAnalyzer(model=model)

    image = Image.new("RGB", (640, 480))
    analysis = analyzer.analyze(image, frame_id="f1")

    import json
    data = json.loads(analysis.to_json())
    assert data["frame_id"] == "f1"
    assert len(data["elements"]) == 1


def test_empty_predictions():
    model = MockModel([])
    analyzer = FrameAnalyzer(model=model)

    image = Image.new("RGB", (640, 480))
    analysis = analyzer.analyze(image)
    assert len(analysis.elements) == 0
    assert analysis.drift_score == 0.0


def test_custom_affordance_map():
    preds = [UIElement("custom_widget", BoundingBox(10, 20, 50, 60), 0.9)]
    model = MockModel(preds)
    analyzer = FrameAnalyzer(
        model=model,
        affordance_map={"custom_widget": ("swipe", "pinch")},
    )

    image = Image.new("RGB", (640, 480))
    analysis = analyzer.analyze(image)
    assert analysis.elements[0].affordances == ("swipe", "pinch")
