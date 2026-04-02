from __future__ import annotations

from vis_ground_lab.base import BoundingBox
from vis_ground_lab.evaluation import Evaluator


def test_evaluator_metrics():
    evaluator = Evaluator()

    pred = BoundingBox(10, 10, 30, 30)
    target = BoundingBox(20, 20, 40, 40)

    iou = evaluator.iou(pred, target)
    dist = evaluator.center_distance_px(pred, target)

    assert round(iou, 6) == round(100 / 700, 6)
    assert round(dist, 6) == round((200) ** 0.5, 6)


def test_evaluator_aggregate_dict_shape():
    evaluator = Evaluator()
    pred = [BoundingBox(0, 0, 10, 10)]
    target = [BoundingBox(0, 0, 10, 10)]

    result = evaluator.evaluate(pred, target)
    assert set(result.keys()) == {"mean_iou", "mean_distance_px"}
    assert result["mean_iou"] == 1.0
    assert result["mean_distance_px"] == 0.0


def test_evaluator_classification_metrics():
    evaluator = Evaluator()
    result = evaluator.evaluate_classification(predictions=[0, 1, 1, 0], targets=[0, 1, 0, 0])

    assert round(result["accuracy"], 6) == 0.75
    assert round(result["macro_f1"], 6) == round((0.8 + (2 / 3)) / 2, 6)
