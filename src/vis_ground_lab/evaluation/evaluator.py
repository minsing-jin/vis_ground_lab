"""Evaluator for UI-oriented visual grounding metrics."""

from __future__ import annotations

from math import sqrt

from vis_ground_lab.base import BoundingBox


class Evaluator:
    """Computes IoU and center-point pixel distance across predictions."""

    @staticmethod
    def iou(pred: BoundingBox, target: BoundingBox) -> float:
        inter_x1 = max(pred.x_min, target.x_min)
        inter_y1 = max(pred.y_min, target.y_min)
        inter_x2 = min(pred.x_max, target.x_max)
        inter_y2 = min(pred.y_max, target.y_max)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        pred_area = max(0.0, pred.x_max - pred.x_min) * max(0.0, pred.y_max - pred.y_min)
        target_area = max(0.0, target.x_max - target.x_min) * max(0.0, target.y_max - target.y_min)
        union = pred_area + target_area - inter_area

        if union <= 0.0:
            return 0.0
        return inter_area / union

    @staticmethod
    def center_distance_px(pred: BoundingBox, target: BoundingBox) -> float:
        pred_cx = (pred.x_min + pred.x_max) / 2.0
        pred_cy = (pred.y_min + pred.y_max) / 2.0
        target_cx = (target.x_min + target.x_max) / 2.0
        target_cy = (target.y_min + target.y_max) / 2.0

        return sqrt((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2)

    def evaluate(self, predictions: list[BoundingBox], targets: list[BoundingBox]) -> dict[str, float]:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have equal lengths")
        if not predictions:
            return {"mean_iou": 0.0, "mean_distance_px": 0.0}

        ious = [self.iou(pred, tgt) for pred, tgt in zip(predictions, targets)]
        distances = [self.center_distance_px(pred, tgt) for pred, tgt in zip(predictions, targets)]

        return {
            "mean_iou": sum(ious) / len(ious),
            "mean_distance_px": sum(distances) / len(distances),
        }
