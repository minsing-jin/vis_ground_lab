"""Tests for ralph_self_improvement.core.types."""

import json

from ralph_self_improvement.core.types import (
    IterationResult,
    Judgment,
    PreferencePair,
    WeightSnapshot,
)


class TestJudgment:
    def test_roundtrip(self):
        j = Judgment(
            sample_id="sample_000001",
            vlm_score=0.8,
            heuristic_score=0.7,
            ensemble_score=0.76,
            iou_with_judge=0.8,
            click_inside_bbox=True,
            bbox_area_ratio=0.05,
            aspect_ratio=1.5,
            confidence=0.9,
            details={"key": "value"},
        )
        d = j.to_dict()
        j2 = Judgment.from_dict(d)
        assert j2.sample_id == "sample_000001"
        assert j2.ensemble_score == 0.76
        assert j2.details == {"key": "value"}

    def test_json_roundtrip(self):
        j = Judgment(sample_id="s1", ensemble_score=0.5)
        text = j.to_json()
        d = json.loads(text)
        j2 = Judgment.from_dict(d)
        assert j2.sample_id == "s1"
        assert j2.ensemble_score == 0.5

    def test_defaults(self):
        j = Judgment(sample_id="s2")
        assert j.vlm_score == 0.0
        assert j.heuristic_score == 0.0
        assert j.click_inside_bbox is False
        assert j.details == {}


class TestPreferencePair:
    def test_roundtrip(self):
        p = PreferencePair(
            chosen_sample_id="s1",
            rejected_sample_id="s2",
            chosen_score=0.9,
            rejected_score=0.3,
            chosen_prompt="detect",
            chosen_bbox=[10, 20, 30, 40],
            rejected_prompt="detect",
            rejected_bbox=[50, 60, 70, 80],
        )
        d = p.to_dict()
        p2 = PreferencePair.from_dict(d)
        assert p2.chosen_sample_id == "s1"
        assert p2.chosen_bbox == [10, 20, 30, 40]


class TestWeightSnapshot:
    def test_roundtrip(self):
        ws = WeightSnapshot(
            weights={"vlm": 0.4, "diff": 0.3, "click_proximity": 0.2, "ocr": 0.0, "profile_hint": 0.1},
            objective_value=0.85,
            n_trials=30,
            n_samples=50,
        )
        d = ws.to_dict()
        ws2 = WeightSnapshot.from_dict(d)
        assert ws2.weights["vlm"] == 0.4
        assert ws2.objective_value == 0.85


class TestIterationResult:
    def test_roundtrip(self):
        ws = WeightSnapshot(weights={"vlm": 0.5}, objective_value=0.9)
        r = IterationResult(
            iteration=1,
            mean_iou=0.7,
            mean_ensemble_score=0.75,
            n_samples=100,
            n_preference_pairs=50,
            weight_snapshot=ws,
            checkpoint_path="checkpoints/ralph/sft",
            improved=True,
        )
        d = r.to_dict()
        r2 = IterationResult.from_dict(d)
        assert r2.iteration == 1
        assert r2.mean_iou == 0.7
        assert r2.weight_snapshot is not None
        assert r2.weight_snapshot.weights["vlm"] == 0.5
        assert r2.checkpoint_path == "checkpoints/ralph/sft"

    def test_without_snapshot(self):
        r = IterationResult(iteration=2)
        d = r.to_dict()
        r2 = IterationResult.from_dict(d)
        assert r2.weight_snapshot is None
