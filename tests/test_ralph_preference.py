"""Tests for ralph_self_improvement.reward.preference."""

from pathlib import Path

from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    HarvestSample,
    LabelResult,
)
from ralph_self_improvement.core.types import Judgment, PreferencePair
from ralph_self_improvement.reward.preference import PreferencePairGenerator


def _make_judged_sample(
    sample_id: str,
    score: float,
    bbox: tuple[float, float, float, float] = (10, 20, 30, 40),
) -> tuple[Judgment, HarvestSample]:
    event = ActionEvent(timestamp_ms=0, action=ActionType.click, x=20, y=30)
    label = LabelResult(
        bbox_x_min=bbox[0], bbox_y_min=bbox[1],
        bbox_x_max=bbox[2], bbox_y_max=bbox[3],
        confidence=0.8, semantic_text="button",
    )
    sample = HarvestSample(
        sample_id=sample_id,
        sample_dir=Path(f"/tmp/{sample_id}"),
        event=event,
        label=label,
    )
    judgment = Judgment(sample_id=sample_id, ensemble_score=score)
    return judgment, sample


class TestPreferencePairGenerator:
    def test_top_vs_bottom(self):
        items = [
            _make_judged_sample("s1", 0.9),
            _make_judged_sample("s2", 0.8),
            _make_judged_sample("s3", 0.5),
            _make_judged_sample("s4", 0.2),
            _make_judged_sample("s5", 0.1),
        ]
        judgments = [j for j, _ in items]
        samples = [s for _, s in items]

        gen = PreferencePairGenerator(strategy="top_vs_bottom", margin=0.1, top_k_ratio=0.4)
        pairs = gen.generate(judgments, samples)

        assert len(pairs) > 0
        for p in pairs:
            assert p.chosen_score > p.rejected_score

    def test_pairwise(self):
        items = [
            _make_judged_sample("s1", 0.9),
            _make_judged_sample("s2", 0.5),
            _make_judged_sample("s3", 0.1),
        ]
        judgments = [j for j, _ in items]
        samples = [s for _, s in items]

        gen = PreferencePairGenerator(strategy="pairwise", margin=0.1)
        pairs = gen.generate(judgments, samples)

        assert len(pairs) > 0
        for p in pairs:
            assert p.chosen_score - p.rejected_score >= 0.1

    def test_no_pairs_when_scores_similar(self):
        items = [
            _make_judged_sample("s1", 0.5),
            _make_judged_sample("s2", 0.5),
        ]
        judgments = [j for j, _ in items]
        samples = [s for _, s in items]

        gen = PreferencePairGenerator(margin=0.2)
        pairs = gen.generate(judgments, samples)
        assert len(pairs) == 0

    def test_jsonl_save_load(self, tmp_path: Path):
        pairs = [
            PreferencePair(
                chosen_sample_id="s1", rejected_sample_id="s2",
                chosen_score=0.9, rejected_score=0.1,
                chosen_prompt="detect", chosen_bbox=[10, 20, 30, 40],
                rejected_prompt="detect", rejected_bbox=[50, 60, 70, 80],
            ),
        ]
        path = tmp_path / "pairs.jsonl"
        PreferencePairGenerator.save_jsonl(pairs, path)
        loaded = PreferencePairGenerator.load_jsonl(path)

        assert len(loaded) == 1
        assert loaded[0].chosen_sample_id == "s1"
        assert loaded[0].chosen_bbox == [10, 20, 30, 40]
