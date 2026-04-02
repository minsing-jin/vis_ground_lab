"""Tests for ralph_self_improvement.reward.scoring."""

from ralph_self_improvement.core.types import Judgment
from ralph_self_improvement.reward.scoring import compute_reward, rank_judgments


class TestComputeReward:
    def test_normal(self):
        j = Judgment(sample_id="s1", ensemble_score=0.75)
        assert compute_reward(j) == 0.75

    def test_clamp_high(self):
        j = Judgment(sample_id="s1", ensemble_score=1.5)
        assert compute_reward(j) == 1.0

    def test_clamp_low(self):
        j = Judgment(sample_id="s1", ensemble_score=-0.5)
        assert compute_reward(j) == 0.0


class TestRankJudgments:
    def test_ranking(self):
        judgments = [
            Judgment(sample_id="s1", ensemble_score=0.3),
            Judgment(sample_id="s2", ensemble_score=0.9),
            Judgment(sample_id="s3", ensemble_score=0.6),
        ]
        ranked = rank_judgments(judgments)
        assert ranked[0].sample_id == "s2"
        assert ranked[1].sample_id == "s3"
        assert ranked[2].sample_id == "s1"

    def test_empty(self):
        assert rank_judgments([]) == []
