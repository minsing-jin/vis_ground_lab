"""Reward computation and judgment ranking."""

from __future__ import annotations

from ralph_self_improvement.core.types import Judgment


def compute_reward(judgment: Judgment) -> float:
    """Compute a scalar reward from a Judgment.

    Uses ensemble_score directly as the reward signal, bounded to [0, 1].
    """
    return max(0.0, min(1.0, judgment.ensemble_score))


def rank_judgments(judgments: list[Judgment]) -> list[Judgment]:
    """Rank judgments by ensemble_score in descending order."""
    return sorted(judgments, key=lambda j: j.ensemble_score, reverse=True)
