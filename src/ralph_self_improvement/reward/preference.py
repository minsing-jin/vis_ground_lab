"""Preference pair generation for DPO training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_harvest.core.types import HarvestSample
from ralph_self_improvement.core.types import Judgment, PreferencePair
from ralph_self_improvement.reward.scoring import rank_judgments

logger = logging.getLogger(__name__)


class PreferencePairGenerator:
    """Generate chosen/rejected pairs from judgments for DPO training.

    Strategies:
        pairwise: all (i, j) pairs where score(i) > score(j) + margin
        top_vs_bottom: top-k vs bottom-k pairs
    """

    def __init__(
        self,
        strategy: str = "top_vs_bottom",
        margin: float = 0.1,
        top_k_ratio: float = 0.3,
    ) -> None:
        self.strategy = strategy
        self.margin = margin
        self.top_k_ratio = top_k_ratio

    def generate(
        self,
        judgments: list[Judgment],
        samples: list[HarvestSample],
    ) -> list[PreferencePair]:
        """Generate preference pairs from judged samples."""
        sample_map = {s.sample_id: s for s in samples}
        ranked = rank_judgments(judgments)

        if self.strategy == "pairwise":
            return self._pairwise(ranked, sample_map)
        return self._top_vs_bottom(ranked, sample_map)

    def _top_vs_bottom(
        self,
        ranked: list[Judgment],
        sample_map: dict[str, HarvestSample],
    ) -> list[PreferencePair]:
        n = len(ranked)
        k = max(1, int(n * self.top_k_ratio))
        top = ranked[:k]
        bottom = ranked[-k:]

        pairs = []
        for chosen_j in top:
            for rejected_j in bottom:
                if chosen_j.ensemble_score - rejected_j.ensemble_score < self.margin:
                    continue
                pair = self._make_pair(chosen_j, rejected_j, sample_map)
                if pair is not None:
                    pairs.append(pair)
        return pairs

    def _pairwise(
        self,
        ranked: list[Judgment],
        sample_map: dict[str, HarvestSample],
    ) -> list[PreferencePair]:
        pairs = []
        for i, chosen_j in enumerate(ranked):
            for rejected_j in ranked[i + 1:]:
                if chosen_j.ensemble_score - rejected_j.ensemble_score < self.margin:
                    break
                pair = self._make_pair(chosen_j, rejected_j, sample_map)
                if pair is not None:
                    pairs.append(pair)
        return pairs

    @staticmethod
    def _make_pair(
        chosen_j: Judgment,
        rejected_j: Judgment,
        sample_map: dict[str, HarvestSample],
    ) -> PreferencePair | None:
        chosen_s = sample_map.get(chosen_j.sample_id)
        rejected_s = sample_map.get(rejected_j.sample_id)
        if chosen_s is None or rejected_s is None:
            return None
        if chosen_s.label is None or rejected_s.label is None:
            return None

        return PreferencePair(
            chosen_sample_id=chosen_j.sample_id,
            rejected_sample_id=rejected_j.sample_id,
            chosen_score=chosen_j.ensemble_score,
            rejected_score=rejected_j.ensemble_score,
            chosen_prompt=chosen_s.label.semantic_text or "detect the UI element",
            chosen_bbox=chosen_s.label.bbox_xyxy,
            rejected_prompt=rejected_s.label.semantic_text or "detect the UI element",
            rejected_bbox=rejected_s.label.bbox_xyxy,
            chosen_image_path=str(chosen_s.pre_frame_path),
            rejected_image_path=str(rejected_s.pre_frame_path),
        )

    @staticmethod
    def save_jsonl(pairs: list[PreferencePair], path: str | Path) -> None:
        """Append preference pairs to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for pair in pairs:
                f.write(pair.to_json() + "\n")
        logger.info("Saved %d preference pairs to %s", len(pairs), path)

    @staticmethod
    def load_jsonl(path: str | Path) -> list[PreferencePair]:
        """Load preference pairs from a JSONL file."""
        path = Path(path)
        if not path.exists():
            return []
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(PreferencePair.from_dict(json.loads(line)))
        return pairs
