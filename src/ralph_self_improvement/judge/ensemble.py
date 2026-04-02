"""Ensemble judge combining VLM and heuristic scores."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from data_harvest.core.types import HarvestSample
from ralph_self_improvement.core.config import JudgeConfig
from ralph_self_improvement.core.types import Judgment
from ralph_self_improvement.judge.heuristic_judge import HeuristicJudge
from ralph_self_improvement.judge.vlm_judge import VLMJudge

logger = logging.getLogger(__name__)


class EnsembleJudge:
    """Weighted combination of VLM judge (0.6) and heuristic judge (0.4)."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self.vlm_judge = VLMJudge(config)
        self.heuristic_judge = HeuristicJudge(config)

    def judge(self, sample: HarvestSample) -> Judgment:
        """Produce a Judgment for a single labeled sample."""
        label = sample.label
        event = sample.event

        if label is None or event is None:
            return Judgment(sample_id=sample.sample_id)

        # Get image dimensions for heuristic judge
        pre_path = sample.pre_frame_path
        image_width, image_height = 1920, 1080  # fallback
        if pre_path.exists():
            try:
                with Image.open(pre_path) as img:
                    image_width, image_height = img.size
            except Exception:
                pass

        # VLM judge
        vlm_result = self.vlm_judge.judge(sample)
        vlm_score = vlm_result["score"]

        # Heuristic judge
        h_result = self.heuristic_judge.judge(sample, image_width, image_height)
        heuristic_score = h_result["score"]

        # Ensemble
        ensemble_score = (
            self.config.vlm_weight * vlm_score
            + self.config.heuristic_weight * heuristic_score
        )

        return Judgment(
            sample_id=sample.sample_id,
            vlm_score=vlm_score,
            heuristic_score=heuristic_score,
            ensemble_score=ensemble_score,
            iou_with_judge=vlm_score,  # VLM score is already IoU-based
            click_inside_bbox=h_result["click_inside"],
            bbox_area_ratio=h_result["area_ratio"],
            aspect_ratio=h_result["aspect_ratio"],
            confidence=h_result["confidence"],
            details={
                "vlm": vlm_result,
                "heuristic": h_result,
            },
        )

    def judge_batch(self, samples: list[HarvestSample]) -> list[Judgment]:
        """Judge a batch of samples."""
        judgments = []
        for i, sample in enumerate(samples):
            logger.info("Judging sample %d/%d: %s", i + 1, len(samples), sample.sample_id)
            judgments.append(self.judge(sample))
        return judgments
