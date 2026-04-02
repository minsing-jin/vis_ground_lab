"""VLM-based judge using Florence-2-large for re-prediction and IoU comparison.

Strategy: re-predict bbox with a larger model → compute IoU against auto-label.
High agreement ≈ high quality.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from data_harvest.core.types import HarvestSample
from ralph_self_improvement.core.config import JudgeConfig
from vis_ground_lab.base import BoundingBox
from vis_ground_lab.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)

_JUDGE_MODEL = None


def _get_judge_model(config: JudgeConfig) -> Any:
    """Lazy-load the Florence-2-large judge model (singleton)."""
    global _JUDGE_MODEL
    if _JUDGE_MODEL is not None:
        return _JUDGE_MODEL

    from vis_ground_lab.models.florence2 import Florence2Wrapper

    wrapper = Florence2Wrapper(
        model_name=config.vlm_model_name,
        device_map=config.vlm_device_map,
        use_lora=False,
    )
    wrapper.load_model()
    _JUDGE_MODEL = wrapper
    return _JUDGE_MODEL


class VLMJudge:
    """Re-predict bboxes with a larger VLM and score by IoU agreement."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self._evaluator = Evaluator()

    def judge(self, sample: HarvestSample) -> dict[str, Any]:
        """Evaluate a single labeled sample by re-predicting with the judge model.

        Returns:
            score (float): mean IoU across prompts [0, 1]
            iou_scores (list[float]): per-prompt IoU values
            judge_bboxes (list[dict]): per-prompt predicted bboxes
        """
        label = sample.label
        if label is None:
            return {"score": 0.0, "iou_scores": [], "judge_bboxes": []}

        pre_path = sample.pre_frame_path
        if not pre_path.exists():
            return {"score": 0.0, "iou_scores": [], "judge_bboxes": []}

        try:
            image = Image.open(pre_path).convert("RGB")
        except Exception:
            logger.warning("Failed to open image: %s", pre_path)
            return {"score": 0.0, "iou_scores": [], "judge_bboxes": []}

        model = _get_judge_model(self.config)

        auto_bbox = BoundingBox(
            x_min=label.bbox_x_min,
            y_min=label.bbox_y_min,
            x_max=label.bbox_x_max,
            y_max=label.bbox_y_max,
        )

        iou_scores: list[float] = []
        judge_bboxes: list[dict[str, float]] = []

        for prompt in self.config.vlm_prompts:
            try:
                pred_bbox = model.predict(image, prompt)
                iou = self._evaluator.iou(pred_bbox, auto_bbox)
                iou_scores.append(iou)
                judge_bboxes.append({
                    "x_min": pred_bbox.x_min,
                    "y_min": pred_bbox.y_min,
                    "x_max": pred_bbox.x_max,
                    "y_max": pred_bbox.y_max,
                    "prompt": prompt,
                })
            except Exception:
                logger.debug("VLM judge prediction failed for prompt: %s", prompt, exc_info=True)
                iou_scores.append(0.0)

        mean_iou = sum(iou_scores) / max(len(iou_scores), 1)

        return {
            "score": mean_iou,
            "iou_scores": iou_scores,
            "judge_bboxes": judge_bboxes,
        }
