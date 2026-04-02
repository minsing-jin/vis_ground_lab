"""Optuna-based Bayesian optimization of fusion weights.

Uses a subset of samples (default 50) × N trials (default 30) to find
fusion weights that maximize ensemble judge score.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from data_harvest.core.config import HarvestConfig, LabelerConfig
from data_harvest.core.types import HarvestSample
from ralph_self_improvement.core.config import JudgeConfig, WeightTunerConfig
from ralph_self_improvement.core.types import WeightSnapshot

logger = logging.getLogger(__name__)


class WeightTuner:
    """Optimize fusion weights via Optuna Bayesian search."""

    def __init__(
        self,
        config: WeightTunerConfig,
        judge_config: JudgeConfig,
        harvest_config: HarvestConfig,
    ) -> None:
        self.config = config
        self.judge_config = judge_config
        self.harvest_config = harvest_config

    def tune(self, samples: list[HarvestSample]) -> WeightSnapshot:
        """Run Optuna optimization on a subset of samples.

        Returns a WeightSnapshot with the best fusion weights found.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Subsample
        if len(samples) > self.config.n_samples:
            subset = random.sample(samples, self.config.n_samples)
        else:
            subset = list(samples)

        logger.info(
            "Starting weight tuning: %d samples, %d trials",
            len(subset),
            self.config.n_trials,
        )

        def objective(trial: optuna.Trial) -> float:
            weights = {
                "click_proximity": trial.suggest_float("click_proximity", 0.0, 1.0),
                "diff": trial.suggest_float("diff", 0.0, 1.0),
                "vlm": trial.suggest_float("vlm", 0.0, 1.0),
                "ocr": trial.suggest_float("ocr", 0.0, 0.5),
                "profile_hint": trial.suggest_float("profile_hint", 0.0, 0.5),
            }
            # Normalize to sum to 1
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            return self._evaluate_weights(weights, subset)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
        )

        best_params = study.best_params
        total = sum(best_params.values())
        if total > 0:
            best_weights = {k: v / total for k, v in best_params.items()}
        else:
            best_weights = best_params

        logger.info("Best weights: %s (score=%.4f)", best_weights, study.best_value)

        return WeightSnapshot(
            weights=best_weights,
            objective_value=study.best_value,
            n_trials=len(study.trials),
            n_samples=len(subset),
        )

    def _evaluate_weights(
        self,
        weights: dict[str, float],
        samples: list[HarvestSample],
    ) -> float:
        """Re-label samples with given weights and score via heuristic judge."""
        from data_harvest.labeler.fusion import AutoLabeler
        from ralph_self_improvement.judge.heuristic_judge import HeuristicJudge

        # Create a labeler config with trial weights
        labeler_config = self.harvest_config.labeler.model_copy(deep=True)
        labeler_config.fusion_weights = weights

        labeler = AutoLabeler(config=labeler_config)
        h_judge = HeuristicJudge(self.judge_config)

        scores: list[float] = []
        for sample in samples:
            try:
                result = labeler.label_sample(sample)
                if result is None:
                    scores.append(0.0)
                    continue

                # Temporarily assign the result to get heuristic score
                original_label = sample.label
                sample.label = result
                from PIL import Image

                w, h = 1920, 1080
                if sample.pre_frame_path.exists():
                    try:
                        with Image.open(sample.pre_frame_path) as img:
                            w, h = img.size
                    except Exception:
                        pass

                h_result = h_judge.judge(sample, w, h)
                scores.append(h_result["score"])
                sample.label = original_label
            except Exception:
                scores.append(0.0)

        return sum(scores) / max(len(scores), 1)
