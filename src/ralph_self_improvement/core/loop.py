"""ImprovementLoop: main orchestrator for the RLAIF self-improvement cycle."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from ralph_self_improvement.core.config import RalphConfig
from ralph_self_improvement.core.types import IterationResult, Judgment
from ralph_self_improvement.judge.ensemble import EnsembleJudge
from ralph_self_improvement.optimizer.dpo_trainer import DPOTrainer
from ralph_self_improvement.optimizer.weight_tuner import WeightTuner
from ralph_self_improvement.reward.preference import PreferencePairGenerator
from ralph_self_improvement.reward.scoring import compute_reward
from ralph_self_improvement.tracker.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class ImprovementLoop:
    """Orchestrate: judge → preference pairs → weight tuning → DPO/SFT → evaluate → repeat."""

    def __init__(self, config: RalphConfig) -> None:
        self.config = config
        self.harvest_config = HarvestConfig.from_yaml(config.harvest_config_path)
        self.session = HarvestSession(self.harvest_config)
        self.tracker = MetricsTracker(config.metrics_path)
        self.judge = EnsembleJudge(config.judge)

    def run(self) -> list[IterationResult]:
        """Run the full improvement loop up to max_iterations."""
        results: list[IterationResult] = []
        start_iter = self.tracker.last_iteration + 1

        for i in range(start_iter, start_iter + self.config.loop.max_iterations):
            logger.info("=== Iteration %d ===", i)

            result = self._run_iteration(i)
            self.tracker.append(result)
            results.append(result)

            if self.tracker.improvement_stalled(
                self.config.loop.patience,
                self.config.loop.improvement_threshold,
            ):
                logger.info("Improvement stalled for %d iterations. Stopping.", self.config.loop.patience)
                break

        return results

    def _run_iteration(self, iteration: int) -> IterationResult:
        """Execute a single iteration of the improvement loop."""
        samples = self.session.labeled_samples()
        if not samples:
            logger.warning("No labeled samples found.")
            return IterationResult(iteration=iteration)

        logger.info("Judging %d labeled samples...", len(samples))

        # Step 1: Judge
        judgments = self.judge.judge_batch(samples)
        self._save_judgments(judgments)

        # Compute mean scores
        rewards = [compute_reward(j) for j in judgments]
        mean_score = sum(rewards) / max(len(rewards), 1)
        mean_iou = sum(j.iou_with_judge for j in judgments) / max(len(judgments), 1)

        # Step 2: Generate preference pairs
        pair_gen = PreferencePairGenerator()
        pairs = pair_gen.generate(judgments, samples)
        if pairs:
            PreferencePairGenerator.save_jsonl(pairs, self.config.preferences_path)
        logger.info("Generated %d preference pairs.", len(pairs))

        # Step 3: Weight tuning (optional)
        weight_snapshot = None
        if self.config.loop.run_weight_tuning and len(samples) >= 5:
            tuner = WeightTuner(
                config=self.config.weight_tuner,
                judge_config=self.config.judge,
                harvest_config=self.harvest_config,
            )
            weight_snapshot = tuner.tune(samples)

        # Step 4: DPO/SFT training (optional)
        checkpoint_path = None
        if self.config.loop.run_dpo and pairs:
            dpo = DPOTrainer(self.config.dpo)
            checkpoint_path = dpo.train(pairs)

        # Determine improvement
        best = self.tracker.best_iteration()
        improved = best is None or mean_score > best.mean_ensemble_score

        return IterationResult(
            iteration=iteration,
            mean_iou=mean_iou,
            mean_distance_px=0.0,
            mean_ensemble_score=mean_score,
            n_samples=len(samples),
            n_preference_pairs=len(pairs),
            weight_snapshot=weight_snapshot,
            checkpoint_path=checkpoint_path,
            improved=improved,
        )

    def _save_judgments(self, judgments: list[Judgment]) -> None:
        """Append judgments to the JSONL file."""
        path = Path(self.config.judgments_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for j in judgments:
                f.write(j.to_json() + "\n")
