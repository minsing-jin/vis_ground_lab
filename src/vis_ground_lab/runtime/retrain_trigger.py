"""Evaluate retrain conditions and trigger incremental model updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RetrainDecision:
    """Result of evaluating whether retraining should be triggered."""

    should_retrain: bool
    reason: str
    failure_count: int
    correction_count: int
    drift_detected: bool
    recommended_strategy: str  # "incremental"|"full"|"none"


class RetrainTrigger:
    """Evaluate retrain conditions based on failure store and review queue state."""

    def __init__(
        self,
        failure_threshold: int = 50,
        correction_threshold: int = 20,
        drift_threshold: float = 0.5,
        cooldown_hours: float = 24.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.correction_threshold = correction_threshold
        self.drift_threshold = drift_threshold
        self.cooldown_hours = cooldown_hours
        self._last_retrain_time: str | None = None

    def evaluate(
        self,
        failure_store: Any,
        review_queue: Any,
        monitor: Any | None = None,
    ) -> RetrainDecision:
        """Decide whether to trigger retraining based on current state."""
        failure_count = failure_store.count()
        queue_stats = review_queue.stats()
        correction_count = queue_stats.get("with_corrections", 0)

        drift_detected = False
        if monitor is not None:
            drift_detected = getattr(monitor, "_drift_detected", False)

        # Check cooldown
        if self._last_retrain_time:
            last = datetime.fromisoformat(self._last_retrain_time)
            now = datetime.now(tz=timezone.utc)
            hours_since = (now - last).total_seconds() / 3600.0
            if hours_since < self.cooldown_hours:
                return RetrainDecision(
                    should_retrain=False,
                    reason=f"cooldown active ({hours_since:.1f}h < {self.cooldown_hours}h)",
                    failure_count=failure_count,
                    correction_count=correction_count,
                    drift_detected=drift_detected,
                    recommended_strategy="none",
                )

        reasons: list[str] = []
        if failure_count >= self.failure_threshold:
            reasons.append(f"failures={failure_count}>={self.failure_threshold}")
        if correction_count >= self.correction_threshold:
            reasons.append(f"corrections={correction_count}>={self.correction_threshold}")
        if drift_detected:
            reasons.append("drift_detected")

        should = len(reasons) > 0
        strategy = "none"
        if should:
            strategy = "full" if drift_detected else "incremental"

        return RetrainDecision(
            should_retrain=should,
            reason="; ".join(reasons) if reasons else "no retrain conditions met",
            failure_count=failure_count,
            correction_count=correction_count,
            drift_detected=drift_detected,
            recommended_strategy=strategy,
        )

    def record_retrain(self) -> None:
        """Record that a retrain was executed (for cooldown tracking)."""
        self._last_retrain_time = datetime.now(tz=timezone.utc).isoformat()
