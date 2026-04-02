"""JSONL-based metrics tracker for iteration results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ralph_self_improvement.core.types import IterationResult

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Append/load IterationResult records to a JSONL file.

    Supports resume by loading existing records on init.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._results: list[IterationResult] = []
        self._load()

    def _load(self) -> None:
        """Load existing results from disk."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._results.append(IterationResult.from_dict(json.loads(line)))
        logger.info("Loaded %d existing iteration results from %s", len(self._results), self.path)

    def append(self, result: IterationResult) -> None:
        """Append a single iteration result."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(result.to_json() + "\n")
        self._results.append(result)

    @property
    def results(self) -> list[IterationResult]:
        return list(self._results)

    @property
    def last_iteration(self) -> int:
        """Return the last iteration number, or 0 if no results."""
        if not self._results:
            return 0
        return self._results[-1].iteration

    def best_iteration(self) -> IterationResult | None:
        """Return the iteration with the highest mean_iou."""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.mean_ensemble_score)

    def improvement_stalled(self, patience: int, threshold: float) -> bool:
        """Check if improvement has stalled for `patience` iterations.

        Returns True if the last `patience` iterations have not improved
        mean_ensemble_score by more than `threshold` over the best seen.
        """
        if len(self._results) < patience + 1:
            return False

        best = self.best_iteration()
        if best is None:
            return False

        recent = self._results[-patience:]
        return all(
            best.mean_ensemble_score - r.mean_ensemble_score > -threshold
            and r.mean_ensemble_score - best.mean_ensemble_score < threshold
            for r in recent
            if r.iteration != best.iteration
        )
