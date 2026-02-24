"""Runtime drift detection and confidence tracking."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image

from vis_ground_lab.base import FrameAnalysis
from vis_ground_lab.data.dedup import hamming_distance, phash_from_image


class RuntimeMonitor:
    """Monitor runtime predictions for drift and confidence degradation."""

    def __init__(
        self,
        reference_frames: list[str | Path] | None = None,
        drift_hash_threshold: int = 12,
        confidence_window: int = 100,
        low_confidence_threshold: float = 0.3,
    ) -> None:
        self.drift_hash_threshold = drift_hash_threshold
        self.confidence_window = confidence_window
        self.low_confidence_threshold = low_confidence_threshold

        self._reference_hashes: list[int] = []
        if reference_frames:
            for frame_path in reference_frames:
                with Image.open(frame_path) as img:
                    self._reference_hashes.append(phash_from_image(img))

        self._confidence_history: deque[float] = deque(maxlen=confidence_window)
        self._drift_detected = False

    def observe(self, analysis: FrameAnalysis, frame_image: Image.Image | None = None) -> dict[str, Any]:
        """Process a new frame analysis and return monitoring signals."""
        # Track confidence
        scores = [e.score for e in analysis.elements]
        mean_conf = sum(scores) / len(scores) if scores else 0.0
        self._confidence_history.append(mean_conf)

        # Uncertain elements
        uncertain = [e for e in analysis.elements if e.score < self.low_confidence_threshold]

        # Drift detection
        drift_score = 0.0
        if frame_image is not None:
            drift_score = self.detect_drift(frame_image)
        is_drifted = drift_score > (self.drift_hash_threshold / 64.0)
        self._drift_detected = self._drift_detected or is_drifted

        # Should we collect this as a failure?
        should_collect = (
            is_drifted
            or mean_conf < self.low_confidence_threshold
            or len(uncertain) > len(analysis.elements) / 2
        )

        return {
            "is_drifted": is_drifted,
            "drift_score": round(drift_score, 4),
            "mean_confidence": round(mean_conf, 4),
            "uncertain_count": len(uncertain),
            "should_collect_failure": should_collect,
        }

    def detect_drift(self, frame_image: Image.Image) -> float:
        """Compare frame against reference set via pHash. Returns 0.0-1.0 drift score."""
        if not self._reference_hashes:
            return 0.0

        current_hash = phash_from_image(frame_image)
        distances = [hamming_distance(current_hash, ref) for ref in self._reference_hashes]
        min_distance = min(distances)
        # Normalize to 0-1 range (64-bit hash)
        return min_distance / 64.0

    def save_state(self, path: str | Path) -> None:
        """Persist monitor state to JSON."""
        state = {
            "reference_hashes": self._reference_hashes,
            "confidence_history": list(self._confidence_history),
            "drift_detected": self._drift_detected,
        }
        Path(path).write_text(json.dumps(state), encoding="utf-8")

    def load_state(self, path: str | Path) -> None:
        """Restore monitor state from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._reference_hashes = data.get("reference_hashes", [])
        self._confidence_history = deque(
            data.get("confidence_history", []), maxlen=self.confidence_window
        )
        self._drift_detected = data.get("drift_detected", False)
