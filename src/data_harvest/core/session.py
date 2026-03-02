"""HarvestSession: workdir management, sample ID generation, iteration."""

from __future__ import annotations

import logging
from pathlib import Path

from data_harvest.core.config import HarvestConfig
from data_harvest.core.types import HarvestSample

logger = logging.getLogger(__name__)


class HarvestSession:
    """Manages the working directory, sample enumeration, and iteration."""

    def __init__(self, config: HarvestConfig) -> None:
        self.config = config
        self.workdir = Path(config.workdir)
        self.samples_dir = self.workdir / "samples"

    def setup(self) -> None:
        """Create workdir and samples directory."""
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Session workdir: %s", self.workdir)

    def next_sample_id(self) -> str:
        """Return the next sequential sample ID (zero-padded 6 digits)."""
        existing = sorted(self.samples_dir.iterdir()) if self.samples_dir.exists() else []
        dirs = [d for d in existing if d.is_dir() and d.name.startswith("sample_")]
        idx = len(dirs) + 1
        return f"sample_{idx:06d}"

    def create_sample(self) -> HarvestSample:
        """Create a new empty HarvestSample with the next ID."""
        sid = self.next_sample_id()
        sample_dir = self.samples_dir / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        return HarvestSample(sample_id=sid, sample_dir=sample_dir)

    def iter_samples(self) -> list[HarvestSample]:
        """Load all existing samples from the workdir."""
        if not self.samples_dir.exists():
            return []
        dirs = sorted(
            d for d in self.samples_dir.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        )
        samples = []
        for d in dirs:
            try:
                samples.append(HarvestSample.load(d))
            except Exception:
                logger.warning("Failed to load sample: %s", d)
        return samples

    def unlabeled_samples(self) -> list[HarvestSample]:
        """Return samples that have events but no labels."""
        return [s for s in self.iter_samples() if s.event is not None and s.label is None]

    def labeled_samples(self) -> list[HarvestSample]:
        """Return samples that have both events and labels."""
        return [s for s in self.iter_samples() if s.event is not None and s.label is not None]

    @property
    def sample_count(self) -> int:
        if not self.samples_dir.exists():
            return 0
        return sum(
            1
            for d in self.samples_dir.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        )
