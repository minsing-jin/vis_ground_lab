"""Config loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from vis_ground_lab.config.schema import FactoryConfig, TrainRunConfig


def load_train_config(path: str | Path) -> TrainRunConfig:
    """Load YAML config into a validated TrainRunConfig object."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TrainRunConfig.model_validate(raw)


def load_factory_config(path: str | Path) -> FactoryConfig:
    """Load YAML config into a validated FactoryConfig object."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return FactoryConfig.model_validate(raw)
