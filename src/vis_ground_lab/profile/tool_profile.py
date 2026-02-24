"""ToolProfile: reproducible per-tool bundle for model lifecycle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vis_ground_lab.config.schema import DataConfig, ModelConfig


class ToolProfile(BaseModel):
    """Reproducible per-tool pack containing model, data, and runtime config."""

    tool_id: str
    tool_version: str
    package_dir: str
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    data_config: DataConfig | None = None
    training_history: list[dict[str, Any]] = Field(default_factory=list)
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    failure_store_dir: str | None = None
    review_queue_dir: str | None = None
    reference_frames_dir: str | None = None
    created_at: str = ""
    updated_at: str = ""
    dataset_hash: str = ""

    @classmethod
    def load(cls, path: str | Path) -> ToolProfile:
        """Load a ToolProfile from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        """Save profile to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(tz=timezone.utc).isoformat()
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_package(cls, package_dir: str | Path, tool_id: str, tool_version: str) -> ToolProfile:
        """Create a ToolProfile from an existing model package directory."""
        package_dir = Path(package_dir)
        meta_path = package_dir / "metadata.json"

        model_cfg = ModelConfig()
        dataset_hash = ""
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tool_meta = meta.get("tool_metadata", {})
            dataset_hash = tool_meta.get("dataset_hash", "")

        now = datetime.now(tz=timezone.utc).isoformat()
        return cls(
            tool_id=tool_id,
            tool_version=tool_version,
            package_dir=str(package_dir),
            model_cfg=model_cfg,
            dataset_hash=dataset_hash,
            created_at=now,
            updated_at=now,
        )

    def record_training_run(self, metrics: dict[str, Any]) -> None:
        """Append a training run record to history."""
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metrics,
        }
        self.training_history.append(record)

    def get_model_wrapper(self) -> Any:
        """Instantiate the appropriate model wrapper from stored config."""
        from vis_ground_lab.models.factory import create_model_wrapper

        wrapper = create_model_wrapper(self.model_cfg)
        model_path = Path(self.package_dir) / "model.pt"
        if model_path.exists():
            wrapper.load_model(weights=str(model_path))
        else:
            wrapper.load_model()
        return wrapper
