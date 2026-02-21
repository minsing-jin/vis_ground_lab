"""Factory for prelabel plugins."""

from __future__ import annotations

from vis_ground_lab.prelabel.base import Prelabeler
from vis_ground_lab.prelabel.florence_teacher import FlorenceTeacherPrelabeler


def create_prelabeler(
    backend: str,
    model_name: str,
    adapter_path_or_repo: str | None = None,
    prompts: list[str] | None = None,
) -> Prelabeler:
    key = backend.lower()
    if key in {"florence2_teacher", "florence2"}:
        return FlorenceTeacherPrelabeler(
            model_name=model_name,
            adapter_path_or_repo=adapter_path_or_repo,
            prompts=prompts,
        )
    raise ValueError(
        f"Unsupported prelabel backend: {backend}. "
        "Register it in vis_ground_lab.prelabel.factory.create_prelabeler()."
    )
