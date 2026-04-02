"""Model package exporter for tool-specific detector artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_dataset_hash(dataset_dir: str | Path) -> str:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return ""
    files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    return _hash_files(files)


def create_model_package(
    outdir: str | Path,
    artifacts: dict[str, str],
    label_map: dict[str, int],
    preprocessing: dict[str, Any],
    postprocessing: dict[str, Any],
    metrics: dict[str, Any],
    latency: dict[str, Any],
    tool_metadata: dict[str, Any],
) -> Path:
    """Create a self-contained model package directory."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fmt, src in artifacts.items():
        src_path = Path(src)
        if not src_path.exists():
            continue
        target_name = f"model.{fmt}" if fmt != "pt" else "model.pt"
        shutil.copy2(src_path, outdir / target_name)

    (outdir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (outdir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    (outdir / "postprocessing.json").write_text(json.dumps(postprocessing, indent=2), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
    (outdir / "tool_metadata.json").write_text(json.dumps(tool_metadata, indent=2), encoding="utf-8")
    (outdir / "metadata.json").write_text(
        json.dumps(
            {
                "tool_metadata": tool_metadata,
                "metrics": metrics,
                "latency": latency,
                "preprocessing": preprocessing,
                "postprocessing": postprocessing,
                "label_map": label_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return outdir
