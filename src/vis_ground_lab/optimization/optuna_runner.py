"""Optuna runner for detector hyperparameter search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from vis_ground_lab.base import BoundingBox, UIElement
from vis_ground_lab.config.schema import ModelConfig
from vis_ground_lab.data import coco_bbox_to_xyxy, load_coco
from vis_ground_lab.export import compute_dataset_hash, create_model_package
from vis_ground_lab.models.factory import create_model_wrapper


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    union = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _compute_click_success(
    model: Any,
    coco_path: str | Path,
    image_dir: str | Path,
    iou_threshold: float = 0.5,
) -> float:
    coco = load_coco(coco_path)
    image_dir = Path(image_dir)

    categories = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    images = {int(i["id"]): i for i in coco.get("images", [])}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        grouped.setdefault(int(ann["image_id"]), []).append(ann)

    total = 0
    success = 0

    for image_id, anns in grouped.items():
        image_info = images.get(image_id)
        if not image_info:
            continue
        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            continue
        preds: list[UIElement] = model.predict(image_path)
        for ann in anns:
            total += 1
            gt_name = categories.get(int(ann["category_id"]), "")
            gt_bbox = coco_bbox_to_xyxy(ann["bbox"])
            gt_box = BoundingBox(*map(float, gt_bbox))

            matched = False
            for pred in preds:
                if pred.class_name != gt_name:
                    continue
                if _iou(pred.bbox, gt_box) >= iou_threshold:
                    matched = True
                    break
            if matched:
                success += 1

    if total == 0:
        return 0.0
    return float(success / total)


def _latency_penalty(latency_ms: float, budget_ms: float = 30.0) -> float:
    if latency_ms <= budget_ms:
        return 0.0
    return (latency_ms - budget_ms) / 1000.0


def _extract_map50(metrics: dict[str, Any]) -> float:
    value = metrics.get("mAP50")
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def render_leaderboard_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No trials."

    headers = ["trial", "score", "mAP50", "click_success", "latency_ms"]
    line = " | ".join(headers)
    sep = "-+-".join("-" * len(h) for h in headers)
    body = []
    for row in rows:
        body.append(
            " | ".join(
                [
                    str(row.get("trial", "")),
                    f"{float(row.get('score', 0.0)):.4f}",
                    f"{float(row.get('mAP50', 0.0)):.4f}",
                    f"{float(row.get('click_success', 0.0)):.4f}",
                    f"{float(row.get('latency_ms', 0.0)):.2f}",
                ]
            )
        )
    return "\n".join([line, sep, *body])


def run_optimization(
    model_name: str,
    data_yaml: str,
    val_coco: str,
    image_dir: str,
    workdir: str,
    class_names: list[str],
    n_trials: int = 10,
    timeout_sec: int | None = None,
    export_formats: list[str] | None = None,
    tool_id: str = "tool",
    tool_version: str = "v1",
) -> dict[str, Any]:
    """Run Optuna HPO and export best detector package."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required. Install with: pip install optuna") from exc

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    leaderboard: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        cfg = {
            "epochs": trial.suggest_int("epochs", 5, 30),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "imgsz": trial.suggest_categorical("imgsz", [416, 512, 640]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "patience": trial.suggest_int("patience", 3, 10),
            "run_name": f"trial_{trial.number:03d}",
        }

        wrapper = create_model_wrapper(ModelConfig(backend="yolo_ultralytics", name=model_name))
        train_metrics = wrapper.train(dataset=data_yaml, cfg=cfg, workdir=str(workdir_path))

        sample_images = sorted(Path(image_dir).glob("*.png"))[:8]
        if not sample_images:
            sample_images = sorted(Path(image_dir).glob("*.jpg"))[:8]
        latency = wrapper.benchmark_latency([str(p) for p in sample_images], repeats=1)

        map50 = _extract_map50(train_metrics)
        click_success = _compute_click_success(model=wrapper, coco_path=val_coco, image_dir=image_dir)
        latency_ms = float(latency.get("latency_ms_mean", 0.0))

        score = 0.6 * map50 + 0.4 * click_success - _latency_penalty(latency_ms)
        trial.report(score, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        row = {
            "trial": trial.number,
            "score": score,
            "mAP50": map50,
            "click_success": click_success,
            "latency_ms": latency_ms,
            "train_metrics": train_metrics,
            "latency": latency,
            "cfg": cfg,
            "weights": str(wrapper.weights_path) if wrapper.weights_path else None,
        }
        leaderboard.append(row)
        return score

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    leaderboard_sorted = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
    leaderboard_path = workdir_path / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(leaderboard_sorted, indent=2), encoding="utf-8")

    if not leaderboard_sorted:
        return {
            "leaderboard_path": str(leaderboard_path),
            "table": render_leaderboard_table([]),
            "best": None,
        }

    best = leaderboard_sorted[0]
    best_weights = best.get("weights")
    best_wrapper = create_model_wrapper(ModelConfig(backend="yolo_ultralytics", name=model_name))
    if best_weights:
        best_wrapper.load_model(weights=str(best_weights))
    else:
        best_wrapper.load_model()

    export_dir = workdir_path / "best_export"
    artifacts = best_wrapper.export(export_dir, formats=export_formats or ["pt", "onnx"])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    tool_metadata = {
        "tool_id": tool_id,
        "tool_version": tool_version,
        "dataset_hash": compute_dataset_hash(Path(data_yaml).parent),
    }

    package_dir = workdir_path / "best_package"
    create_model_package(
        outdir=package_dir,
        artifacts=artifacts,
        label_map=label_map,
        preprocessing={"imgsz": int(best["cfg"]["imgsz"])},
        postprocessing={"conf_threshold": 0.25, "iou_threshold": 0.5},
        metrics={
            "objective": float(best["score"]),
            "mAP50": float(best["mAP50"]),
            "click_success": float(best["click_success"]),
        },
        latency=best["latency"],
        tool_metadata=tool_metadata,
    )

    return {
        "leaderboard_path": str(leaderboard_path),
        "table": render_leaderboard_table(leaderboard_sorted),
        "best": best,
        "package_dir": str(package_dir),
    }
