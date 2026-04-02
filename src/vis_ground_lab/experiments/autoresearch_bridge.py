"""Bridge functions for autoresearch-style experiment runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from vis_ground_lab.config.loader import load_train_config
from vis_ground_lab.data import load_coco
from vis_ground_lab.data_manager import RouterClassificationDataset
from vis_ground_lab.evaluation import Evaluator
from vis_ground_lab.export import compute_dataset_hash
from vis_ground_lab.models.factory import create_model_wrapper
from vis_ground_lab.optimization import run_optimization
from vis_ground_lab.optimization.optuna_runner import evaluate_detector_checkpoint
from vis_ground_lab.profile import ToolProfile
from vis_ground_lab.training import RouterTrainer


def run_router_autoresearch_experiment(config_path: str, *, profile_path: str | None = None) -> dict[str, Any]:
    """Train/evaluate one router experiment and persist its profile metadata."""
    cfg = load_train_config(config_path)
    if cfg.task.name.lower() != "router_classification":
        raise ValueError("run_router_autoresearch_experiment requires task.name=router_classification")

    train_dataset = RouterClassificationDataset(
        source=cfg.data.train_csv,
        image_root=cfg.data.image_root,
        image_size=cfg.model.router_image_size,
        label_column=cfg.data.label_column,
        aux_label_columns=cfg.data.aux_label_columns,
    )
    val_dataset = RouterClassificationDataset(
        source=cfg.data.val_csv,
        image_root=cfg.data.image_root,
        image_size=cfg.model.router_image_size,
        label_column=cfg.data.label_column,
        aux_label_columns=cfg.data.aux_label_columns,
        label_to_index=train_dataset.label_to_index,
        aux_label_to_index=train_dataset.aux_label_to_index,
    )

    wrapper = create_model_wrapper(cfg.model)
    trainer = RouterTrainer(wrapper, cfg.trainer, aux_loss_weight=cfg.model.router_aux_loss_weight)
    train_result = trainer.train(train_dataset, val_dataset)

    wrapper.load_model(
        checkpoint_path=train_result["checkpoint_path"],
        label_to_index=train_dataset.label_to_index,
        aux_label_to_index=train_dataset.aux_label_to_index,
    )
    loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        collate_fn=RouterTrainer._collate_fn,
    )
    eval_metrics = trainer.evaluate(loader)
    result = {
        "task": "router_classification",
        "primary_metric": "primitive_macro_f1",
        "metrics": eval_metrics,
        "checkpoint_path": train_result["checkpoint_path"],
        "artifacts": {
            "label_maps_path": train_result["label_maps_path"],
            "metrics_path": str(Path(cfg.trainer.checkpoint_dir) / "router_metrics.json"),
        },
    }

    dataset_hash = compute_dataset_hash(Path(cfg.data.train_csv).parent)
    _record_experiment_profile(
        profile_path=profile_path,
        config_path=config_path,
        package_dir=cfg.trainer.checkpoint_dir,
        model_cfg=cfg.model,
        data_cfg=cfg.data,
        dataset_hash=dataset_hash,
        record={
            "task_name": "router_classification",
            "primary_metric": result["primary_metric"],
            "metrics": result["metrics"],
            "checkpoint_path": result["checkpoint_path"],
            "artifacts": result["artifacts"],
            "dataset_hash": dataset_hash,
        },
    )
    return result


def run_detector_autoresearch_experiment(
    config_path: str,
    *,
    n_trials: int = 5,
    profile_path: str | None = None,
) -> dict[str, Any]:
    """Run one detector optimization/evaluation cycle and persist profile metadata."""
    cfg = load_train_config(config_path)
    if cfg.task.name.lower() != "tool_button_detection":
        raise ValueError("run_detector_autoresearch_experiment requires task.name=tool_button_detection")

    if not cfg.data.val_coco or not cfg.data.image_root:
        raise ValueError("Detector autoresearch bridge requires data.val_coco and data.image_root")

    dataset_yaml = cfg.data.dataset_yaml or cfg.data.train_jsonl
    if not dataset_yaml:
        raise ValueError("Detector autoresearch bridge requires data.dataset_yaml or data.train_jsonl")

    coco = load_coco(cfg.data.val_coco)
    class_names = [str(category["name"]) for category in coco.get("categories", [])]
    if not class_names:
        class_names = ["button"]

    optimize_result = run_optimization(
        model_name=cfg.model.name,
        data_yaml=dataset_yaml,
        val_coco=cfg.data.val_coco,
        image_dir=cfg.data.image_root,
        workdir=cfg.trainer.checkpoint_dir,
        class_names=class_names,
        n_trials=n_trials,
    )

    best_package = Path(optimize_result["package_dir"])
    weights_path = best_package / "model.pt"
    metrics = evaluate_detector_checkpoint(
        model_name=cfg.model.name,
        weights=str(weights_path),
        val_coco=cfg.data.val_coco,
        image_dir=cfg.data.image_root,
        reported_map50=float((optimize_result.get("best") or {}).get("mAP50", 0.0)),
    )
    result = {
        "task": "tool_button_detection",
        "primary_metric": "score",
        "metrics": metrics,
        "checkpoint_path": str(weights_path),
        "artifacts": {
            "leaderboard_path": optimize_result["leaderboard_path"],
            "package_dir": optimize_result["package_dir"],
        },
    }

    dataset_hash = compute_dataset_hash(Path(dataset_yaml).parent)
    _record_experiment_profile(
        profile_path=profile_path,
        config_path=config_path,
        package_dir=optimize_result["package_dir"],
        model_cfg=cfg.model,
        data_cfg=cfg.data,
        dataset_hash=dataset_hash,
        record={
            "task_name": "tool_button_detection",
            "primary_metric": result["primary_metric"],
            "metrics": result["metrics"],
            "checkpoint_path": result["checkpoint_path"],
            "artifacts": result["artifacts"],
            "dataset_hash": dataset_hash,
        },
    )
    return result


def _record_experiment_profile(
    *,
    profile_path: str | None,
    config_path: str,
    package_dir: str,
    model_cfg,
    data_cfg,
    dataset_hash: str,
    record: dict[str, Any],
) -> None:
    path = Path(profile_path) if profile_path else Path(package_dir) / "tool_profile.json"
    if path.exists():
        profile = ToolProfile.load(path)
        profile.package_dir = package_dir
        profile.dataset_hash = dataset_hash
    else:
        profile = ToolProfile(
            tool_id=Path(config_path).stem,
            tool_version="autoresearch",
            package_dir=package_dir,
            model_cfg=model_cfg,
            data_config=data_cfg,
            dataset_hash=dataset_hash,
        )
    profile.record_training_run(record)
    profile.save(path)
