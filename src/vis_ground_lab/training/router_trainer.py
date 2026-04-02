"""Training loop for primitive routing classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vis_ground_lab.config.schema import TrainerConfig
from vis_ground_lab.data_manager import IGNORE_INDEX, RouterClassificationDataset
from vis_ground_lab.evaluation import Evaluator
from vis_ground_lab.models.timm_router import TimmRouterWrapper


class RouterTrainer:
    """Small training engine for routing classification."""

    def __init__(self, model_wrapper: TimmRouterWrapper, config: TrainerConfig, *, aux_loss_weight: float = 0.2) -> None:
        self.model_wrapper = model_wrapper
        self.config = config
        self.aux_loss_weight = aux_loss_weight

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.CrossEntropyLoss()
        self.aux_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def train(
        self,
        train_dataset: RouterClassificationDataset,
        val_dataset: RouterClassificationDataset,
    ) -> dict[str, Any]:
        self.model_wrapper.load_model(
            label_to_index=train_dataset.label_to_index,
            aux_label_to_index=train_dataset.aux_label_to_index,
        )
        assert self.model_wrapper.model is not None
        self.model_wrapper.model.to(self.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )

        optimizer = AdamW(self.model_wrapper.model.parameters(), lr=self.config.learning_rate)
        history: list[dict[str, Any]] = []
        best_metrics = {"primitive_macro_f1": -1.0}
        best_checkpoint = self.checkpoint_dir / "best_router.pt"

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(val_loader)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
            }
            history.append(epoch_metrics)
            if val_metrics["primitive_macro_f1"] >= best_metrics["primitive_macro_f1"]:
                best_metrics = dict(epoch_metrics)
                self.model_wrapper.save_checkpoint(best_checkpoint, metrics=best_metrics)

        metrics_path = self.checkpoint_dir / "router_metrics.json"
        label_maps_path = self.checkpoint_dir / "router_label_maps.json"
        metrics_payload = {
            "best": best_metrics,
            "history": history,
            "checkpoint_path": str(best_checkpoint),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        label_maps_path.write_text(
            json.dumps(
                {
                    "label_to_index": train_dataset.label_to_index,
                    "aux_label_to_index": train_dataset.aux_label_to_index,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            **metrics_payload,
            "label_maps_path": str(label_maps_path),
        }

    def evaluate(self, loader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        assert self.model_wrapper.model is not None
        self.model_wrapper.model.eval()
        primary_predictions: list[int] = []
        primary_targets: list[int] = []
        aux_predictions: dict[str, list[int]] = {name: [] for name in self.model_wrapper.aux_label_to_index}
        aux_targets: dict[str, list[int]] = {name: [] for name in self.model_wrapper.aux_label_to_index}

        with torch.inference_mode():
            for batch in loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model_wrapper.model(pixel_values)

                pred = torch.argmax(outputs["primary_logits"], dim=-1)
                primary_predictions.extend(pred.cpu().tolist())
                primary_targets.extend(labels.cpu().tolist())

                for name, logits in outputs["aux_logits"].items():
                    targets = batch["aux_labels"][name].to(self.device)
                    if logits.shape[-1] == 0:
                        continue
                    aux_pred = torch.argmax(logits, dim=-1)
                    for pred_idx, target_idx in zip(aux_pred.cpu().tolist(), targets.cpu().tolist()):
                        if target_idx == IGNORE_INDEX:
                            continue
                        aux_predictions[name].append(pred_idx)
                        aux_targets[name].append(target_idx)

        metrics = Evaluator().evaluate_classification(primary_predictions, primary_targets)
        result = {
            "primitive_macro_f1": metrics["macro_f1"],
            "primitive_accuracy": metrics["accuracy"],
        }
        for name, targets in aux_targets.items():
            if not targets:
                result[f"{name}_accuracy"] = 0.0
                continue
            aux_metric = Evaluator().evaluate_classification(aux_predictions[name], targets)
            result[f"{name}_accuracy"] = aux_metric["accuracy"]
        return result

    def _run_train_epoch(self, loader: DataLoader[dict[str, Any]], optimizer: AdamW) -> float:
        assert self.model_wrapper.model is not None
        self.model_wrapper.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad(set_to_none=True)
            outputs = self.model_wrapper.model(pixel_values)
            loss = self.loss_fn(outputs["primary_logits"], labels)

            for name, logits in outputs["aux_logits"].items():
                targets = batch["aux_labels"][name].to(self.device)
                if logits.shape[-1] == 0:
                    continue
                loss = loss + (self.aux_loss_weight * self.aux_loss_fn(logits, targets))

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return total_loss / max(total_batches, 1)

    @staticmethod
    def _collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
        aux_labels: dict[str, torch.Tensor] = {}
        if items:
            for name in items[0]["aux_labels"]:
                aux_labels[name] = torch.tensor([item["aux_labels"][name] for item in items], dtype=torch.long)
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in items]),
            "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
            "aux_labels": aux_labels,
            "sample_ids": [item["sample_id"] for item in items],
            "image_paths": [item["image_path"] for item in items],
        }
