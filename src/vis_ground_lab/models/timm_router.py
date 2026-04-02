"""Small image classifier for primitive routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn


class _RouterHeadModel(nn.Module):
    """Shared image backbone with one primary and multiple auxiliary heads."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, aux_dims: dict[str, int], dropout: float):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.primary_head = nn.Linear(feature_dim, num_classes)
        self.aux_heads = nn.ModuleDict({name: nn.Linear(feature_dim, size) for name, size in aux_dims.items()})

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(pixel_values)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = self.dropout(features)
        return {
            "primary_logits": self.primary_head(features),
            "aux_logits": {name: head(features) for name, head in self.aux_heads.items()},
        }


class TimmRouterWrapper:
    """Wrapper around a timm image encoder for routing classification."""

    def __init__(
        self,
        model_name: str = "resnet18",
        *,
        pretrained: bool = False,
        image_size: int = 224,
        dropout: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.image_size = image_size
        self.dropout = dropout

        self.model: _RouterHeadModel | None = None
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: dict[int, str] = {}
        self.aux_label_to_index: dict[str, dict[str, int]] = {}
        self.index_to_aux_label: dict[str, dict[int, str]] = {}

    def load_model(
        self,
        *,
        checkpoint_path: str | None = None,
        label_to_index: dict[str, int] | None = None,
        aux_label_to_index: dict[str, dict[str, int]] | None = None,
    ) -> None:
        checkpoint = None
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            label_to_index = label_to_index or checkpoint.get("label_to_index")
            aux_label_to_index = aux_label_to_index or checkpoint.get("aux_label_to_index")

        if not label_to_index:
            raise ValueError("Router model requires label_to_index before load_model().")

        self.label_to_index = dict(label_to_index)
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.aux_label_to_index = {key: dict(value) for key, value in (aux_label_to_index or {}).items()}
        self.index_to_aux_label = {
            key: {idx: label for label, idx in value.items()}
            for key, value in self.aux_label_to_index.items()
        }

        backbone = self._create_backbone()
        feature_dim = getattr(backbone, "num_features", None)
        if not feature_dim:
            raise RuntimeError(f"Unable to infer feature dimension for router backbone '{self.model_name}'")

        self.model = _RouterHeadModel(
            backbone=backbone,
            feature_dim=int(feature_dim),
            num_classes=len(self.label_to_index),
            aux_dims={name: len(mapping) for name, mapping in self.aux_label_to_index.items()},
            dropout=self.dropout,
        )

        if checkpoint is not None:
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.model.load_state_dict(state_dict)

    def save_checkpoint(self, path: str | Path, *, metrics: dict[str, Any] | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("Model is unavailable. Call load_model() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "label_to_index": self.label_to_index,
                "aux_label_to_index": self.aux_label_to_index,
                "model_name": self.model_name,
                "image_size": self.image_size,
                "metrics": metrics or {},
            },
            path,
        )
        return path

    def predict(self, image: str | Path | Image.Image) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is unavailable. Call load_model() first.")

        device = next(self.model.parameters()).device
        tensor = self._image_to_tensor(image).unsqueeze(0).to(device)

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(tensor)

        primary_probs = torch.softmax(outputs["primary_logits"], dim=-1)[0]
        primary_index = int(torch.argmax(primary_probs).item())
        prediction: dict[str, Any] = {
            "primitive_id": self.index_to_label[primary_index],
            "primitive_confidence": float(primary_probs[primary_index].item()),
        }
        for name, logits in outputs["aux_logits"].items():
            if logits.shape[-1] == 0:
                continue
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(torch.argmax(probs).item())
            prediction[name] = self.index_to_aux_label[name][idx]
            prediction[f"{name}_confidence"] = float(probs[idx].item())
        return prediction

    def _create_backbone(self) -> nn.Module:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for backend=timm_router. Install with: pip install timm") from exc

        kwargs = {"pretrained": self.pretrained, "num_classes": 0, "global_pool": "avg"}
        try:
            return timm.create_model(self.model_name, **kwargs)
        except Exception:
            if not self.pretrained:
                raise
            kwargs["pretrained"] = False
            return timm.create_model(self.model_name, **kwargs)

    def _image_to_tensor(self, image: str | Path | Image.Image) -> torch.Tensor:
        pil_image = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        resized = pil_image.resize((self.image_size, self.image_size))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
