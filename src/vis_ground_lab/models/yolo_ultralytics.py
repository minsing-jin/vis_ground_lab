"""Ultralytics YOLO wrapper for tool-specific UI button detection."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image

from vis_ground_lab.base import BoundingBox, UIElement


class YoloUltralyticsWrapper:
    """Lightweight detector wrapper for small tool/game-specific datasets."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self.model: Any | None = None
        self.class_names: dict[int, str] = {}
        self.weights_path: Path | None = None

    def load_model(self, weights: str | None = None) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for backend=yolo_ultralytics. Install with: pip install ultralytics"
            ) from exc

        source = weights or self.model_name
        self.model = YOLO(source)

        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}

    def train(self, dataset: str, cfg: dict[str, Any], workdir: str) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        result = self.model.train(
            data=dataset,
            epochs=int(cfg.get("epochs", 20)),
            batch=int(cfg.get("batch_size", 8)),
            imgsz=int(cfg.get("imgsz", 640)),
            lr0=float(cfg.get("learning_rate", 1e-3)),
            patience=int(cfg.get("patience", 10)),
            project=str(Path(workdir) / "yolo_runs"),
            name=str(cfg.get("run_name", "train")),
            exist_ok=True,
            verbose=False,
        )

        save_dir = Path(getattr(result, "save_dir", Path(workdir) / "yolo_runs" / str(cfg.get("run_name", "train"))))
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            self.weights_path = best_pt
            self.load_model(weights=str(best_pt))

        metrics = self._extract_metrics(result)
        metrics["save_dir"] = str(save_dir)
        if self.weights_path:
            metrics["best_weights"] = str(self.weights_path)
        return metrics

    def predict(self, image: str | Path | Image.Image) -> list[UIElement]:
        if self.model is None:
            self.load_model()

        result = self.model.predict(source=image, verbose=False)[0]
        boxes = result.boxes

        outputs: list[UIElement] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            name = self.class_names.get(cls_id, str(cls_id))
            outputs.append(
                UIElement(
                    class_name=name,
                    bbox=BoundingBox(
                        x_min=float(xyxy[0]),
                        y_min=float(xyxy[1]),
                        x_max=float(xyxy[2]),
                        y_max=float(xyxy[3]),
                    ),
                    score=conf,
                )
            )
        return outputs

    def export(self, outdir: str | Path, formats: list[str] | None = None) -> dict[str, str]:
        if self.model is None:
            self.load_model(weights=str(self.weights_path) if self.weights_path else None)

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        formats = formats or ["pt", "onnx"]
        artifacts: dict[str, str] = {}

        if "pt" in formats and self.weights_path and self.weights_path.exists():
            target = outdir / "model.pt"
            shutil.copy2(self.weights_path, target)
            artifacts["pt"] = str(target)

        if "onnx" in formats:
            exported_path = self.model.export(format="onnx", imgsz=640)
            onnx_path = Path(str(exported_path))
            target = outdir / "model.onnx"
            shutil.copy2(onnx_path, target)
            artifacts["onnx"] = str(target)

        (outdir / "export_artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
        return artifacts

    def benchmark_latency(self, images: list[str | Path], repeats: int = 1) -> dict[str, float]:
        if self.model is None:
            self.load_model(weights=str(self.weights_path) if self.weights_path else None)

        durations: list[float] = []
        for _ in range(max(1, repeats)):
            for image in images:
                start = time.perf_counter()
                self.model.predict(source=str(image), verbose=False)
                durations.append((time.perf_counter() - start) * 1000.0)

        if not durations:
            return {"latency_ms_mean": 0.0, "latency_ms_p95": 0.0}

        sorted_values = sorted(durations)
        p95_index = min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1))))
        return {
            "latency_ms_mean": float(sum(sorted_values) / len(sorted_values)),
            "latency_ms_p95": float(sorted_values[p95_index]),
        }

    @staticmethod
    def _extract_metrics(result: Any) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metrics_box = getattr(result, "box", None)
        if metrics_box is not None:
            map50 = getattr(metrics_box, "map50", None)
            if map50 is not None:
                metrics["mAP50"] = float(map50)
        return metrics
