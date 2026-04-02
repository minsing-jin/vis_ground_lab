# vis-ground-lab (`myvg`, `vis-ground-lab`)

[English](README.md) | [한국어](README.ko.md)

`vis-ground-lab` is an experimental toolkit for building small, task-specific UI vision systems locally.

The repository currently has three practical tracks:

- `data-harvest`: routing-first dataset collection and review for game UI actions
- `router_classification`: lightweight vision router training from harvested data
- `grounding` / `tool_button_detection`: legacy or side-path model pipelines kept for compatibility

## Overview

The active focus is not generic grounding.
It is producing stable training data for small vision routers that predict:

- `screen_type`
- `situation_id`
- `primitive_id`

For the current Civ6 workflow, the target is reliable page-level routing data rather than dense button semantics or general-purpose object detection.

High-level flow:

```text
record -> label-auto -> review -> export -> train/evaluate
```

## Installation

Base install:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Recommended extras by workflow:

```bash
python -m pip install -e ".[harvest]"
python -m pip install -e ".[detector]"
```

CLI entry points:

- `myvg`
- `vis-ground-lab`
- `data-harvest`
- `ralph`

## Routing-First Data Harvest

`data-harvest` captures gameplay frames and input events, runs automatic labeling, exposes a review UI, and exports routing datasets.

Current default mode is `routing_only`.
The main labels are:

- `screen_type`: coarse UI page type such as `main_map`, `popup`, or `tech_tree`
- `situation_id`: current game/UI situation
- `primitive_id`: action class the router should call next

Legacy exports such as grounding, COCO, YOLO, and ROI-state are still available, but they are not the primary path.

### Hardware Guidance

Minimum:

- CPU: 4 cores
- RAM: 8 GB
- GPU: optional
- Storage: 20 GB+

Recommended:

- CPU: 8 cores
- RAM: 16 GB+
- GPU: 8 GB+ VRAM or Apple Silicon MPS
- Storage: SSD, 50 GB+

Comfortable for repeated experiments:

- CPU: 8 to 12 cores
- RAM: 32 GB
- GPU: 12 GB+ VRAM
- Storage: SSD, 100 GB+

### Recommended Settings for M2 Pro

In `configs/harvest.yaml`:

- `recorder.capture_fps: 5~10`
- `recorder.enable_hover: false`
- `labeler.vlm.device_map: "auto"`
- `labeler.use_ocr: false`
- `labeler.legacy_weak_signals: false`

Operational notes:

- keep the machine plugged in during long recording sessions
- periodically clean `runs/harvest_session_01/samples` if disk usage grows too much

### Required Setup

Before recording:

- enable macOS `Screen Recording`
- enable macOS `Accessibility`
- keep Civ6 in fullscreen or borderless fullscreen if possible
- keep UI scale fixed if you want consistent crops

Main config files:

- `configs/harvest.yaml`: run configuration
- `configs/harvest_taxonomy/civ6.yaml`: Civ6 routing taxonomy source of truth

Gemini API key:

```bash
export GEMINI_API_KEY=YOUR_KEY
```

You can also place it in `.env`. The tool loads `.env` automatically.

### Quickstart

```bash
data-harvest record -c configs/harvest.yaml
data-harvest label-auto -c configs/harvest.yaml
data-harvest review -c configs/harvest.yaml
data-harvest export -c configs/harvest.yaml --format all
```

What each step does:

- `record`: captures screen frames and input-triggered samples
- `label-auto`: runs Gemini-first automatic labeling with local fallback
- `review`: launches the Gradio review UI for human correction
- `export`: writes routing datasets such as `router_full` and `router_roi`

Other useful commands:

```bash
data-harvest relabel -c configs/harvest.yaml
data-harvest filter -c configs/harvest.yaml
data-harvest stats -c configs/harvest.yaml
data-harvest profiles
```

### Review UI

The review UI is routing-focused.

Editable fields:

- `screen_type`
- `situation_id`
- `primitive_id`
- `router_roi`

Behavior:

- Gemini candidates are shown first
- taxonomy-backed candidates are also available
- the main screenshot and current ROI crop are shown together
- ROI can be changed by two clicks or numeric coordinate edits

Primary actions:

- `Approve`
- `Save Edit + Next`
- `Reject`
- `Update Preview`

### Outputs

If `workdir` is `runs/harvest_session_01`, the typical output layout is:

- raw samples: `runs/harvest_session_01/samples/sample_xxxxxx/`
- auto labels: `label.json`
- review results: `review.json`
- routing exports:
  - `runs/harvest_session_01/export/router_full/`
  - `runs/harvest_session_01/export/router_roi/`

Primary export formats:

- `router_full`: full screenshot routing dataset
- `router_roi`: primary ROI crop routing dataset
- `unified`: routing JSONL with `screen_type`, `situation_id`, and `primitive_id`

Legacy export formats:

- `grounding`
- `coco`
- `yolo`
- `roi_state`

## Router Training

Example config:

- `configs/config.router_classification.yaml`

Train:

```bash
myvg train --config configs/config.router_classification.yaml
```

Evaluate:

```bash
myvg evaluate --config configs/config.router_classification.yaml
```

The current router path expects:

- `task.name: router_classification`
- `model.backend: timm_router`
- CSV-based train and validation splits

## Tool-Specific Button Detector

This path is still useful when you want a small detector for a single tool or game UI.
It is not intended to generalize across applications.

Task config:

```yaml
task:
  name: tool_button_detection
model:
  backend: yolo_ultralytics
  name: yolov8n.pt
```

Quickstart:

```bash
vis-ground-lab extract \
  --video data/raw/excel_session.mp4 \
  --workdir runs/excel_file_btn \
  --fps 2 \
  --every-nth 2 \
  --dedup-threshold 8 \
  --sample-count 400

vis-ground-lab prelabel \
  --image-dir runs/excel_file_btn/extract/sample \
  --out-coco runs/excel_file_btn/prelabel/candidates.coco.json \
  --backend florence2_teacher \
  --model-name microsoft/Florence-2-base \
  --class-name file_button

vis-ground-lab label \
  --image-dir runs/excel_file_btn/extract/sample \
  --candidate-coco runs/excel_file_btn/prelabel/candidates.coco.json \
  --out-coco runs/excel_file_btn/labels/gt.coco.json \
  --class-names file_button,save_button

vis-ground-lab optimize \
  --data-yaml data/tool_dataset/data.yaml \
  --val-coco runs/excel_file_btn/labels/val.coco.json \
  --image-dir data/tool_dataset/images/val \
  --workdir runs/excel_file_btn \
  --class-names file_button,save_button \
  --n-trials 20 \
  --tool-id excel \
  --tool-version 365
```

Packaging and inference:

```bash
vis-ground-lab export \
  --weights runs/excel_file_btn/yolo_runs/trial_000/weights/best.pt \
  --outdir runs/excel_file_btn/package \
  --class-names file_button,save_button \
  --tool-id excel \
  --tool-version 365 \
  --dataset-dir data/tool_dataset/images/val

vis-ground-lab run \
  --package-dir runs/excel_file_btn/package \
  --image-path data/tool_dataset/images/val/example_001.png
```

## Florence-2 Grounding Path

The original grounding path is still supported.

Train:

```bash
myvg train --config configs/config.example.yaml
```

Evaluate:

```bash
myvg evaluate \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --eval-jsonl data/eval.jsonl \
  --image-root data/images \
  --normalize-mode 0-1000
```

Infer:

```bash
myvg infer \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --image-path data/images/eval_000.png \
  --prompt "click the File button"
```

## Ralph Self-Improvement Loop

`ralph` is an RLAIF-style loop for improving harvested labels and fusion weights.

Main config:

- `configs/ralph.yaml`

Useful commands:

```bash
ralph run -c configs/ralph.yaml
ralph judge -c configs/ralph.yaml
ralph tune-weights -c configs/ralph.yaml
ralph report -c configs/ralph.yaml
ralph apply-weights -c configs/ralph.yaml
```

## Project Structure

- `src/data_harvest/`: routing-first capture, auto-label, review, and export pipeline
- `src/vis_ground_lab/`: training, evaluation, packaging, and model backends
- `src/ralph_self_improvement/`: judgment, scoring, and self-improvement loop
- `configs/`: training and harvest configs
- `experiments/`: focused experiment programs
- `third_party/autoresearch/`: vendored experiment support metadata

## Development Notes

Recommended workflow:

- start with a small dataset
- review failure cases early
- train the router before scaling data volume
- use `myvg evaluate` or `data-harvest stats` before larger iterations

Project tracking files:

- `docs/WORK_LOG.md`
- `docs/WORKBOARD.md`
