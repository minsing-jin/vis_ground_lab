#!/usr/bin/env bash
set -euo pipefail

# 1) Create toy dataset
python3 examples/prepare_dummy_data.py

# 2) Train (LoRA)
myvg train --config configs/config.example.yaml

# 3) Evaluate local checkpoints adapter
myvg evaluate \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --eval-jsonl data/eval.jsonl \
  --image-root data/images \
  --normalize-mode 0-1000

# 4) One-shot inference
myvg infer \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --image-path data/images/eval_000.png \
  --prompt "click the File button"
