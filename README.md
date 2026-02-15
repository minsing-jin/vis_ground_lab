# myvg

Local-first visual grounding toolkit for UI automation.

## CLI

```bash
myvg train --config config.yaml
```

```bash
myvg push \
  --base-model microsoft/Florence-2-base \
  --adapter-path checkpoints \
  --repo-name <hf_user>/<adapter_repo> \
  --token <hf_token>
```

```bash
myvg infer \
  --base-model microsoft/Florence-2-base \
  --adapter-repo <hf_user>/<adapter_repo> \
  --image-path ./ui.png \
  --prompt "click the File button"
```

```bash
myvg evaluate \
  --base-model microsoft/Florence-2-base \
  --adapter-repo <hf_user>/<adapter_repo> \
  --eval-jsonl data/eval.jsonl \
  --image-root data/images \
  --normalize-mode 0-1000
```
