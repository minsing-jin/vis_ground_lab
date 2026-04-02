# autoresearch

Minimal vendored scaffold reserved for `karpathy/autoresearch`-style experiment loops.

This tree is intentionally kept thin in this repo:

- external orchestration concepts live here
- task-specific adapters live under `experiments/autoresearch/`
- repo modeling/data code stays under `src/vis_ground_lab/`

When syncing with upstream, record the source ref in `VENDORED_FROM.json` and keep this directory read-only during normal experiment runs.
