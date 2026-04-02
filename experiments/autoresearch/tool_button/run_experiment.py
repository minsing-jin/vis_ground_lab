from __future__ import annotations

import argparse
import json
from pathlib import Path

from vis_ground_lab.experiments import run_detector_autoresearch_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one detector autoresearch experiment")
    parser.add_argument("--config", default="configs/config.tool_button_detection.yaml")
    parser.add_argument("--profile-path", default=None)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--out", default="experiments/autoresearch/tool_button/latest_metrics.json")
    args = parser.parse_args()

    result = run_detector_autoresearch_experiment(
        args.config,
        n_trials=args.n_trials,
        profile_path=args.profile_path,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
