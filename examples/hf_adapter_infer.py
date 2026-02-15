"""Load LoRA adapter from Hugging Face and run one prediction.

Usage:
python3 examples/hf_adapter_infer.py \
  --base-model microsoft/Florence-2-base \
  --adapter-repo <user>/<repo> \
  --image data/images/eval_000.png \
  --prompt "click the File button"
"""

from __future__ import annotations

import argparse

from PIL import Image

from vis_ground_lab.models.florence2 import Florence2Wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    wrapper = Florence2Wrapper.from_pretrained_adapter(
        base_model_name=args.base_model,
        adapter_path_or_repo=args.adapter_repo,
    )

    image = Image.open(args.image).convert("RGB")
    pred = wrapper.predict(image=image, text=args.prompt)

    print(
        {
            "x1": pred.x_min,
            "y1": pred.y_min,
            "x2": pred.x_max,
            "y2": pred.y_max,
        }
    )


if __name__ == "__main__":
    main()
