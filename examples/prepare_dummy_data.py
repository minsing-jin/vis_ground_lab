"""Create a tiny dummy UI grounding dataset for local smoke tests.

Output:
- data/images/*.png
- data/train.jsonl
- data/eval.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_sample(idx: int, split: str, out_dir: Path) -> dict[str, object]:
    width, height = 640, 360
    image = Image.new("RGB", (width, height), color=(242, 244, 247))
    draw = ImageDraw.Draw(image)

    # Simple menu bar region
    draw.rectangle([(0, 0), (width, 48)], fill=(220, 226, 234))

    # Synthetic "File" button bbox
    x1, y1, x2, y2 = 20 + idx * 2, 10, 100 + idx * 2, 36
    draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255), outline=(50, 60, 70), width=2)
    draw.text((x1 + 20, y1 + 7), "File", fill=(10, 10, 10))

    image_name = f"{split}_{idx:03d}.png"
    image_path = out_dir / "images" / image_name
    image.save(image_path)

    return {
        "image_path": image_name,
        "prompt": "click the File button",
        "bbox": [x1, y1, x2, y2],
        "image_id": f"{split}-{idx}",
    }


def write_split(split: str, n: int, out_dir: Path) -> None:
    records = [make_sample(i, split, out_dir) for i in range(n)]
    jsonl_path = out_dir / f"{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    base = Path("data")
    (base / "images").mkdir(parents=True, exist_ok=True)

    write_split("train", n=12, out_dir=base)
    write_split("eval", n=4, out_dir=base)

    print("Created:")
    print("- data/train.jsonl")
    print("- data/eval.jsonl")
    print("- data/images/*.png")
