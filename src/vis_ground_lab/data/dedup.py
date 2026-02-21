"""Image de-duplication utilities based on perceptual hash."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _dct_matrix(n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float32)
    factor = np.pi / (2.0 * n)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)
    for k in range(n):
        coeff = scale0 if k == 0 else scale
        for i in range(n):
            matrix[k, i] = coeff * np.cos((2 * i + 1) * k * factor)
    return matrix


def phash_from_image(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Compute a simple pHash integer from a PIL image."""
    size = hash_size * highfreq_factor
    gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)

    dct_n = _dct_matrix(size)
    dct = dct_n @ arr @ dct_n.T
    low = dct[:hash_size, :hash_size]

    flattened = low.flatten()
    median = float(np.median(flattened[1:])) if flattened.size > 1 else float(flattened[0])
    bits = (flattened > median).astype(np.uint8)

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def hamming_distance(x: int, y: int) -> int:
    return int((x ^ y).bit_count())


def deduplicate_images(
    image_paths: list[Path],
    distance_threshold: int = 8,
) -> tuple[list[Path], list[Path]]:
    """Return (kept, dropped) image paths using pHash hamming threshold."""
    kept: list[Path] = []
    dropped: list[Path] = []
    hashes: list[int] = []

    for path in image_paths:
        with Image.open(path) as img:
            current = phash_from_image(img)
        is_dup = any(hamming_distance(current, existing) <= distance_threshold for existing in hashes)
        if is_dup:
            dropped.append(path)
            continue
        hashes.append(current)
        kept.append(path)

    return kept, dropped
