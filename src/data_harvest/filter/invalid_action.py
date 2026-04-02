"""Invalid action filter: remove samples where pre/post frames are nearly identical."""

from __future__ import annotations

import cv2

from data_harvest.core.types import HarvestSample
from data_harvest.labeler.diff_detector import diff_ratio


def is_invalid_action(sample: HarvestSample, min_diff_ratio: float = 0.005) -> bool:
    """Return True if the sample should be removed (no visible change)."""
    pre_path = sample.pre_frame_path
    post_path = sample.post_frame_path
    if not pre_path.exists() or not post_path.exists():
        return True

    pre = cv2.imread(str(pre_path))
    post = cv2.imread(str(post_path))
    if pre is None or post is None:
        return True

    return diff_ratio(pre, post, align=True) < min_diff_ratio
