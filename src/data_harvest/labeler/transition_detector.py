"""Transition detector: detect large-area screen transitions between pre/post frames."""

from __future__ import annotations

from data_harvest.labeler.diff_detector import diff_ratio


def is_screen_transition(
    pre_frame: "np.ndarray",
    post_frame: "np.ndarray",
    max_diff_ratio: float = 0.4,
) -> bool:
    """Return True if the diff ratio exceeds the threshold (large screen change).

    A large diff ratio typically means a scene or menu transition rather
    than a localized UI interaction.
    """
    import numpy as np  # noqa: F811

    ratio = diff_ratio(pre_frame, post_frame)
    return ratio > max_diff_ratio
