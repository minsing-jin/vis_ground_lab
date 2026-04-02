"""VLM-based semantic descriptor using Florence-2.

Replaces OCR as the primary semantic signal.  Given a crop around the
click point, the VLM returns both a grounding bbox and a caption that
serves as ``semantic_text``.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from data_harvest.core.types import BBoxCandidate

logger = logging.getLogger(__name__)

# Module-level singleton — heavy to init, so we cache it.
_vlm_cache: dict[str, Any] = {}


def _get_vlm(model_name: str, device_map: str = "auto") -> Any:
    """Lazy-load and cache Florence2Wrapper (import torch only here)."""
    if model_name in _vlm_cache:
        return _vlm_cache[model_name]

    from vis_ground_lab.models.florence2 import Florence2Wrapper

    wrapper = Florence2Wrapper(
        model_name=model_name,
        use_lora=False,
        device_map=device_map,
    )
    wrapper.load_model()
    _vlm_cache[model_name] = wrapper
    logger.info("VLM loaded: %s", model_name)
    return wrapper


def _to_pixel_bbox(
    x_min: float, y_min: float, x_max: float, y_max: float,
    width: int, height: int,
) -> tuple[float, float, float, float]:
    """Normalise Florence-2 bbox output to absolute pixels."""
    vmax = max(abs(x_min), abs(y_min), abs(x_max), abs(y_max))
    if vmax <= 1.5:
        return x_min * width, y_min * height, x_max * width, y_max * height
    if vmax <= 1200.0:
        return (
            x_min / 1000.0 * width,
            y_min / 1000.0 * height,
            x_max / 1000.0 * width,
            y_max / 1000.0 * height,
        )
    return x_min, y_min, x_max, y_max


def _crop_with_roi(
    frame: np.ndarray,
    roi_norm_xyxy: list[float] | tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, int, int]:
    """Crop frame to ROI and return (crop, x_offset, y_offset)."""
    if roi_norm_xyxy is None:
        return frame, 0, 0
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(round(float(roi_norm_xyxy[0]) * w))))
    y1 = max(0, min(h, int(round(float(roi_norm_xyxy[1]) * h))))
    x2 = max(x1 + 1, min(w, int(round(float(roi_norm_xyxy[2]) * w))))
    y2 = max(y1 + 1, min(h, int(round(float(roi_norm_xyxy[3]) * h))))
    return frame[y1:y2, x1:x2], x1, y1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vlm_describe_click(
    frame: np.ndarray,
    click_x: float,
    click_y: float,
    crop_radius: int = 120,
    model_name: str = "microsoft/Florence-2-base",
    device_map: str = "auto",
    prompts: list[str] | None = None,
) -> list[BBoxCandidate]:
    """Run Florence-2 on a crop around the click and return candidates.

    Strategy
    --------
    1. Crop the frame around ``(click_x, click_y)`` with ``crop_radius``.
    2. For each prompt, call ``wrapper.predict`` → get a bbox + implicit caption.
    3. Also run a caption-style prompt to extract ``semantic_text``.
    4. Return merged :class:`BBoxCandidate` list (signal="vlm").

    The prompts are game-UI oriented by default.
    """
    if prompts is None:
        prompts = [
            "detect the UI element",
            "detect the clickable button",
        ]

    h, w = frame.shape[:2]
    cx, cy = int(click_x), int(click_y)

    # Crop
    x1 = max(cx - crop_radius, 0)
    y1 = max(cy - crop_radius, 0)
    x2 = min(cx + crop_radius, w)
    y2 = min(cy + crop_radius, h)

    crop_bgr = frame[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return []

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    crop_w, crop_h = pil_crop.size

    try:
        wrapper = _get_vlm(model_name, device_map=device_map)
    except Exception:
        logger.warning("VLM load failed — skipping vlm signal.", exc_info=True)
        return []

    # --- Grounding prompts → bbox candidates --------------------------------
    candidates: list[BBoxCandidate] = []
    for prompt in prompts:
        try:
            pred = wrapper.predict(image=pil_crop, text=prompt)
        except Exception:
            continue

        bx1, by1, bx2, by2 = _to_pixel_bbox(
            pred.x_min, pred.y_min, pred.x_max, pred.y_max,
            crop_w, crop_h,
        )

        # Shift back to full-frame coords
        abs_x1 = bx1 + x1
        abs_y1 = by1 + y1
        abs_x2 = bx2 + x1
        abs_y2 = by2 + y1

        # Clamp
        abs_x1 = max(0.0, min(float(w), abs_x1))
        abs_y1 = max(0.0, min(float(h), abs_y1))
        abs_x2 = max(0.0, min(float(w), abs_x2))
        abs_y2 = max(0.0, min(float(h), abs_y2))

        if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
            continue

        candidates.append(
            BBoxCandidate(
                x_min=abs_x1,
                y_min=abs_y1,
                x_max=abs_x2,
                y_max=abs_y2,
                signal="vlm",
                confidence=0.7,
            )
        )

    # --- Caption prompt → semantic text --------------------------------------
    caption_text: str | None = None
    try:
        caption_pred = wrapper.predict(image=pil_crop, text="describe this UI element")
        # predict returns a BoundingBox, but the underlying model also generates
        # text.  Florence-2 caption task returns text in the generation; however
        # the wrapper only parses bbox.  We can still use the prompt itself as a
        # coarse description; the grounding bbox is the main value.
    except Exception:
        pass

    # Try a simpler approach: use the raw model generation for a caption
    try:
        from vis_ground_lab.base import BoundingBox as _BB, VGSample as _VS

        dummy_sample = _VS(image=pil_crop, text="<CAPTION>", bbox=_BB(0, 0, 1, 1))
        batch = wrapper.preprocess(dummy_sample)
        # Move tensors to model device
        import torch

        device = next(wrapper.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = wrapper.model.generate(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            max_new_tokens=30,
        )
        caption_text = wrapper.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        logger.debug("VLM caption failed — using prompt as fallback.", exc_info=True)

    # Attach semantic_text to the best candidate
    if caption_text and candidates:
        best = candidates[0]
        candidates[0] = BBoxCandidate(
            x_min=best.x_min,
            y_min=best.y_min,
            x_max=best.x_max,
            y_max=best.y_max,
            signal="vlm",
            confidence=best.confidence,
            semantic_text=caption_text,
        )

    return candidates


def vlm_detect_prompt(
    frame: np.ndarray,
    prompt: str,
    model_name: str = "microsoft/Florence-2-base",
    device_map: str = "auto",
    roi_norm_xyxy: list[float] | tuple[float, float, float, float] | None = None,
    confidence: float = 0.65,
) -> BBoxCandidate | None:
    """Run Florence-2 with a single grounding prompt on the full frame or ROI crop."""
    crop_bgr, off_x, off_y = _crop_with_roi(frame, roi_norm_xyxy)
    if crop_bgr.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    crop_w, crop_h = pil_crop.size

    try:
        wrapper = _get_vlm(model_name, device_map=device_map)
        pred = wrapper.predict(image=pil_crop, text=prompt)
    except Exception:
        logger.debug("VLM prompt detection failed.", exc_info=True)
        return None

    bx1, by1, bx2, by2 = _to_pixel_bbox(
        pred.x_min,
        pred.y_min,
        pred.x_max,
        pred.y_max,
        crop_w,
        crop_h,
    )
    x1 = max(0.0, bx1 + off_x)
    y1 = max(0.0, by1 + off_y)
    x2 = max(x1 + 1.0, bx2 + off_x)
    y2 = max(y1 + 1.0, by2 + off_y)
    if x2 <= x1 or y2 <= y1:
        return None

    return BBoxCandidate(
        x_min=x1,
        y_min=y1,
        x_max=x2,
        y_max=y2,
        signal="vlm_catalog",
        confidence=confidence,
        semantic_text=prompt,
    )
