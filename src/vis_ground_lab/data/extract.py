"""Video frame extraction pipeline with ffmpeg/cv2 fallback."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image


def _extract_frames_ffmpeg(video_path: Path, out_dir: Path, fps: float | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.png"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path)]
    if fps is not None and fps > 0:
        cmd.extend(["-vf", f"fps={fps}"])
    cmd.extend([str(pattern)])
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.png"))


def _extract_frames_opencv(video_path: Path, out_dir: Path, fps: float | None = None) -> list[Path]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required as a fallback when ffmpeg is unavailable.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = 1
    if fps is not None and fps > 0 and native_fps > 0:
        stride = max(1, int(round(native_fps / fps)))

    frames: list[Path] = []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride != 0:
            idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        out_path = out_dir / f"frame_{saved:06d}.png"
        image.save(out_path)
        frames.append(out_path)

        saved += 1
        idx += 1

    cap.release()
    return frames


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float | None = None,
    every_nth: int = 1,
    max_frames: int | None = None,
) -> list[Path]:
    """Extract frames from video and return sampled frame paths."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)

    if shutil.which("ffmpeg"):
        frames = _extract_frames_ffmpeg(video_path=video_path, out_dir=out_dir, fps=fps)
    else:
        frames = _extract_frames_opencv(video_path=video_path, out_dir=out_dir, fps=fps)

    if every_nth > 1:
        frames = [p for i, p in enumerate(frames) if i % every_nth == 0]

    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]

    return frames
