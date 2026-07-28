from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops

from submd.errors import DependencyError, MediaError
from submd.models import FrameRef, MediaInfo, Roi


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise DependencyError(f"Missing system dependencies: {', '.join(missing)}")


def probe_video(path: Path) -> MediaInfo:
    require_media_tools()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise MediaError(f"ffprobe failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        duration_ms = round(float(payload["format"]["duration"]) * 1000)
        numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
        fps = float(numerator) / float(denominator) if float(denominator) else None
        return MediaInfo(
            duration_ms=duration_ms,
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=fps,
        )
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise MediaError(f"Could not parse ffprobe output for {path}") from exc


def extract_frames(video_path: Path, frames_dir: Path, roi: Roi, fps: float) -> list[FrameRef]:
    require_media_tools()
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    crop = (
        f"crop=floor(iw*{roi.width}/2)*2:"
        f"floor(ih*{roi.height}/2)*2:"
        f"floor(iw*{roi.x}/2)*2:"
        f"floor(ih*{roi.y}/2)*2"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        f"{crop},fps={fps}",
        "-q:v",
        "2",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise MediaError(f"FFmpeg frame extraction failed: {completed.stderr.strip()}")
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise MediaError("FFmpeg did not produce any frames")
    interval_ms = 1000.0 / fps
    return [
        FrameRef(index=index, timestamp_ms=round((index - 1) * interval_ms), path=path)
        for index, path in enumerate(paths, start=1)
    ]


def _load_difference_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("L").resize((320, 120), Image.Resampling.BILINEAR)
    except OSError as exc:
        raise MediaError(f"Could not read extracted frame: {path}") from exc


def _difference_score(previous: Image.Image, current: Image.Image) -> float:
    histogram = ImageChops.difference(previous, current).histogram()
    changed = sum(count for value, count in enumerate(histogram) if value > 24)
    return changed / (previous.width * previous.height)


def select_ocr_frames(
    frames: list[FrameRef],
    change_threshold: float,
    max_interval_seconds: float,
) -> list[FrameRef]:
    if change_threshold <= 0 or len(frames) <= 1:
        return frames

    selected_indexes = {0}
    scores: dict[int, float] = {0: 1.0}
    previous = _load_difference_image(frames[0].path)
    last_selected_ms = frames[0].timestamp_ms

    for index in range(1, len(frames)):
        current = _load_difference_image(frames[index].path)
        score = _difference_score(previous, current)
        elapsed = (frames[index].timestamp_ms - last_selected_ms) / 1000
        if score >= change_threshold or elapsed >= max_interval_seconds:
            selected_indexes.add(index)
            scores[index] = score
            last_selected_ms = frames[index].timestamp_ms
            # A second sample after a transition improves OCR voting on fades.
            if score >= change_threshold and index + 1 < len(frames):
                selected_indexes.add(index + 1)
                scores[index + 1] = score
        previous = current

    selected: list[FrameRef] = []
    for index in sorted(selected_indexes):
        frame = frames[index].model_copy()
        frame.diff_score = scores.get(index, 0.0)
        selected.append(frame)
    return selected
