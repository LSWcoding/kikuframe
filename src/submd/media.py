from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from bisect import bisect_left
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from submd.errors import DependencyError, ExtractionCancelled, MediaError
from submd.models import FrameRef, MediaInfo, Roi


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise DependencyError(f"Missing system dependencies: {', '.join(missing)}")


CancelCheck = Callable[[], bool]


def _run_media_command(
    command: list[str], cancelled: CancelCheck | None = None
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            if not cancelled or not cancelled():
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=1)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            raise ExtractionCancelled("处理已由用户中止") from None


def probe_video(path: Path, cancelled: CancelCheck | None = None) -> MediaInfo:
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
    completed = _run_media_command(command, cancelled)
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


def extract_audio(
    video_path: Path, output_path: Path, cancelled: CancelCheck | None = None
) -> Path:
    """Save a browser-compatible AAC/M4A copy of the source audio."""
    require_media_tools()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.part.m4a")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = _run_media_command(command, cancelled)
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "source video has no usable audio stream"
        raise MediaError(f"FFmpeg audio extraction failed: {detail}")
    temporary.replace(output_path)
    return output_path


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    roi: Roi,
    fps: float,
    cancelled: CancelCheck | None = None,
) -> list[FrameRef]:
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
    completed = _run_media_command(command, cancelled)
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


@dataclass
class _SubtitleSignature:
    mask: Image.Image
    pixel_count: int
    quality: float
    line_count: int


def _load_gray_frame(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
    except OSError as exc:
        raise MediaError(f"Could not read extracted frame: {path}") from exc
    target_width = min(640, gray.width)
    scale = target_width / gray.width
    target_height = max(1, round(gray.height * scale))
    return gray.resize((target_width, target_height), Image.Resampling.BILINEAR)


def _binary_threshold(image: Image.Image, threshold: int) -> Image.Image:
    return image.point(lambda value: 255 if value >= threshold else 0)


def _white_pixel_count(image: Image.Image) -> int:
    return image.histogram()[255]


def _subtitle_signature(path: Path) -> _SubtitleSignature:
    """Extract a lightweight mask for bright, outlined burned-in subtitle glyphs."""
    gray = _load_gray_frame(path)
    softened = gray.filter(ImageFilter.GaussianBlur(radius=0.8))
    local_min = softened.filter(ImageFilter.MinFilter(7))
    local_max = softened.filter(ImageFilter.MaxFilter(7))

    bright_core = _binary_threshold(softened, 145)
    above_neighbor = _binary_threshold(ImageChops.subtract(softened, local_min), 38)
    local_contrast = _binary_threshold(ImageChops.subtract(local_max, local_min), 55)
    mask = ImageChops.multiply(
        ImageChops.multiply(bright_core, above_neighbor),
        local_contrast,
    )
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))

    pixel_count = _white_pixel_count(mask)
    minimum_row_pixels = max(3, round(mask.width * 0.006))
    occupied_rows = [
        _white_pixel_count(mask.crop((0, row, mask.width, row + 1))) >= minimum_row_pixels
        for row in range(mask.height)
    ]
    line_count = 0
    active_rows = 0
    for occupied in [*occupied_rows, False]:
        if occupied:
            active_rows += 1
            continue
        if active_rows >= 2:
            line_count += 1
        active_rows = 0
    edge_histogram = gray.filter(ImageFilter.FIND_EDGES).histogram()
    sharpness = sum(value * count for value, count in enumerate(edge_histogram)) / (
        gray.width * gray.height
    )
    return _SubtitleSignature(
        mask=mask,
        pixel_count=pixel_count,
        quality=sharpness,
        line_count=line_count,
    )


def _signature_similarity(left: _SubtitleSignature, right: _SubtitleSignature) -> float:
    minimum_pixels = max(12, round(left.mask.width * left.mask.height * 0.0002))
    left_empty = left.pixel_count < minimum_pixels
    right_empty = right.pixel_count < minimum_pixels
    if left_empty or right_empty:
        return 1.0 if left_empty and right_empty else 0.0

    expanded_left = left.mask.filter(ImageFilter.MaxFilter(3))
    expanded_right = right.mask.filter(ImageFilter.MaxFilter(3))
    left_covered = _white_pixel_count(ImageChops.multiply(left.mask, expanded_right))
    right_covered = _white_pixel_count(ImageChops.multiply(right.mask, expanded_left))
    return min(left_covered / left.pixel_count, right_covered / right.pixel_count)


def select_ocr_frames(
    frames: list[FrameRef],
    change_threshold: float,
    max_interval_seconds: float,
    hint_timestamps_ms: list[int] | None = None,
    coverage_windows_ms: list[tuple[int, int]] | None = None,
) -> list[FrameRef]:
    if len(frames) <= 1:
        return frames

    # The historical pixel threshold remains accepted for config compatibility. Subtitle
    # signatures use a fixed, deliberately conservative similarity threshold instead.
    del change_threshold

    # A higher threshold is intentional: subtitles can change while keeping the same position,
    # length and overall silhouette. Requiring two consecutive samples below the threshold
    # filters one-frame flashes without swallowing those same-layout text changes.
    similarity_threshold = 0.86
    selected_indexes = {0}
    scores: dict[int, float] = {0: 1.0}
    signature_cache: dict[int, _SubtitleSignature] = {}

    def signature(index: int) -> _SubtitleSignature:
        if index not in signature_cache:
            signature_cache[index] = _subtitle_signature(frames[index].path)
        return signature_cache[index]

    anchor_signature = signature(0)
    pending_index: int | None = None
    pending_signature: _SubtitleSignature | None = None

    # A fixed anchor prevents transitive drift: A≈B and B≈C must never imply A≈C.
    for index in range(1, len(frames)):
        current_signature = signature(index)
        similarity = _signature_similarity(anchor_signature, current_signature)
        if similarity >= similarity_threshold:
            pending_index = None
            pending_signature = None
            continue

        if pending_signature is None:
            pending_index = index
            pending_signature = current_signature
            continue

        pending_similarity = _signature_similarity(pending_signature, current_signature)
        if pending_similarity < similarity_threshold:
            pending_index = index
            pending_signature = current_signature
            continue

        assert pending_index is not None
        selected_indexes.add(pending_index)
        scores[pending_index] = 1.0 - similarity
        anchor_signature = pending_signature
        pending_index = None
        pending_signature = None

    # YouTube cue starts are pronunciation-timing hints. Including their nearest sampled frame
    # prevents a locally imperfect visual mask from swallowing a whole spoken/subtitle passage.
    timestamps = [frame.timestamp_ms for frame in frames]
    for hint in sorted(set(hint_timestamps_ms or [])):
        insertion = bisect_left(timestamps, max(0, hint))
        candidates = [value for value in (insertion - 1, insertion) if 0 <= value < len(frames)]
        if candidates:
            nearest = min(candidates, key=lambda value: abs(timestamps[value] - hint))
            selected_indexes.add(nearest)
            scores.setdefault(nearest, 0.0)

    # YouTube cue spans provide coverage targets, not OCR text. Probe long spans locally and add
    # only visually novel subtitle states, so a missed transition is repaired without uploading
    # repeated copies of an unchanged subtitle.
    interval_ms = max(100, round(max_interval_seconds * 1000))
    for start_ms, end_ms in coverage_windows_ms or []:
        probe_ms = max(0, start_ms)
        safe_end = max(probe_ms, end_ms)
        while probe_ms <= safe_end:
            insertion = bisect_left(timestamps, probe_ms)
            candidates = [
                value for value in (insertion - 1, insertion) if 0 <= value < len(frames)
            ]
            if candidates:
                candidate = min(
                    candidates, key=lambda value: abs(timestamps[value] - probe_ms)
                )
                nearby_selected = [
                    value
                    for value in selected_indexes
                    if abs(timestamps[value] - timestamps[candidate]) <= interval_ms * 2
                ]
                is_duplicate = any(
                    _signature_similarity(signature(value), signature(candidate)) >= 0.96
                    for value in nearby_selected
                )
                if not is_duplicate:
                    selected_indexes.add(candidate)
                    scores.setdefault(candidate, 0.0)
            probe_ms += interval_ms

    selected: list[FrameRef] = []
    for index in sorted(selected_indexes):
        frame = frames[index].model_copy()
        frame.diff_score = scores.get(index, 0.0)
        frame.visual_line_count = signature(index).line_count
        selected.append(frame)
    return selected
