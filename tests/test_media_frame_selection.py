from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from submd.media import select_ocr_frames
from submd.models import FrameRef


def _write_frame(
    path: Path,
    background: int,
    text: str | None,
    *,
    foreground_flash: bool = False,
) -> None:
    image = Image.new("RGB", (640, 180), (background, background, background))
    if foreground_flash:
        ImageDraw.Draw(image).rectangle((10, 10, 220, 55), fill="white", outline="black", width=4)
    if text:
        font = ImageFont.load_default(size=44)
        ImageDraw.Draw(image).text(
            (80, 65),
            text,
            font=font,
            fill="white",
            stroke_width=4,
            stroke_fill="black",
        )
    image.save(path, quality=95)


def test_repeated_subtitle_keeps_one_representative_despite_background_change(
    tmp_path: Path,
) -> None:
    frames: list[FrameRef] = []
    for index, background in enumerate((20, 45, 70, 95), start=1):
        path = tmp_path / f"frame_{index:06d}.jpg"
        _write_frame(path, background, "SAME SUBTITLE")
        frames.append(FrameRef(index=index, timestamp_ms=(index - 1) * 333, path=path))

    selected = select_ocr_frames(frames, change_threshold=0.012, max_interval_seconds=0.5)

    assert len(selected) == 1


def test_changed_subtitle_creates_one_representative_per_text(tmp_path: Path) -> None:
    texts = ("FIRST LINE", "FIRST LINE", "SECOND LINE", "SECOND LINE")
    frames: list[FrameRef] = []
    for index, text in enumerate(texts, start=1):
        path = tmp_path / f"frame_{index:06d}.jpg"
        _write_frame(path, 30 + index * 8, text)
        frames.append(FrameRef(index=index, timestamp_ms=(index - 1) * 333, path=path))

    selected = select_ocr_frames(frames, change_threshold=0.012, max_interval_seconds=0.5)

    assert len(selected) == 2


def test_single_frame_background_flash_does_not_create_duplicate_subtitle(
    tmp_path: Path,
) -> None:
    frames: list[FrameRef] = []
    for index in range(1, 5):
        path = tmp_path / f"frame_{index:06d}.jpg"
        _write_frame(
            path,
            35,
            "SAME SUBTITLE",
            foreground_flash=index == 2,
        )
        frames.append(FrameRef(index=index, timestamp_ms=(index - 1) * 333, path=path))

    selected = select_ocr_frames(frames, change_threshold=0.012, max_interval_seconds=0.5)

    assert len(selected) == 1


def test_youtube_timing_hint_forces_exact_frame_without_rewriting_its_timestamp(
    tmp_path: Path,
) -> None:
    frames: list[FrameRef] = []
    for index, text in enumerate(("FIRST", "FIRST", "SECOND", "SECOND"), start=1):
        path = tmp_path / f"frame_{index:06d}.jpg"
        image = Image.new("RGB", (640, 180), "black")
        ImageDraw.Draw(image).text(
            (80, 65),
            text,
            font=ImageFont.load_default(size=44),
            fill=(100, 100, 100),
        )
        image.save(path, quality=95)
        frames.append(FrameRef(index=index, timestamp_ms=(index - 1) * 333, path=path))

    selected = select_ocr_frames(
        frames,
        change_threshold=0.012,
        max_interval_seconds=2,
        hint_timestamps_ms=[666],
    )

    assert [(item.index, item.timestamp_ms, item.path.name) for item in selected] == [
        (1, 0, "frame_000001.jpg"),
        (3, 666, "frame_000003.jpg"),
    ]


def test_coverage_window_recovers_a_single_sample_subtitle_state(tmp_path: Path) -> None:
    frames: list[FrameRef] = []
    for index, text in enumerate(("FIRST", "FIRST", "MISSING", "FIRST", "FIRST"), start=1):
        path = tmp_path / f"frame_{index:06d}.jpg"
        _write_frame(path, 30, text)
        frames.append(FrameRef(index=index, timestamp_ms=(index - 1) * 333, path=path))

    selected = select_ocr_frames(
        frames,
        change_threshold=0.012,
        max_interval_seconds=2,
        coverage_windows_ms=[(666, 666)],
    )

    assert [item.index for item in selected] == [1, 3]
