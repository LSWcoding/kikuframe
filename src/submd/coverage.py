from __future__ import annotations

import re
from bisect import bisect_left
from difflib import SequenceMatcher

from submd.models import (
    FrameRef,
    OcrObservation,
    SubtitleSegment,
    YouTubeCaptionTrack,
)
from submd.text import normalize_text

_NON_TEXT = re.compile(r"[\s\W_]+", re.UNICODE)


def _key(value: str) -> str:
    return _NON_TEXT.sub("", normalize_text(value)).lower()


def _text_supports(reference: str, visual: str, threshold: float) -> bool:
    left = _key(reference)
    right = _key(visual)
    if not left or not right:
        return False
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 4 and shorter in longer and len(shorter) / len(longer) >= 0.30:
        return True
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= threshold


def find_reference_coverage_windows(
    track: YouTubeCaptionTrack,
    segments: list[SubtitleSegment],
    threshold: float = 0.46,
) -> list[tuple[int, int]]:
    """Return YouTube cue spans that have no plausible visual OCR evidence.

    This check never changes transcript text. It only decides where another small group of
    already-extracted local frames should be sent to OCR.
    """

    uncovered: list[tuple[int, int]] = []
    for cue in track.cues:
        candidates = [
            segment
            for segment in segments
            if segment.end_ms >= cue.start_ms - 700 and segment.start_ms <= cue.end_ms + 700
        ]
        if not any(_text_supports(cue.text, segment.text, threshold) for segment in candidates):
            uncovered.append((cue.start_ms, cue.end_ms))

    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in uncovered:
        if merged and start_ms <= merged[-1][1] + 500:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return merged


def select_supplemental_frames(
    frames: list[FrameRef],
    selected: list[FrameRef],
    observations: list[OcrObservation],
    coverage_windows: list[tuple[int, int]],
) -> list[FrameRef]:
    """Pick a bounded number of unprocessed frames inside suspected OCR coverage gaps."""

    if not frames:
        return []
    observed_ids = {item.frame_index for item in observations}
    timestamps = [frame.timestamp_ms for frame in frames]
    wanted_indexes: set[int] = set()

    def add_nearest(timestamp_ms: int) -> None:
        insertion = bisect_left(timestamps, max(0, timestamp_ms))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(frames)]
        if not candidates:
            return
        nearest = min(candidates, key=lambda index: abs(timestamps[index] - timestamp_ms))
        if frames[nearest].index not in observed_ids:
            wanted_indexes.add(nearest)

    for start_ms, end_ms in coverage_windows:
        start_ms = max(0, start_ms - 400)
        end_ms = max(start_ms, end_ms + 400)
        span = end_ms - start_ms
        for fraction in (0.20, 0.50, 0.80):
            add_nearest(round(start_ms + span * fraction))

    observation_by_frame = {item.frame_index: item for item in observations}
    for frame in selected:
        observation = observation_by_frame.get(frame.index)
        if observation is not None and frame.visual_line_count > max(0, observation.line_count):
            position = bisect_left(timestamps, frame.timestamp_ms)
            for neighbor in (position - 1, position + 1):
                if 0 <= neighbor < len(frames) and frames[neighbor].index not in observed_ids:
                    wanted_indexes.add(neighbor)

    return [frames[index].model_copy() for index in sorted(wanted_indexes)]
