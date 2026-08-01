from pathlib import Path

from submd.coverage import find_reference_coverage_windows, select_supplemental_frames
from submd.models import (
    FrameRef,
    OcrObservation,
    SubtitleSegment,
    YouTubeCaptionCue,
    YouTubeCaptionTrack,
)


def _track() -> YouTubeCaptionTrack:
    return YouTubeCaptionTrack(
        video_id="sample",
        language="ja",
        source="automatic",
        cues=[
            YouTubeCaptionCue(
                cue_id="yt000001",
                start_ms=1000,
                end_ms=2000,
                text="お昼も一人で食べています",
            )
        ],
    )


def test_reference_gap_is_detected_and_correct_visual_text_is_covered() -> None:
    unrelated = [
        SubtitleSegment(
            start_ms=900,
            end_ms=2100,
            text="前の字幕",
            confidence=0.9,
        )
    ]
    correct = [unrelated[0].model_copy(update={"text": "お昼も1人で食べています"})]

    assert find_reference_coverage_windows(_track(), unrelated) == [(1000, 2000)]
    assert find_reference_coverage_windows(_track(), correct) == []


def test_supplemental_selector_never_reuploads_observed_frames(tmp_path: Path) -> None:
    frames = [
        FrameRef(index=index, timestamp_ms=(index - 1) * 500, path=tmp_path / f"{index}.jpg")
        for index in range(1, 7)
    ]
    observation = OcrObservation(
        timestamp_ms=1000,
        frame_index=3,
        frame_file="3.jpg",
        diff_score=1,
        model="vision",
        batch_index=1,
        text="前の字幕",
        confidence=0.9,
    )

    selected = select_supplemental_frames(
        frames,
        [frames[2]],
        [observation],
        [(1000, 2000)],
    )

    assert selected
    assert all(frame.index != 3 for frame in selected)
