from submd.models import OcrObservation
from submd.segments import build_segments


def observation(timestamp: int, text: str, confidence: float = 0.9) -> OcrObservation:
    return OcrObservation(
        timestamp_ms=timestamp,
        frame_index=timestamp // 500 + 1,
        frame_file="frame.jpg",
        diff_score=0.1,
        model="vision-ocr",
        batch_index=1,
        text=text,
        confidence=confidence,
    )


def test_merges_similar_observations_and_uses_majority_text() -> None:
    segments = build_segments(
        [
            observation(0, ""),
            observation(500, "今日は学校に行きます"),
            observation(1000, "今日は学枝に行きます", 0.6),
            observation(1500, "今日は学校に行きます"),
            observation(2000, ""),
        ],
        duration_ms=3000,
        similarity_threshold=80,
        review_confidence=0.55,
        sample_interval_ms=500,
    )
    assert len(segments) == 1
    assert segments[0].text == "今日は学校に行きます"
    assert segments[0].start_ms == 500
    assert segments[0].end_ms == 2000
    assert segments[0].observation_count == 3
    assert not segments[0].needs_review


def test_creates_distinct_segments() -> None:
    segments = build_segments(
        [observation(0, "hello"), observation(1000, "goodbye")],
        duration_ms=2000,
        similarity_threshold=85,
        review_confidence=0.5,
        sample_interval_ms=500,
    )
    assert [segment.text for segment in segments] == ["hello", "goodbye"]


def test_containment_prefers_complete_visual_subtitle() -> None:
    segments = build_segments(
        [
            observation(0, "本当に自分がやりたいことだけ"),
            observation(500, "本当に自分がやりたいことだけをするようになり"),
            observation(1000, "本当に自分がやりたいことだけ"),
        ],
        duration_ms=1500,
        similarity_threshold=82,
        review_confidence=0.5,
        sample_interval_ms=500,
    )

    assert [item.text for item in segments] == [
        "本当に自分がやりたいことだけをするようになり"
    ]


def test_adjacent_rolling_overlap_is_emitted_once() -> None:
    segments = build_segments(
        [
            observation(0, "上京してそれ以降は"),
            observation(500, "それ以降はずっと非正規雇用で働いています"),
        ],
        duration_ms=1000,
        similarity_threshold=82,
        review_confidence=0.5,
        sample_interval_ms=500,
    )

    assert [item.text for item in segments] == [
        "上京してそれ以降はずっと非正規雇用で働いています"
    ]
