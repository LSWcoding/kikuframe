import json
from pathlib import Path

import httpx

from submd.models import (
    CloudOcrConfig,
    ExtractionConfig,
    ReconciledScreenAnnotation,
    ReconciledSpokenSegment,
    SubtitleDocument,
    SubtitleSegment,
    TextLlmConfig,
    VideoMetadata,
    YouTubeCaptionCue,
    YouTubeCaptionTrack,
    YoutubePrimaryReconciliation,
)
from submd.reconcile import (
    OpenAICompatibleReconciliationEngine,
    YoutubePrimaryReconciler,
)


class FakeEngine:
    def reconcile(
        self,
        youtube_cues,
        context_cues,
        ocr_segments,
        target_ocr_ids,
        previous_tail,
        visual_rechecks=None,
        *,
        final_pass=False,
    ):
        del context_cues, target_ocr_ids, previous_tail, visual_rechecks, final_pass
        assert [item["cue_id"] for item in youtube_cues] == ["yt000001", "yt000002"]
        assert ocr_segments[0]["text"] == "挨拶がない"
        return (
            YoutubePrimaryReconciliation(
                spoken=[
                    ReconciledSpokenSegment(
                        cue_ids=["yt000001", "yt000002"],
                        text="接客がないところです",
                        reason="视觉汉字与读音一致",
                    )
                ],
                screen_annotations=[
                    ReconciledScreenAnnotation(
                        ocr_segment_ids=["ocr000002"],
                        text="ここだけ画面の説明",
                        reason="YouTube 字幕中没有",
                    )
                ],
            ),
            "req-test",
            {"total_tokens": 10},
        )


def test_youtube_is_backbone_and_visual_extra_stays_separate(tmp_path: Path) -> None:
    video = VideoMetadata(
        video_id="sample",
        original_title="Sample",
        duration_ms=3000,
        webpage_url="https://youtu.be/sample",
    )
    document = SubtitleDocument(
        video=video,
        config=ExtractionConfig(
            source_url=video.webpage_url,
            ocr=CloudOcrConfig(base_url="https://vendor.example/v1", model="vision"),
        ),
        segments=[
            SubtitleSegment(start_ms=0, end_ms=2000, text="挨拶がない", confidence=0.8),
            SubtitleSegment(
                start_ms=2200,
                end_ms=2600,
                text="ここだけ画面の説明",
                confidence=0.9,
            ),
        ],
    )
    track = YouTubeCaptionTrack(
        video_id="sample",
        language="ja",
        source="automatic",
        cues=[
            YouTubeCaptionCue(cue_id="yt000001", start_ms=0, end_ms=1000, text="接客がない"),
            YouTubeCaptionCue(cue_id="yt000002", start_ms=900, end_ms=2000, text="ところです"),
        ],
    )

    result = YoutubePrimaryReconciler(engine=FakeEngine()).run(
        document,
        track,
        TextLlmConfig(base_url="https://vendor.example/v1", model="text"),
        tmp_path / "corrected.json",
        tmp_path / "checkpoint.json",
    )

    assert [(item.source, item.text) for item in result.segments] == [
        ("youtube_primary", "接客がないところです"),
        ("screen_annotation", "ここだけ画面の説明"),
    ]
    assert result.segments[0].start_ms == 0
    assert result.segments[0].end_ms == 2000


class WindowEngine:
    def __init__(self) -> None:
        self.calls = []

    def reconcile(
        self,
        target_cues,
        context_cues,
        ocr_segments,
        target_ocr_ids,
        previous_tail,
        visual_rechecks=None,
        *,
        final_pass=False,
    ):
        self.calls.append(
            {
                "target": [item["cue_id"] for item in target_cues],
                "context": [item["cue_id"] for item in context_cues],
                "previous_tail": previous_tail,
                "visual_rechecks": visual_rechecks,
                "final_pass": final_pass,
            }
        )
        ocr_id = target_ocr_ids[0] if target_ocr_ids else None
        return (
            YoutubePrimaryReconciliation(
                spoken=[
                    ReconciledSpokenSegment(
                        cue_ids=[item["cue_id"]],
                        ocr_segment_ids=[ocr_id] if ocr_id else [],
                        text=item["text"],
                    )
                    for item in target_cues
                ]
            ),
            f"request-{len(self.calls)}",
            {"total_tokens": 1},
        )


def test_reconciliation_uses_20_second_windows_and_reuses_checkpoints(
    tmp_path: Path,
) -> None:
    video = VideoMetadata(
        video_id="long",
        original_title="Long",
        duration_ms=41_000,
        webpage_url="https://youtu.be/long",
    )
    document = SubtitleDocument(
        video=video,
        config=ExtractionConfig(
            source_url=video.webpage_url,
            ocr=CloudOcrConfig(base_url="https://vendor.example/v1", model="vision"),
        ),
        segments=[
            SubtitleSegment(start_ms=1000, end_ms=3000, text="一", confidence=0.9),
            SubtitleSegment(start_ms=21_000, end_ms=23_000, text="二", confidence=0.9),
            SubtitleSegment(start_ms=40_000, end_ms=41_000, text="三", confidence=0.9),
        ],
    )
    track = YouTubeCaptionTrack(
        video_id="long",
        language="ja",
        source="automatic",
        cues=[
            YouTubeCaptionCue(cue_id="yt000001", start_ms=1000, end_ms=3000, text="一"),
            YouTubeCaptionCue(cue_id="yt000002", start_ms=21_000, end_ms=23_000, text="二"),
            YouTubeCaptionCue(cue_id="yt000003", start_ms=40_000, end_ms=41_000, text="三"),
        ],
    )
    engine = WindowEngine()
    reconciler = YoutubePrimaryReconciler(engine=engine)
    args = (
        document,
        track,
        TextLlmConfig(base_url="https://vendor.example/v1", model="text"),
        tmp_path / "corrected.json",
        tmp_path / "checkpoint.json",
    )
    result = reconciler.run(*args)

    assert [item["target"] for item in engine.calls] == [
        ["yt000001"],
        ["yt000002"],
        ["yt000003"],
    ]
    assert engine.calls[1]["previous_tail"][0]["text"] == "一"
    assert [item.text for item in result.segments] == ["一", "二", "三"]
    assert len(list((tmp_path / "reconciliation-windows").glob("*.json"))) == 3

    reconciler.run(*args)
    assert len(engine.calls) == 3


class ConflictEngine:
    def __init__(self) -> None:
        self.calls = []

    def reconcile(
        self,
        target_cues,
        context_cues,
        ocr_segments,
        target_ocr_ids,
        previous_tail,
        visual_rechecks=None,
        *,
        final_pass=False,
    ):
        del context_cues, ocr_segments, target_ocr_ids, previous_tail
        self.calls.append((visual_rechecks, final_pass))
        return (
            YoutubePrimaryReconciliation(
                spoken=[
                    ReconciledSpokenSegment(
                        cue_ids=[target_cues[0]["cue_id"]],
                        ocr_segment_ids=["ocr000001"],
                        text="接客" if final_pass else "挨拶",
                        needs_visual_recheck=not final_pass,
                    )
                ]
            ),
            "request",
            {},
        )


class FakeVisualReviewer:
    def __init__(self) -> None:
        self.segment_ids = []

    def review(self, segment_ids, evidence_path):
        assert evidence_path.is_file()
        self.segment_ids = segment_ids
        return ({"ocr000001": [{"text": "接客", "timestamp_ms": 1000}]}, [])


def test_ambiguous_text_is_reread_visually_then_returned_to_adjudicator(
    tmp_path: Path,
) -> None:
    video = VideoMetadata(
        video_id="review",
        original_title="Review",
        duration_ms=3000,
        webpage_url="https://youtu.be/review",
    )
    document = SubtitleDocument(
        video=video,
        config=ExtractionConfig(
            source_url=video.webpage_url,
            ocr=CloudOcrConfig(base_url="https://vendor.example/v1", model="vision"),
        ),
        segments=[SubtitleSegment(start_ms=0, end_ms=2000, text="挨拶", confidence=0.7)],
    )
    track = YouTubeCaptionTrack(
        video_id="review",
        language="ja",
        source="automatic",
        cues=[YouTubeCaptionCue(cue_id="yt000001", start_ms=0, end_ms=2000, text="接客")],
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}", encoding="utf-8")
    engine = ConflictEngine()
    reviewer = FakeVisualReviewer()
    result = YoutubePrimaryReconciler(
        engine=engine, visual_reviewer=reviewer
    ).run(
        document,
        track,
        TextLlmConfig(base_url="https://vendor.example/v1", model="text"),
        tmp_path / "corrected.json",
        tmp_path / "checkpoint.json",
        evidence_path=evidence_path,
    )

    assert reviewer.segment_ids == ["ocr000001"]
    assert engine.calls == [
        (None, False),
        ({"ocr000001": [{"text": "接客", "timestamp_ms": 1000}]}, True),
    ]
    assert result.segments[0].text == "接客"
    assert result.segments[0].needs_review is False


def test_invalid_cue_ids_are_sent_back_to_model_for_format_repair() -> None:
    requests = []
    responses = [
        {
            "spoken": [{"cue_ids": ["yt000001"], "text": "前半"}],
            "screen_annotations": [],
        },
        {
            "spoken": [
                {
                    "cue_ids": ["yt000001", "yt000002"],
                    "text": "前半後半",
                }
            ],
            "screen_annotations": [],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        result = responses[len(requests) - 1]
        return httpx.Response(
            200,
            json={
                "id": f"request-{len(requests)}",
                "choices": [{"message": {"content": json.dumps(result)}}],
            },
            request=request,
        )

    engine = OpenAICompatibleReconciliationEngine(
        TextLlmConfig(
            base_url="https://vendor.example/v1",
            model="text",
            max_retries=0,
        ),
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result, request_id, _usage = engine.reconcile(
        [
            {"cue_id": "yt000001", "start_ms": 0, "end_ms": 1000, "text": "前半"},
            {"cue_id": "yt000002", "start_ms": 900, "end_ms": 2000, "text": "後半"},
        ],
        [],
        [],
        [],
        [],
    )

    assert request_id == "request-2"
    assert result.spoken[0].cue_ids == ["yt000001", "yt000002"]
    assert len(requests) == 2
    assert "exactly once" in requests[1]["messages"][-1]["content"]
