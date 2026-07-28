from __future__ import annotations

import json
from pathlib import Path

import httpx

from submd.models import TextLlmConfig, YouTubeCaptionCue, YouTubeCaptionTrack
from submd.organize import (
    BoundaryResult,
    OpenAICompatibleBoundaryEngine,
    SubtitleOrganizer,
    SubtitleUnit,
    extract_subtitle_fragments,
)


class PunctuationBoundaryEngine:
    def __init__(self) -> None:
        self.calls = 0

    def decide_boundaries(
        self,
        before_context: list[SubtitleUnit],
        target_units: list[SubtitleUnit],
        after_context: list[SubtitleUnit],
    ) -> BoundaryResult:
        del before_context, after_context
        self.calls += 1
        breaks = {unit.unit_id for unit in target_units if unit.text.endswith(("。", "！", "？"))}
        return BoundaryResult(
            break_after=frozenset(breaks),
            request_id=f"request-{self.calls}",
            usage={"total_tokens": 10},
        )


class FailIfCalledEngine:
    def decide_boundaries(self, *_args, **_kwargs) -> BoundaryResult:
        raise AssertionError("completed chunk should have been restored from checkpoint")


def subtitle_markdown() -> str:
    return """---
title: "示例"
segments: 3
---

# 示例

- [00:00.000–00:01.000] 今日は
- [00:01.000–00:02.000] 学校へ行きます。
- [00:02.000–00:03.000] 明日は<br>休みです。 ⚠️
"""


def test_extracts_only_clean_subtitle_fragments() -> None:
    fragments = extract_subtitle_fragments(subtitle_markdown())
    assert fragments == ["今日は", "学校へ行きます。", "明日は休みです。"]


def test_organizer_outputs_only_sentences_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "示例.md"
    source.write_text(subtitle_markdown(), encoding="utf-8")
    config = TextLlmConfig(base_url="https://vendor.example/v1", model="text-model")
    engine = PunctuationBoundaryEngine()
    first = SubtitleOrganizer(boundary_engine=engine).run(
        source,
        config,
        workspace_root=tmp_path / "workspace",
        output_dir=tmp_path / "output",
        overwrite=True,
    )

    assert engine.calls == 1
    assert first.api_call_count == 1
    assert first.reused_chunk_count == 0
    assert first.sentence_count == 2
    assert first.markdown_path.read_text(encoding="utf-8") == (
        "今日は学校へ行きます。\n明日は休みです。\n"
    )
    assert first.sentences_path is not None
    timed = json.loads(first.sentences_path.read_text(encoding="utf-8"))
    assert timed["schema_version"] == 1
    assert timed["sentences"] == [
        {
            "sentence_id": "s000001",
            "text": "今日は学校へ行きます。",
            "start_ms": 0,
            "end_ms": 2000,
            "source_unit_ids": ["u000001", "u000002"],
            "source_segment_ids": ["seg000001", "seg000002"],
        },
        {
            "sentence_id": "s000002",
            "text": "明日は休みです。",
            "start_ms": 2000,
            "end_ms": 3000,
            "source_unit_ids": ["u000003"],
            "source_segment_ids": ["seg000003"],
        },
    ]
    checkpoint = json.loads(first.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["model"] == "text-model"
    assert len(checkpoint["chunks"]) == 1

    second = SubtitleOrganizer(boundary_engine=FailIfCalledEngine()).run(
        source,
        config,
        workspace_root=tmp_path / "workspace",
        output_dir=tmp_path / "output",
        overwrite=True,
    )
    assert second.api_call_count == 0
    assert second.reused_chunk_count == 1
    assert second.markdown_path.read_text(encoding="utf-8") == (
        "今日は学校へ行きます。\n明日は休みです。\n"
    )
    assert second.sentences_path == first.sentences_path


def test_boundary_engine_sends_units_and_parses_json() -> None:
    target = [SubtitleUnit(unit_id="u000001", text="今日は学校へ行きます")]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://vendor.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret-key"
        body = json.loads(request.content)
        assert body["model"] == "text-model"
        assert body["response_format"] == {"type": "json_object"}
        user_payload = json.loads(body["messages"][1]["content"])
        assert user_payload["target_units"] == [
            {"unit_id": "u000001", "text": "今日は学校へ行きます", "text_length": 10}
        ]
        return httpx.Response(
            200,
            headers={"x-request-id": "boundary-request"},
            json={
                "choices": [{"message": {"content": '```json\n{"break_after":["u000001"]}\n```'}}],
                "usage": {"total_tokens": 20},
            },
        )

    engine = OpenAICompatibleBoundaryEngine(
        TextLlmConfig(base_url="https://vendor.example/v1", model="text-model"),
        "secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = engine.decide_boundaries([], target, [])
    assert result.break_after == frozenset({"u000001"})
    assert result.request_id == "boundary-request"
    assert result.usage == {"total_tokens": 20}


def test_boundary_engine_ignores_context_ids() -> None:
    before = [SubtitleUnit(unit_id="u000001", text="前の文です")]
    target = [SubtitleUnit(unit_id="u000002", text="次の文です")]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"break_after":["u000001","u000002"]}'}}]},
        )

    engine = OpenAICompatibleBoundaryEngine(
        TextLlmConfig(base_url="https://vendor.example/v1", model="text-model"),
        "secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = engine.decide_boundaries(before, target, [])
    assert result.break_after == frozenset({"u000002"})


def test_organizer_can_split_inside_an_ocr_unit_and_interpolate_timing(
    tmp_path: Path,
) -> None:
    class CharacterBoundaryEngine:
        def decide_boundaries(self, _before, target, _after) -> BoundaryResult:
            assert [unit.text for unit in target] == ["フリーダー", "ですさっき病院に行ったら"]
            return BoundaryResult(
                break_after=frozenset({"u000002"}),
                split_after=frozenset({("u000002", 2)}),
            )

    source = tmp_path / "字符断句.md"
    source.write_text(
        "- [00:00.000–00:01.000] フリーダー\n"
        "- [00:01.000–00:04.000] ですさっき病院に行ったら\n",
        encoding="utf-8",
    )
    result = SubtitleOrganizer(boundary_engine=CharacterBoundaryEngine()).run(
        source,
        TextLlmConfig(base_url="https://vendor.example/v1", model="text-model"),
        workspace_root=tmp_path / "workspace",
        output_dir=tmp_path / "output",
        overwrite=True,
    )
    assert result.markdown_path.read_text(encoding="utf-8") == (
        "フリーダーです\nさっき病院に行ったら\n"
    )
    document = json.loads(result.sentences_path.read_text(encoding="utf-8"))
    first, second = document["sentences"]
    assert first["end_ms"] == second["start_ms"]
    assert 1000 < first["end_ms"] < 4000
    checkpoint = json.loads(result.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["chunks"]["000001"]["split_after"] == [
        {"unit_id": "u000002", "after_char": 2}
    ]


def test_youtube_reading_reference_adds_only_aligned_sentence_boundaries(
    tmp_path: Path,
) -> None:
    class FinalBoundaryOnlyEngine:
        def decide_boundaries(self, _before, target, _after) -> BoundaryResult:
            return BoundaryResult(break_after=frozenset({target[-1].unit_id}))

    source = tmp_path / "读音参考.md"
    source.write_text(
        "- [00:00.000–00:01.000] こんにちは\n"
        "- [00:01.000–00:02.000] フリーター\n"
        "- [00:02.000–00:04.000] です さっき病院に行ったら\n",
        encoding="utf-8",
    )
    track = YouTubeCaptionTrack(
        video_id="sample",
        language="ja",
        source="automatic",
        cues=[
            YouTubeCaptionCue(
                cue_id="yt000001",
                start_ms=0,
                end_ms=4000,
                text="こんにちは。フリーターです。さっき病院に行ったら。",
            )
        ],
    )
    result = SubtitleOrganizer(boundary_engine=FinalBoundaryOnlyEngine()).run(
        source,
        TextLlmConfig(base_url="https://vendor.example/v1", model="text-model"),
        workspace_root=tmp_path / "workspace",
        output_dir=tmp_path / "output",
        overwrite=True,
        reference_track=track,
    )
    assert result.markdown_path.read_text(encoding="utf-8") == (
        "こんにちは\nフリーターです\nさっき病院に行ったら\n"
    )
    document = json.loads(result.sentences_path.read_text(encoding="utf-8"))
    assert document["sentences"][1]["end_ms"] == document["sentences"][2]["start_ms"]
