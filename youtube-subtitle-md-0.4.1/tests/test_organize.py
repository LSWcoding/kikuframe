from __future__ import annotations

import json
from pathlib import Path

import httpx

from submd.models import TextLlmConfig
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
            {"unit_id": "u000001", "text": "今日は学校へ行きます"}
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
