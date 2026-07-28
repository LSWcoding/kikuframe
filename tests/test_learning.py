from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from submd.errors import LearningAnalysisError
from submd.learning import SentenceAnalyzer
from submd.models import LanguageLearningConfig


def learning_response(
    *, include_reading: bool = True, translation: str = "短期离职明明会被否定。"
) -> dict:
    return {
        "id": "learning-test",
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "translation": translation,
                            "vocabulary": [
                                {
                                    "expression": "短期離職",
                                    "reading": "たんきりしょく" if include_reading else "",
                                    "meaning": "短期离职",
                                },
                                {
                                    "expression": "否定される",
                                    "reading": "ひていされる" if include_reading else "",
                                    "meaning": "被否定",
                                },
                            ],
                            "grammar": [
                                {
                                    "pattern": "～のに",
                                    "explanation": "表示与预期相反的转折，相当于“明明……却……”。",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ],
    }


def test_sentence_analyzer_returns_structured_result_and_reuses_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url == httpx.URL("https://vendor.example/v1/chat/completions")
        assert request.headers["Authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        assert payload["messages"][1]["content"] == "短期離職は否定されるのに"
        assert "hiragana" in payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json=learning_response(
                translation=(
                    "重新分析后的翻译。" if calls == 2 else "短期离职明明会被否定。"
                )
            ),
        )

    analyzer = SentenceAnalyzer(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    config = LanguageLearningConfig(base_url="https://vendor.example/v1", model="study-model")

    first, first_cached = analyzer.analyze(
        "短期離職は否定されるのに", config, tmp_path / "learning"
    )
    second, second_cached = analyzer.analyze(
        "短期離職は否定されるのに", config, tmp_path / "learning"
    )
    refreshed, refreshed_cached = analyzer.analyze(
        "短期離職は否定されるのに", config, tmp_path / "learning", force=True
    )
    after_refresh, after_refresh_cached = analyzer.analyze(
        "短期離職は否定されるのに", config, tmp_path / "learning"
    )

    assert first.translation == "短期离职明明会被否定。"
    assert first.vocabulary[0].reading == "たんきりしょく"
    assert first.grammar[0].pattern == "～のに"
    assert second == first
    assert refreshed.translation == "重新分析后的翻译。"
    assert after_refresh == refreshed
    assert first_cached is False
    assert second_cached is True
    assert refreshed_cached is False
    assert after_refresh_cached is True
    assert calls == 2


def test_failed_forced_analysis_preserves_previous_cache(tmp_path: Path) -> None:
    fail = False

    def handler(_request: httpx.Request) -> httpx.Response:
        if fail:
            return httpx.Response(500, text="temporary failure")
        return httpx.Response(200, json=learning_response())

    analyzer = SentenceAnalyzer(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    config = LanguageLearningConfig(
        base_url="https://vendor.example/v1", model="study-model", max_retries=0
    )
    original, cached = analyzer.analyze("短期離職", config, tmp_path / "learning")
    assert cached is False

    fail = True
    with pytest.raises(LearningAnalysisError, match="重试后仍然失败"):
        analyzer.analyze("短期離職", config, tmp_path / "learning", force=True)

    preserved, preserved_cached = analyzer.analyze("短期離職", config, tmp_path / "learning")
    assert preserved == original
    assert preserved_cached is True


def test_sentence_analyzer_rejects_kanji_vocabulary_without_reading(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=learning_response(include_reading=False))

    analyzer = SentenceAnalyzer(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    config = LanguageLearningConfig(base_url="https://vendor.example/v1", model="study-model")

    with pytest.raises(LearningAnalysisError, match="无法解析"):
        analyzer.analyze("短期離職", config, tmp_path / "learning")
