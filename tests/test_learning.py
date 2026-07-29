from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from submd.errors import LearningAnalysisError
from submd.learning import SentenceAnalyzer
from submd.models import LanguageLearningConfig, VocabularyAnalysisItem


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
                                    "kind": "word",
                                    "expression": "短期離職",
                                    "lemma": "短期離職",
                                    "reading": "たんきりしょく" if include_reading else "",
                                    "meaning": "短期离职",
                                },
                                {
                                    "kind": "word",
                                    "expression": "否定される",
                                    "lemma": "否定する",
                                    "reading": "ひていされる" if include_reading else "",
                                    "meaning": "被否定",
                                },
                            ],
                            "grammar": [
                                {
                                    "pattern": "～のに",
                                    "lemma": "～のに",
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
    assert first.vocabulary[1].lemma == "否定する"
    assert first.grammar[0].pattern == "～のに"
    assert second == first
    assert refreshed.translation == "重新分析后的翻译。"
    assert after_refresh == refreshed
    assert first_cached is False
    assert second_cached is True
    assert refreshed_cached is False
    assert after_refresh_cached is True
    assert calls == 2


def test_sentence_analyzer_supplies_existing_library_to_model(tmp_path: Path) -> None:
    known_items = [
        {
            "kind": "word",
            "lemma": "否定する",
            "reading": "ひていする",
            "meanings": ["否定"],
            "forms": ["否定される"],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user_content = json.loads(payload["messages"][1]["content"])
        assert user_content == {
            "sentence": "否定されました",
            "learning_library": known_items,
        }
        assert "dictionary lemma" in payload["messages"][0]["content"]
        return httpx.Response(200, json=learning_response())

    analyzer = SentenceAnalyzer(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    config = LanguageLearningConfig(base_url="https://vendor.example/v1", model="study-model")
    analysis, cached = analyzer.analyze(
        "否定されました",
        config,
        tmp_path / "learning",
        known_items=known_items,
    )

    assert cached is False
    assert analysis.vocabulary[1].lemma == "否定する"


def test_sentence_cache_is_stable_after_library_changes_for_same_context(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=learning_response())

    analyzer = SentenceAnalyzer(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    config = LanguageLearningConfig(base_url="https://vendor.example/v1", model="study-model")
    first, first_cached = analyzer.analyze(
        "否定されました",
        config,
        tmp_path / "learning",
        cache_scope="video-a/sentence-1",
    )
    second, second_cached = analyzer.analyze(
        "否定されました",
        config,
        tmp_path / "learning",
        known_items=[{"kind": "word", "lemma": "否定する"}],
        cache_scope="video-a/sentence-1",
    )
    _, other_context_cached = analyzer.analyze(
        "否定されました",
        config,
        tmp_path / "learning",
        known_items=[{"kind": "word", "lemma": "否定する"}],
        cache_scope="video-b/sentence-1",
    )

    assert first == second
    assert first_cached is False
    assert second_cached is True
    assert other_context_cached is False
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


def test_sentence_analyzer_repairs_missing_kanji_readings(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            payload = json.loads(request.content)
            repair_prompt = payload["messages"][-1]["content"]
            assert "zero-based indices [0, 1]" in repair_prompt
            assert "complete corrected JSON object" in repair_prompt
        return httpx.Response(
            200,
            json=learning_response(include_reading=calls > 1),
        )

    analyzer = SentenceAnalyzer(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    config = LanguageLearningConfig(base_url="https://vendor.example/v1", model="study-model")

    analysis, cached = analyzer.analyze("短期離職", config, tmp_path / "learning")

    assert cached is False
    assert calls == 2
    assert analysis.vocabulary[0].reading == "たんきりしょく"


def test_sentence_analyzer_keeps_other_results_if_reading_repair_still_fails(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=learning_response(include_reading=False))

    analyzer = SentenceAnalyzer(
        api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    config = LanguageLearningConfig(base_url="https://vendor.example/v1", model="study-model")

    analysis, cached = analyzer.analyze("短期離職", config, tmp_path / "learning")
    cached_analysis, cached_again = analyzer.analyze(
        "短期離職", config, tmp_path / "learning"
    )

    assert cached is False
    assert cached_again is True
    assert cached_analysis == analysis
    assert analysis.translation == "短期离职明明会被否定。"
    assert analysis.vocabulary[0].reading == ""
    assert calls == 2

    with pytest.raises(ValueError, match="must include a hiragana reading"):
        VocabularyAnalysisItem.model_validate(
            {
                "kind": "word",
                "expression": "短期離職",
                "lemma": "短期離職",
                "reading": "",
                "meaning": "短期离职",
            }
        )
