from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from submd import __version__
from submd.errors import LearningAnalysisError
from submd.json_io import write_json
from submd.models import LanguageLearningConfig, SentenceLearningAnalysis

PROMPT_VERSION = "japanese-learning-v1"
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_SYSTEM_PROMPT = """You are a meticulous Japanese teacher for native Chinese speakers.
Analyze exactly the Japanese sentence supplied by the user. The sentence may originate from OCR;
do not silently replace it with a different sentence. Explain the text that is actually supplied.

Requirements:
1. Give a natural Simplified Chinese translation of the whole sentence.
2. Extract all meaningful words and fixed or common collocations. Do not omit functionally
   important words merely because they are elementary.
3. Every vocabulary expression containing Japanese kanji MUST have its reading written entirely
   in hiragana. Kana-only expressions may use an empty reading.
4. Give a concise Simplified Chinese meaning for every vocabulary or collocation item.
5. Extract and explain every grammar pattern used in the sentence in Simplified Chinese. Explain
   its role in this exact sentence, not only a dictionary definition.
6. Do not use Markdown and do not add sections outside the JSON schema.

Return JSON only in this exact shape:
{
  "translation": "整句的简体中文翻译",
  "vocabulary": [
    {"expression": "日语单词或搭配", "reading": "含汉字时必须填写平假名", "meaning": "中文释义"}
  ],
  "grammar": [
    {"pattern": "文法形式", "explanation": "该文法的中文说明及在本句中的作用"}
  ]
}
"""


class SentenceAnalyzer:
    """Analyze one saved subtitle sentence with an OpenAI-compatible text model."""

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise LearningAnalysisError("语言学习模型 API Key 为空")
        self.__api_key = api_key.strip()
        self._client = client
        self._sleep = sleep

    def analyze(
        self,
        sentence: str,
        config: LanguageLearningConfig,
        cache_root: Path,
        force: bool = False,
    ) -> tuple[SentenceLearningAnalysis, bool]:
        clean_sentence = sentence.strip()
        if not clean_sentence:
            raise LearningAnalysisError("待分析的句子为空")
        cache_path = self._cache_path(cache_root, clean_sentence, config.model)
        if not force:
            cached = self._read_cache(cache_path, clean_sentence, config.model)
            if cached is not None:
                return cached, True

        payload: dict[str, Any] = {
            "model": config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": clean_sentence},
            ],
        }
        if config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for _format_attempt in range(2):
            response = self._post_with_retries(payload, config)
            try:
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                parsed = self._extract_json(self._content_text(content))
                analysis = SentenceLearningAnalysis(
                    prompt_version=PROMPT_VERSION,
                    sentence=clean_sentence,
                    model=config.model,
                    translation=parsed["translation"],
                    vocabulary=parsed["vocabulary"],
                    grammar=parsed["grammar"],
                )
                write_json(cache_path, analysis.model_dump(mode="json"))
                return analysis, False
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise LearningAnalysisError(f"无法解析语言学习模型的分析结果：{last_error}") from last_error

    def _post_with_retries(
        self, payload: dict[str, Any], config: LanguageLearningConfig
    ) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"kikuframe/{__version__}",
        }
        client = self._client or httpx.Client(timeout=config.timeout_seconds)
        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            try:
                response = client.post(
                    self._endpoint(config.base_url), headers=headers, json=payload
                )
                if response.status_code not in {408, 409, 429} and response.status_code < 500:
                    if response.is_error:
                        body = response.text[:500].replace("\n", " ")
                        raise LearningAnalysisError(
                            f"语言学习模型返回 HTTP {response.status_code}：{body}"
                        )
                    return response
                last_error = LearningAnalysisError(
                    f"语言学习模型返回 HTTP {response.status_code}"
                )
                retry_after = self._retry_after_seconds(response)
            except LearningAnalysisError:
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                retry_after = None
            if attempt < config.max_retries:
                self._sleep(retry_after if retry_after is not None else min(2**attempt, 8))
        message = str(last_error) if last_error else "unknown request failure"
        raise LearningAnalysisError(f"语言学习模型请求重试后仍然失败：{message}") from last_error

    @staticmethod
    def _cache_path(cache_root: Path, sentence: str, model: str) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {"prompt_version": PROMPT_VERSION, "model": model, "sentence": sentence},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        return cache_root / f"{digest}.json"

    @staticmethod
    def _read_cache(
        path: Path, sentence: str, model: str
    ) -> SentenceLearningAnalysis | None:
        if not path.is_file():
            return None
        try:
            value = SentenceLearningAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            value.prompt_version != PROMPT_VERSION
            or value.sentence != sentence
            or value.model != model
        ):
            return None
        return value

    @staticmethod
    def _endpoint(base_url: str) -> str:
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        raise ValueError("message content is not text")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        clean = _JSON_FENCE.sub("", text.strip()).strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(clean[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("response JSON must be an object")
        return value

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            return min(30.0, max(0.0, float(value)))
        except ValueError:
            return None
