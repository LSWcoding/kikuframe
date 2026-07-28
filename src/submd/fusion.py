from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

from submd import __version__
from submd.errors import OrganizeError
from submd.json_io import write_json
from submd.models import (
    CaptionCorrection,
    SubtitleDocument,
    SubtitleSegment,
    TextLlmConfig,
    YouTubeCaptionCue,
    YouTubeCaptionTrack,
)

StatusCallback = Callable[[str], None]
_PROMPT_VERSION = "ocr-youtube-reading-fusion-v3"
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_SYSTEM_PROMPT = """You correct burned-in subtitle OCR with time-aligned YouTube captions as a
pronunciation reference. The burned-in subtitle remains the visual source of truth; YouTube
captions describe what was spoken and may omit, summarize, or phrase the on-screen text
differently.

Return only target segments whose OCR text should actually be changed. Omitted target segments are
kept verbatim. Every returned item must use the same segment_id supplied in target_segments.
Rules:
1. Correct likely OCR character errors, missing characters, and obvious repeated fragments only
   when confidence, alternatives, neighboring OCR, and the spoken reference support the change.
2. Use the YouTube caption primarily for pronunciation/reading evidence (especially homophones,
   kanji, names, and particles). Never blindly replace burned text with the YouTube caption.
3. Preserve information that exists only in the burned subtitle. Do not translate, summarize,
   censor, modernize, or improve its writing style.
4. Do not add sentence punctuation merely to create sentence boundaries; boundary analysis is a
   later stage.
5. Correct each segment independently. Never move words into or out of neighboring segments. If a
   fix would require redistributing text across segments, omit that correction.
6. If evidence is insufficient, omit the segment and keep the original OCR text unchanged.

Return JSON only:
{"corrections":[{"segment_id":"seg000001","corrected_text":"verbatim result",
"reason":"short reason or unchanged"}]}
"""


class FusionEngine(Protocol):
    def correct(
        self,
        before_context: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        after_context: list[dict[str, Any]],
    ) -> tuple[list[CaptionCorrection], str | None, dict[str, Any]]: ...


class OpenAICompatibleFusionEngine:
    def __init__(
        self,
        config: TextLlmConfig,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise OrganizeError("Text LLM API key is empty")
        self.config = config
        self.__api_key = api_key.strip()
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._sleep = sleep

    def correct(
        self,
        before_context: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        after_context: list[dict[str, Any]],
    ) -> tuple[list[CaptionCorrection], str | None, dict[str, Any]]:
        target_ids = {str(item["segment_id"]) for item in targets}
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "before_context": before_context,
                            "target_segments": targets,
                            "after_context": after_context,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for format_attempt in range(2):
            response = self._post_with_retries(payload)
            try:
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                parsed = self._extract_json(self._content_text(content))
                raw_items = parsed.get("corrections")
                if not isinstance(raw_items, list):
                    raise ValueError("corrections must be an array")
                corrections = [CaptionCorrection.model_validate(item) for item in raw_items]
                # Models occasionally also answer for the supplied neighboring context. Context
                # corrections belong to the adjacent chunk, so only exact target IDs are applied.
                target_corrections = {
                    item.segment_id: item
                    for item in corrections
                    if item.segment_id in target_ids
                }
                return (
                    list(target_corrections.values()),
                    response.headers.get("x-request-id") or envelope.get("id"),
                    envelope.get("usage") or {},
                )
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if format_attempt == 0:
                    continue
        raise OrganizeError(f"无法解析字幕综合纠错结果：{last_error}") from last_error

    def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"youtube-subtitle-md/{__version__}",
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(self._endpoint(), headers=headers, json=payload)
                if response.status_code not in {408, 409, 429} and response.status_code < 500:
                    if response.is_error:
                        body = response.text[:500].replace("\n", " ")
                        raise OrganizeError(
                            f"Text LLM API returned HTTP {response.status_code}: {body}"
                        )
                    return response
                last_error = OrganizeError(f"Text LLM API returned HTTP {response.status_code}")
                retry_after = self._retry_after_seconds(response)
            except OrganizeError:
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                retry_after = None
            if attempt < self.config.max_retries:
                self._sleep(retry_after if retry_after is not None else min(2**attempt, 8))
        message = str(last_error) if last_error else "unknown request failure"
        raise OrganizeError(f"字幕综合纠错请求重试后仍然失败：{message}") from last_error

    def _endpoint(self) -> str:
        if self.config.base_url.endswith("/chat/completions"):
            return self.config.base_url
        return f"{self.config.base_url}/chat/completions"

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


def aligned_cues(
    track: YouTubeCaptionTrack, start_ms: int, end_ms: int, padding_ms: int = 650
) -> list[YouTubeCaptionCue]:
    low = max(0, start_ms - padding_ms)
    high = end_ms + padding_ms
    return [cue for cue in track.cues if cue.end_ms >= low and cue.start_ms <= high]


def infer_caption_language(document: SubtitleDocument) -> str:
    sample = "".join(segment.text for segment in document.segments[:80])
    if re.search(r"[\u3040-\u30ff]", sample):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", sample):
        return "ko"
    if re.search(r"[\u3400-\u9fff]", sample):
        return "zh"
    if re.search(r"[\u0400-\u04ff]", sample):
        return "ru"
    return "en" if re.search(r"[A-Za-z]", sample) else "auto"


class SubtitleFusion:
    def __init__(
        self,
        engine: FusionEngine | None = None,
        api_key: str | None = None,
        status: StatusCallback | None = None,
    ) -> None:
        self.engine = engine
        self.api_key = api_key
        self.status = status or (lambda _message: None)

    def run(
        self,
        document: SubtitleDocument,
        track: YouTubeCaptionTrack,
        config: TextLlmConfig,
        output_path: Path,
        checkpoint_path: Path,
    ) -> SubtitleDocument:
        if not document.segments:
            write_json(output_path, document)
            return document
        payloads = [
            self._segment_payload(index, segment, track)
            for index, segment in enumerate(document.segments, 1)
        ]
        fingerprint = self._fingerprint(document, track, config)
        checkpoint = self._load_checkpoint(checkpoint_path, fingerprint, config)
        engine = self.engine or OpenAICompatibleFusionEngine(config, self.api_key or "")
        chunk_size = max(10, min(config.chunk_size, 80))
        chunks = [
            payloads[index : index + chunk_size]
            for index in range(0, len(payloads), chunk_size)
        ]
        correction_by_id: dict[str, CaptionCorrection] = {}
        for chunk_index, targets in enumerate(chunks, 1):
            key = f"{chunk_index:06d}"
            signature = self._payload_signature(targets)
            saved = checkpoint["chunks"].get(key)
            if isinstance(saved, dict) and saved.get("target_sha256") == signature:
                try:
                    restored = [
                        CaptionCorrection.model_validate(item) for item in saved["corrections"]
                    ]
                    if {item.segment_id for item in restored}.issubset(
                        {str(item["segment_id"]) for item in targets}
                    ):
                        correction_by_id.update({item.segment_id: item for item in restored})
                        self.status(f"综合纠错 {chunk_index}/{len(chunks)}（复用检查点）")
                        continue
                except (KeyError, TypeError, ValueError):
                    pass
            start = (chunk_index - 1) * chunk_size
            before = payloads[max(0, start - config.context_size) : start]
            end = start + len(targets)
            after = payloads[end : end + config.context_size]
            corrections, request_id, usage = engine.correct(before, targets, after)
            correction_by_id.update({item.segment_id: item for item in corrections})
            checkpoint["chunks"][key] = {
                "target_sha256": signature,
                "corrections": [item.model_dump(mode="json") for item in corrections],
                "request_id": request_id,
                "usage": usage,
            }
            write_json(checkpoint_path, checkpoint)
            self.status(f"综合纠错 {chunk_index}/{len(chunks)}")

        corrected: list[SubtitleSegment] = []
        changed = 0
        rejected = 0
        for index, segment in enumerate(document.segments, 1):
            segment_id = f"seg{index:06d}"
            item = correction_by_id.get(
                segment_id,
                CaptionCorrection(
                    segment_id=segment_id,
                    corrected_text=segment.text,
                    reason="unchanged",
                ),
            )
            references = [cue.text for cue in aligned_cues(track, segment.start_ms, segment.end_ms)]
            text = item.corrected_text.strip() or segment.text
            if text != segment.text and not self._safe_correction(segment.text, text):
                text = segment.text
                rejected += 1
            is_changed = text != segment.text
            changed += int(is_changed)
            corrected.append(
                segment.model_copy(
                    update={
                        "text": text,
                        "source": "burned_ocr_corrected",
                        "original_text": segment.text if is_changed else None,
                        "youtube_reference": list(dict.fromkeys(references)),
                        "correction_reason": item.reason.strip() if is_changed else None,
                    }
                )
            )
        result = document.model_copy(update={"segments": corrected})
        write_json(output_path, result)
        self.status(
            f"YouTube 读音参考综合完成：采用 {changed} 处修正，"
            f"安全拦截 {rejected} 处大幅改写"
        )
        return result

    @staticmethod
    def _safe_correction(original: str, corrected: str) -> bool:
        def comparable(value: str) -> str:
            return re.sub(r"[\s、。，．,.!?！？:：;；'\"“”‘’]+", "", value)

        old = comparable(original)
        new = comparable(corrected)
        if not old or not new:
            return False
        allowed_delta = max(2, round(len(old) * 0.2))
        return abs(len(new) - len(old)) <= allowed_delta

    @staticmethod
    def _segment_payload(
        index: int, segment: SubtitleSegment, track: YouTubeCaptionTrack
    ) -> dict[str, Any]:
        references = aligned_cues(track, segment.start_ms, segment.end_ms)
        return {
            "segment_id": f"seg{index:06d}",
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "ocr_text": segment.text,
            "ocr_confidence": round(segment.confidence, 4),
            "ocr_alternatives": segment.alternatives,
            "youtube_reading_reference": [
                {"start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": cue.text}
                for cue in references
            ],
        }

    @staticmethod
    def _fingerprint(
        document: SubtitleDocument, track: YouTubeCaptionTrack, config: TextLlmConfig
    ) -> str:
        stable = {
            "prompt_version": _PROMPT_VERSION,
            "model": config.model,
            "base_url": config.base_url,
            "segments": [segment.model_dump(mode="json") for segment in document.segments],
            "track": track.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _payload_signature(payloads: list[dict[str, Any]]) -> str:
        return hashlib.sha256(
            json.dumps(payloads, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _load_checkpoint(
        path: Path, fingerprint: str, config: TextLlmConfig
    ) -> dict[str, Any]:
        expected = {
            "schema_version": 1,
            "prompt_version": _PROMPT_VERSION,
            "fingerprint": fingerprint,
            "model": config.model,
        }
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                saved = None
            if (
                isinstance(saved, dict)
                and all(saved.get(key) == value for key, value in expected.items())
                and isinstance(saved.get("chunks"), dict)
            ):
                return saved
        return {**expected, "chunks": {}}
