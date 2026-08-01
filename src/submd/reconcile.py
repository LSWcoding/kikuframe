from __future__ import annotations

import hashlib
import json
import math
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
    CloudOcrConfig,
    FrameRef,
    SubtitleDocument,
    SubtitleSegment,
    TextLlmConfig,
    YouTubeCaptionTrack,
    YoutubePrimaryReconciliation,
)
from submd.ocr.openai_compatible import OpenAICompatibleOcrEngine

StatusCallback = Callable[[str], None]
WINDOW_MS = 20_000
CONTEXT_MS = 3_000
_PROMPT_VERSION = "windowed-text-adjudication-v2"
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_SYSTEM_PROMPT = """You are a text adjudicator for Japanese video subtitles. You receive:
- target YouTube caption cues: the complete spoken-text baseline for this 20-second window;
- surrounding YouTube cues: read-only context;
- independent burned-subtitle OCR candidates with timestamps and alternative readings;
- the already resolved tail from the previous window;
- optionally, a second visual OCR result for disputed source frames.

Rules:
1. Return every target YouTube cue_id exactly once and in chronological order. Never return a
   context-only cue. You may combine adjacent target cues and remove rolling-caption overlap.
2. YouTube is the spoken-content baseline. Never delete its content merely because OCR missed it.
3. Use OCR to correct homophones, unclear pronunciation, kanji, names, particles, small kana and
   gemination only when timing, reading, grammar and context support the correction.
4. Do not vote. Compare the two textual sources and the context. Never invent wording absent from
   both YouTube and OCR candidates.
5. When OCR repeats a spoken phrase but YouTube contains it once, return it once.
6. OCR-only explanatory text that is genuinely not spoken belongs in screen_annotations and must
   use only target_ocr_segment_ids. Do not turn ordinary verification OCR into annotations.
7. Set needs_visual_recheck=true only when both competing readings remain plausible and the source
   glyphs must be reread. Include the relevant ocr_segment_ids. If visual_rechecks are supplied,
   use them for the final choice and normally clear this flag; if still impossible, keep YouTube.
8. Preserve Japanese text. Do not translate, summarize, split into semantic sentences, or add
   content. Sentence-boundary organization happens in a later, separate stage.

Return JSON only:
{"spoken":[{"cue_ids":["yt000001"],"ocr_segment_ids":["ocr000001"],
"text":"resolved text","reason":"short evidence-based reason","needs_visual_recheck":false}],
"screen_annotations":[{"ocr_segment_ids":["ocr000002"],"text":"screen-only text",
"reason":"not spoken","needs_visual_recheck":false}]}
"""


class ReconciliationEngine(Protocol):
    def reconcile(
        self,
        target_youtube_cues: list[dict[str, Any]],
        context_youtube_cues: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
        target_ocr_segment_ids: list[str],
        previous_resolved_tail: list[dict[str, Any]],
        visual_rechecks: dict[str, list[dict[str, Any]]] | None = None,
        *,
        final_pass: bool = False,
    ) -> tuple[YoutubePrimaryReconciliation, str | None, dict[str, Any]]: ...


class VisualReviewer(Protocol):
    def review(
        self, segment_ids: list[str], evidence_path: Path
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]: ...


class OpenAICompatibleReconciliationEngine:
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

    def reconcile(
        self,
        target_youtube_cues: list[dict[str, Any]],
        context_youtube_cues: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
        target_ocr_segment_ids: list[str],
        previous_resolved_tail: list[dict[str, Any]],
        visual_rechecks: dict[str, list[dict[str, Any]]] | None = None,
        *,
        final_pass: bool = False,
    ) -> tuple[YoutubePrimaryReconciliation, str | None, dict[str, Any]]:
        expected_cue_ids = [str(item["cue_id"]) for item in target_youtube_cues]
        valid_ocr_ids = {str(item["segment_id"]) for item in ocr_segments}
        target_ocr_ids = set(target_ocr_segment_ids)
        user_input = {
            "target_youtube_cues": target_youtube_cues,
            "surrounding_youtube_context": context_youtube_cues,
            "target_ocr_segment_ids": target_ocr_segment_ids,
            "visual_ocr_segments": ocr_segments,
            "previous_resolved_tail": previous_resolved_tail,
            "visual_rechecks": visual_rechecks or {},
            "pass": "final_after_visual_recheck" if final_pass else "initial_text_only",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        user_input, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for format_attempt in range(3):
            response = self._post_with_retries(payload)
            content_text = ""
            try:
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                content_text = self._content_text(content)
                parsed = YoutubePrimaryReconciliation.model_validate(
                    self._extract_json(content_text)
                )
                self._validate_result(
                    parsed,
                    expected_cue_ids=expected_cue_ids,
                    valid_ocr_ids=valid_ocr_ids,
                    target_ocr_ids=target_ocr_ids,
                )
                return (
                    parsed,
                    response.headers.get("x-request-id") or envelope.get("id"),
                    envelope.get("usage") or {},
                )
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if format_attempt < 2:
                    if content_text:
                        payload["messages"].append(
                            {"role": "assistant", "content": content_text}
                        )
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous JSON failed validation: "
                                f"{exc}. Rebuild and return the complete JSON object. "
                                "spoken must contain these target cue_ids exactly once in this "
                                f"order: {json.dumps(expected_cue_ids, ensure_ascii=False)}. "
                                "screen_annotations may use only these target OCR ids: "
                                f"{json.dumps(sorted(target_ocr_ids), ensure_ascii=False)}. "
                                "Do not omit, duplicate, reorder, or add IDs."
                            ),
                        }
                    )
        raise OrganizeError(f"无法解析 20 秒字幕取舍结果：{last_error}") from last_error

    @staticmethod
    def _validate_result(
        result: YoutubePrimaryReconciliation,
        *,
        expected_cue_ids: list[str],
        valid_ocr_ids: set[str],
        target_ocr_ids: set[str],
    ) -> None:
        returned_cue_ids = [cue_id for item in result.spoken for cue_id in item.cue_ids]
        if returned_cue_ids != expected_cue_ids:
            raise ValueError(
                "spoken cue_ids must contain every target YouTube cue exactly once and in order"
            )
        spoken_ocr_ids = {
            segment_id for item in result.spoken for segment_id in item.ocr_segment_ids
        }
        unknown_spoken = spoken_ocr_ids - valid_ocr_ids
        if unknown_spoken:
            raise ValueError("unknown spoken OCR ids: " + ", ".join(sorted(unknown_spoken)))
        annotation_ids = [
            segment_id
            for item in result.screen_annotations
            for segment_id in item.ocr_segment_ids
        ]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("an OCR segment was used in multiple screen annotations")
        unknown_annotations = set(annotation_ids) - target_ocr_ids
        if unknown_annotations:
            raise ValueError(
                "annotation used context-only or unknown OCR ids: "
                + ", ".join(sorted(unknown_annotations))
            )

    def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"kikuframe/{__version__}",
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
        raise OrganizeError(f"20 秒字幕取舍请求重试后仍然失败：{message}") from last_error

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


class CloudEvidenceVisualReviewer:
    """Reread only preserved source frames for OCR segments disputed by the text model."""

    def __init__(self, config: CloudOcrConfig, api_key: str, language_hint: str) -> None:
        self.config = config
        self.engine = OpenAICompatibleOcrEngine(config, api_key, language_hint=language_hint)

    def review(
        self, segment_ids: list[str], evidence_path: Path
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        try:
            manifest = json.loads(evidence_path.read_text(encoding="utf-8"))
            stored = manifest["segments"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OrganizeError("视觉复核来源帧清单不可用") from exc
        evidence_root = evidence_path.parent
        results: dict[str, list[dict[str, Any]]] = {}
        api_calls: list[dict[str, Any]] = []
        for segment_id in dict.fromkeys(segment_ids):
            items = stored.get(segment_id, []) if isinstance(stored, dict) else []
            frames = [
                FrameRef(
                    index=int(item["frame_index"]),
                    timestamp_ms=int(item["timestamp_ms"]),
                    path=evidence_root / str(item["path"]),
                )
                for item in items
                if isinstance(item, dict)
                and (evidence_root / str(item.get("path") or "")).is_file()
            ]
            if not frames:
                results[segment_id] = []
                continue
            batch = self.engine.recognize_batch(frames)
            by_id = {item.frame_id: item for item in batch.frames}
            reread: list[dict[str, Any]] = []
            for frame, original in zip(frames, items, strict=False):
                recognized = by_id.get(f"{frame.index:06d}")
                if recognized is None:
                    continue
                reread.append(
                    {
                        "timestamp_ms": frame.timestamp_ms,
                        "text": recognized.text,
                        "confidence": recognized.confidence,
                        "first_pass_text": str(original.get("first_pass_text") or ""),
                    }
                )
            results[segment_id] = reread
            api_calls.append(
                {
                    "segment_id": segment_id,
                    "model": self.config.model,
                    "request_id": batch.request_id,
                    "usage": batch.usage,
                    "frame_count": len(frames),
                }
            )
        return results, api_calls


class YoutubePrimaryReconciler:
    def __init__(
        self,
        engine: ReconciliationEngine | None = None,
        visual_reviewer: VisualReviewer | None = None,
        api_key: str | None = None,
        status: StatusCallback | None = None,
    ) -> None:
        self.engine = engine
        self.visual_reviewer = visual_reviewer
        self.api_key = api_key
        self.status = status or (lambda _message: None)

    def run(
        self,
        document: SubtitleDocument,
        track: YouTubeCaptionTrack,
        config: TextLlmConfig,
        output_path: Path,
        checkpoint_path: Path,
        *,
        evidence_path: Path | None = None,
        visual_review_config: CloudOcrConfig | None = None,
        visual_review_api_key: str | None = None,
    ) -> SubtitleDocument:
        cue_payloads = [cue.model_dump(mode="json") for cue in track.cues]
        ocr_payloads = [
            {
                "segment_id": f"ocr{index:06d}",
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
                "confidence": round(segment.confidence, 4),
                "alternatives": segment.alternatives,
            }
            for index, segment in enumerate(document.segments, 1)
        ]
        fingerprint = self._fingerprint(document, track, config, visual_review_config)
        engine = self.engine or OpenAICompatibleReconciliationEngine(config, self.api_key or "")
        reviewer = self.visual_reviewer
        if reviewer is None and visual_review_config is not None and visual_review_api_key:
            reviewer = CloudEvidenceVisualReviewer(
                visual_review_config, visual_review_api_key, track.language
            )
        evidence_path = evidence_path or (
            checkpoint_path.parent / "evidence" / "segment_evidence.json"
        )
        window_dir = checkpoint_path.parent / "reconciliation-windows"
        window_dir.mkdir(parents=True, exist_ok=True)
        duration_ms = max(
            document.video.duration_ms,
            max((cue.end_ms for cue in track.cues), default=0),
            max((segment.end_ms for segment in document.segments), default=0),
        )
        window_count = max(1, math.ceil(duration_ms / WINDOW_MS))
        reconciliations: list[YoutubePrimaryReconciliation] = []
        previous_tail: list[dict[str, Any]] = []
        completed_windows: list[dict[str, Any]] = []

        for window_index in range(window_count):
            start_ms = window_index * WINDOW_MS
            end_ms = min(duration_ms, start_ms + WINDOW_MS)
            target_cues = [
                item
                for item in cue_payloads
                if start_ms <= int(item["start_ms"]) < start_ms + WINDOW_MS
            ]
            target_ocr = [
                item
                for item in ocr_payloads
                if start_ms <= int(item["start_ms"]) < start_ms + WINDOW_MS
            ]
            if not target_cues and not target_ocr:
                continue
            context_cues = [
                item
                for item in cue_payloads
                if int(item["end_ms"]) >= max(0, start_ms - CONTEXT_MS)
                and int(item["start_ms"]) <= end_ms + CONTEXT_MS
            ]
            context_ocr = [
                item
                for item in ocr_payloads
                if int(item["end_ms"]) >= max(0, start_ms - CONTEXT_MS)
                and int(item["start_ms"]) <= end_ms + CONTEXT_MS
            ]
            target_ocr_ids = [str(item["segment_id"]) for item in target_ocr]
            window_fingerprint = self._window_fingerprint(
                fingerprint,
                start_ms,
                target_cues,
                context_cues,
                context_ocr,
                previous_tail,
            )
            window_path = window_dir / f"window_{start_ms:09d}_{end_ms:09d}.json"
            reconciliation: YoutubePrimaryReconciliation | None = None
            checkpoint: dict[str, Any] = {}
            if window_path.is_file():
                try:
                    checkpoint = json.loads(window_path.read_text(encoding="utf-8"))
                    if checkpoint.get("fingerprint") == window_fingerprint:
                        reconciliation = YoutubePrimaryReconciliation.model_validate(
                            checkpoint["result"]
                        )
                        self.status(
                            f"20 秒字幕取舍 {start_ms // 1000:>4}–"
                            f"{end_ms // 1000:>4} 秒（复用检查点）"
                        )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    reconciliation = None
            if reconciliation is None:
                self.status(
                    f"20 秒字幕取舍 {start_ms // 1000:>4}–{end_ms // 1000:>4} 秒…"
                )
                reconciliation, request_id, usage = engine.reconcile(
                    target_cues,
                    context_cues,
                    context_ocr,
                    target_ocr_ids,
                    previous_tail,
                )
                checkpoint = {
                    "schema_version": 2,
                    "prompt_version": _PROMPT_VERSION,
                    "fingerprint": window_fingerprint,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "model": config.model,
                    "request_id": request_id,
                    "usage": usage,
                    "initial_result": reconciliation.model_dump(mode="json"),
                }
                review_ids = self._visual_review_ids(reconciliation, target_ocr_ids)
                if review_ids and reviewer is not None and evidence_path.is_file():
                    self.status(
                        f"该窗口有 {len(review_ids)} 个文字冲突，回读来源帧后再次取舍…"
                    )
                    visual_rechecks, review_calls = reviewer.review(review_ids, evidence_path)
                    if any(visual_rechecks.values()):
                        reconciliation, final_request_id, final_usage = engine.reconcile(
                            target_cues,
                            context_cues,
                            context_ocr,
                            target_ocr_ids,
                            previous_tail,
                            visual_rechecks,
                            final_pass=True,
                        )
                        checkpoint["visual_rechecks"] = visual_rechecks
                        checkpoint["visual_review_calls"] = review_calls
                        checkpoint["final_request_id"] = final_request_id
                        checkpoint["final_usage"] = final_usage
                checkpoint["result"] = reconciliation.model_dump(mode="json")
                write_json(window_path, checkpoint)

            reconciliations.append(reconciliation)
            previous_tail = [
                item.model_dump(mode="json") for item in reconciliation.spoken[-3:]
            ]
            completed_windows.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "path": str(window_path.relative_to(checkpoint_path.parent)),
                    "fingerprint": window_fingerprint,
                }
            )
            write_json(
                checkpoint_path,
                {
                    "schema_version": 2,
                    "prompt_version": _PROMPT_VERSION,
                    "fingerprint": fingerprint,
                    "window_ms": WINDOW_MS,
                    "completed_windows": completed_windows,
                },
            )

        result = self._build_document(document, track, reconciliations)
        write_json(output_path, result)
        spoken_count = sum(len(item.spoken) for item in reconciliations)
        annotation_count = sum(len(item.screen_annotations) for item in reconciliations)
        unresolved = sum(1 for segment in result.segments if segment.needs_review)
        self.status(
            f"20 秒分批综合完成：{spoken_count} 个语音段，{annotation_count} 个画面补充"
            + (f"，{unresolved} 段标记待人工复核" if unresolved else "")
        )
        return result

    @staticmethod
    def _visual_review_ids(
        reconciliation: YoutubePrimaryReconciliation, target_ocr_ids: list[str]
    ) -> list[str]:
        requested = [
            segment_id
            for item in [*reconciliation.spoken, *reconciliation.screen_annotations]
            if item.needs_visual_recheck
            for segment_id in item.ocr_segment_ids
        ]
        if any(item.needs_visual_recheck for item in reconciliation.spoken) and not requested:
            requested.extend(target_ocr_ids)
        return list(dict.fromkeys(requested))

    @staticmethod
    def _build_document(
        document: SubtitleDocument,
        track: YouTubeCaptionTrack,
        reconciliations: list[YoutubePrimaryReconciliation],
    ) -> SubtitleDocument:
        cue_by_id = {cue.cue_id: cue for cue in track.cues}
        ocr_by_id = {
            f"ocr{index:06d}": segment for index, segment in enumerate(document.segments, 1)
        }
        segments: list[SubtitleSegment] = []
        for reconciliation in reconciliations:
            for item in reconciliation.spoken:
                cues = [cue_by_id[cue_id] for cue_id in item.cue_ids]
                visuals = [
                    ocr_by_id[segment_id]
                    for segment_id in item.ocr_segment_ids
                    if segment_id in ocr_by_id
                ]
                segments.append(
                    SubtitleSegment(
                        start_ms=min(cue.start_ms for cue in cues),
                        end_ms=max(cue.end_ms for cue in cues),
                        text=item.text.strip(),
                        source="youtube_primary",
                        confidence=0.72 if item.needs_visual_recheck else 0.92,
                        observation_count=len(cues),
                        needs_review=item.needs_visual_recheck,
                        original_text=(" / ".join(segment.text for segment in visuals) or None),
                        youtube_reference=[cue.text for cue in cues],
                        correction_reason=item.reason.strip() or None,
                    )
                )
            for item in reconciliation.screen_annotations:
                visuals = [ocr_by_id[segment_id] for segment_id in item.ocr_segment_ids]
                segments.append(
                    SubtitleSegment(
                        start_ms=min(segment.start_ms for segment in visuals),
                        end_ms=max(segment.end_ms for segment in visuals),
                        text=item.text.strip(),
                        source="screen_annotation",
                        confidence=(
                            sum(segment.confidence for segment in visuals) / len(visuals)
                        ),
                        observation_count=sum(
                            segment.observation_count for segment in visuals
                        ),
                        needs_review=item.needs_visual_recheck,
                        original_text=" / ".join(segment.text for segment in visuals),
                        correction_reason=item.reason.strip() or None,
                    )
                )
        segments.sort(
            key=lambda segment: (
                segment.start_ms,
                1 if segment.source == "screen_annotation" else 0,
            )
        )
        return document.model_copy(update={"segments": segments})

    @staticmethod
    def _window_fingerprint(
        global_fingerprint: str,
        start_ms: int,
        target_cues: list[dict[str, Any]],
        context_cues: list[dict[str, Any]],
        context_ocr: list[dict[str, Any]],
        previous_tail: list[dict[str, Any]],
    ) -> str:
        stable = {
            "global": global_fingerprint,
            "start_ms": start_ms,
            "target_cues": target_cues,
            "context_cues": context_cues,
            "context_ocr": context_ocr,
            "previous_tail": previous_tail,
        }
        return hashlib.sha256(
            json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _fingerprint(
        document: SubtitleDocument,
        track: YouTubeCaptionTrack,
        config: TextLlmConfig,
        visual_review_config: CloudOcrConfig | None,
    ) -> str:
        stable = {
            "prompt_version": _PROMPT_VERSION,
            "window_ms": WINDOW_MS,
            "model": config.model,
            "base_url": config.base_url,
            "review_model": visual_review_config.model if visual_review_config else None,
            "segments": [segment.model_dump(mode="json") for segment in document.segments],
            "track": track.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
