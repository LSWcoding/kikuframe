from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

import httpx

from submd import __version__
from submd.errors import OrganizeError
from submd.exporters import sanitize_filename
from submd.json_io import write_json, write_text
from submd.models import (
    OrganizedSentence,
    OrganizedSubtitleDocument,
    OrganizeResult,
    TextLlmConfig,
    YouTubeCaptionTrack,
)

StatusCallback = Callable[[str], None]

_PROMPT_VERSION = "semantic-character-boundaries-v6"
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_TIMESTAMP = r"\d{2}(?::\d{2})?:\d{2}\.\d{3}"
_TIMED_SUBTITLE = re.compile(
    rf"^\s*[-*+]\s+\[(?P<start>{_TIMESTAMP})\s*[–—-]\s*"
    rf"(?P<end>{_TIMESTAMP})\]\s*(?P<body>.*?)\s*$"
)
_REVIEW_MARK = re.compile(r"\s*⚠\ufe0f?\s*$")
_TERMINAL_UNIT = re.compile(r".+?(?:[。！？!?]+|\.(?=\s|$)|$)", re.DOTALL)

_SYSTEM_PROMPT = """You determine semantic sentence boundaries in subtitle fragments.

You receive three ordered arrays: before_context, target_units, and after_context.
Each item has an immutable unit_id, verbatim text, and sometimes a time-aligned
youtube_reading_reference. Context exists only to understand sentences crossing a chunk edge.

Rules:
1. Return a boundary when a grammatically and semantically complete spoken sentence ends. A
   subtitle screen change is not a sentence boundary. Boundaries may occur inside a target unit.
2. Each boundary contains a target unit_id and after_char: the 1-based Unicode character count
   after which the sentence ends. It must be between 1 and the supplied text_length. A boundary at
   the end of a unit is exactly after_char=text_length, never text_length+1.
3. Return IDs only from target_units, never from either context array.
4. Do not translate, rewrite, correct, summarize, or reproduce subtitle text.
5. Do not mark the last target merely because it is the end of a chunk; use after_context.
6. Treat terminal punctuation as strong evidence, but use meaning and grammar when punctuation
   is absent or OCR punctuation is unreliable.
7. Do not end after an unfinished topic, modifier, conjunction, filler, or Japanese connective
   such as は, が, を, に, で, と, ので, から, けど, or という unless it is clearly a
   deliberate standalone utterance.
8. Prefer joining uncertain short fragments into the surrounding sentence. Do not create a
   one-line sentence from a phrase that depends on the next unit for its meaning.
9. Avoid merging multiple complete claims into one very long sentence. A change from one complete
   claim or question to the next should normally be a boundary even when punctuation is missing.

Examples:
- u000001="フリーダー", u000002="ですさっき病院に行ったら" -> boundaries after
  character 2 of u000002 ("です") and at the true end of the following sentence.
- u000001="2026年7月20日", u000002="月曜日天気は", u000003="晴れ" -> break only
  after u000003.
- u000001="今日も私が誰かにおすすめしたい", u000002="お得な話をお届けします" ->
  break only after u000002.
- u000001="最近の話で言うと", u000002="まあなんか", u000003="呪われてますね" ->
  break only after u000003.

Return JSON only:
{"boundaries":[{"unit_id":"u000012","after_char":5},{"unit_id":"u000019","after_char":9}]}
"""


@dataclass(frozen=True)
class SubtitleUnit:
    unit_id: str
    text: str
    start_ms: int = 0
    end_ms: int = 0
    source_segment_ids: tuple[str, ...] = ()
    youtube_reading_reference: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "unit_id": self.unit_id,
            "text": self.text,
            "text_length": len(self.text),
        }
        if self.youtube_reading_reference:
            payload["youtube_reading_reference"] = list(self.youtube_reading_reference)
        return payload


@dataclass(frozen=True)
class TimedSubtitleFragment:
    segment_id: str
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class BoundaryResult:
    break_after: frozenset[str]
    split_after: frozenset[tuple[str, int]] = frozenset()
    request_id: str | None = None
    usage: dict[str, Any] | None = None


class BoundaryEngine(Protocol):
    def decide_boundaries(
        self,
        before_context: list[SubtitleUnit],
        target_units: list[SubtitleUnit],
        after_context: list[SubtitleUnit],
    ) -> BoundaryResult: ...


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def extract_timed_subtitle_fragments(markdown: str) -> list[TimedSubtitleFragment]:
    """Extract clean subtitle text together with its source screen timing."""
    fragments: list[TimedSubtitleFragment] = []
    for line in markdown.splitlines():
        match = _TIMED_SUBTITLE.match(line)
        if not match:
            continue
        raw_text = _REVIEW_MARK.sub("", match.group("body"))
        parser = _VisibleTextParser()
        try:
            parser.feed(raw_text)
            parser.close()
        except Exception as exc:
            raise OrganizeError(f"无法解析字幕中的 HTML：{exc}") from exc
        pieces = [piece.strip() for piece in "".join(parser.parts).splitlines() if piece.strip()]
        clean = _join_pieces(pieces)
        if clean:
            fragments.append(
                TimedSubtitleFragment(
                    segment_id=f"seg{len(fragments) + 1:06d}",
                    text=clean,
                    start_ms=_timestamp_to_ms(match.group("start")),
                    end_ms=_timestamp_to_ms(match.group("end")),
                )
            )
    if not fragments:
        raise OrganizeError("输入文件中没有找到带时间戳的字幕行")
    return fragments


def extract_subtitle_fragments(markdown: str) -> list[str]:
    """Backward-compatible text-only view of timestamped subtitle fragments."""
    return [fragment.text for fragment in extract_timed_subtitle_fragments(markdown)]


def build_subtitle_units(fragments: list[str]) -> list[SubtitleUnit]:
    raw_units: list[str] = []
    for fragment in fragments:
        raw_units.extend(_fragment_unit_texts(fragment))
    return [
        SubtitleUnit(unit_id=f"u{index:06d}", text=text)
        for index, text in enumerate(raw_units, start=1)
    ]


def build_timed_subtitle_units(
    fragments: list[TimedSubtitleFragment],
    reference_track: YouTubeCaptionTrack | None = None,
) -> list[SubtitleUnit]:
    units: list[SubtitleUnit] = []
    for fragment in fragments:
        texts = _fragment_unit_texts(fragment.text)
        total_characters = max(1, sum(len(text) for text in texts))
        consumed = 0
        duration = max(0, fragment.end_ms - fragment.start_ms)
        for text in texts:
            start_ms = fragment.start_ms + round(duration * consumed / total_characters)
            consumed += len(text)
            end_ms = fragment.start_ms + round(duration * consumed / total_characters)
            units.append(
                SubtitleUnit(
                    unit_id=f"u{len(units) + 1:06d}",
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    source_segment_ids=(fragment.segment_id,),
                    youtube_reading_reference=tuple(
                        dict.fromkeys(
                            cue.text
                            for cue in (reference_track.cues if reference_track else [])
                            if cue.end_ms >= fragment.start_ms - 650
                            and cue.start_ms <= fragment.end_ms + 650
                        )
                    ),
                )
            )
    return units


def _fragment_unit_texts(text: str) -> list[str]:
    values: list[str] = []
    for match in _TERMINAL_UNIT.finditer(text):
        value = match.group(0).strip()
        if not value:
            continue
        if re.search(r"[\u3040-\u9fff]\s+[\u3040-\u9fff]", value):
            values.extend(part for part in re.split(r"\s+", value) if part)
        else:
            values.append(value)
    return values


class OpenAICompatibleBoundaryEngine:
    """Semantic boundary decisions through an OpenAI-compatible text model."""

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
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    def decide_boundaries(
        self,
        before_context: list[SubtitleUnit],
        target_units: list[SubtitleUnit],
        after_context: list[SubtitleUnit],
    ) -> BoundaryResult:
        if not target_units:
            return BoundaryResult(break_after=frozenset(), usage={})
        target_ids = {unit.unit_id for unit in target_units}
        context_ids = {unit.unit_id for unit in [*before_context, *after_context]}
        user_payload = {
            "before_context": [unit.as_payload() for unit in before_context],
            "target_units": [unit.as_payload() for unit in target_units],
            "after_context": [unit.as_payload() for unit in after_context],
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
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
                raw_boundaries = parsed.get("boundaries")
                legacy_values = parsed.get("break_after")
                if raw_boundaries is None and isinstance(legacy_values, list):
                    raw_boundaries = [
                        {"unit_id": unit_id, "after_char": len(next(
                            unit.text
                            for unit in [*before_context, *target_units, *after_context]
                            if unit.unit_id == unit_id
                        ))}
                        for unit_id in legacy_values
                    ]
                if not isinstance(raw_boundaries, list):
                    raise ValueError("boundaries must be an array")
                all_units = {
                    unit.unit_id: unit
                    for unit in [*before_context, *target_units, *after_context]
                }
                parsed_boundaries: list[tuple[str, int]] = []
                for item in raw_boundaries:
                    if not isinstance(item, dict):
                        raise ValueError("each boundary must be an object")
                    unit_id = item.get("unit_id")
                    after_char = item.get("after_char")
                    if not isinstance(unit_id, str) or not isinstance(after_char, int):
                        raise ValueError("boundary unit_id/after_char is invalid")
                    parsed_boundaries.append((unit_id, after_char))
                unknown = sorted(
                    {unit_id for unit_id, _offset in parsed_boundaries}.difference(
                        target_ids | context_ids
                    )
                )
                if unknown:
                    raise ValueError(f"response contains unknown unit IDs: {unknown}")
                normalized_boundaries = [
                    (
                        unit_id,
                        len(all_units[unit_id].text)
                        if offset == len(all_units[unit_id].text) + 1
                        else offset,
                    )
                    for unit_id, offset in parsed_boundaries
                ]
                invalid = [
                    (unit_id, offset)
                    for unit_id, offset in normalized_boundaries
                    if offset < 1 or offset > len(all_units[unit_id].text)
                ]
                if invalid:
                    raise ValueError(f"response contains invalid character offsets: {invalid}")
                target_boundaries = [
                    (unit_id, offset)
                    for unit_id, offset in normalized_boundaries
                    if unit_id in target_ids
                ]
                return BoundaryResult(
                    break_after=frozenset(
                        unit_id
                        for unit_id, offset in target_boundaries
                        if offset == len(all_units[unit_id].text)
                    ),
                    split_after=frozenset(
                        (unit_id, offset)
                        for unit_id, offset in target_boundaries
                        if offset < len(all_units[unit_id].text)
                    ),
                    request_id=response.headers.get("x-request-id") or envelope.get("id"),
                    usage=envelope.get("usage") or {},
                )
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if format_attempt == 0:
                    continue
        raise OrganizeError(f"无法解析文本模型的断句结果：{last_error}") from last_error

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
        raise OrganizeError(f"文本模型请求重试后仍然失败：{message}") from last_error

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


class SubtitleOrganizer:
    def __init__(
        self,
        boundary_engine: BoundaryEngine | None = None,
        api_key: str | None = None,
        status: StatusCallback | None = None,
    ) -> None:
        self.boundary_engine = boundary_engine
        self.api_key = api_key
        self.status = status or (lambda _message: None)

    def run(
        self,
        source_path: Path,
        config: TextLlmConfig,
        workspace_root: Path = Path("workspace"),
        output_dir: Path | None = None,
        overwrite: bool = False,
        reference_track: YouTubeCaptionTrack | None = None,
    ) -> OrganizeResult:
        source_path = source_path.expanduser().resolve()
        if not source_path.is_file():
            raise OrganizeError(f"字幕 Markdown 不存在：{source_path}")
        source_bytes = source_path.read_bytes()
        try:
            markdown = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OrganizeError("字幕 Markdown 必须使用 UTF-8 编码") from exc

        fragments = extract_timed_subtitle_fragments(markdown)
        units = build_timed_subtitle_units(fragments, reference_track)
        if not units:
            raise OrganizeError("清理后没有可供断句的字幕文字")
        self.status(f"清理得到 {len(fragments)} 个字幕片段、{len(units)} 个断句单元")

        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        config_fingerprint = self._config_fingerprint(source_sha256, config)
        job_stem = sanitize_filename(source_path.stem, maximum_length=100)
        job_dir = (
            workspace_root.expanduser().resolve()
            / "organize"
            / f"{job_stem}-{config_fingerprint[:12]}"
        )
        checkpoint_path = job_dir / "organize_checkpoint.json"
        checkpoint = self._load_checkpoint(
            checkpoint_path, source_sha256, config_fingerprint, len(units), config
        )

        chunks = [
            units[index : index + config.chunk_size]
            for index in range(0, len(units), config.chunk_size)
        ]
        engine = self.boundary_engine or OpenAICompatibleBoundaryEngine(
            config=config, api_key=self.api_key or ""
        )
        all_breaks: set[str] = set()
        all_splits: set[tuple[str, int]] = set()
        reused_chunks = 0
        new_api_calls = 0
        for chunk_index, target in enumerate(chunks, start=1):
            start = (chunk_index - 1) * config.chunk_size
            before = units[max(0, start - config.context_size) : start]
            end = start + len(target)
            after = units[end : end + config.context_size]
            chunk_key = f"{chunk_index:06d}"
            target_signature = self._target_signature(target)
            saved = checkpoint["chunks"].get(chunk_key)
            if (
                isinstance(saved, dict)
                and saved.get("target_sha256") == target_signature
                and isinstance(saved.get("break_after"), list)
                and isinstance(saved.get("split_after"), list)
            ):
                target_ids = {unit.unit_id for unit in target}
                saved_breaks = set(saved["break_after"])
                saved_splits = {
                    (str(item["unit_id"]), int(item["after_char"]))
                    for item in saved["split_after"]
                    if isinstance(item, dict)
                    and isinstance(item.get("unit_id"), str)
                    and isinstance(item.get("after_char"), int)
                }
                if saved_breaks.issubset(target_ids) and all(
                    unit_id in target_ids for unit_id, _offset in saved_splits
                ):
                    all_breaks.update(saved_breaks)
                    all_splits.update(saved_splits)
                    reused_chunks += 1
                    self.status(f"断句进度 {chunk_index}/{len(chunks)}（复用检查点）")
                    continue

            result = engine.decide_boundaries(before, target, after)
            target_ids = {unit.unit_id for unit in target}
            if not result.break_after.issubset(target_ids):
                raise OrganizeError("文本模型返回了当前分块以外的断句编号")
            all_breaks.update(result.break_after)
            all_splits.update(result.split_after)
            checkpoint["chunks"][chunk_key] = {
                "target_sha256": target_signature,
                "target_ids": [unit.unit_id for unit in target],
                "break_after": sorted(result.break_after),
                "split_after": [
                    {"unit_id": unit_id, "after_char": offset}
                    for unit_id, offset in sorted(result.split_after)
                ],
                "request_id": result.request_id,
                "usage": result.usage or {},
            }
            write_json(checkpoint_path, checkpoint)
            new_api_calls += 1
            self.status(f"断句进度 {chunk_index}/{len(chunks)}")

        all_breaks.add(units[-1].unit_id)
        reference_boundaries = self._reference_boundaries(units, reference_track)
        for unit_id, offset in reference_boundaries:
            unit = next(item for item in units if item.unit_id == unit_id)
            if offset >= len(unit.text):
                all_breaks.add(unit_id)
            else:
                all_splits.add((unit_id, offset))
        if reference_boundaries:
            self.status(f"采用 {len(reference_boundaries)} 个 YouTube 读音参考句界")
        sentences = self._build_sentences(units, all_breaks, all_splits)
        self._validate_conservation(units, sentences)
        target_dir = output_dir.expanduser().resolve() if output_dir else source_path.parent
        output_path = self._output_path(source_path, target_dir, overwrite)
        write_text(output_path, "\n".join(sentence.text for sentence in sentences) + "\n")
        sentences_path = job_dir / "organized_segments.json"
        write_json(
            sentences_path,
            OrganizedSubtitleDocument(
                source_markdown=source_path.name,
                sentences=sentences,
            ),
        )
        self.status(f"完成：{len(sentences)} 句话")
        return OrganizeResult(
            source_path=source_path,
            markdown_path=output_path,
            checkpoint_path=checkpoint_path,
            source_fragment_count=len(fragments),
            sentence_count=len(sentences),
            api_call_count=new_api_calls,
            reused_chunk_count=reused_chunks,
            sentences_path=sentences_path,
        )

    @staticmethod
    def _load_checkpoint(
        path: Path,
        source_sha256: str,
        config_fingerprint: str,
        unit_count: int,
        config: TextLlmConfig,
    ) -> dict[str, Any]:
        expected = {
            "schema_version": 1,
            "prompt_version": _PROMPT_VERSION,
            "source_sha256": source_sha256,
            "config_fingerprint": config_fingerprint,
            "model": config.model,
            "unit_count": unit_count,
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

    @staticmethod
    def _config_fingerprint(source_sha256: str, config: TextLlmConfig) -> str:
        stable = {
            "source_sha256": source_sha256,
            "prompt_version": _PROMPT_VERSION,
            "base_url": config.base_url,
            "model": config.model,
            "chunk_size": config.chunk_size,
            "context_size": config.context_size,
        }
        encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _target_signature(units: list[SubtitleUnit]) -> str:
        encoded = json.dumps(
            [unit.as_payload() for unit in units], ensure_ascii=False, sort_keys=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _build_sentences(
        units: list[SubtitleUnit],
        breaks: set[str],
        split_after: set[tuple[str, int]] | None = None,
    ) -> list[OrganizedSentence]:
        sentences: list[OrganizedSentence] = []
        pieces: list[SubtitleUnit] = []
        split_after = split_after or set()
        for unit in units:
            offsets = sorted(
                offset
                for unit_id, offset in split_after
                if unit_id == unit.unit_id and 0 < offset < len(unit.text)
            )
            if unit.unit_id in breaks:
                offsets.append(len(unit.text))
            start_offset = 0
            for end_offset in [*offsets, len(unit.text)]:
                if end_offset <= start_offset:
                    continue
                duration = max(0, unit.end_ms - unit.start_ms)
                length = max(1, len(unit.text))
                pieces.append(
                    SubtitleUnit(
                        unit_id=unit.unit_id,
                        text=unit.text[start_offset:end_offset],
                        start_ms=unit.start_ms + round(duration * start_offset / length),
                        end_ms=unit.start_ms + round(duration * end_offset / length),
                        source_segment_ids=unit.source_segment_ids,
                        youtube_reading_reference=unit.youtube_reading_reference,
                    )
                )
                if end_offset in offsets:
                    sentence = SubtitleOrganizer._sentence_from_units(
                        pieces, len(sentences) + 1
                    )
                    if sentence is not None:
                        sentences.append(sentence)
                    pieces = []
                start_offset = end_offset
        if pieces:
            sentence = SubtitleOrganizer._sentence_from_units(pieces, len(sentences) + 1)
            if sentence is not None:
                sentences.append(sentence)
        return sentences

    @staticmethod
    def _sentence_from_units(
        units: list[SubtitleUnit], sentence_index: int
    ) -> OrganizedSentence | None:
        text = _join_pieces([unit.text for unit in units])
        if not text:
            return None
        source_ids = list(
            dict.fromkeys(
                source_id for unit in units for source_id in unit.source_segment_ids
            )
        )
        return OrganizedSentence(
            sentence_id=f"s{sentence_index:06d}",
            text=text,
            start_ms=min(unit.start_ms for unit in units),
            end_ms=max(unit.end_ms for unit in units),
            source_unit_ids=list(dict.fromkeys(unit.unit_id for unit in units)),
            source_segment_ids=source_ids,
        )

    @staticmethod
    def _validate_conservation(
        units: list[SubtitleUnit], sentences: list[OrganizedSentence]
    ) -> None:
        source = _conservation_key(_join_pieces([unit.text for unit in units]))
        output = _conservation_key("\n".join(sentence.text for sentence in sentences))
        if source != output:
            raise OrganizeError("整理结果未通过字符守恒校验，已拒绝写出")

    @staticmethod
    def _reference_boundaries(
        units: list[SubtitleUnit], reference_track: YouTubeCaptionTrack | None
    ) -> set[tuple[str, int]]:
        if reference_track is None or not units:
            return set()
        source_characters: list[str] = []
        source_positions: list[tuple[str, int]] = []
        for unit in units:
            for offset, character in enumerate(unit.text, 1):
                for comparable in _comparable_characters(character):
                    source_characters.append(comparable)
                    source_positions.append((unit.unit_id, offset))

        caption_text = "".join(cue.text for cue in reference_track.cues)
        caption_text = re.sub(r"\[[^\]]*]|【[^】]*】", "", caption_text)
        raw_sentences = [
            match.group(0)
            for match in re.finditer(r".+?(?:[。！？!?]+|$)", caption_text)
            if match.group(0).strip()
        ]
        reference_characters: list[str] = []
        ranges: list[tuple[int, int]] = []
        for sentence in raw_sentences:
            start = len(reference_characters)
            for character in sentence:
                reference_characters.extend(_comparable_characters(character))
            end = len(reference_characters)
            if end > start:
                ranges.append((start, end))
        if not source_characters or not reference_characters:
            return set()

        matcher = SequenceMatcher(
            None,
            "".join(reference_characters),
            "".join(source_characters),
            autojunk=False,
        )
        reference_to_source: dict[int, int] = {}
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                reference_to_source[block.a + offset] = block.b + offset

        boundaries: set[tuple[str, int]] = set()
        for start, end in ranges:
            length = end - start
            if length < 3:
                continue
            mapped = [
                reference_to_source[index]
                for index in range(start, end)
                if index in reference_to_source
            ]
            if len(mapped) / length < 0.72 or not mapped:
                continue
            if mapped[-1] - mapped[0] + 1 > max(length + 4, round(length * 1.8)):
                continue
            last_reference_index = end - 1
            source_index = reference_to_source.get(last_reference_index)
            if source_index is None:
                continue
            boundaries.add(source_positions[source_index])
        return boundaries

    @staticmethod
    def _output_path(source_path: Path, output_dir: Path, overwrite: bool) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize_filename(source_path.stem)
        path = output_dir / f"{stem}（整理版）.md"
        if overwrite or not path.exists():
            return path
        counter = 2
        while True:
            candidate = output_dir / f"{stem}（整理版-{counter}）.md"
            if not candidate.exists():
                return candidate
            counter += 1


def write_fallback_player_document(source_path: Path, output_path: Path) -> Path:
    """Build a clickable screen-by-screen transcript when semantic grouping fails."""
    markdown = source_path.read_text(encoding="utf-8")
    units = build_timed_subtitle_units(extract_timed_subtitle_fragments(markdown))
    sentences = [
        sentence
        for index, unit in enumerate(units, start=1)
        if (sentence := SubtitleOrganizer._sentence_from_units([unit], index)) is not None
    ]
    write_json(
        output_path,
        OrganizedSubtitleDocument(
            source_markdown=source_path.name,
            sentences=sentences,
        ),
    )
    return output_path


def _join_pieces(pieces: list[str]) -> str:
    result = ""
    for piece in pieces:
        clean = re.sub(r"\s+", " ", piece).strip()
        if not clean:
            continue
        if result and _needs_space(result, clean):
            result += " "
        result += clean
    return result


def _timestamp_to_ms(value: str) -> int:
    clock, milliseconds = value.rsplit(".", maxsplit=1)
    parts = [int(part) for part in clock.split(":")]
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise OrganizeError(f"无效字幕时间戳：{value}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + int(milliseconds)


def _needs_space(left: str, right: str) -> bool:
    left_char = left[-1]
    right_char = right[0]
    left_ascii_word = left_char.isascii() and (left_char.isalnum() or left_char in ",.;:!?)]}'\"")
    right_ascii_word = right_char.isascii() and (right_char.isalnum() or right_char in "([{'\"")
    return left_ascii_word and right_ascii_word


def _conservation_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", normalized)


def _comparable_characters(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return [
        character
        for character in normalized
        if not character.isspace() and character not in "、。，．,.!?！？:：;；'\"“”‘’"
    ]
