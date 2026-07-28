from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from submd.models import OrganizedSentence, OrganizedSubtitleDocument, SubtitleSegment

_PERIODS = "。．."
_BOUNDARY_SPLIT = re.compile(r"([。．.]+|\n+)")


@dataclass(frozen=True)
class _CharacterTiming:
    start_ms: int
    end_ms: int
    source_unit_ids: tuple[str, ...]
    source_segment_ids: tuple[str, ...]


def preview_manual_sentences(edited_text: str) -> list[str]:
    pieces: list[str] = []
    pending = ""
    for part in _BOUNDARY_SPLIT.split(edited_text.replace("\r\n", "\n")):
        if not part:
            continue
        if re.fullmatch(r"[。．.]+", part):
            pending += part
            if pending:
                pieces.append(pending)
                pending = ""
            continue
        if "\n" in part:
            if pending:
                pieces.append(pending)
                pending = ""
            continue
        clean = re.sub(r"\s+", "", part)
        if clean:
            pending += clean
        if pending and pending[-1] in _PERIODS:
            pieces.append(pending)
            pending = ""
    if pending:
        pieces.append(pending)
    return pieces


def apply_manual_resegmentation(
    document: OrganizedSubtitleDocument,
    sentence_ids: list[str],
    edited_text: str,
    source_segments: list[SubtitleSegment] | None = None,
) -> OrganizedSubtitleDocument:
    if not sentence_ids:
        raise ValueError("请先选择要修改断句的句子")
    if len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError("选中的句子存在重复")
    indexes = [
        index
        for index, sentence in enumerate(document.sentences)
        if sentence.sentence_id in set(sentence_ids)
    ]
    if len(indexes) != len(sentence_ids):
        raise ValueError("选中的句子不属于当前视频")
    if indexes != list(range(indexes[0], indexes[-1] + 1)):
        raise ValueError("只能连续选择前后相邻的句子")
    selected = document.sentences[indexes[0] : indexes[-1] + 1]
    source_text = "".join(sentence.text for sentence in selected)
    if _content_key(source_text) != _content_key(edited_text):
        raise ValueError("这里只能增删或移动句号，不能修改字幕文字")
    new_texts = preview_manual_sentences(edited_text)
    if not new_texts:
        raise ValueError("修改后没有可用的句子")

    timings = _source_character_timings(selected, source_segments or [])
    if timings is None:
        timings = _character_timings(selected)
    cursor = 0
    replacements: list[OrganizedSentence] = []
    for text in new_texts:
        length = len(_content_key(text))
        if length <= 0 or cursor + length > len(timings):
            raise ValueError("句号位置无法映射到原始字幕时间")
        covered = timings[cursor : cursor + length]
        replacements.append(
            OrganizedSentence(
                sentence_id="pending",
                text=text,
                start_ms=covered[0].start_ms,
                end_ms=covered[-1].end_ms,
                source_unit_ids=list(
                    dict.fromkeys(
                        source_id for item in covered for source_id in item.source_unit_ids
                    )
                ),
                source_segment_ids=list(
                    dict.fromkeys(
                        source_id for item in covered for source_id in item.source_segment_ids
                    )
                ),
            )
        )
        cursor += length
    if cursor != len(timings):
        raise ValueError("修改后的句子未覆盖全部原始字幕")

    combined = [
        *document.sentences[: indexes[0]],
        *replacements,
        *document.sentences[indexes[-1] + 1 :],
    ]
    reindexed = [
        sentence.model_copy(update={"sentence_id": f"s{index:06d}"})
        for index, sentence in enumerate(combined, 1)
    ]
    return document.model_copy(update={"sentences": reindexed})


def _character_timings(sentences: list[OrganizedSentence]) -> list[_CharacterTiming]:
    result: list[_CharacterTiming] = []
    for sentence in sentences:
        content = _content_key(sentence.text)
        if not content:
            continue
        duration = max(1, sentence.end_ms - sentence.start_ms)
        for index in range(len(content)):
            result.append(
                _CharacterTiming(
                    start_ms=sentence.start_ms + round(duration * index / len(content)),
                    end_ms=sentence.start_ms + round(duration * (index + 1) / len(content)),
                    source_unit_ids=tuple(sentence.source_unit_ids),
                    source_segment_ids=tuple(sentence.source_segment_ids),
                )
            )
    return result


def _source_character_timings(
    sentences: list[OrganizedSentence], source_segments: list[SubtitleSegment]
) -> list[_CharacterTiming] | None:
    if not source_segments:
        return None
    selected_ids = list(
        dict.fromkeys(
            segment_id for sentence in sentences for segment_id in sentence.source_segment_ids
        )
    )
    source_characters = ""
    source_timings: list[_CharacterTiming] = []
    for segment_id in selected_ids:
        match = re.fullmatch(r"seg(\d{6})", segment_id)
        if not match:
            return None
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(source_segments):
            return None
        segment = source_segments[index]
        content = _content_key(segment.text)
        if not content:
            continue
        duration = max(1, segment.end_ms - segment.start_ms)
        for offset, character in enumerate(content):
            source_characters += character
            source_timings.append(
                _CharacterTiming(
                    start_ms=segment.start_ms + round(duration * offset / len(content)),
                    end_ms=segment.start_ms + round(duration * (offset + 1) / len(content)),
                    source_unit_ids=tuple(
                        dict.fromkeys(
                            unit_id
                            for sentence in sentences
                            if segment_id in sentence.source_segment_ids
                            for unit_id in sentence.source_unit_ids
                        )
                    ),
                    source_segment_ids=(segment_id,),
                )
            )
    selected_text = _content_key("".join(sentence.text for sentence in sentences))
    start = source_characters.find(selected_text)
    if start < 0:
        return None
    end = start + len(selected_text)
    return source_timings[start:end]


def _content_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(rf"[\s{re.escape(_PERIODS)}]+", "", normalized)
