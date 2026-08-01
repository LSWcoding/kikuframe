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
    new_texts = preview_manual_sentences(edited_text)
    if not new_texts:
        raise ValueError("修改后没有可用的句子")

    if _content_key(source_text) == _content_key(edited_text):
        replacements = _replacements_from_character_timings(
            selected, new_texts, source_segments or []
        )
    else:
        replacements = _proportional_replacements(selected, new_texts)

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


def delete_organized_sentence(
    document: OrganizedSubtitleDocument, sentence_id: str
) -> OrganizedSubtitleDocument:
    """Delete one player sentence without changing the learning library."""
    if not sentence_id:
        raise ValueError("句子 ID 为空")
    if not any(sentence.sentence_id == sentence_id for sentence in document.sentences):
        raise ValueError("该句子不属于当前视频")
    remaining = [
        sentence
        for sentence in document.sentences
        if sentence.sentence_id != sentence_id
    ]
    reindexed = [
        sentence.model_copy(update={"sentence_id": f"s{index:06d}"})
        for index, sentence in enumerate(remaining, 1)
    ]
    return document.model_copy(update={"sentences": reindexed})


def _replacements_from_character_timings(
    selected: list[OrganizedSentence],
    new_texts: list[str],
    source_segments: list[SubtitleSegment],
) -> list[OrganizedSentence]:
    timings = _source_character_timings(selected, source_segments)
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
    return replacements


def _proportional_replacements(
    selected: list[OrganizedSentence], new_texts: list[str]
) -> list[OrganizedSentence]:
    """Keep the selected audio span when the user also corrects subtitle characters."""
    start_ms = selected[0].start_ms
    end_ms = max(start_ms, selected[-1].end_ms)
    duration = max(1, end_ms - start_ms)
    weights = [max(1, len(_content_key(text))) for text in new_texts]
    total_weight = sum(weights)
    source_unit_ids = list(
        dict.fromkeys(source_id for sentence in selected for source_id in sentence.source_unit_ids)
    )
    source_segment_ids = list(
        dict.fromkeys(
            source_id for sentence in selected for source_id in sentence.source_segment_ids
        )
    )
    replacements: list[OrganizedSentence] = []
    consumed = 0
    for index, (text, weight) in enumerate(zip(new_texts, weights, strict=True)):
        item_start = start_ms + round(duration * consumed / total_weight)
        consumed += weight
        item_end = (
            end_ms
            if index == len(new_texts) - 1
            else start_ms + round(duration * consumed / total_weight)
        )
        replacements.append(
            OrganizedSentence(
                sentence_id="pending",
                text=text,
                start_ms=item_start,
                end_ms=max(item_start, item_end),
                source_unit_ids=source_unit_ids,
                source_segment_ids=source_segment_ids,
            )
        )
    return replacements


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
