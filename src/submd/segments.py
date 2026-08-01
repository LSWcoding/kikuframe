from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.fuzz import ratio

from submd.models import OcrObservation, SubtitleSegment
from submd.text import normalize_text


@dataclass
class _ActiveSegment:
    start_ms: int
    observations: list[OcrObservation] = field(default_factory=list)


def _similar(left: str, right: str, threshold: float) -> bool:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if ratio(normalized_left, normalized_right) >= threshold:
        return True
    compact_left = normalized_left.replace("\n", "").replace(" ", "")
    compact_right = normalized_right.replace("\n", "").replace(" ", "")
    shorter, longer = sorted((compact_left, compact_right), key=len)
    return (
        len(shorter) >= 4
        and shorter in longer
        and len(shorter) / max(1, len(longer)) >= 0.30
    )


def _representative(observations: list[OcrObservation]) -> tuple[str, float, list[str]]:
    grouped_scores: dict[str, float] = defaultdict(float)
    grouped_counts: dict[str, int] = defaultdict(int)
    confidences: list[float] = []
    for observation in observations:
        text = normalize_text(observation.text)
        grouped_scores[text] += max(observation.confidence, 0.01)
        grouped_counts[text] += 1
        confidences.append(observation.confidence)
    ranked = sorted(
        grouped_scores,
        key=lambda text: (
            sum(
                grouped_counts[other]
                for other in grouped_scores
                if _similar(text, other, 72.0)
            ),
            len(text.replace("\n", "").replace(" ", "")),
            grouped_counts[text],
            grouped_scores[text],
        ),
        reverse=True,
    )
    representative = ranked[0]
    alternatives = [text for text in ranked[1:4] if text != representative]
    return representative, sum(confidences) / len(confidences), alternatives


def _overlap_join(left: str, right: str) -> str | None:
    clean_left = normalize_text(left).replace("\n", "")
    clean_right = normalize_text(right).replace("\n", "")
    maximum = min(len(clean_left), len(clean_right))
    for size in range(maximum, 3, -1):
        if (
            clean_left[-size:] == clean_right[:size]
            and size / max(1, min(len(clean_left), len(clean_right))) >= 0.30
        ):
            return clean_left + clean_right[size:]
    return None


def build_segments(
    observations: list[OcrObservation],
    duration_ms: int,
    similarity_threshold: float,
    review_confidence: float,
    sample_interval_ms: int,
) -> list[SubtitleSegment]:
    ordered = sorted(observations, key=lambda item: item.timestamp_ms)
    segments: list[SubtitleSegment] = []
    active: _ActiveSegment | None = None

    def finish(end_ms: int) -> None:
        nonlocal active
        if active is None or not active.observations:
            active = None
            return
        text, confidence, alternatives = _representative(active.observations)
        safe_end = min(duration_ms, max(active.start_ms + sample_interval_ms, end_ms))
        segments.append(
            SubtitleSegment(
                start_ms=active.start_ms,
                end_ms=safe_end,
                text=text,
                confidence=confidence,
                observation_count=len(active.observations),
                alternatives=alternatives,
                needs_review=confidence < review_confidence,
            )
        )
        active = None

    for observation in ordered:
        text = normalize_text(observation.text)
        if not text:
            finish(observation.timestamp_ms)
            continue
        if active is None:
            active = _ActiveSegment(start_ms=observation.timestamp_ms, observations=[observation])
            continue
        representative, _, _ = _representative(active.observations)
        if _similar(representative, text, similarity_threshold):
            active.observations.append(observation)
        else:
            finish(observation.timestamp_ms)
            active = _ActiveSegment(start_ms=observation.timestamp_ms, observations=[observation])
    finish(duration_ms)

    # A final pass joins directly adjacent duplicate segments produced by an OCR dropout.
    merged: list[SubtitleSegment] = []
    for segment in segments:
        if (
            merged
            and segment.start_ms - merged[-1].end_ms <= sample_interval_ms
            and _similar(merged[-1].text, segment.text, similarity_threshold)
        ):
            previous = merged[-1]
            count = previous.observation_count + segment.observation_count
            previous.confidence = (
                previous.confidence * previous.observation_count
                + segment.confidence * segment.observation_count
            ) / count
            previous.end_ms = segment.end_ms
            previous.observation_count = count
            previous.needs_review = (
                previous.confidence < review_confidence
            )
            previous.alternatives = list(
                dict.fromkeys([*previous.alternatives, *segment.alternatives])
            )[:3]
            if len(normalize_text(segment.text)) > len(normalize_text(previous.text)):
                previous.text = segment.text
        else:
            joined = (
                _overlap_join(merged[-1].text, segment.text)
                if merged
                and segment.start_ms - merged[-1].end_ms <= sample_interval_ms
                else None
            )
            if joined is None:
                merged.append(segment)
                continue
            previous = merged[-1]
            count = previous.observation_count + segment.observation_count
            previous.text = joined
            previous.end_ms = segment.end_ms
            previous.confidence = (
                previous.confidence * previous.observation_count
                + segment.confidence * segment.observation_count
            ) / count
            previous.observation_count = count
            previous.needs_review = previous.confidence < review_confidence
            previous.alternatives = list(
                dict.fromkeys(
                    [
                        *previous.alternatives,
                        segment.text,
                        *segment.alternatives,
                    ]
                )
            )[:3]
    return merged
