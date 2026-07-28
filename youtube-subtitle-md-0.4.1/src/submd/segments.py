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
    return ratio(normalize_text(left), normalize_text(right)) >= threshold


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
        key=lambda text: (grouped_counts[text], grouped_scores[text], len(text)),
        reverse=True,
    )
    representative = ranked[0]
    alternatives = [text for text in ranked[1:4] if text != representative]
    return representative, sum(confidences) / len(confidences), alternatives


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
                needs_review=confidence < review_confidence or len(active.observations) < 2,
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
                previous.confidence < review_confidence or previous.observation_count < 2
            )
            previous.alternatives = list(
                dict.fromkeys([*previous.alternatives, *segment.alternatives])
            )[:3]
        else:
            merged.append(segment)
    return merged
