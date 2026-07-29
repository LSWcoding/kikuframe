from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class Roi(BaseModel):
    """Normalized rectangle measured against the full video frame."""

    x: float = Field(default=0.0, ge=0.0, le=1.0)
    y: float = Field(default=0.65, ge=0.0, le=1.0)
    width: float = Field(default=1.0, gt=0.0, le=1.0)
    height: float = Field(default=0.35, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Roi:
        if self.x + self.width > 1.000001:
            raise ValueError("ROI x + width must be at most 1")
        if self.y + self.height > 1.000001:
            raise ValueError("ROI y + height must be at most 1")
        return self

    @classmethod
    def parse(cls, value: str) -> Roi:
        try:
            parts = [float(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError("ROI must contain four numbers: x,y,width,height") from exc
        if len(parts) != 4:
            raise ValueError("ROI must contain four numbers: x,y,width,height")
        return cls(x=parts[0], y=parts[1], width=parts[2], height=parts[3])


class CloudOcrConfig(BaseModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    batch_size: int = Field(default=4, ge=1, le=16)
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    image_max_side: int = Field(default=1600, ge=256, le=4096)
    jpeg_quality: int = Field(default=88, ge=50, le=100)
    json_mode: bool = True

    @model_validator(mode="after")
    def normalize_url(self) -> CloudOcrConfig:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("OCR API base URL must start with http:// or https://")
        return self


class TextLlmConfig(BaseModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    chunk_size: int = Field(default=100, ge=10, le=300)
    context_size: int = Field(default=10, ge=0, le=50)
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    json_mode: bool = True

    @model_validator(mode="after")
    def normalize_url(self) -> TextLlmConfig:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Text LLM API base URL must start with http:// or https://")
        return self


class LanguageLearningConfig(BaseModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    json_mode: bool = True

    @model_validator(mode="after")
    def normalize_url(self) -> LanguageLearningConfig:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Learning API base URL must start with http:// or https://")
        return self


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_url: str
    cookies_from_browser: str | None = None
    workspace_root: Path = Path("workspace")
    output_dir: Path = Path("output")
    roi: Roi = Field(default_factory=Roi)
    language: str = "auto"
    ocr: CloudOcrConfig
    sample_fps: float = Field(default=3.0, gt=0.0, le=30.0)
    change_threshold: float = Field(default=0.012, ge=0.0, le=1.0)
    max_ocr_interval: float = Field(default=2.0, gt=0.0, le=60.0)
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=82.0, ge=0.0, le=100.0)
    max_height: int = Field(default=720, ge=144, le=4320)
    keep_cache: bool = False
    overwrite: bool = False


class VideoMetadata(BaseModel):
    video_id: str
    original_title: str
    uploader: str | None = None
    duration_ms: int
    webpage_url: str
    width: int | None = None
    height: int | None = None


class MediaInfo(BaseModel):
    duration_ms: int
    width: int
    height: int
    fps: float | None = None


class FrameRef(BaseModel):
    index: int
    timestamp_ms: int
    path: Path
    diff_score: float = 1.0


class CloudOcrFrameResult(BaseModel):
    frame_id: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class CloudOcrBatchResult(BaseModel):
    frames: list[CloudOcrFrameResult]
    request_id: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class OcrObservation(BaseModel):
    timestamp_ms: int
    frame_index: int
    frame_file: str
    diff_score: float
    backend: Literal["cloud_vlm"] = "cloud_vlm"
    model: str
    batch_index: int
    request_id: str | None = None
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class SubtitleSegment(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    source: Literal["burned_ocr", "burned_ocr_corrected"] = "burned_ocr"
    confidence: float = Field(ge=0.0, le=1.0)
    observation_count: int = 1
    alternatives: list[str] = Field(default_factory=list)
    needs_review: bool = False
    original_text: str | None = None
    youtube_reference: list[str] = Field(default_factory=list)
    correction_reason: str | None = None


class SubtitleDocument(BaseModel):
    video: VideoMetadata
    config: ExtractionConfig
    segments: list[SubtitleSegment]


class ExtractionResult(BaseModel):
    metadata_path: Path
    config_path: Path
    observations_path: Path
    api_calls_path: Path
    segments_path: Path
    markdown_path: Path
    segment_count: int
    observation_count: int
    audio_path: Path | None = None


class YouTubeCaptionCue(BaseModel):
    cue_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timing(self) -> YouTubeCaptionCue:
        if self.end_ms < self.start_ms:
            raise ValueError("YouTube caption end_ms must be at least start_ms")
        return self


class YouTubeCaptionTrack(BaseModel):
    schema_version: int = 1
    video_id: str
    language: str
    source: Literal["manual", "automatic"]
    name: str | None = None
    cues: list[YouTubeCaptionCue]


class CaptionCorrection(BaseModel):
    segment_id: str
    corrected_text: str = Field(min_length=1)
    reason: str = ""


class OrganizedSentence(BaseModel):
    sentence_id: str
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    source_unit_ids: list[str] = Field(default_factory=list)
    source_segment_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timing(self) -> OrganizedSentence:
        if self.end_ms < self.start_ms:
            raise ValueError("organized sentence end_ms must be at least start_ms")
        return self


class OrganizedSubtitleDocument(BaseModel):
    schema_version: int = 1
    source_markdown: str
    sentences: list[OrganizedSentence]


class VocabularyAnalysisItem(BaseModel):
    kind: Literal["word", "collocation"]
    expression: str = Field(min_length=1)
    lemma: str = Field(min_length=1)
    reading: str = ""
    meaning: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_reading_for_kanji(self, info: ValidationInfo) -> VocabularyAnalysisItem:
        allow_missing = bool(
            isinstance(info.context, dict) and info.context.get("allow_missing_reading")
        )
        if (
            re.search(r"[\u3400-\u9fff]", f"{self.expression}{self.lemma}")
            and not self.reading.strip()
            and not allow_missing
        ):
            raise ValueError("Japanese vocabulary containing kanji must include a hiragana reading")
        return self


class GrammarAnalysisItem(BaseModel):
    pattern: str = Field(min_length=1)
    lemma: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class SentenceLearningAnalysis(BaseModel):
    schema_version: int = 1
    prompt_version: str
    sentence: str = Field(min_length=1)
    model: str = Field(min_length=1)
    translation: str = Field(min_length=1)
    vocabulary: list[VocabularyAnalysisItem]
    grammar: list[GrammarAnalysisItem]


class OrganizeResult(BaseModel):
    source_path: Path
    markdown_path: Path
    checkpoint_path: Path
    source_fragment_count: int
    sentence_count: int
    api_call_count: int
    reused_chunk_count: int
    sentences_path: Path | None = None
