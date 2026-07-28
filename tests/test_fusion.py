from pathlib import Path

from submd.fusion import SubtitleFusion
from submd.models import (
    CaptionCorrection,
    CloudOcrConfig,
    ExtractionConfig,
    SubtitleDocument,
    SubtitleSegment,
    TextLlmConfig,
    VideoMetadata,
    YouTubeCaptionCue,
    YouTubeCaptionTrack,
)


class ReferenceAwareEngine:
    def correct(self, before_context, targets, after_context):
        del before_context, after_context
        assert targets[0]["ocr_text"] == "孤賃で働いています"
        assert targets[0]["ocr_alternatives"] == ["個人で働いています"]
        assert targets[0]["youtube_reading_reference"][0]["text"] == "個人で働いています"
        return (
            [
                CaptionCorrection(
                    segment_id="seg000001",
                    corrected_text="個人で働いています",
                    reason="读音参考与 OCR 候选一致",
                )
            ],
            "fusion-request",
            {"total_tokens": 42},
        )


def test_fuses_visual_ocr_with_youtube_reading_reference(tmp_path: Path) -> None:
    document = SubtitleDocument(
        video=VideoMetadata(
            video_id="sample",
            original_title="示例",
            duration_ms=3000,
            webpage_url="https://youtu.be/sample",
        ),
        config=ExtractionConfig(
            source_url="https://youtu.be/sample",
            ocr=CloudOcrConfig(base_url="https://vendor.example/v1", model="vision"),
        ),
        segments=[
            SubtitleSegment(
                start_ms=1000,
                end_ms=2500,
                text="孤賃で働いています",
                confidence=0.62,
                alternatives=["個人で働いています"],
            )
        ],
    )
    track = YouTubeCaptionTrack(
        video_id="sample",
        language="ja",
        source="automatic",
        cues=[
            YouTubeCaptionCue(
                cue_id="yt000001",
                start_ms=900,
                end_ms=2600,
                text="個人で働いています",
            )
        ],
    )
    output = tmp_path / "corrected_segments.json"
    corrected = SubtitleFusion(engine=ReferenceAwareEngine()).run(
        document,
        track,
        TextLlmConfig(base_url="https://vendor.example/v1", model="text"),
        output,
        tmp_path / "fusion_checkpoint.json",
    )
    segment = corrected.segments[0]
    assert segment.text == "個人で働いています"
    assert segment.original_text == "孤賃で働いています"
    assert segment.youtube_reference == ["個人で働いています"]
    assert segment.source == "burned_ocr_corrected"
    assert output.is_file()


def test_fusion_safety_rejects_large_deletions_and_insertions() -> None:
    assert SubtitleFusion._safe_correction("フリーダー", "フリーター") is True
    assert SubtitleFusion._safe_correction("ですさっき病院に行ったら", "です") is False
    assert SubtitleFusion._safe_correction(
        "海外に行くって言ったら寂",
        "海外に行くって言ったら褒めてくれるんだろうね",
    ) is False
