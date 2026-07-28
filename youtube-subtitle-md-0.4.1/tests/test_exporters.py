from pathlib import Path

from submd.exporters import export_markdown, format_timestamp, sanitize_filename
from submd.models import (
    CloudOcrConfig,
    ExtractionConfig,
    SubtitleDocument,
    SubtitleSegment,
    VideoMetadata,
)


def test_sanitize_filename() -> None:
    assert sanitize_filename(' A/B: "test" ') == "A_B_ _test_"
    assert sanitize_filename("...") == "untitled"


def test_timestamp() -> None:
    assert format_timestamp(65_432) == "01:05.432"
    assert format_timestamp(3_665_432) == "01:01:05.432"


def test_markdown_export(tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="abc",
        original_title="示例 / video",
        duration_ms=3000,
        webpage_url="https://youtu.be/abc",
    )
    config = ExtractionConfig(
        source_url=metadata.webpage_url,
        output_dir=tmp_path,
        ocr=CloudOcrConfig(base_url="https://vendor.example/v1", model="vision-ocr"),
    )
    document = SubtitleDocument(
        video=metadata,
        config=config,
        segments=[
            SubtitleSegment(
                start_ms=1000,
                end_ms=2500,
                text="第一行\nsecond",
                confidence=0.9,
                observation_count=2,
            )
        ],
    )
    path = export_markdown(document, tmp_path, overwrite=False)
    content = path.read_text(encoding="utf-8")
    assert path.name == "示例 _ video.md"
    assert "第一行<br>second" in content
    assert "00:01.000–00:02.500" in content
    assert 'ocr_model: "vision-ocr"' in content
