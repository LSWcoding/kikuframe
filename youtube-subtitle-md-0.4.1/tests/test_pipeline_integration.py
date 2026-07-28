from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from submd.models import (
    CloudOcrBatchResult,
    CloudOcrConfig,
    CloudOcrFrameResult,
    ExtractionConfig,
    FrameRef,
    VideoMetadata,
)
from submd.pipeline import BurnedSubtitlePipeline


class FakeDownloader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.metadata = VideoMetadata(
            video_id="synthetic",
            original_title="Cloud OCR pipeline test",
            uploader="tests",
            duration_ms=2000,
            webpage_url="https://youtu.be/synthetic",
            width=320,
            height=180,
        )

    def inspect(self, _url: str, cookies_from_browser: str | None = None) -> VideoMetadata:
        del cookies_from_browser
        return self.metadata.model_copy(deep=True)

    def download(
        self,
        _url: str,
        _target_dir: Path,
        _max_height: int,
        cookies_from_browser: str | None = None,
    ) -> tuple[Path, VideoMetadata]:
        del cookies_from_browser
        return self.video_path, self.metadata.model_copy(deep=True)


class FakeCloudOcr:
    def recognize_batch(self, frames: list[FrameRef]) -> CloudOcrBatchResult:
        return CloudOcrBatchResult(
            frames=[
                CloudOcrFrameResult(
                    frame_id=f"{frame.index:06d}",
                    text="HELLO" if frame.timestamp_ms < 1000 else "WORLD",
                    confidence=0.95,
                )
                for frame in frames
            ],
            request_id=f"batch-{frames[0].index}",
            usage={"input_tokens": len(frames) * 10},
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_pipeline_writes_json_and_markdown(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:r=2:d=2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    config = ExtractionConfig(
        source_url="https://youtu.be/synthetic",
        workspace_root=tmp_path / "workspace",
        output_dir=tmp_path / "output",
        sample_fps=2,
        change_threshold=0.01,
        max_ocr_interval=0.5,
        keep_cache=True,
        ocr=CloudOcrConfig(
            base_url="https://vendor.example/v1",
            model="vision-ocr",
            batch_size=2,
        ),
    )
    result = BurnedSubtitlePipeline(
        downloader=FakeDownloader(video),
        ocr_engine=FakeCloudOcr(),
    ).run(config)

    assert result.segment_count == 2
    assert result.observation_count == 4
    assert result.markdown_path.is_file()
    assert "HELLO" in result.markdown_path.read_text(encoding="utf-8")
    assert "WORLD" in result.markdown_path.read_text(encoding="utf-8")

    saved_config = result.config_path.read_text(encoding="utf-8")
    assert "vision-ocr" in saved_config
    assert "secret" not in saved_config
    calls = json.loads(result.api_calls_path.read_text(encoding="utf-8"))
    assert len(calls) == 2
