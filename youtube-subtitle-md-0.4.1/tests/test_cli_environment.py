import os
from pathlib import Path

from typer.testing import CliRunner

from submd.cli import app, load_environment
from submd.models import ExtractionResult, OrganizeResult


def test_loads_dotenv_without_overriding_exported_values(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        'SUBMD_OCR_BASE_URL="https://dotenv.example/v1"\n'
        'SUBMD_OCR_MODEL="dotenv-model"\n'
        'SUBMD_OCR_API_KEY="dotenv-key"\n'
        'SUBMD_YOUTUBE_URL="https://youtu.be/dotenv-video"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SUBMD_OCR_BASE_URL", raising=False)
    monkeypatch.setenv("SUBMD_OCR_MODEL", "exported-model")
    monkeypatch.delenv("SUBMD_OCR_API_KEY", raising=False)
    monkeypatch.delenv("SUBMD_YOUTUBE_URL", raising=False)

    load_environment()

    assert os.environ["SUBMD_OCR_BASE_URL"] == "https://dotenv.example/v1"
    assert os.environ["SUBMD_OCR_MODEL"] == "exported-model"
    assert os.environ["SUBMD_OCR_API_KEY"] == "dotenv-key"
    assert os.environ["SUBMD_YOUTUBE_URL"] == "https://youtu.be/dotenv-video"


def test_extract_reads_youtube_url_from_environment(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakePipeline:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, config) -> ExtractionResult:
            captured["url"] = config.source_url
            captured["cookies_from_browser"] = config.cookies_from_browser
            result_file = tmp_path / "result.json"
            markdown_file = tmp_path / "result.md"
            return ExtractionResult(
                metadata_path=result_file,
                config_path=result_file,
                observations_path=result_file,
                api_calls_path=result_file,
                segments_path=result_file,
                markdown_path=markdown_file,
                segment_count=0,
                observation_count=0,
            )

    monkeypatch.setattr("submd.cli.BurnedSubtitlePipeline", FakePipeline)
    result = CliRunner().invoke(
        app,
        ["extract"],
        env={
            "SUBMD_OCR_BASE_URL": "https://vendor.example/v1",
            "SUBMD_OCR_MODEL": "vision-ocr",
            "SUBMD_OCR_API_KEY": "test-key",
            "SUBMD_YOUTUBE_URL": "https://youtu.be/from-dotenv",
            "SUBMD_YOUTUBE_COOKIES_FROM_BROWSER": "chrome",
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["url"] == "https://youtu.be/from-dotenv"
    assert captured["cookies_from_browser"] == "chrome"


def test_organize_reuses_ocr_environment_by_default(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.md"
    source.write_text("- [00:00.000–00:01.000] 字幕\n", encoding="utf-8")
    captured: dict[str, str] = {}

    class FakeOrganizer:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, source_path, config, **_kwargs) -> OrganizeResult:
            captured["source"] = str(source_path)
            captured["base_url"] = config.base_url
            captured["model"] = config.model
            output = tmp_path / "organized.md"
            checkpoint = tmp_path / "checkpoint.json"
            return OrganizeResult(
                source_path=source_path,
                markdown_path=output,
                checkpoint_path=checkpoint,
                source_fragment_count=1,
                sentence_count=1,
                api_call_count=1,
                reused_chunk_count=0,
            )

    monkeypatch.setattr("submd.cli.SubtitleOrganizer", FakeOrganizer)
    result = CliRunner().invoke(
        app,
        ["organize", str(source)],
        env={
            "SUBMD_OCR_BASE_URL": "https://vendor.example/v1",
            "SUBMD_OCR_MODEL": "shared-model",
            "SUBMD_OCR_API_KEY": "test-key",
            "SUBMD_TEXT_BASE_URL": "",
            "SUBMD_TEXT_MODEL": "",
        },
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "source": str(source),
        "base_url": "https://vendor.example/v1",
        "model": "shared-model",
    }
