from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from submd.errors import DownloadError, OrganizeError
from submd.models import ExtractionResult, OrganizeResult, SentenceLearningAnalysis
from submd.web import EnvironmentStore, ExtractionJobManager, create_ui_server


def config_payload() -> dict[str, str]:
    return {
        "SUBMD_YOUTUBE_URL": "https://youtu.be/ui-test",
        "SUBMD_OCR_BASE_URL": "https://vendor.example/v1",
        "SUBMD_OCR_MODEL": "vision-model",
        "SUBMD_OCR_API_KEY": "secret-key",
        "SUBMD_YOUTUBE_COOKIES_FROM_BROWSER": "chrome",
        "SUBMD_TEXT_BASE_URL": "",
        "SUBMD_TEXT_MODEL": "text-model",
        "SUBMD_LEARNING_BASE_URL": "",
        "SUBMD_LEARNING_MODEL": "",
        "SUBMD_LEARNING_API_KEY": "",
    }


def wait_for_job(manager: ExtractionJobManager, job_id: str) -> dict:
    for _ in range(100):
        job = manager.job(job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("background job did not finish")


class SuccessfulOrganizer:
    def run(self, source_path, config, workspace_root, output_dir, overwrite):
        del config, overwrite
        organized = output_dir / f"{source_path.stem}（整理版）.md"
        organized.write_text("第一句话。\n第二句话。\n", encoding="utf-8")
        checkpoint = workspace_root / "organize" / "checkpoint.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("{}\n", encoding="utf-8")
        sentences = workspace_root / "organize" / "organized_segments.json"
        sentences.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_markdown": source_path.name,
                    "sentences": [
                        {
                            "sentence_id": "s000001",
                            "text": "第一句话。",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "source_unit_ids": ["u000001"],
                            "source_segment_ids": ["seg000001"],
                        },
                        {
                            "sentence_id": "s000002",
                            "text": "第二句话。",
                            "start_ms": 1000,
                            "end_ms": 2000,
                            "source_unit_ids": ["u000002"],
                            "source_segment_ids": ["seg000002"],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return OrganizeResult(
            source_path=source_path,
            markdown_path=organized,
            checkpoint_path=checkpoint,
            source_fragment_count=2,
            sentence_count=2,
            api_call_count=1,
            reused_chunk_count=0,
            sentences_path=sentences,
        )


class SuccessfulAnalyzer:
    def __init__(self, api_key: str, captured: dict | None = None) -> None:
        self.api_key = api_key
        self.captured = captured if captured is not None else {}

    def analyze(self, sentence, config, cache_root, force=False):
        self.captured.update(
            sentence=sentence,
            base_url=config.base_url,
            model=config.model,
            api_key=self.api_key,
            cache_root=str(cache_root),
            force=force,
        )
        return (
            SentenceLearningAnalysis(
                prompt_version="test-v1",
                sentence=sentence,
                model=config.model,
                translation="第二句话的翻译。",
                vocabulary=[
                    {"expression": "第二句", "reading": "だいにく", "meaning": "第二句"}
                ],
                grammar=[{"pattern": "です", "explanation": "礼貌判断句。"}],
            ),
            False,
        )


def test_environment_store_never_returns_api_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        'SUBMD_OCR_BASE_URL="https://vendor.example/v1"\nSUBMD_OCR_API_KEY="stored-secret"\n',
        encoding="utf-8",
    )
    store = EnvironmentStore(env_path)

    public = store.read_public()
    assert public["SUBMD_OCR_API_KEY_CONFIGURED"] is True
    assert "SUBMD_OCR_API_KEY" not in public
    assert "stored-secret" not in json.dumps(public)

    store.update({"SUBMD_OCR_BASE_URL": "https://new.example/v1", "SUBMD_OCR_API_KEY": ""})
    private = store.read_private()
    assert private["SUBMD_OCR_BASE_URL"] == "https://new.example/v1"
    assert private["SUBMD_OCR_API_KEY"] == "stored-secret"


def test_job_manager_records_success_and_download(tmp_path: Path) -> None:
    learning_capture: dict = {}

    class SuccessfulPipeline:
        def __init__(self, status) -> None:
            self.status = status

        def run(self, config) -> ExtractionResult:
            self.status("正在识别测试字幕…")
            result = config.output_dir / "测试视频.md"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text("测试字幕\n", encoding="utf-8")
            audio = result.with_suffix(".m4a")
            audio.write_bytes(b"0123456789")
            intermediate = config.workspace_root / "test.json"
            intermediate.parent.mkdir(parents=True, exist_ok=True)
            intermediate.write_text("{}\n", encoding="utf-8")
            return ExtractionResult(
                metadata_path=intermediate,
                config_path=intermediate,
                observations_path=intermediate,
                api_calls_path=intermediate,
                segments_path=intermediate,
                markdown_path=result,
                segment_count=1,
                observation_count=1,
                audio_path=audio,
            )

    manager = ExtractionJobManager(
        tmp_path,
        pipeline_factory=lambda status, _key: SuccessfulPipeline(status),
        organizer_factory=lambda _status, _key: SuccessfulOrganizer(),
        analyzer_factory=lambda key: SuccessfulAnalyzer(key, learning_capture),
    )
    started = manager.start(config_payload())
    finished = wait_for_job(manager, started["job_id"])

    assert finished["status"] == "succeeded"
    assert finished["result_name"] == "测试视频.md"
    assert finished["download_url"] == f"/api/files/{started['job_id']}"
    assert [item["kind"] for item in finished["results"]] == [
        "raw",
        "audio",
        "organized",
    ]
    assert finished["player_url"] == f"/api/player/{started['job_id']}"
    assert manager.result_file(started["job_id"]).read_text(encoding="utf-8") == "测试字幕\n"
    assert manager.result_file(started["job_id"], "organized").read_text(
        encoding="utf-8"
    ) == "第一句话。\n第二句话。\n"
    history = manager.history()
    assert len(history) == 1
    assert history[0]["status"] == "succeeded"
    assert "result_path" not in history[0]
    assert all("path" not in item for item in history[0]["results"])
    player = manager.player_data(started["job_id"])
    assert player["sentence_count"] == 2
    assert player["sentences"][1]["start_ms"] == 1000
    assert player["analysis_url"] == f"/api/player/{started['job_id']}/analysis"
    assert player["resegment_url"] == f"/api/player/{started['job_id']}/resegment"
    analysis = manager.analyze_sentence(started["job_id"], "s000002")
    assert analysis["translation"] == "第二句话的翻译。"
    assert analysis["cached"] is False
    assert learning_capture == {
        "sentence": "第二句话。",
        "base_url": "https://vendor.example/v1",
        "model": "text-model",
        "api_key": "secret-key",
        "cache_root": str(tmp_path / "workspace" / "learning"),
        "force": False,
    }
    resegmented = manager.resegment_sentences(
        started["job_id"],
        ["s000001", "s000002"],
        "第一句话第二句话。",
    )
    assert resegmented["saved"] is True
    assert resegmented["sentence_count"] == 1
    assert resegmented["sentences"][0]["start_ms"] == 0
    assert resegmented["sentences"][0]["end_ms"] == 2000
    assert (tmp_path / "workspace" / "manual-edits").is_dir()


def test_job_manager_records_failure_reason(tmp_path: Path) -> None:
    class FailedPipeline:
        def run(self, _config):
            raise DownloadError("YouTube 拒绝访问测试视频")

    manager = ExtractionJobManager(
        tmp_path,
        pipeline_factory=lambda _status, _key: FailedPipeline(),
    )
    started = manager.start(config_payload())
    finished = wait_for_job(manager, started["job_id"])

    assert finished["status"] == "failed"
    assert finished["error"] == "YouTube 拒绝访问测试视频"
    assert manager.history()[0]["error"] == "YouTube 拒绝访问测试视频"


def test_retry_reuses_raw_markdown_after_organizer_failure(tmp_path: Path) -> None:
    pipeline_calls = 0
    organizer_calls = 0

    class CountingPipeline:
        def run(self, config) -> ExtractionResult:
            nonlocal pipeline_calls
            pipeline_calls += 1
            result = config.output_dir / "复用测试.md"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(
                f'---\nsource_url: "{config.source_url}"\n---\n\n'
                "- [00:00.000–00:01.000] 测试字幕\n",
                encoding="utf-8",
            )
            intermediate = config.workspace_root / "test.json"
            intermediate.parent.mkdir(parents=True, exist_ok=True)
            intermediate.write_text("{}\n", encoding="utf-8")
            return ExtractionResult(
                metadata_path=intermediate,
                config_path=intermediate,
                observations_path=intermediate,
                api_calls_path=intermediate,
                segments_path=intermediate,
                markdown_path=result,
                segment_count=1,
                observation_count=1,
            )

    class RetryOrganizer(SuccessfulOrganizer):
        def run(self, *args, **kwargs):
            nonlocal organizer_calls
            organizer_calls += 1
            if organizer_calls == 1:
                raise OrganizeError("视觉模型不接受纯文本请求")
            return super().run(*args, **kwargs)

    manager = ExtractionJobManager(
        tmp_path,
        pipeline_factory=lambda _status, _key: CountingPipeline(),
        organizer_factory=lambda _status, _key: RetryOrganizer(),
    )
    first = wait_for_job(manager, manager.start(config_payload())["job_id"])
    assert first["status"] == "partial"
    assert [item["kind"] for item in first["results"]] == ["raw"]
    assert "不会重复视觉 OCR" in first["error"]

    retry_payload = config_payload() | {
        "SUBMD_YOUTUBE_URL": "https://www.youtube.com/watch?v=ui-test&si=different",
        "SUBMD_TEXT_MODEL": "working-text-model",
    }
    second = wait_for_job(manager, manager.start(retry_payload)["job_id"])
    assert second["status"] == "succeeded"
    assert [item["kind"] for item in second["results"]] == ["raw", "organized"]
    assert pipeline_calls == 1
    assert organizer_calls == 2


def test_invalid_youtube_url_is_rejected_without_overwriting_env(tmp_path: Path) -> None:
    manager = ExtractionJobManager(tmp_path)
    manager.save_config(config_payload())
    invalid = config_payload() | {"SUBMD_YOUTUBE_URL": "https://example.com/video"}

    try:
        manager.start(invalid)
    except ValueError as exc:
        assert "YouTube URL 无效" in str(exc)
    else:
        raise AssertionError("non-YouTube URL should be rejected")

    assert manager.config()["SUBMD_YOUTUBE_URL"] == "https://youtu.be/ui-test"


def test_existing_subtitle_markdown_is_imported_into_history(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    subtitle = output / "现有字幕.md"
    subtitle.write_text(
        '---\nsource_url: "https://youtu.be/existing"\n---\n\n- [00:00.000–00:01.000] 字幕\n',
        encoding="utf-8",
    )
    (output / "现有字幕（整理版）.md").write_text("字幕\n", encoding="utf-8")

    manager = ExtractionJobManager(tmp_path)
    history = manager.history()

    assert len(history) == 1
    assert history[0]["source_url"] == "https://youtu.be/existing"
    assert history[0]["result_name"] == "现有字幕.md"
    assert history[0]["download_url"].startswith("/api/files/imported-")
    assert [item["kind"] for item in history[0]["results"]] == ["raw", "organized"]


def test_http_ui_serves_assets_config_and_errors(tmp_path: Path) -> None:
    manager = ExtractionJobManager(tmp_path)
    manager.save_config(config_payload())
    server = create_ui_server(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base}/", timeout=2) as response:
            html = response.read().decode()
        assert "YouTube 烧录字幕提取" in html
        assert 'id="extract-button"' in html
        assert 'id="history-body"' in html
        assert 'id="error-dialog"' in html
        assert 'id="player-panel"' in html
        assert 'id="loop-sentence"' in html
        assert 'id="analysis-dialog"' in html
        assert 'id="reanalyze-sentence"' in html
        assert 'id="boundary-toolbar"' in html
        assert 'id="boundary-dialog"' in html
        assert 'id="boundary-editor-text"' in html

        with urlopen(f"{base}/assets/app.js", timeout=2) as response:
            script = response.read().decode()
        assert '"failed", ["partial", "failed", "interrupted"]' in script
        assert "data-sentence-index" in script
        assert "data-analysis-index" in script
        assert "sentenceLoopEnabled" in script
        assert "force" in script
        assert "boundarySelectionAnchor" in script
        assert "activeResegmentUrl" in script
        assert "saveBoundaryEdit" in script

        with urlopen(f"{base}/api/config", timeout=2) as response:
            config = json.loads(response.read())
        assert config["SUBMD_OCR_API_KEY_CONFIGURED"] is True
        assert config["SUBMD_LEARNING_API_KEY_CONFIGURED"] is False
        assert "SUBMD_OCR_API_KEY" not in config
        assert "SUBMD_LEARNING_API_KEY" not in config
        assert "secret-key" not in json.dumps(config)

        request = Request(
            f"{base}/api/jobs",
            data=json.dumps({key: "" for key in config_payload()}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=2)
        except HTTPError as exc:
            payload = json.loads(exc.read())
            assert exc.code == 400
            assert "请填写" in payload["error"]
        else:
            raise AssertionError("invalid extraction request should fail")
    finally:
        server.shutdown()
        server.server_close()


def test_http_player_supports_clickable_sentences_and_ranges(tmp_path: Path) -> None:
    class SuccessfulPipeline:
        def run(self, config) -> ExtractionResult:
            raw = config.output_dir / "播放器测试.md"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("- [00:00.000–00:01.000] 第一句话。\n", encoding="utf-8")
            audio = raw.with_suffix(".m4a")
            audio.write_bytes(b"0123456789")
            intermediate = config.workspace_root / "test.json"
            intermediate.parent.mkdir(parents=True, exist_ok=True)
            intermediate.write_text("{}\n", encoding="utf-8")
            return ExtractionResult(
                metadata_path=intermediate,
                config_path=intermediate,
                observations_path=intermediate,
                api_calls_path=intermediate,
                segments_path=intermediate,
                markdown_path=raw,
                segment_count=1,
                observation_count=1,
                audio_path=audio,
            )

    manager = ExtractionJobManager(
        tmp_path,
        pipeline_factory=lambda _status, _key: SuccessfulPipeline(),
        organizer_factory=lambda _status, _key: SuccessfulOrganizer(),
        analyzer_factory=lambda key: SuccessfulAnalyzer(key),
    )
    job = wait_for_job(manager, manager.start(config_payload())["job_id"])
    server = create_ui_server(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base}{job['player_url']}", timeout=2) as response:
            player = json.loads(response.read())
        assert player["sentences"][0]["text"] == "第一句话。"
        assert player["audio_url"].endswith("/audio")
        assert player["analysis_url"].endswith("/analysis")

        analysis_request = Request(
            f"{base}{player['analysis_url']}",
            data=json.dumps({"sentence_id": "s000001", "force": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(analysis_request, timeout=2) as response:
            analysis = json.loads(response.read())
        assert analysis["translation"] == "第二句话的翻译。"
        assert analysis["vocabulary"][0]["reading"] == "だいにく"

        request = Request(
            f"{base}{player['audio_url']}",
            headers={"Range": "bytes=2-5"},
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.read() == b"2345"
    finally:
        server.shutdown()
        server.server_close()
