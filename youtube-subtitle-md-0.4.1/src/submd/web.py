from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import threading
import uuid
import webbrowser
from collections.abc import Callable
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from dotenv import dotenv_values, set_key

from submd.errors import SubmdError
from submd.json_io import write_json
from submd.models import CloudOcrConfig, ExtractionConfig, TextLlmConfig
from submd.organize import SubtitleOrganizer
from submd.pipeline import BurnedSubtitlePipeline

PipelineFactory = Callable[[Callable[[str], None], str], BurnedSubtitlePipeline]
OrganizerFactory = Callable[[Callable[[str], None], str], SubtitleOrganizer]

_ENV_FIELDS = (
    "SUBMD_YOUTUBE_URL",
    "SUBMD_OCR_BASE_URL",
    "SUBMD_OCR_MODEL",
    "SUBMD_OCR_API_KEY",
    "SUBMD_YOUTUBE_COOKIES_FROM_BROWSER",
    "SUBMD_TEXT_BASE_URL",
    "SUBMD_TEXT_MODEL",
)
_SECRET_FIELD = "SUBMD_OCR_API_KEY"
_MAX_REQUEST_BYTES = 64 * 1024


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class EnvironmentStore:
    """Read and update the project's .env without disclosing stored secrets."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def read_private(self) -> dict[str, str]:
        with self._lock:
            values = dotenv_values(self.path) if self.path.is_file() else {}
        return {
            key: str(values[key] or "") if key in values else str(os.environ.get(key) or "")
            for key in _ENV_FIELDS
        }

    def read_public(self) -> dict[str, Any]:
        values = self.read_private()
        return {key: value for key, value in values.items() if key != _SECRET_FIELD} | {
            "SUBMD_OCR_API_KEY_CONFIGURED": bool(values[_SECRET_FIELD])
        }

    def update(self, submitted: dict[str, Any]) -> dict[str, str]:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            current = dotenv_values(self.path)
            for key in _ENV_FIELDS:
                value = submitted.get(key)
                if value is None:
                    continue
                clean = str(value).strip()
                if key == _SECRET_FIELD and not clean and current.get(key):
                    continue
                set_key(str(self.path), key, clean, quote_mode="always")
        return self.read_private()


class ExtractionJobManager:
    def __init__(
        self,
        project_root: Path,
        pipeline_factory: PipelineFactory | None = None,
        organizer_factory: OrganizerFactory | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.environment = EnvironmentStore(self.project_root / ".env")
        self.history_path = self.project_root / "workspace" / "ui_history.json"
        self.pipeline_factory = pipeline_factory or self._default_pipeline
        self.organizer_factory = organizer_factory or self._default_organizer
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._history = self._load_history()

    @staticmethod
    def _default_pipeline(status: Callable[[str], None], api_key: str) -> BurnedSubtitlePipeline:
        return BurnedSubtitlePipeline(status=status, api_key=api_key)

    @staticmethod
    def _default_organizer(status: Callable[[str], None], api_key: str) -> SubtitleOrganizer:
        return SubtitleOrganizer(status=status, api_key=api_key)

    def config(self) -> dict[str, Any]:
        return self.environment.read_public()

    def save_config(self, submitted: dict[str, Any]) -> dict[str, Any]:
        self.environment.update(submitted)
        return self.config()

    def start(self, submitted: dict[str, Any]) -> dict[str, Any]:
        current = self.environment.read_private()
        candidate = dict(current)
        for key in _ENV_FIELDS:
            if key not in submitted:
                continue
            clean = str(submitted[key]).strip()
            if key == _SECRET_FIELD and not clean and current.get(key):
                continue
            candidate[key] = clean
        self._validate_required(candidate)
        with self._lock:
            if any(item.get("status") == "running" for item in self._jobs.values()):
                raise ValueError("已有字幕提取任务正在运行，请等待当前任务完成")
        values = self.environment.update(submitted)
        job_id = uuid.uuid4().hex
        record: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "source_url": values["SUBMD_YOUTUBE_URL"],
            "started_at": _now(),
            "finished_at": None,
            "message": "任务已创建，正在读取视频信息…",
            "error": None,
            "result_name": None,
            "result_path": None,
            "results": [],
        }
        with self._lock:
            self._jobs[job_id] = record
            self._history.insert(0, dict(record))
            self._persist_history_locked()
        thread = threading.Thread(
            target=self._run,
            args=(job_id, values),
            name=f"submd-extract-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return self.job(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id) or next(
                (item for item in self._history if item.get("job_id") == job_id), None
            )
            if record is None:
                raise KeyError(job_id)
            return self._public_record(record)

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_record(item) for item in self._history]

    def result_file(self, job_id: str, result_kind: str | None = None) -> Path:
        with self._lock:
            record = self._jobs.get(job_id) or next(
                (item for item in self._history if item.get("job_id") == job_id), None
            )
            raw_path = None
            if record and result_kind:
                result = next(
                    (
                        item
                        for item in record.get("results", [])
                        if item.get("kind") == result_kind
                    ),
                    None,
                )
                raw_path = result.get("path") if result else None
            elif record:
                raw_path = record.get("result_path")
        if not raw_path:
            raise FileNotFoundError(job_id)
        path = Path(str(raw_path)).expanduser().resolve()
        output_root = (self.project_root / "output").resolve()
        if not path.is_relative_to(output_root) or not path.is_file():
            raise FileNotFoundError(job_id)
        return path

    def _run(self, job_id: str, values: dict[str, str]) -> None:
        def status(message: str) -> None:
            self._update_job(job_id, message=message)

        raw_path: Path | None = None
        raw_results: list[dict[str, str]] = []
        try:
            raw_path = self._find_reusable_raw(values["SUBMD_YOUTUBE_URL"])
            if raw_path is None:
                config = ExtractionConfig(
                    source_url=values["SUBMD_YOUTUBE_URL"],
                    cookies_from_browser=values["SUBMD_YOUTUBE_COOKIES_FROM_BROWSER"] or None,
                    workspace_root=self.project_root / "workspace",
                    output_dir=self.project_root / "output",
                    ocr=CloudOcrConfig(
                        base_url=values["SUBMD_OCR_BASE_URL"],
                        model=values["SUBMD_OCR_MODEL"],
                        batch_size=16,
                    ),
                )
                pipeline = self.pipeline_factory(status, values[_SECRET_FIELD])
                extraction = pipeline.run(config)
                raw_path = extraction.markdown_path.resolve()
                extract_message = f"OCR 完成：{extraction.segment_count} 个字幕段；正在语义断句…"
            else:
                extract_message = "找到该视频已有的原始字幕，已跳过下载和视觉 OCR；正在语义断句…"

            raw_results = [self._result_entry("raw", "原始字幕（含时间戳）", raw_path)]
            self._update_job(
                job_id,
                message=extract_message,
                result_name=raw_path.name,
                result_path=str(raw_path),
                results=raw_results,
            )

            text_config = TextLlmConfig(
                base_url=values["SUBMD_TEXT_BASE_URL"] or values["SUBMD_OCR_BASE_URL"],
                model=values["SUBMD_TEXT_MODEL"] or values["SUBMD_OCR_MODEL"],
            )
            organizer = self.organizer_factory(status, values[_SECRET_FIELD])
            organized = organizer.run(
                source_path=raw_path,
                config=text_config,
                workspace_root=self.project_root / "workspace",
                output_dir=self.project_root / "output",
                overwrite=True,
            )
            organized_path = organized.markdown_path.resolve()
            results = [
                *raw_results,
                self._result_entry("organized", "整理版（只含字幕）", organized_path),
            ]
            self._finish_job(
                job_id,
                status="succeeded",
                message=f"处理完成：已生成原始版和 {organized.sentence_count} 句整理版",
                result_name=raw_path.name,
                result_path=str(raw_path),
                results=results,
            )
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if raw_path is not None and raw_path.is_file():
                self._finish_job(
                    job_id,
                    status="partial",
                    message="原始字幕已生成，但整理版生成失败",
                    error=(
                        f"{message}\n\n请在“整理版字幕模型”中填写可处理纯文本的模型后，"
                        "用同一个 YouTube URL 再试；将直接复用原始字幕，不会重复视觉 OCR。"
                    ),
                    result_name=raw_path.name,
                    result_path=str(raw_path),
                    results=raw_results,
                )
                return
            self._finish_job(
                job_id,
                status="failed",
                message="提取失败",
                error=message,
            )

    @staticmethod
    def _result_entry(kind: str, label: str, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        return {"kind": kind, "label": label, "name": resolved.name, "path": str(resolved)}

    def _find_reusable_raw(self, source_url: str) -> Path | None:
        with self._lock:
            records = [dict(item) for item in self._history]
        for record in records:
            if not self._same_youtube_video(source_url, str(record.get("source_url") or "")):
                continue
            raw_result = next(
                (item for item in record.get("results", []) if item.get("kind") == "raw"),
                None,
            )
            candidate = raw_result.get("path") if raw_result else record.get("result_path")
            if candidate:
                path = Path(str(candidate)).expanduser().resolve()
                if path.is_file() and path.parent == (self.project_root / "output").resolve():
                    return path

        for path in (self.project_root / "output").glob("*.md"):
            if "（整理版" in path.stem:
                continue
            saved_url = self._source_url_from_markdown(path)
            if saved_url and self._same_youtube_video(source_url, saved_url):
                return path.resolve()
        return None

    @staticmethod
    def _source_url_from_markdown(path: Path) -> str:
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[:20]:
                if line.startswith("source_url:"):
                    return str(json.loads(line.partition(":")[2].strip()))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ""
        return ""

    @classmethod
    def _same_youtube_video(cls, left: str, right: str) -> bool:
        left_id = cls._youtube_video_id(left)
        right_id = cls._youtube_video_id(right)
        return bool(left_id and right_id and left_id == right_id) or left.strip() == right.strip()

    @staticmethod
    def _youtube_video_id(url: str) -> str | None:
        parsed = urlsplit(url.strip())
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if host == "youtu.be":
            return parsed.path.strip("/").partition("/")[0] or None
        if host in {"youtube.com", "m.youtube.com"}:
            video_id = (parse_qs(parsed.query).get("v") or [""])[0].strip()
            if video_id:
                return video_id
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                return parts[1]
        return None

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.update(changes)

    def _finish_job(self, job_id: str, status: str, message: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.update(
                status=status,
                message=message,
                finished_at=_now(),
                **changes,
            )
            for index, history_item in enumerate(self._history):
                if history_item.get("job_id") == job_id:
                    self._history[index] = dict(record)
                    break
            self._persist_history_locked()

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.is_file():
            discovered = self._discover_existing_outputs()
            if discovered:
                write_json(self.history_path, discovered)
            return discovered
        try:
            value = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        changed = False
        history: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            record = dict(item)
            legacy_path = record.get("result_path")
            if legacy_path and not record.get("results"):
                raw_path = Path(str(legacy_path)).expanduser().resolve()
                if raw_path.is_file():
                    results = [self._result_entry("raw", "原始字幕（含时间戳）", raw_path)]
                    organized_path = raw_path.with_name(f"{raw_path.stem}（整理版）.md")
                    if organized_path.is_file():
                        results.append(
                            self._result_entry(
                                "organized", "整理版（只含字幕）", organized_path
                            )
                        )
                    record["results"] = results
                    changed = True
            if record.get("status") == "running":
                record.update(
                    status="interrupted",
                    message="上次运行被中断；再次提取会自动复用已有检查点",
                    error="应用在任务完成前退出",
                    finished_at=_now(),
                )
                changed = True
            history.append(record)
        if changed:
            write_json(self.history_path, history)
        return history[:100]

    def _discover_existing_outputs(self) -> list[dict[str, Any]]:
        output_root = self.project_root / "output"
        records: list[dict[str, Any]] = []
        for path in sorted(
            output_root.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True
        ):
            if "（整理版" in path.stem:
                continue
            source_url = ""
            try:
                for line in path.read_text(encoding="utf-8").splitlines()[:20]:
                    if line.startswith("source_url:"):
                        source_url = str(json.loads(line.partition(":")[2].strip()))
                        break
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            timestamp = (
                datetime.fromtimestamp(path.stat().st_mtime)
                .astimezone()
                .isoformat(timespec="seconds")
            )
            digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
            results = [self._result_entry("raw", "原始字幕（含时间戳）", path)]
            organized_path = path.with_name(f"{path.stem}（整理版）.md")
            if organized_path.is_file():
                results.append(
                    self._result_entry("organized", "整理版（只含字幕）", organized_path)
                )
            records.append(
                {
                    "job_id": f"imported-{digest}",
                    "status": "succeeded",
                    "source_url": source_url,
                    "started_at": timestamp,
                    "finished_at": timestamp,
                    "message": "已发现现有字幕文件",
                    "error": None,
                    "result_name": path.name,
                    "result_path": str(path.resolve()),
                    "results": results,
                }
            )
        return records[:100]

    def _persist_history_locked(self) -> None:
        write_json(self.history_path, self._history[:100])

    @staticmethod
    def _validate_required(values: dict[str, str]) -> None:
        required = {
            "SUBMD_YOUTUBE_URL": "YouTube URL",
            "SUBMD_OCR_BASE_URL": "OCR API 地址",
            "SUBMD_OCR_MODEL": "视觉模型名称",
            _SECRET_FIELD: "API Key",
        }
        missing = [label for key, label in required.items() if not values.get(key)]
        if missing:
            raise ValueError(f"请填写：{'、'.join(missing)}")
        parsed_url = urlsplit(values["SUBMD_YOUTUBE_URL"])
        hostname = (parsed_url.hostname or "").lower().removeprefix("www.")
        if parsed_url.scheme not in {"http", "https"} or hostname not in {
            "youtube.com",
            "m.youtube.com",
            "youtu.be",
        }:
            raise ValueError("YouTube URL 无效，请输入 youtube.com 或 youtu.be 视频地址")

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value for key, value in record.items() if key not in {"result_path", "results"}
        }
        public_results: list[dict[str, str]] = []
        for item in record.get("results", []):
            kind = str(item.get("kind") or "")
            if not kind or not item.get("path"):
                continue
            public_results.append(
                {
                    "kind": kind,
                    "label": str(item.get("label") or item.get("name") or "结果文件"),
                    "name": str(item.get("name") or Path(str(item["path"])).name),
                    "download_url": f"/api/files/{record['job_id']}/{kind}",
                }
            )
        if public_results:
            public["results"] = public_results
        if record.get("result_path"):
            public["download_url"] = f"/api/files/{record['job_id']}"
        return public


class UiHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    manager: ExtractionJobManager
    static_root: Path


class UiRequestHandler(BaseHTTPRequestHandler):
    server: UiHttpServer

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/health":
                self._send_json({"ok": True})
            elif path == "/api/config":
                self._send_json(self.server.manager.config())
            elif path == "/api/history":
                self._send_json({"items": self.server.manager.history()})
            elif path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/")
                self._send_json(self.server.manager.job(job_id))
            elif path.startswith("/api/files/"):
                parts = path.removeprefix("/api/files/").split("/", maxsplit=1)
                job_id = parts[0]
                result_kind = parts[1] if len(parts) == 2 else None
                self._send_file(
                    self.server.manager.result_file(job_id, result_kind), download=True
                )
            elif path in {"/", "/index.html"}:
                self._send_file(self.server.static_root / "index.html")
            elif path.startswith("/assets/"):
                relative = Path(path.removeprefix("/assets/"))
                if ".." in relative.parts:
                    raise FileNotFoundError(path)
                self._send_file(self.server.static_root / relative)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "页面不存在")
        except KeyError:
            self._send_error(HTTPStatus.NOT_FOUND, "任务不存在")
        except FileNotFoundError:
            self._send_error(HTTPStatus.NOT_FOUND, "文件不存在")
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/config":
                self._send_json(self.server.manager.save_config(payload))
            elif path == "/api/jobs":
                self._send_json(self.server.manager.start(payload), status=HTTPStatus.ACCEPTED)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "接口不存在")
        except (SubmdError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError as exc:
            raise ValueError("请求长度无效") from exc
        if length <= 0 or length > _MAX_REQUEST_BYTES:
            raise ValueError("请求内容为空或过大")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("请求不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("请求必须是 JSON 对象")
        return value

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, download: bool = False) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if download:
            encoded = quote(path.name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded}")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


def create_ui_server(
    manager: ExtractionJobManager,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> UiHttpServer:
    static_root = Path(__file__).with_name("ui")
    if not (static_root / "index.html").is_file():
        raise RuntimeError("UI 静态文件缺失")
    server = UiHttpServer((host, port), UiRequestHandler)
    server.manager = manager
    server.static_root = static_root
    return server


def open_ui_in_browser(url: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(  # noqa: S603
                ["open", "-a", "Google Chrome", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        webbrowser.open(url)
    except OSError:
        webbrowser.open(url)


def serve_ui(
    project_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    manager = ExtractionJobManager(project_root)
    try:
        server = create_ui_server(manager, host=host, port=port)
    except OSError as exc:
        raise SubmdError(f"无法启动 UI 服务：{exc}") from exc
    url = f"http://{host}:{server.server_port}"
    if open_browser:
        threading.Timer(0.5, open_ui_in_browser, args=(url,)).start()
    print(f"SubMD UI 已启动：{url}")
    print("关闭这个窗口会停止 UI；字幕提取期间请保持窗口开启。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
