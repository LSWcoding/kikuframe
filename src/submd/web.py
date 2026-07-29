from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
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

from submd.editing import apply_manual_resegmentation
from submd.errors import SubmdError
from submd.exporters import export_markdown
from submd.fusion import SubtitleFusion, infer_caption_language
from submd.json_io import write_json, write_text
from submd.learning import SentenceAnalyzer
from submd.models import (
    CloudOcrConfig,
    ExtractionConfig,
    GrammarAnalysisItem,
    LanguageLearningConfig,
    OrganizedSubtitleDocument,
    SubtitleDocument,
    TextLlmConfig,
    VocabularyAnalysisItem,
    YouTubeCaptionTrack,
)
from submd.organize import SubtitleOrganizer, write_fallback_player_document
from submd.pipeline import BurnedSubtitlePipeline
from submd.study_library import StudyLibrary

PipelineFactory = Callable[[Callable[[str], None], str], BurnedSubtitlePipeline]
OrganizerFactory = Callable[[Callable[[str], None], str], SubtitleOrganizer]
AnalyzerFactory = Callable[[str], SentenceAnalyzer]

_ENV_FIELDS = (
    "SUBMD_YOUTUBE_URL",
    "SUBMD_OCR_BASE_URL",
    "SUBMD_OCR_MODEL",
    "SUBMD_OCR_API_KEY",
    "SUBMD_YOUTUBE_COOKIES_FROM_BROWSER",
    "SUBMD_TEXT_BASE_URL",
    "SUBMD_TEXT_MODEL",
    "SUBMD_LEARNING_BASE_URL",
    "SUBMD_LEARNING_MODEL",
    "SUBMD_LEARNING_API_KEY",
)
_PRIMARY_SECRET_FIELD = "SUBMD_OCR_API_KEY"
_SECRET_FIELDS = {_PRIMARY_SECRET_FIELD, "SUBMD_LEARNING_API_KEY"}
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
        return {key: value for key, value in values.items() if key not in _SECRET_FIELDS} | {
            "SUBMD_OCR_API_KEY_CONFIGURED": bool(values[_PRIMARY_SECRET_FIELD]),
            "SUBMD_LEARNING_API_KEY_CONFIGURED": bool(values["SUBMD_LEARNING_API_KEY"]),
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
                if key in _SECRET_FIELDS and not clean and current.get(key):
                    continue
                set_key(str(self.path), key, clean, quote_mode="always")
        return self.read_private()


class ExtractionJobManager:
    def __init__(
        self,
        project_root: Path,
        pipeline_factory: PipelineFactory | None = None,
        organizer_factory: OrganizerFactory | None = None,
        analyzer_factory: AnalyzerFactory | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.environment = EnvironmentStore(self.project_root / ".env")
        self.history_path = self.project_root / "workspace" / "ui_history.json"
        self.study_library = StudyLibrary(
            self.project_root / "workspace" / "learning" / "library.sqlite3"
        )
        self.pipeline_factory = pipeline_factory or self._default_pipeline
        self.organizer_factory = organizer_factory or self._default_organizer
        self.analyzer_factory = analyzer_factory or self._default_analyzer
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._history = self._load_history()

    @staticmethod
    def _default_pipeline(status: Callable[[str], None], api_key: str) -> BurnedSubtitlePipeline:
        return BurnedSubtitlePipeline(status=status, api_key=api_key)

    @staticmethod
    def _default_organizer(status: Callable[[str], None], api_key: str) -> SubtitleOrganizer:
        return SubtitleOrganizer(status=status, api_key=api_key)

    @staticmethod
    def _default_analyzer(api_key: str) -> SentenceAnalyzer:
        return SentenceAnalyzer(api_key=api_key)

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
            if key in _SECRET_FIELDS and not clean and current.get(key):
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
            "audio_path": None,
            "sentences_path": None,
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

    def learning_library(self) -> dict[str, Any]:
        items = self.study_library.list_entries()
        return {
            "items": items,
            "entry_count": len(items),
            "encounter_count": sum(int(item["encounter_count"]) for item in items),
            "export_url": "/api/library/export",
        }

    def learning_library_entry(self, entry_id: int) -> dict[str, Any]:
        if entry_id < 1:
            raise ValueError("词库条目 ID 无效")
        entry = self.study_library.entry_details(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        with self._lock:
            records = {
                str(record.get("job_id") or ""): dict(record)
                for record in [*self._history, *self._jobs.values()]
            }
        enriched_encounters: list[dict[str, Any]] = []
        for encounter in entry["encounters"]:
            record = records.get(str(encounter.get("job_id") or ""), {})
            result_name = str(record.get("result_name") or "").strip()
            source_url = str(encounter.get("source_url") or "")
            enriched_encounters.append(
                encounter
                | {
                    "article_title": Path(result_name).stem if result_name else source_url,
                }
            )
        return entry | {"encounters": enriched_encounters}

    def export_learning_library(self) -> Path:
        items = self.study_library.list_entries()
        lines: list[str] = []
        for item in items:
            lemma = str(item["display"])
            reading = str(item["reading"])
            headword = f"{lemma}（{reading}）" if reading else lemma
            meanings = "；".join(str(value) for value in item["meanings"])
            lines.append(f"{headword}：{meanings}")
        path = self.project_root / "output" / "KikuFrame-单词库.md"
        write_text(path, "\n".join(lines) + ("\n" if lines else ""))
        return path

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
        return self._validated_project_file(raw_path, "output")

    def player_data(self, job_id: str) -> dict[str, Any]:
        record = self._record(job_id)
        self._validated_project_file(record.get("audio_path"), "output")
        sentences_path = self._validated_project_file(record.get("sentences_path"), "workspace")
        try:
            document = OrganizedSubtitleDocument.model_validate_json(
                sentences_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(job_id) from exc
        raw_name = str(record.get("result_name") or document.source_markdown)
        return {
            "title": Path(raw_name).stem,
            "sentence_count": len(document.sentences),
            "audio_url": f"/api/player/{job_id}/audio",
            "analysis_url": f"/api/player/{job_id}/analysis",
            "library_url": f"/api/player/{job_id}/library",
            "resegment_url": f"/api/player/{job_id}/resegment",
            "sentences": [sentence.model_dump() for sentence in document.sentences],
        }

    def resegment_sentences(
        self, job_id: str, sentence_ids: list[str], edited_text: str
    ) -> dict[str, Any]:
        if not isinstance(sentence_ids, list) or not all(
            isinstance(item, str) for item in sentence_ids
        ):
            raise ValueError("sentence_ids 必须是句子 ID 数组")
        if not isinstance(edited_text, str) or not edited_text.strip():
            raise ValueError("请输入修改后的断句文本")
        record = self._record(job_id)
        sentences_path = self._validated_project_file(record.get("sentences_path"), "workspace")
        organized_result = next(
            (item for item in record.get("results", []) if item.get("kind") == "organized"),
            None,
        )
        if not organized_result or not organized_result.get("path"):
            raise FileNotFoundError("organized")
        organized_path = self._validated_project_file(organized_result["path"], "output")
        try:
            document = OrganizedSubtitleDocument.model_validate_json(
                sentences_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError("当前播放器字幕数据已损坏") from exc
        selected_before = [
            sentence.model_dump(mode="json")
            for sentence in document.sentences
            if sentence.sentence_id in set(sentence_ids)
        ]
        source_document = self._load_source_document(
            str(record.get("source_url") or ""), prefer_corrected=True
        )
        updated = apply_manual_resegmentation(
            document,
            sentence_ids,
            edited_text,
            source_segments=source_document.segments if source_document else None,
        )
        write_json(sentences_path, updated)
        write_text(
            organized_path,
            "\n".join(sentence.text for sentence in updated.sentences) + "\n",
        )
        self._append_manual_edit(
            job_id=job_id,
            record=record,
            sentence_ids=sentence_ids,
            edited_text=edited_text,
            before=selected_before,
            after=[sentence.model_dump(mode="json") for sentence in updated.sentences],
        )
        self._update_persisted_record(
            job_id,
            message=f"手动断句已保存：当前共 {len(updated.sentences)} 句话",
        )
        return self.player_data(job_id) | {"saved": True}

    def analyze_sentence(
        self, job_id: str, sentence_id: str, force: bool = False
    ) -> dict[str, Any]:
        record, sentence = self._learning_sentence(job_id, sentence_id)

        values = self.environment.read_private()
        base_url = (
            values["SUBMD_LEARNING_BASE_URL"]
            or values["SUBMD_TEXT_BASE_URL"]
            or values["SUBMD_OCR_BASE_URL"]
        )
        model = (
            values["SUBMD_LEARNING_MODEL"]
            or values["SUBMD_TEXT_MODEL"]
            or values["SUBMD_OCR_MODEL"]
        )
        api_key = values["SUBMD_LEARNING_API_KEY"] or values[_PRIMARY_SECRET_FIELD]
        if not base_url or not model or not api_key:
            raise ValueError("请先配置语言学习模型的地址、模型名称和 API Key")
        analyzer = self.analyzer_factory(api_key)
        known_items = self.study_library.context_for_analysis(sentence.text)
        context_key = self._learning_context_key(record, sentence.sentence_id, sentence.text)
        analysis, cached = analyzer.analyze(
            sentence=sentence.text,
            config=LanguageLearningConfig(base_url=base_url, model=model),
            cache_root=self.project_root / "workspace" / "learning",
            force=force,
            known_items=known_items,
            cache_scope=context_key,
        )
        payload = analysis.model_dump(mode="json")
        payload["vocabulary"] = [
            item.model_dump(mode="json")
            | {
                "library": self.study_library.state_for(
                    kind=item.kind,
                    lemma=item.lemma,
                    reading=item.reading,
                    meaning=item.meaning,
                    context_key=context_key,
                )
            }
            for item in analysis.vocabulary
        ]
        payload["grammar"] = [
            item.model_dump(mode="json")
            | {
                "library": self.study_library.state_for(
                    kind="grammar",
                    lemma=item.lemma,
                    reading="",
                    meaning=item.explanation,
                    context_key=context_key,
                )
            }
            for item in analysis.grammar
        ]
        return payload | {
            "sentence_id": sentence.sentence_id,
            "cached": cached,
            "library_context_count": len(known_items),
        }

    def save_learning_item(
        self,
        job_id: str,
        sentence_id: str,
        item_type: str,
        raw_item: Any,
    ) -> dict[str, Any]:
        record, sentence = self._learning_sentence(job_id, sentence_id)
        if not isinstance(raw_item, dict):
            raise ValueError("学习词库条目必须是对象")
        if item_type == "vocabulary":
            item = VocabularyAnalysisItem.model_validate(raw_item)
            kind = item.kind
            lemma = item.lemma
            surface = item.expression
            reading = item.reading
            meaning = item.meaning
        elif item_type == "grammar":
            grammar = GrammarAnalysisItem.model_validate(raw_item)
            kind = "grammar"
            lemma = grammar.lemma
            surface = grammar.pattern
            reading = ""
            meaning = grammar.explanation
        else:
            raise ValueError("item_type 必须是 vocabulary 或 grammar")
        state = self.study_library.save(
            kind=kind,
            lemma=lemma,
            surface=surface,
            reading=reading,
            meaning=meaning,
            context_key=self._learning_context_key(record, sentence.sentence_id, sentence.text),
            source_url=str(record.get("source_url") or ""),
            job_id=job_id,
            sentence_id=sentence.sentence_id,
            sentence=sentence.text,
        )
        return {
            "sentence_id": sentence.sentence_id,
            "item_type": item_type,
            "lemma": lemma,
            "library": state,
        }

    def _learning_sentence(self, job_id: str, sentence_id: str) -> tuple[dict[str, Any], Any]:
        if not sentence_id.strip():
            raise ValueError("请选择要分析的句子")
        record = self._record(job_id)
        sentences_path = self._validated_project_file(record.get("sentences_path"), "workspace")
        try:
            document = OrganizedSubtitleDocument.model_validate_json(
                sentences_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(job_id) from exc
        sentence = next(
            (item for item in document.sentences if item.sentence_id == sentence_id), None
        )
        if sentence is None:
            raise ValueError("该句子不属于当前视频")
        return record, sentence

    def _learning_context_key(self, record: dict[str, Any], sentence_id: str, text: str) -> str:
        source_url = str(record.get("source_url") or "")
        source_key = self._youtube_video_id(source_url) or source_url
        return hashlib.sha256(
            json.dumps(
                {"source": source_key, "sentence_id": sentence_id, "text": text},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def player_audio_file(self, job_id: str) -> Path:
        return self._validated_project_file(self._record(job_id).get("audio_path"), "output")

    def _append_manual_edit(
        self,
        job_id: str,
        record: dict[str, Any],
        sentence_ids: list[str],
        edited_text: str,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
    ) -> None:
        video_id = self._youtube_video_id(str(record.get("source_url") or ""))
        stem = video_id or hashlib.sha256(job_id.encode()).hexdigest()[:16]
        path = self.project_root / "workspace" / "manual-edits" / f"{stem}.json"
        entries: list[dict[str, Any]] = []
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    entries = loaded
            except (OSError, json.JSONDecodeError):
                pass
        entries.append(
            {
                "edited_at": _now(),
                "job_id": job_id,
                "selected_sentence_ids": sentence_ids,
                "edited_text": edited_text,
                "selected_before": before,
                "document_after": after,
            }
        )
        write_json(path, entries)

    def _update_persisted_record(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)
            for index, item in enumerate(self._history):
                if item.get("job_id") == job_id:
                    self._history[index] = dict(item) | changes
                    break
            self._persist_history_locked()

    def _record(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id) or next(
                (item for item in self._history if item.get("job_id") == job_id), None
            )
            if record is None:
                raise KeyError(job_id)
            return dict(record)

    def _validated_project_file(self, raw_path: Any, root_name: str) -> Path:
        if not raw_path:
            raise FileNotFoundError(root_name)
        path = Path(str(raw_path)).expanduser().resolve()
        allowed_root = (self.project_root / root_name).resolve()
        if path.is_relative_to(allowed_root) and path.is_file():
            return path

        # History stores absolute paths. If the project directory was renamed or moved,
        # safely resolve the suffix below its former output/workspace directory again.
        if root_name in path.parts:
            reversed_parts = tuple(reversed(path.parts))
            root_index = len(path.parts) - 1 - reversed_parts.index(root_name)
            relocated = allowed_root.joinpath(*path.parts[root_index + 1 :]).resolve()
            if relocated.is_relative_to(allowed_root) and relocated.is_file():
                return relocated
        raise FileNotFoundError(path)

    def _run(self, job_id: str, values: dict[str, str]) -> None:
        def status(message: str) -> None:
            self._update_job(job_id, message=message)

        raw_path: Path | None = None
        audio_path: Path | None = None
        sentences_path: Path | None = None
        raw_results: list[dict[str, str]] = []
        try:
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
            pipeline = self.pipeline_factory(status, values[_PRIMARY_SECRET_FIELD])
            raw_path = self._find_reusable_raw(values["SUBMD_YOUTUBE_URL"])
            if raw_path is None:
                extraction = pipeline.run(config)
                raw_path = extraction.markdown_path.resolve()
                audio_path = extraction.audio_path.resolve() if extraction.audio_path else None
                extract_message = f"OCR 完成：{extraction.segment_count} 个字幕段；正在语义断句…"
            else:
                audio_path = self._find_reusable_audio(values["SUBMD_YOUTUBE_URL"], raw_path)
                if audio_path is None and hasattr(pipeline, "ensure_audio"):
                    audio_path = pipeline.ensure_audio(config, raw_path).resolve()
                extract_message = (
                    "找到该视频已有的原始字幕，已跳过视觉 OCR；正在语义断句…"
                )

            raw_results = [self._result_entry("raw", "原始字幕（含时间戳）", raw_path)]
            if audio_path is not None and audio_path.is_file():
                raw_results.append(self._result_entry("audio", "视频音频（M4A）", audio_path))
            self._update_job(
                job_id,
                message=extract_message,
                result_name=raw_path.name,
                result_path=str(raw_path),
                results=raw_results,
                audio_path=str(audio_path) if audio_path else None,
            )

            text_config = TextLlmConfig(
                base_url=values["SUBMD_TEXT_BASE_URL"] or values["SUBMD_OCR_BASE_URL"],
                model=values["SUBMD_TEXT_MODEL"] or values["SUBMD_OCR_MODEL"],
            )
            organizer_source = raw_path
            reference_track: YouTubeCaptionTrack | None = None
            source_document = self._load_source_document(values["SUBMD_YOUTUBE_URL"])
            if source_document is not None and hasattr(pipeline, "ensure_youtube_captions"):
                language_hint = (
                    config.language
                    if config.language.lower() != "auto"
                    else infer_caption_language(source_document)
                )
                reference_track, _caption_path = pipeline.ensure_youtube_captions(
                    config,
                    source_document.video.video_id,
                    language_hint,
                )
                if reference_track is not None:
                    job_dir = (
                        self.project_root / "workspace" / source_document.video.video_id
                    )
                    corrected_document = SubtitleFusion(
                        api_key=values[_PRIMARY_SECRET_FIELD], status=status
                    ).run(
                        document=source_document,
                        track=reference_track,
                        config=text_config,
                        output_path=job_dir / "corrected_segments.json",
                        checkpoint_path=job_dir / "fusion_checkpoint.json",
                    )
                    organizer_source = export_markdown(
                        corrected_document,
                        self.project_root / "output",
                        overwrite=True,
                        name_suffix="（综合校正版）",
                    ).resolve()
                    raw_results.append(
                        self._result_entry(
                            "corrected",
                            "OCR + YouTube 读音参考（含时间戳）",
                            organizer_source,
                        )
                    )
                    self._update_job(job_id, results=raw_results)
            organizer = self.organizer_factory(status, values[_PRIMARY_SECRET_FIELD])
            organize_kwargs: dict[str, Any] = {
                "source_path": organizer_source,
                "config": text_config,
                "workspace_root": self.project_root / "workspace",
                "output_dir": self.project_root / "output",
                "overwrite": True,
            }
            if reference_track is not None:
                organize_kwargs["reference_track"] = reference_track
            organized = organizer.run(
                **organize_kwargs,
            )
            organized_path = organized.markdown_path.resolve()
            sentences_path = (
                organized.sentences_path.resolve() if organized.sentences_path else None
            )
            results = [
                *raw_results,
                self._result_entry("organized", "整理版（只含字幕）", organized_path),
            ]
            self._finish_job(
                job_id,
                status="succeeded",
                message=(
                    f"处理完成：YouTube 读音参考综合 + {organized.sentence_count} 句整理版"
                    if reference_track is not None
                    else (
                        "处理完成：无可用 YouTube 字幕，已生成 "
                        f"{organized.sentence_count} 句整理版"
                    )
                ),
                result_name=raw_path.name,
                result_path=str(raw_path),
                results=results,
                audio_path=str(audio_path) if audio_path else None,
                sentences_path=str(sentences_path) if sentences_path else None,
            )
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if raw_path is not None and raw_path.is_file():
                if audio_path is not None and audio_path.is_file():
                    try:
                        fallback_dir = self.project_root / "workspace" / "player-fallback"
                        fallback_dir.mkdir(parents=True, exist_ok=True)
                        fallback_name = hashlib.sha256(str(raw_path).encode()).hexdigest()[:20]
                        sentences_path = write_fallback_player_document(
                            raw_path, fallback_dir / f"{fallback_name}.json"
                        )
                    except (OSError, UnicodeError, ValueError, SubmdError):
                        sentences_path = None
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
                    audio_path=str(audio_path) if audio_path else None,
                    sentences_path=str(sentences_path) if sentences_path else None,
                )
                return
            self._finish_job(
                job_id,
                status="failed",
                message="提取失败",
                error=message,
            )

    def _load_source_document(
        self, source_url: str, prefer_corrected: bool = False
    ) -> SubtitleDocument | None:
        video_id = self._youtube_video_id(source_url)
        if not video_id:
            return None
        job_dir = self.project_root / "workspace" / video_id
        corrected_path = job_dir / "corrected_segments.json"
        path = (
            corrected_path
            if prefer_corrected and corrected_path.is_file()
            else job_dir / "segments.json"
        )
        if not path.is_file():
            return None
        try:
            return SubtitleDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

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
            if "（整理版" in path.stem or "综合校正版" in path.stem:
                continue
            saved_url = self._source_url_from_markdown(path)
            if saved_url and self._same_youtube_video(source_url, saved_url):
                return path.resolve()
        return None

    def _find_reusable_audio(self, source_url: str, raw_path: Path) -> Path | None:
        sidecar = raw_path.with_suffix(".m4a")
        if sidecar.is_file() and sidecar.parent == (self.project_root / "output").resolve():
            return sidecar
        with self._lock:
            records = [dict(item) for item in self._history]
        for record in records:
            if not self._same_youtube_video(source_url, str(record.get("source_url") or "")):
                continue
            item = next(
                (result for result in record.get("results", []) if result.get("kind") == "audio"),
                None,
            )
            if item and item.get("path"):
                candidate = Path(str(item["path"])).expanduser().resolve()
                output_root = (self.project_root / "output").resolve()
                if candidate.is_file() and candidate.parent == output_root:
                    return candidate
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
                    audio_path = raw_path.with_suffix(".m4a")
                    if audio_path.is_file():
                        results.append(
                            self._result_entry("audio", "视频音频（M4A）", audio_path)
                        )
                        record["audio_path"] = str(audio_path)
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
            if "（整理版" in path.stem or "综合校正版" in path.stem:
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
            audio_path = path.with_suffix(".m4a")
            if audio_path.is_file():
                results.append(self._result_entry("audio", "视频音频（M4A）", audio_path))
            sentences_path = self._find_organized_sentences(path.name)
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
                    "audio_path": str(audio_path.resolve()) if audio_path.is_file() else None,
                    "sentences_path": (
                        str(sentences_path.resolve()) if sentences_path is not None else None
                    ),
                }
            )
        return records[:100]

    def _find_organized_sentences(self, source_name: str) -> Path | None:
        for path in (self.project_root / "workspace" / "organize").glob(
            "*/organized_segments.json"
        ):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("source_markdown") == source_name:
                return path.resolve()
        return None

    def _persist_history_locked(self) -> None:
        write_json(self.history_path, self._history[:100])

    @staticmethod
    def _validate_required(values: dict[str, str]) -> None:
        required = {
            "SUBMD_YOUTUBE_URL": "YouTube URL",
            "SUBMD_OCR_BASE_URL": "OCR API 地址",
            "SUBMD_OCR_MODEL": "视觉模型名称",
            _PRIMARY_SECRET_FIELD: "API Key",
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

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value
            for key, value in record.items()
            if key not in {"result_path", "results", "audio_path", "sentences_path"}
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
        try:
            self._validated_project_file(record.get("audio_path"), "output")
            self._validated_project_file(record.get("sentences_path"), "workspace")
        except FileNotFoundError:
            pass
        else:
            public["player_url"] = f"/api/player/{record['job_id']}"
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
            elif path == "/api/library":
                self._send_json(self.server.manager.learning_library())
            elif path == "/api/library/export":
                self._send_file(
                    self.server.manager.export_learning_library(), download=True
                )
            elif path.startswith("/api/library/"):
                raw_entry_id = path.removeprefix("/api/library/")
                try:
                    entry_id = int(raw_entry_id)
                except ValueError as exc:
                    raise FileNotFoundError(path) from exc
                self._send_json(self.server.manager.learning_library_entry(entry_id))
            elif path.startswith("/api/player/"):
                parts = path.removeprefix("/api/player/").split("/", maxsplit=1)
                job_id = parts[0]
                if len(parts) == 2 and parts[1] == "audio":
                    self._send_media(self.server.manager.player_audio_file(job_id))
                elif len(parts) == 1:
                    self._send_json(self.server.manager.player_data(job_id))
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "播放器资源不存在")
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
            elif path.startswith("/api/player/") and path.endswith("/resegment"):
                job_id = (
                    path.removeprefix("/api/player/")
                    .removesuffix("/resegment")
                    .strip("/")
                )
                if not job_id:
                    raise ValueError("播放器任务 ID 为空")
                self._send_json(
                    self.server.manager.resegment_sentences(
                        job_id,
                        payload.get("sentence_ids") or [],
                        payload.get("edited_text") or "",
                    )
                )
            elif path.startswith("/api/player/") and path.endswith("/analysis"):
                job_id = path.removeprefix("/api/player/").removesuffix("/analysis").strip("/")
                if not job_id:
                    raise ValueError("播放器任务 ID 为空")
                force = payload.get("force", False)
                if not isinstance(force, bool):
                    raise ValueError("force 必须是布尔值")
                self._send_json(
                    self.server.manager.analyze_sentence(
                        job_id,
                        str(payload.get("sentence_id") or ""),
                        force=force,
                    )
                )
            elif path.startswith("/api/player/") and path.endswith("/library"):
                job_id = path.removeprefix("/api/player/").removesuffix("/library").strip("/")
                if not job_id:
                    raise ValueError("播放器任务 ID 为空")
                self._send_json(
                    self.server.manager.save_learning_item(
                        job_id=job_id,
                        sentence_id=str(payload.get("sentence_id") or ""),
                        item_type=str(payload.get("item_type") or ""),
                        raw_item=payload.get("item"),
                    )
                )
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
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type = f"{content_type}; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if download:
            encoded = quote(path.name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded}")
        self.end_headers()
        self.wfile.write(payload)

    def _send_media(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        file_size = path.stat().st_size
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = int(right) if right else end
            elif right:
                suffix_length = min(file_size, int(right))
                start = file_size - suffix_length
            if start >= file_size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            end = min(end, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        content_type = mimetypes.guess_type(path.name)[0] or "audio/mp4"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers routinely cancel an audio range request after seeking.
                    return
                remaining -= len(chunk)

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
    print(f"KikuFrame UI 已启动：{url}")
    print("关闭这个窗口会停止 UI；字幕提取期间请保持窗口开启。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
