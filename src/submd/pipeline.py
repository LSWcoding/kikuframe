from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from submd.exporters import export_markdown
from submd.json_io import write_json
from submd.media import extract_audio, extract_frames, probe_video, select_ocr_frames
from submd.models import (
    ExtractionConfig,
    ExtractionResult,
    OcrObservation,
    SubtitleDocument,
    YouTubeCaptionTrack,
)
from submd.ocr.base import OcrEngine
from submd.ocr.openai_compatible import OpenAICompatibleOcrEngine
from submd.segments import build_segments
from submd.youtube import YouTubeDownloader

StatusCallback = Callable[[str], None]


class BurnedSubtitlePipeline:
    def __init__(
        self,
        downloader: YouTubeDownloader | None = None,
        ocr_engine: OcrEngine | None = None,
        api_key: str | None = None,
        status: StatusCallback | None = None,
    ) -> None:
        self.status = status or (lambda _message: None)
        self.downloader = downloader or YouTubeDownloader(callback=self.status)
        self.ocr_engine = ocr_engine
        self.api_key = api_key

    def run(self, config: ExtractionConfig) -> ExtractionResult:
        self.status("读取视频信息…")
        metadata = self.downloader.inspect(
            config.source_url, cookies_from_browser=config.cookies_from_browser
        )
        job_dir = config.workspace_root.expanduser().resolve() / metadata.video_id
        cache_dir = job_dir / "cache"
        frames_dir = job_dir / "frames"
        preview_dir = job_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = job_dir / "metadata.json"
        config_path = job_dir / "config.json"
        observations_path = job_dir / "observations.json"
        api_calls_path = job_dir / "api_calls.json"
        segments_path = job_dir / "segments.json"
        write_json(metadata_path, metadata)
        write_json(config_path, config)

        cached_videos = sorted(
            path
            for path in cache_dir.glob("source.*")
            if path.is_file() and path.suffix not in {".part", ".ytdl"}
        )
        if cached_videos:
            video_path = cached_videos[0]
            self.status(f"复用已下载视频：{video_path.name}")
        else:
            self.status(f"下载视频（最高 {config.max_height}p）…")
            video_path, metadata = self.downloader.download(
                config.source_url,
                cache_dir,
                config.max_height,
                cookies_from_browser=config.cookies_from_browser,
            )
        media_info = probe_video(video_path)
        metadata.duration_ms = media_info.duration_ms
        metadata.width = media_info.width
        metadata.height = media_info.height
        write_json(metadata_path, metadata)

        self.status(
            f"FFmpeg 抽帧：{media_info.width}×{media_info.height}，"
            f"{config.sample_fps:g} FPS，ROI={config.roi.model_dump()}"
        )
        frames = extract_frames(video_path, frames_dir, config.roi, config.sample_fps)
        shutil.copy2(frames[0].path, preview_dir / "roi-first-frame.jpg")
        selected = select_ocr_frames(frames, config.change_threshold, config.max_ocr_interval)
        self.status(f"抽取 {len(frames)} 帧，选择 {len(selected)} 帧执行云端 OCR")

        engine = self.ocr_engine or OpenAICompatibleOcrEngine(
            config=config.ocr,
            api_key=self.api_key or "",
            language_hint=config.language,
        )
        batches = [
            selected[index : index + config.ocr.batch_size]
            for index in range(0, len(selected), config.ocr.batch_size)
        ]
        self.status(
            f"云端 OCR：{len(selected)} 帧，{len(batches)} 个 API 请求，模型={config.ocr.model}"
        )

        observations = self._load_observations(observations_path, selected, config.ocr.model)
        api_calls = self._load_api_calls(api_calls_path)
        completed_ids = {observation.frame_index for observation in observations}
        pending = [frame for frame in selected if frame.index not in completed_ids]
        batches = [
            pending[index : index + config.ocr.batch_size]
            for index in range(0, len(pending), config.ocr.batch_size)
        ]
        if observations:
            self.status(
                f"断点续跑：复用 {len(observations)} 帧结果，"
                f"剩余 {len(pending)} 帧、{len(batches)} 个请求"
            )
        completed_frames = len(observations)
        saved_batch_indexes = [observation.batch_index for observation in observations]
        saved_batch_indexes.extend(
            int(call.get("batch_index", 0))
            for call in api_calls
            if isinstance(call.get("batch_index"), int)
        )
        first_batch_index = max(saved_batch_indexes, default=0) + 1
        total_batch_count = first_batch_index - 1 + len(batches)
        for batch_index, batch in enumerate(batches, start=first_batch_index):
            batch_result = engine.recognize_batch(batch)
            result_by_id = {item.frame_id: item for item in batch_result.frames}
            for frame in batch:
                item = result_by_id[f"{frame.index:06d}"]
                observations.append(
                    OcrObservation(
                        timestamp_ms=frame.timestamp_ms,
                        frame_index=frame.index,
                        frame_file=frame.path.name,
                        diff_score=frame.diff_score,
                        model=config.ocr.model,
                        batch_index=batch_index,
                        request_id=batch_result.request_id,
                        text=item.text,
                        confidence=item.confidence,
                    )
                )
            completed_frames += len(batch)
            api_calls.append(
                {
                    "batch_index": batch_index,
                    "frame_ids": [f"{frame.index:06d}" for frame in batch],
                    "request_id": batch_result.request_id,
                    "usage": batch_result.usage,
                }
            )
            write_json(observations_path, observations)
            write_json(api_calls_path, api_calls)
            self.status(
                f"云端 OCR 进度 {completed_frames}/{len(selected)} "
                f"（请求 {batch_index}/{total_batch_count}）"
            )

        sample_interval_ms = max(1, round(1000 / config.sample_fps))
        segments = build_segments(
            observations=observations,
            duration_ms=metadata.duration_ms,
            similarity_threshold=config.similarity_threshold,
            review_confidence=config.min_confidence,
            sample_interval_ms=sample_interval_ms,
        )
        document = SubtitleDocument(video=metadata, config=config, segments=segments)
        write_json(segments_path, document)
        markdown_path = export_markdown(
            document, config.output_dir.expanduser().resolve(), config.overwrite
        )
        self.status("保存视频音频…")
        audio_path = extract_audio(video_path, markdown_path.with_suffix(".m4a"))

        if not config.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
            shutil.rmtree(frames_dir, ignore_errors=True)

        self.status(f"完成：{len(segments)} 个字幕段")
        return ExtractionResult(
            metadata_path=metadata_path,
            config_path=config_path,
            observations_path=observations_path,
            api_calls_path=api_calls_path,
            segments_path=segments_path,
            markdown_path=markdown_path,
            segment_count=len(segments),
            observation_count=len(observations),
            audio_path=audio_path,
        )

    def ensure_audio(self, config: ExtractionConfig, raw_markdown_path: Path) -> Path:
        """Create the audio sidecar for a reusable OCR result without rerunning OCR."""
        raw_markdown_path = raw_markdown_path.expanduser().resolve()
        audio_path = raw_markdown_path.with_suffix(".m4a")
        if audio_path.is_file():
            self.status(f"复用已保存音频：{audio_path.name}")
            return audio_path

        self.status("已有字幕缺少音频；只下载音视频源，不会重新执行视觉 OCR…")
        metadata = self.downloader.inspect(
            config.source_url, cookies_from_browser=config.cookies_from_browser
        )
        cache_dir = config.workspace_root.expanduser().resolve() / metadata.video_id / "cache"
        cached_videos = sorted(
            path
            for path in cache_dir.glob("source.*")
            if path.is_file() and path.suffix not in {".part", ".ytdl"}
        )
        if cached_videos:
            video_path = cached_videos[0]
            self.status(f"复用已下载视频：{video_path.name}")
        else:
            video_path, _metadata = self.downloader.download(
                config.source_url,
                cache_dir,
                config.max_height,
                cookies_from_browser=config.cookies_from_browser,
            )
        result = extract_audio(video_path, audio_path)
        if not config.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        return result

    def ensure_youtube_captions(
        self,
        config: ExtractionConfig,
        video_id: str,
        language_hint: str,
    ) -> tuple[YouTubeCaptionTrack | None, Path]:
        path = config.workspace_root.expanduser().resolve() / video_id / "youtube_captions.json"
        self.status("读取 YouTube 字幕作为读音参考…")
        track = self.downloader.fetch_caption_track(
            config.source_url,
            path,
            language_hint=language_hint,
            cookies_from_browser=config.cookies_from_browser,
        )
        if track is None:
            self.status("该视频没有可用的 YouTube 字幕；继续使用纯 OCR 结果")
        else:
            source_label = "人工字幕" if track.source == "manual" else "自动字幕"
            self.status(
                f"已取得 YouTube {source_label}（{track.language}，{len(track.cues)} 条）"
            )
        return track, path

    @staticmethod
    def _load_observations(path, selected, model: str) -> list[OcrObservation]:
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected_ids = {frame.index for frame in selected}
        observations = [
            OcrObservation.model_validate(item)
            for item in payload
            if item.get("frame_index") in selected_ids
        ]
        if any(item.model != model for item in observations):
            return []
        unique: dict[int, OcrObservation] = {}
        for observation in observations:
            unique[observation.frame_index] = observation
        return sorted(unique.values(), key=lambda item: item.timestamp_ms)

    @staticmethod
    def _load_api_calls(path) -> list[dict[str, object]]:
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
