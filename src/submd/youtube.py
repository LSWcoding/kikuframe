from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from submd.errors import DownloadError
from submd.json_io import write_json
from submd.models import VideoMetadata, YouTubeCaptionCue, YouTubeCaptionTrack

ProgressCallback = Callable[[str], None]


class _YtDlpLogger:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self.callback = callback

    def debug(self, message: str) -> None:
        if self.callback and message.startswith("[download]"):
            self.callback(message)

    def info(self, message: str) -> None:
        if self.callback:
            self.callback(message)

    def warning(self, message: str) -> None:
        if self.callback:
            self.callback(f"yt-dlp warning: {message}")

    def error(self, message: str) -> None:
        if self.callback:
            self.callback(f"yt-dlp error: {message}")


class YouTubeDownloader:
    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self.callback = callback

    def _base_options(self, cookies_from_browser: str | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "logger": _YtDlpLogger(self.callback),
            "js_runtimes": {"deno": {}},
        }
        if cookies_from_browser:
            browser, separator, profile = cookies_from_browser.partition(":")
            options["cookiesfrombrowser"] = (
                browser.strip(),
                profile.strip() if separator and profile.strip() else None,
                None,
                None,
            )
        return options

    def inspect(self, url: str, cookies_from_browser: str | None = None) -> VideoMetadata:
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL(self._base_options(cookies_from_browser)) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise DownloadError(f"Could not read YouTube metadata: {exc}") from exc
        if not info or info.get("_type") == "playlist":
            raise DownloadError("The URL must point to one public YouTube video")
        return self._metadata_from_info(info, url)

    def download(
        self,
        url: str,
        target_dir: Path,
        max_height: int,
        cookies_from_browser: str | None = None,
    ) -> tuple[Path, VideoMetadata]:
        target_dir.mkdir(parents=True, exist_ok=True)
        options = self._base_options(cookies_from_browser)
        options.update(
            {
                "format": (
                    f"bestvideo[height<={max_height}]+bestaudio/"
                    f"best[height<={max_height}]/bestvideo+bestaudio/best"
                ),
                "outtmpl": str(target_dir / "source.%(ext)s"),
                "overwrites": True,
                "merge_output_format": "mkv",
            }
        )
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                prepared = Path(ydl.prepare_filename(info))
        except Exception as exc:
            raise DownloadError(f"Could not download the video: {exc}") from exc

        candidates: list[Path] = []
        requested = info.get("requested_downloads") or []
        for item in requested:
            filepath = item.get("filepath")
            if filepath:
                candidates.append(Path(filepath))
        candidates.extend([prepared, *target_dir.glob("source.*")])
        video_path = next((path for path in candidates if path.is_file()), None)
        if video_path is None:
            raise DownloadError("yt-dlp finished but the downloaded video file was not found")
        return video_path, self._metadata_from_info(info, url)

    def fetch_caption_track(
        self,
        url: str,
        target_path: Path,
        language_hint: str = "auto",
        cookies_from_browser: str | None = None,
    ) -> YouTubeCaptionTrack | None:
        """Download one original-language YouTube caption track as a reading reference.

        Creator-provided subtitles are preferred over automatic captions. Translated tracks are
        never mixed together: this method deliberately selects only one track.
        """
        if target_path.is_file():
            try:
                return YouTubeCaptionTrack.model_validate_json(
                    target_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                pass
        try:
            import yt_dlp
            from yt_dlp.networking import Request

            with yt_dlp.YoutubeDL(self._base_options(cookies_from_browser)) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise DownloadError("YouTube did not return video information")
                selected = self._select_caption(info, language_hint)
                if selected is None:
                    return None
                language, source, item = selected
                response = ydl.urlopen(Request(str(item["url"])))
                payload = response.read()
        except DownloadError:
            raise
        except Exception as exc:
            raise DownloadError(f"Could not download YouTube captions: {exc}") from exc

        extension = str(item.get("ext") or "").lower()
        try:
            text = payload.decode("utf-8-sig", errors="replace")
            if extension == "json3" or text.lstrip().startswith("{"):
                cues = self._parse_json3(text)
            else:
                cues = self._parse_vtt(text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DownloadError(f"Could not parse YouTube captions: {exc}") from exc
        if not cues:
            return None
        track = YouTubeCaptionTrack(
            video_id=str(info.get("id") or ""),
            language=language,
            source=source,
            name=str(item.get("name") or "") or None,
            cues=cues,
        )
        write_json(target_path, track)
        return track

    @staticmethod
    def _select_caption(
        info: dict[str, Any], language_hint: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        manual = info.get("subtitles") if isinstance(info.get("subtitles"), dict) else {}
        automatic = (
            info.get("automatic_captions")
            if isinstance(info.get("automatic_captions"), dict)
            else {}
        )
        preferred: list[str] = []
        if language_hint and language_hint.lower() != "auto":
            preferred.append(language_hint)
        for value in (info.get("language"), info.get("original_language")):
            if isinstance(value, str) and value.strip():
                preferred.append(value.strip())

        def matching_languages(tracks: dict[str, Any]) -> list[str]:
            keys = [str(key) for key in tracks if str(key) != "live_chat"]
            result: list[str] = []
            for requested in preferred:
                requested_lower = requested.lower()
                matches = [
                    key
                    for key in keys
                    if key.lower() == requested_lower
                    or key.lower().split("-", 1)[0] == requested_lower.split("-", 1)[0]
                ]
                result.extend(matches)
            return list(dict.fromkeys(result))

        def select_from(
            source: str, tracks: dict[str, Any], languages: list[str]
        ) -> tuple[str, str, dict[str, Any]] | None:
            rank = {"json3": 0, "vtt": 1, "srv3": 2, "ttml": 3}
            for language in languages:
                items = tracks.get(language)
                if not isinstance(items, list):
                    continue
                usable = [item for item in items if isinstance(item, dict) and item.get("url")]
                if usable:
                    item = min(usable, key=lambda value: rank.get(str(value.get("ext")), 99))
                    return language, source, item
            return None

        # Language correctness is more important than whether the reference was manually made.
        # Within the same language, creator-provided captions still win.
        for source, tracks in (("manual", manual), ("automatic", automatic)):
            selected = select_from(source, tracks, matching_languages(tracks))
            if selected is not None:
                return selected
        for source, tracks in (("manual", manual), ("automatic", automatic)):
            languages = [str(key) for key in tracks if str(key) != "live_chat"]
            selected = select_from(source, tracks, languages)
            if selected is not None:
                return selected
        return None

    @staticmethod
    def _parse_json3(text: str) -> list[YouTubeCaptionCue]:
        payload = json.loads(text)
        events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(events, list):
            return []
        cues: list[YouTubeCaptionCue] = []
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
                continue
            body = "".join(
                str(segment.get("utf8") or "")
                for segment in event["segs"]
                if isinstance(segment, dict)
            )
            body = re.sub(r"\s+", " ", html.unescape(body)).strip()
            if not body:
                continue
            start = max(0, int(event.get("tStartMs") or 0))
            duration = max(1, int(event.get("dDurationMs") or 1))
            cues.append(
                YouTubeCaptionCue(
                    cue_id=f"yt{len(cues) + 1:06d}",
                    start_ms=start,
                    end_ms=start + duration,
                    text=body,
                )
            )
        return YouTubeDownloader._merge_caption_cues(cues)

    @staticmethod
    def _parse_vtt(text: str) -> list[YouTubeCaptionCue]:
        timestamp = re.compile(
            r"(?P<start>\d{2}:\d{2}(?::\d{2})?\.\d{3})\s*-->\s*"
            r"(?P<end>\d{2}:\d{2}(?::\d{2})?\.\d{3})"
        )
        lines = text.replace("\r\n", "\n").split("\n")
        cues: list[YouTubeCaptionCue] = []
        index = 0
        while index < len(lines):
            match = timestamp.search(lines[index])
            if not match:
                index += 1
                continue
            index += 1
            body_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                clean = re.sub(r"<[^>]+>", "", lines[index])
                body_lines.append(html.unescape(clean))
                index += 1
            body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
            if body:
                cues.append(
                    YouTubeCaptionCue(
                        cue_id=f"yt{len(cues) + 1:06d}",
                        start_ms=YouTubeDownloader._caption_timestamp_ms(match.group("start")),
                        end_ms=YouTubeDownloader._caption_timestamp_ms(match.group("end")),
                        text=body,
                    )
                )
            index += 1
        return YouTubeDownloader._merge_caption_cues(cues)

    @staticmethod
    def _caption_timestamp_ms(value: str) -> int:
        clock, millis = value.rsplit(".", 1)
        parts = [int(part) for part in clock.split(":")]
        if len(parts) == 2:
            parts.insert(0, 0)
        hours, minutes, seconds = parts
        return ((hours * 60 + minutes) * 60 + seconds) * 1000 + int(millis)

    @staticmethod
    def _merge_caption_cues(cues: list[YouTubeCaptionCue]) -> list[YouTubeCaptionCue]:
        merged: list[YouTubeCaptionCue] = []
        for cue in sorted(cues, key=lambda item: (item.start_ms, item.end_ms)):
            if merged and cue.text == merged[-1].text and cue.start_ms <= merged[-1].end_ms + 250:
                merged[-1].end_ms = max(merged[-1].end_ms, cue.end_ms)
            else:
                cue.cue_id = f"yt{len(merged) + 1:06d}"
                merged.append(cue)
        return merged

    @staticmethod
    def _metadata_from_info(info: dict[str, Any], fallback_url: str) -> VideoMetadata:
        duration = float(info.get("duration") or 0)
        if duration <= 0:
            raise DownloadError("The video duration is missing or invalid")
        video_id = str(info.get("id") or "").strip()
        if not video_id:
            raise DownloadError("The video ID is missing")
        title = str(info.get("title") or video_id).strip()
        return VideoMetadata(
            video_id=video_id,
            original_title=title,
            uploader=info.get("uploader") or info.get("channel"),
            duration_ms=round(duration * 1000),
            webpage_url=str(info.get("webpage_url") or fallback_url),
            width=info.get("width"),
            height=info.get("height"),
        )
