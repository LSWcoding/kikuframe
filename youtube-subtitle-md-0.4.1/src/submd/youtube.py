from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from submd.errors import DownloadError
from submd.models import VideoMetadata

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
                    f"bestvideo[height<={max_height}]/best[height<={max_height}]/bestvideo/best"
                ),
                "outtmpl": str(target_dir / "source.%(ext)s"),
                "overwrites": True,
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
