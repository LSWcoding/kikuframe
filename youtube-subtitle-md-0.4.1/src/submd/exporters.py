from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from submd.models import SubtitleDocument

_UNSAFE_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize_filename(title: str, maximum_length: int = 180) -> str:
    normalized = unicodedata.normalize("NFKC", title)
    normalized = _UNSAFE_FILENAME.sub("_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    if not normalized:
        normalized = "untitled"
    return normalized[:maximum_length].rstrip(" .") or "untitled"


def format_timestamp(milliseconds: int) -> str:
    total_seconds, millis = divmod(max(0, milliseconds), 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def export_markdown(document: SubtitleDocument, output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(document.video.original_title)
    path = output_dir / f"{stem}.md"
    if path.exists() and not overwrite:
        path = output_dir / f"{stem} [{document.video.video_id}].md"
        counter = 2
        while path.exists():
            path = output_dir / f"{stem} [{document.video.video_id}-{counter}].md"
            counter += 1

    metadata = document.video
    lines = [
        "---",
        f"title: {json.dumps(metadata.original_title, ensure_ascii=False)}",
        f"video_id: {json.dumps(metadata.video_id)}",
        f"source_url: {json.dumps(metadata.webpage_url, ensure_ascii=False)}",
        'subtitle_source: "burned_ocr"',
        'ocr_backend: "cloud_vlm"',
        f"ocr_model: {json.dumps(document.config.ocr.model, ensure_ascii=False)}",
        f"language: {json.dumps(document.config.language)}",
        f"segments: {len(document.segments)}",
        "---",
        "",
        f"# {metadata.original_title}",
        "",
    ]
    if not document.segments:
        lines.append("_在配置的字幕区域中未检测到文字。_")
    else:
        for segment in document.segments:
            time_range = f"{format_timestamp(segment.start_ms)}–{format_timestamp(segment.end_ms)}"
            text = segment.text.replace("\n", "<br>")
            review = " ⚠️" if segment.needs_review else ""
            lines.append(f"- [{time_range}] {text}{review}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
