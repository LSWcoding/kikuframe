from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from submd import __version__
from submd.errors import SubmdError
from submd.models import CloudOcrConfig, ExtractionConfig, Roi, TextLlmConfig
from submd.organize import SubtitleOrganizer
from submd.pipeline import BurnedSubtitlePipeline
from submd.web import serve_ui


def load_environment() -> None:
    """Load local secrets without overriding explicitly exported variables."""
    current_env = Path.cwd() / ".env"
    project_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(current_env, override=False)
    if project_env != current_env:
        load_dotenv(project_env, override=False)


load_environment()

app = typer.Typer(
    name="submd",
    help="KikuFrame：把日语视频转换为逐句对齐、可循环和可分析的学习材料。",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="显示版本。"),
    ] = None,
) -> None:
    del version


@app.command()
def extract(
    url: Annotated[
        str | None,
        typer.Argument(
            help="单个公开 YouTube 视频 URL。",
            envvar="SUBMD_YOUTUBE_URL",
        ),
    ] = None,
    output_dir: Annotated[Path, typer.Option(help="Markdown 输出目录。")] = Path("output"),
    workspace_dir: Annotated[Path, typer.Option(help="JSON 中间结果和缓存目录。")] = Path(
        "workspace"
    ),
    roi: Annotated[str, typer.Option(help="归一化字幕区域：x,y,width,height。")] = "0,0.65,1,0.35",
    language: Annotated[str, typer.Option(help="字幕语言提示，例如 auto、zh、en、ja。")] = "auto",
    api_base: Annotated[
        str | None,
        typer.Option(
            help="OpenAI-compatible API 基础地址。",
            envvar="SUBMD_OCR_BASE_URL",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="云厂商视觉模型名称。", envvar="SUBMD_OCR_MODEL"),
    ] = None,
    api_key_env: Annotated[
        str,
        typer.Option(help="保存 API Key 的环境变量名；Key 本身不会写入文件。"),
    ] = "SUBMD_OCR_API_KEY",
    cookies_from_browser: Annotated[
        str | None,
        typer.Option(
            help="允许 yt-dlp 读取浏览器 Cookie，例如 chrome 或 chrome:Profile 1。",
            envvar="SUBMD_YOUTUBE_COOKIES_FROM_BROWSER",
        ),
    ] = None,
    batch_size: Annotated[int, typer.Option(help="每个云 API 请求包含的关键帧数量。")] = 4,
    api_timeout: Annotated[float, typer.Option(help="每个云 API 请求的超时秒数。")] = 120.0,
    api_retries: Annotated[int, typer.Option(help="遇到限流或服务端错误时的重试次数。")] = 3,
    image_max_side: Annotated[int, typer.Option(help="发送到云端前的图片最大边长。")] = 1600,
    jpeg_quality: Annotated[int, typer.Option(help="发送到云端的 JPEG 质量。")] = 88,
    json_mode: Annotated[
        bool,
        typer.Option(
            "--json-mode/--no-json-mode",
            help="请求云厂商启用 JSON 输出；不兼容时可关闭。",
        ),
    ] = True,
    sample_fps: Annotated[float, typer.Option(help="FFmpeg 每秒采样帧数。")] = 10.0,
    change_threshold: Annotated[
        float, typer.Option(help="兼容旧配置保留；字幕帧现在会按文字轮廓自动去重。")
    ] = 0.012,
    max_ocr_interval: Annotated[
        float, typer.Option(help="YouTube 字幕覆盖区间内的本地安全探测间隔。")
    ] = 0.8,
    min_confidence: Annotated[
        float, typer.Option(help="低于此模型自报置信度时标记人工复核。")
    ] = 0.55,
    similarity_threshold: Annotated[
        float, typer.Option(help="相邻文本合并相似度，范围 0–100。")
    ] = 82.0,
    max_height: Annotated[int, typer.Option(help="下载视频流的最高高度。")] = 720,
    keep_cache: Annotated[
        bool, typer.Option("--keep-cache/--no-keep-cache", help="保留视频和抽取帧。")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite/--no-overwrite", help="覆盖同名 Markdown。")
    ] = False,
) -> None:
    try:
        if not url:
            raise ValueError("缺少 YouTube URL：请设置位置参数或 SUBMD_YOUTUBE_URL")
        if not api_base:
            raise ValueError("缺少 API 地址：请设置 --api-base 或 SUBMD_OCR_BASE_URL")
        if not model:
            raise ValueError("缺少模型名：请设置 --model 或 SUBMD_OCR_MODEL")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"环境变量 {api_key_env} 未设置或为空")
        parsed_roi = Roi.parse(roi)
        cloud_ocr = CloudOcrConfig(
            base_url=api_base,
            model=model,
            batch_size=batch_size,
            timeout_seconds=api_timeout,
            max_retries=api_retries,
            image_max_side=image_max_side,
            jpeg_quality=jpeg_quality,
            json_mode=json_mode,
        )
        config = ExtractionConfig(
            source_url=url,
            cookies_from_browser=cookies_from_browser,
            workspace_root=workspace_dir,
            output_dir=output_dir,
            roi=parsed_roi,
            language=language,
            ocr=cloud_ocr,
            sample_fps=sample_fps,
            change_threshold=change_threshold,
            max_ocr_interval=max_ocr_interval,
            min_confidence=min_confidence,
            similarity_threshold=similarity_threshold,
            max_height=max_height,
            keep_cache=keep_cache,
            overwrite=overwrite,
        )
        result = BurnedSubtitlePipeline(status=typer.echo, api_key=api_key).run(config)
    except (SubmdError, ValidationError, ValueError) as exc:
        typer.secho(f"错误：{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Markdown：{result.markdown_path}", fg=typer.colors.GREEN)
    if result.audio_path:
        typer.echo(f"音频：{result.audio_path}")
    typer.echo(f"segments.json：{result.segments_path}")
    typer.echo(f"api_calls.json：{result.api_calls_path}")
    typer.echo(f"识别观察 {result.observation_count} 个，合并字幕段 {result.segment_count} 个。")


@app.command()
def organize(
    source: Annotated[
        Path,
        typer.Argument(help="由 extract 生成的带时间戳字幕 Markdown。"),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(help="整理版输出目录；默认与输入文件相同。"),
    ] = None,
    workspace_dir: Annotated[Path, typer.Option(help="断句检查点目录。")] = Path("workspace"),
    api_base: Annotated[
        str | None,
        typer.Option(
            help="文本模型 API 地址；默认复用 SUBMD_OCR_BASE_URL。",
            envvar="SUBMD_TEXT_BASE_URL",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            help="语义断句模型；默认复用 SUBMD_OCR_MODEL。",
            envvar="SUBMD_TEXT_MODEL",
        ),
    ] = None,
    api_key_env: Annotated[
        str,
        typer.Option(help="保存 API Key 的环境变量名。"),
    ] = "SUBMD_OCR_API_KEY",
    chunk_size: Annotated[int, typer.Option(help="每次交给文本模型判断的字幕单元数。")] = 100,
    context_size: Annotated[int, typer.Option(help="每个分块前后携带的上下文单元数。")] = 10,
    api_timeout: Annotated[float, typer.Option(help="每个文本模型请求的超时秒数。")] = 120.0,
    api_retries: Annotated[int, typer.Option(help="限流、网络错误或服务端错误的重试次数。")] = 3,
    json_mode: Annotated[
        bool,
        typer.Option(
            "--json-mode/--no-json-mode",
            help="请求云厂商启用 JSON 输出。",
        ),
    ] = True,
    overwrite: Annotated[
        bool, typer.Option("--overwrite/--no-overwrite", help="覆盖已有整理版。")
    ] = False,
) -> None:
    try:
        resolved_base = api_base or os.environ.get("SUBMD_OCR_BASE_URL")
        resolved_model = model or os.environ.get("SUBMD_OCR_MODEL")
        if not resolved_base:
            raise ValueError(
                "缺少文本模型 API 地址：请设置 SUBMD_TEXT_BASE_URL 或 SUBMD_OCR_BASE_URL"
            )
        if not resolved_model:
            raise ValueError("缺少文本模型名：请设置 SUBMD_TEXT_MODEL 或 SUBMD_OCR_MODEL")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"环境变量 {api_key_env} 未设置或为空")
        config = TextLlmConfig(
            base_url=resolved_base,
            model=resolved_model,
            chunk_size=chunk_size,
            context_size=context_size,
            timeout_seconds=api_timeout,
            max_retries=api_retries,
            json_mode=json_mode,
        )
        result = SubtitleOrganizer(status=typer.echo, api_key=api_key).run(
            source_path=source,
            config=config,
            workspace_root=workspace_dir,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    except (SubmdError, ValidationError, ValueError) as exc:
        typer.secho(f"错误：{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"整理版 Markdown：{result.markdown_path}", fg=typer.colors.GREEN)
    if result.sentences_path:
        typer.echo(f"句子时间轴：{result.sentences_path}")
    typer.echo(f"断句检查点：{result.checkpoint_path}")
    typer.echo(
        f"原字幕片段 {result.source_fragment_count} 个，整理为 {result.sentence_count} 句话；"
        f"新请求 {result.api_call_count} 个，复用检查点 {result.reused_chunk_count} 个。"
    )


@app.command()
def ui(
    host: Annotated[str, typer.Option(help="本地 UI 监听地址。")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="本地 UI 端口。")] = 8765,
    open_browser: Annotated[
        bool,
        typer.Option("--open-browser/--no-open-browser", help="启动后自动打开浏览器。"),
    ] = True,
) -> None:
    """启动本地网页界面。"""
    try:
        serve_ui(Path.cwd(), host=host, port=port, open_browser=open_browser)
    except (SubmdError, ValueError) as exc:
        typer.secho(f"错误：{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
