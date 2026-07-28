from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Callable
from io import BytesIO
from typing import Any

import httpx
from PIL import Image
from pydantic import ValidationError

from submd import __version__
from submd.errors import OcrError
from submd.models import (
    CloudOcrBatchResult,
    CloudOcrConfig,
    CloudOcrFrameResult,
    FrameRef,
)
from submd.text import normalize_text

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_SYSTEM_PROMPT = """You are a precise OCR engine for burned-in video subtitles.
Each image is already cropped to the configured subtitle region.

Rules:
1. Transcribe only the burned-in subtitle text visible in each frame.
2. Do not translate, summarize, explain, correct grammar, or infer missing words.
3. Preserve line breaks inside a subtitle using \\n.
4. Ignore logos, watermarks, UI, decorative titles, and unrelated background text.
5. If no burned-in subtitle is visible, return an empty text string.
6. Confidence is only a visual-legibility score from 0 to 1, not semantic certainty.
7. Return exactly one result for every supplied frame_id.

Return JSON only:
{"frames":[{"frame_id":"000001","text":"verbatim subtitle or empty","confidence":0.95}]}
"""


class OpenAICompatibleOcrEngine:
    """Cloud vision OCR through an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        config: CloudOcrConfig,
        api_key: str,
        language_hint: str = "auto",
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise OcrError("OCR API key is empty")
        self.config = config
        self._api_key = api_key.strip()
        self.language_hint = language_hint
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    def recognize_batch(self, frames: list[FrameRef]) -> CloudOcrBatchResult:
        if not frames:
            return CloudOcrBatchResult(frames=[])

        expected_frames = {self._frame_id(frame): frame for frame in frames}
        recognized: dict[str, CloudOcrFrameResult] = {}
        combined = CloudOcrBatchResult(frames=[])
        pending = list(frames)
        max_repair_calls = max(1, self.config.max_retries)
        repair_calls = 0

        while pending:
            result = self._recognize_once(pending)
            for item in result.frames:
                if item.frame_id in expected_frames and item.frame_id not in recognized:
                    recognized[item.frame_id] = item
            combined.request_id = (
                ",".join(value for value in (combined.request_id, result.request_id) if value)
                or None
            )
            if combined.usage:
                combined.usage = self._merge_usage(combined.usage, result.usage)
            else:
                combined.usage = dict(result.usage)

            missing = [frame for frame in frames if self._frame_id(frame) not in recognized]
            if not missing:
                break
            if repair_calls >= max_repair_calls:
                missing_ids = ", ".join(self._frame_id(frame) for frame in missing)
                raise OcrError(
                    f"Cloud OCR omitted frames after {repair_calls} repair calls: {missing_ids}"
                )
            repair_calls += 1
            pending = missing

        combined.usage["repair_calls"] = repair_calls
        combined.frames = [recognized[self._frame_id(frame)] for frame in frames]
        return combined

    def _recognize_once(self, frames: list[FrameRef]) -> CloudOcrBatchResult:
        """Send one request; completeness is enforced by recognize_batch."""

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Subtitle language hint: {self.language_hint}. "
                    f"Process these {len(frames)} frames in the given order."
                ),
            }
        ]
        for frame in frames:
            frame_id = self._frame_id(frame)
            content.append({"type": "text", "text": f"frame_id={frame_id}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(frame.path)},
                }
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._post_with_retries(payload)
        return self._parse_response(response)

    @staticmethod
    def _merge_usage(original: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
        merged = dict(original)
        for key, value in repair.items():
            if isinstance(value, (int, float)) and isinstance(merged.get(key, 0), (int, float)):
                merged[key] = merged.get(key, 0) + value
            elif key not in merged:
                merged[key] = value
        return merged

    def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        endpoint = self._endpoint()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"youtube-subtitle-md/{__version__}",
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code not in {408, 409, 429} and response.status_code < 500:
                    if response.is_error:
                        body = response.text[:500].replace("\n", " ")
                        raise OcrError(
                            f"Cloud OCR API returned HTTP {response.status_code}: {body}"
                        )
                    return response
                last_error = OcrError(f"Cloud OCR API returned HTTP {response.status_code}")
                retry_after = self._retry_after_seconds(response)
            except OcrError:
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                retry_after = None

            if attempt < self.config.max_retries:
                self._sleep(retry_after if retry_after is not None else min(2**attempt, 8))

        message = str(last_error) if last_error else "unknown request failure"
        raise OcrError(f"Cloud OCR request failed after retries: {message}") from last_error

    @property
    def _api_key(self) -> str:
        # Kept private and never included in model/config serialization.
        return self.__api_key

    @_api_key.setter
    def _api_key(self, value: str) -> None:
        self.__api_key = value

    def _endpoint(self) -> str:
        if self.config.base_url.endswith("/chat/completions"):
            return self.config.base_url
        return f"{self.config.base_url}/chat/completions"

    def _image_data_url(self, path: Any) -> str:
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail(
                    (self.config.image_max_side, self.config.image_max_side),
                    Image.Resampling.LANCZOS,
                )
                buffer = BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=self.config.jpeg_quality,
                    optimize=True,
                )
        except OSError as exc:
            raise OcrError(f"Could not prepare OCR frame {path}: {exc}") from exc
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _parse_response(self, response: httpx.Response) -> CloudOcrBatchResult:
        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            text = self._content_text(content)
            payload = self._extract_json(text)
            frames = [
                CloudOcrFrameResult(
                    frame_id=str(item["frame_id"]),
                    text=normalize_text(str(item.get("text") or "")),
                    confidence=item.get("confidence", 0.5),
                )
                for item in payload["frames"]
            ]
            return CloudOcrBatchResult(
                frames=frames,
                request_id=response.headers.get("x-request-id") or envelope.get("id"),
                usage=envelope.get("usage") or {},
            )
        except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
            raise OcrError(f"Could not parse cloud OCR JSON response: {exc}") from exc

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        raise ValueError("message content is not text")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        clean = _JSON_FENCE.sub("", text.strip()).strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(clean[start : end + 1])
        if not isinstance(value, dict) or not isinstance(value.get("frames"), list):
            raise ValueError("response JSON must contain a frames array")
        return value

    @staticmethod
    def _frame_id(frame: FrameRef) -> str:
        return f"{frame.index:06d}"

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            return min(30.0, max(0.0, float(value)))
        except ValueError:
            return None
