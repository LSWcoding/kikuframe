from __future__ import annotations

import json
from pathlib import Path

import httpx
from PIL import Image

from submd.models import CloudOcrConfig, FrameRef
from submd.ocr.openai_compatible import OpenAICompatibleOcrEngine


def make_frame(path: Path, index: int) -> FrameRef:
    Image.new("RGB", (320, 120), color=(index * 20, 0, 0)).save(path)
    return FrameRef(index=index, timestamp_ms=(index - 1) * 500, path=path)


def test_cloud_ocr_sends_images_and_parses_fenced_json(tmp_path: Path) -> None:
    frames = [
        make_frame(tmp_path / "one.jpg", 1),
        make_frame(tmp_path / "two.jpg", 2),
    ]
    frames[0].reading_reference = ["メリットの1つ目は接客がないところ"]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://vendor.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret-key"
        body = json.loads(request.content)
        assert body["model"] == "vision-ocr"
        assert body["response_format"] == {"type": "json_object"}
        images = [part for part in body["messages"][1]["content"] if part["type"] == "image_url"]
        text_parts = [
            part["text"]
            for part in body["messages"][1]["content"]
            if part["type"] == "text"
        ]
        assert len(images) == 2
        assert all("接客がないところ" not in value for value in text_parts)
        assert all("youtube_reading_reference" not in value for value in text_parts)
        assert all(
            item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in images
        )
        content = """```json
        {"frames":[
          {"frame_id":"000001","text":"  第一行  ","confidence":0.94,"line_count":1},
          {"frame_id":"000002","text":"","confidence":0.99}
        ]}
        ```"""
        return httpx.Response(
            200,
            headers={"x-request-id": "req-test"},
            json={
                "id": "completion-test",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 30},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = CloudOcrConfig(base_url="https://vendor.example/v1", model="vision-ocr")
    engine = OpenAICompatibleOcrEngine(config, "secret-key", client=client)
    result = engine.recognize_batch(frames)

    assert [item.text for item in result.frames] == ["第一行", ""]
    assert [item.line_count for item in result.frames] == [1, 0]
    assert result.request_id == "req-test"
    assert result.usage["prompt_tokens"] == 123


def test_cloud_ocr_retries_rate_limit(tmp_path: Path) -> None:
    frame = make_frame(tmp_path / "frame.jpg", 1)
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"frames":[{"frame_id":"000001","text":"ok","confidence":0.9}]}'
                            )
                        }
                    }
                ]
            },
        )

    config = CloudOcrConfig(
        base_url="https://vendor.example/v1",
        model="vision-ocr",
        max_retries=1,
    )
    engine = OpenAICompatibleOcrEngine(
        config,
        "secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _seconds: None,
    )
    assert engine.recognize_batch([frame]).frames[0].text == "ok"
    assert attempts == 2


def test_cloud_ocr_repairs_missing_frame_id(tmp_path: Path) -> None:
    frames = [make_frame(tmp_path / "one.jpg", 1), make_frame(tmp_path / "two.jpg", 2)]
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        frame_id = "000001" if attempts == 1 else "000002"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                f'{{"frames":[{{"frame_id":"{frame_id}","text":"repaired",'
                                '"confidence":0.9}]}'
                            )
                        }
                    }
                ]
            },
        )

    config = CloudOcrConfig(base_url="https://vendor.example/v1", model="vision-ocr")
    engine = OpenAICompatibleOcrEngine(
        config,
        "secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = engine.recognize_batch(frames)
    assert [item.frame_id for item in result.frames] == ["000001", "000002"]
    assert attempts == 2
    assert result.usage["repair_calls"] == 1


def test_cloud_ocr_retries_a_repair_that_is_still_missing(tmp_path: Path) -> None:
    frame = make_frame(tmp_path / "one.jpg", 1)
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        frames = []
        if attempts == 3:
            frames = [{"frame_id": "000001", "text": "recovered", "confidence": 0.9}]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"frames": frames})}}],
                "usage": {"total_tokens": 5},
            },
        )

    config = CloudOcrConfig(
        base_url="https://vendor.example/v1",
        model="vision-ocr",
        max_retries=3,
    )
    engine = OpenAICompatibleOcrEngine(
        config,
        "secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = engine.recognize_batch([frame])
    assert result.frames[0].text == "recovered"
    assert result.usage["total_tokens"] == 15
    assert result.usage["repair_calls"] == 2
    assert attempts == 3
