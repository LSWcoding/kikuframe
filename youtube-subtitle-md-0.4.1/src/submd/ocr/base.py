from __future__ import annotations

from typing import Protocol

from submd.models import CloudOcrBatchResult, FrameRef


class OcrEngine(Protocol):
    def recognize_batch(self, frames: list[FrameRef]) -> CloudOcrBatchResult:
        """Recognize subtitle text for a batch of cropped video frames."""
