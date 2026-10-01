"""Whisper adapter on Hugging Face transformers (PLAN T2.3).

Model: openai/whisper-large-v3. English only in v0. In fp16 it needs roughly
3-4 GB, so it fits on the 12 GB RTX 3060.
"""

import time
from typing import Any

import numpy as np

from backend.asr.base import ASRError, TranscriptResult


class WhisperASR:
    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "auto",  # "auto" | "cuda" | "cpu"
        dtype: str = "float16",
    ) -> None:
        import torch

        self.name = name
        self.model_id = model_id
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # fp16 is not supported on CPU.
        self.dtype = "float32" if self.device == "cpu" else dtype
        self._pipe = None

    def load(self) -> None:
        try:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                dtype=getattr(torch, self.dtype),
            )
            # The first CUDA call is much slower; keep it out of the first real turn's asr_ms.
            self._run(np.zeros(16000, dtype=np.float32), 16000)
        except Exception as exc:
            self._pipe = None
            raise ASRError(f"failed to load {self.model_id}: {exc}") from exc

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptResult:
        if self._pipe is None:
            raise ASRError(f"{self.name} is not loaded")
        try:
            start = time.perf_counter()
            out = self._run(audio, sample_rate)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            raise ASRError(f"{self.name} inference failed: {exc}") from exc
        return TranscriptResult(
            text=out["text"].strip(),
            language="en",
            duration_s=len(audio) / sample_rate,
            latency_ms=latency_ms,
        )

    def _run(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        import torch

        with torch.inference_mode():
            return self._pipe(
                {"raw": audio, "sampling_rate": sample_rate},
                generate_kwargs={"language": "en", "task": "transcribe"},
            )
