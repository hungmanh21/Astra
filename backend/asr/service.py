"""Runs ASR off the event loop and reports timings (SPEC 6.3.2, PLAN T2.6).

The session code calls `await service.transcribe(...)` and gets a Transcription back.
It never touches the registry or an adapter directly.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from backend.asr.base import ASRError
from backend.asr.registry import ASRRegistry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcription:
    model: str
    text: str
    audio_s: float
    asr_ms: float  # inference only (SPEC 6.3.2): excludes queueing and first-use load
    model_load_ms: float  # 0.0 unless this call had to load the model


class TranscriptionService:
    def __init__(self, registry: ASRRegistry, max_workers: int = 1) -> None:
        self._registry = registry
        # One worker serializes GPU jobs.
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="asr")

    async def transcribe(self, model_name: str, audio: np.ndarray) -> Transcription:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor, self._transcribe_blocking, model_name, audio
            )
        except ASRError:
            raise
        except Exception as exc:
            log.exception("transcription with %s failed", model_name)
            raise ASRError("transcription failed") from exc

    def _transcribe_blocking(self, model_name: str, audio: np.ndarray) -> Transcription:
        model, load_ms = self._registry.get(model_name)
        result = model.transcribe(audio, sample_rate=16000, language="en")
        return Transcription(
            model=model_name,
            text=result.text,
            audio_s=result.duration_s,
            asr_ms=result.latency_ms,
            model_load_ms=load_ms,
        )

    async def preload(self, model_name: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._registry.get, model_name)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
