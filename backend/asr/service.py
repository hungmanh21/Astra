"""Runs ASR off the event loop and reports timings (SPEC 6.3.2, PLAN T2.6).

The session code calls `await service.transcribe(...)` and gets a Transcription back.
It never touches the registry or an adapter directly.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.asr.base import ASRError, ASRStream, StreamingASRModel, UnsupportedStrategy
from backend.asr.registry import ASRRegistry
from backend.asr.strategies import (
    BatchStream,
    ChunkedStream,
    default_strategy,
    supported_strategies,
)
from backend.config import StreamingSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcription:
    model: str
    text: str
    audio_s: float
    asr_ms: float  # inference only (SPEC 6.3.2): excludes queueing and first-use load
    model_load_ms: float  # 0.0 unless this call had to load the model


@dataclass(frozen=True)
class ModelStrategies:
    supported: list[str]  # in STRATEGIES order
    default: str  # one of `supported`


@dataclass(frozen=True)
class OpenedStream:
    stream: ASRStream
    model_load_ms: float  # 0.0 unless opening the stream had to load the model


class TranscriptionService:
    def __init__(
        self,
        registry: ASRRegistry,
        max_workers: int = 1,
        streaming: StreamingSettings | None = None,
    ) -> None:
        self._registry = registry
        self._streaming = streaming or StreamingSettings()
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

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking `fn(*args)` on the ASR worker thread and return its result.

        This is the `Runner` that StreamingTranscription calls every stream method through.
        Same error rule as transcribe(): an ASRError passes through, anything else becomes a
        generic ASRError (details go to the log, not to the client).
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, fn, *args)
        except ASRError:
            raise
        except Exception as exc:
            log.exception("ASR call failed")
            raise ASRError("transcription failed") from exc

    def strategies(self, model_name: str) -> ModelStrategies:
        """What `model_name` supports and which strategy it defaults to (SPEC_v1 6.4).

        Does not load the model (uses registry.peek), so the session can list every model's
        strategies in the `session` message at connect time.
        Raises: UnknownASRModel for a name that is not in the registry.
        """
        model = self._registry.peek(model_name)  # raises UnknownASRModel if not found
        return ModelStrategies(
            supported=supported_strategies(model),
            default=default_strategy(model, self._streaming.default_strategy),
        )

    async def open_stream(self, model_name: str, strategy: str) -> OpenedStream:
        """Start the stream for one turn. Loads the model on first use, off the event loop.

        Input:  strategy - one of strategies(model_name).supported.
        Output: OpenedStream. "native" gives the adapter's own open_stream(); "chunked" a
                ChunkedStream; every other strategy a BatchStream until S2 replaces them.
        Raises: UnknownASRModel; UnsupportedStrategy (also for a name that is not a strategy
                at all) BEFORE loading anything, so a bad request never costs a model load;
                ASRError for anything else, wrapped like transcribe().
        """
        if strategy not in self.strategies(model_name).supported:
            raise UnsupportedStrategy(f"{model_name!r} does not support strategy {strategy!r}")
        return await self.run(self._open_stream_blocking, model_name, strategy)

    def _open_stream_blocking(self, model_name: str, strategy: str) -> OpenedStream:
        model, load_ms = self._registry.get(model_name)
        if strategy == "native":
            # strategies() only lists "native" for a StreamingASRModel.
            assert isinstance(model, StreamingASRModel)
            return OpenedStream(model.open_stream(16000, "en"), load_ms)
        if strategy == "chunked":
            return OpenedStream(ChunkedStream(model, chunk_ms=self._streaming.chunk_ms), load_ms)
        return OpenedStream(BatchStream(model), load_ms)

    async def preload(self, model_name: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._registry.get, model_name)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
