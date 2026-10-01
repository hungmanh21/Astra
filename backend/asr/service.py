"""Runs ASR off the event loop and reports timings (SPEC 6.3.2, PLAN T2.6).

The session code calls `await service.transcribe(...)` and gets a Transcription back.
It never touches the registry or an adapter directly.
"""

from dataclasses import dataclass

import numpy as np

from backend.asr.registry import ASRRegistry


@dataclass(frozen=True)
class Transcription:
    model: str
    text: str
    audio_s: float
    asr_ms: float  # inference only (SPEC 6.3.2): excludes queueing and first-use load
    model_load_ms: float  # 0.0 unless this call had to load the model


class TranscriptionService:
    def __init__(self, registry: ASRRegistry, max_workers: int = 1) -> None:
        """TODO(you): keep the registry and create the worker pool.

        Input:  registry - the ASRRegistry to get models from.
                max_workers - threads for ASR. 1 serializes GPU jobs, which is what you
                want on one GPU.
        Output: None.
        State to keep: the registry and a ThreadPoolExecutor(max_workers=max_workers).
        """
        raise NotImplementedError

    async def transcribe(self, model_name: str, audio: np.ndarray) -> Transcription:
        """TODO(you): run ASR in the worker thread without blocking the event loop.

        Input:  model_name - registry key chosen by the user (from `start_turn`).
                audio - 1-D float32 array in [-1, 1], 16 kHz mono (from TurnAudioBuffer).
        Output: Transcription (language is fixed to "en" in v0).
        Raises: ASRError, including UnknownASRModel, which pass through unchanged.
                Any other exception is wrapped in ASRError, with the original logged
                (with its traceback) so the user-facing message leaks no internals.

        Steps:
          1. loop = asyncio.get_running_loop()
          2. await loop.run_in_executor(self._executor, self._transcribe_blocking, ...)
          3. Catch non-ASRError exceptions, log them, raise ASRError("transcription failed").
        """
        raise NotImplementedError

    def _transcribe_blocking(self, model_name: str, audio: np.ndarray) -> Transcription:
        """TODO(you): the work that runs in the worker thread.

        Input:  same as transcribe().
        Output: Transcription with
                  asr_ms = TranscriptResult.latency_ms,
                  model_load_ms = the load_ms returned by registry.get(),
                  audio_s = TranscriptResult.duration_s.
        Raises: whatever the registry or the adapter raises (transcribe() handles it).

        Steps:
          1. model, load_ms = registry.get(model_name)   # may load the model
          2. result = model.transcribe(audio, sample_rate=16000, language="en")
          3. Build and return the Transcription.
        """
        raise NotImplementedError

    async def preload(self, model_name: str) -> None:
        """TODO(you), optional: load a model ahead of the first turn (e.g. at server startup).

        Input:  model_name - registry key.
        Output: None. Afterwards registry.is_loaded(model_name) is True.
        Raises: ASRError if loading fails.
        Hint: run registry.get(...) in the executor, same as transcribe().
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """TODO(you): stop the worker pool when the server shuts down.

        Input:  none.
        Output: None.
        """
        raise NotImplementedError
