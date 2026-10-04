"""Shared test helpers for the backend tests."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from backend.asr.registry import ASRModelConfig
from backend.asr.service import Transcription
from backend.audio import SAMPLE_RATE
from backend.config import LLMSettings, Settings

MODELS = ["whisper-large-v3", "other-model"]


def make_settings(
    debug_dir: Path | None = None,
    max_turn_seconds: float = 30,
    models: list[str] = MODELS,
    adapter: str = "tests.test_asr_registry.FakeASR",
    max_history_tokens: int = 1000,
) -> Settings:
    """Settings built by hand. Debug saving is on only when `debug_dir` is given."""
    cfg = ASRModelConfig(adapter=adapter, model_id="fake/model")
    return Settings(
        asr_models={name: cfg for name in models},
        default_asr_model=models[0],
        llm=LLMSettings(
            model="test/model",
            api_base=None,
            system_prompt="be brief",
            max_history_tokens=max_history_tokens,
        ),
        max_turn_seconds=max_turn_seconds,
        debug_save_audio=debug_dir is not None,
        debug_audio_dir=debug_dir or Path("unused"),
    )


def pcm(seconds: float, freq: float = 440.0) -> bytes:
    """A sine wave as raw PCM16 little-endian bytes at 16 kHz."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * 10_000).astype("<i2").tobytes()


def chunks(data: bytes, size: int = 1600) -> list[bytes]:
    """Split into WebSocket-frame-sized pieces (1600 bytes = 50 ms)."""
    return [data[i : i + size] for i in range(0, len(data), size)]


class FakeClock:
    """A clock the test moves by hand. Pass it to Session(clock=...) and to the fakes."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTranscriber:
    """Stands in for TranscriptionService in session tests.

    `gate`: an asyncio.Event; when given, transcribe() waits for it, so a test can hold a turn
    in the WORKING stage. `error`: raised instead of returning a result.
    """

    def __init__(
        self,
        text: str = "hello world",
        asr_ms: float = 12.5,
        model_load_ms: float = 0.0,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
        clock: FakeClock | None = None,
        takes_s: float = 0.0,
    ) -> None:
        self.clock = clock
        self.takes_s = takes_s  # how far transcribe() moves the fake clock
        self.text = text
        self.asr_ms = asr_ms
        self.model_load_ms = model_load_ms
        self.error = error
        self.gate = gate
        self.calls: list[tuple[str, np.ndarray]] = []
        self.cancelled = False
        self.shut_down = False

    async def transcribe(self, model_name: str, audio: np.ndarray) -> Transcription:
        self.calls.append((model_name, audio))
        if self.clock is not None:
            self.clock.advance(self.takes_s)
        try:
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        return Transcription(
            model=model_name,
            text=self.text,
            audio_s=len(audio) / SAMPLE_RATE,
            asr_ms=self.asr_ms,
            model_load_ms=self.model_load_ms,
        )

    def shutdown(self) -> None:
        self.shut_down = True


class FakeLLM:
    """Stands in for LLMClient in session tests.

    `deltas` are yielded in order. `error` is raised after `error_after` deltas (0 = before the
    first one). `gate`: an asyncio.Event awaited after the first delta, to hold a turn
    mid-stream. `calls` records the messages of every request.
    """

    def __init__(
        self,
        deltas: tuple[str, ...] = ("Hi", " there"),
        error: Exception | None = None,
        error_after: int = 0,
        gate: asyncio.Event | None = None,
        clock: FakeClock | None = None,
        delays: tuple[float, ...] = (),
        model: str = "test/model",
    ) -> None:
        self.model = model  # what the session logs as `llm_model`
        self.clock = clock
        self.delays = delays  # fake-clock seconds that pass before each delta
        self.deltas = deltas
        self.error = error
        self.error_after = error_after
        self.gate = gate
        self.calls: list[list[dict[str, str]]] = []
        self.cancelled = False

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append([dict(m) for m in messages])
        try:
            for i, delta in enumerate(self.deltas):
                if self.error is not None and i == self.error_after:
                    raise self.error
                if self.clock is not None and i < len(self.delays):
                    self.clock.advance(self.delays[i])
                yield delta
                if i == 0 and self.gate is not None:
                    await self.gate.wait()
            if self.error is not None and self.error_after >= len(self.deltas):
                raise self.error
        except asyncio.CancelledError:
            self.cancelled = True
            raise
