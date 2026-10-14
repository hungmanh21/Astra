"""Shared test helpers for the backend tests."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from backend.asr.base import (
    ASRError,
    PartialResult,
    TranscriptResult,
    UnknownASRModel,
    UnsupportedStrategy,
)
from backend.asr.registry import ASRModelConfig
from backend.asr.service import ModelStrategies, OpenedStream
from backend.audio import SAMPLE_RATE
from backend.config import LLMSettings, Settings, StreamingSettings

MODELS = ["whisper-large-v3", "other-model"]


def make_settings(
    debug_dir: Path | None = None,
    max_turn_seconds: float = 30,
    models: list[str] = MODELS,
    adapter: str = "tests.test_asr_registry.FakeASR",
    max_history_tokens: int = 1000,
    streaming: StreamingSettings | None = None,
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
        asr_streaming=streaming or StreamingSettings(),
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


class RecordingModel:
    """An ASRModel (batch only) that returns scripted texts, then `w<n>`, and records calls."""

    def __init__(self, texts: list[str] | None = None, fail_on: set[int] | None = None) -> None:
        self.texts = list(texts or [])
        self.fail_on = fail_on or set()  # 1-based call numbers that raise ASRError
        self.calls: list[np.ndarray] = []
        self.kwargs: list[dict] = []
        self.attempts = 0

    def load(self) -> None:  # pragma: no cover - never called
        pass

    def transcribe(self, audio, sample_rate=16000, language=None) -> TranscriptResult:
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise ASRError("decode failed")
        self.calls.append(np.array(audio, copy=True))
        self.kwargs.append({"sample_rate": sample_rate, "language": language})
        text = self.texts.pop(0) if self.texts else f"w{len(self.calls)}"
        return TranscriptResult(text, "en", len(audio) / sample_rate, 5.0)


class FakeTurnStream:
    """The ASRStream a FakeTranscriber hands out: scripted partials, records what it was fed."""

    def __init__(self, model: str, partials: list[PartialResult], feed_error: Exception | None):
        self.model = model
        self._partials = list(partials)
        self._feed_error = feed_error
        self.fed: list[np.ndarray] = []

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        self.fed.append(np.array(audio, copy=True))
        if self._feed_error is not None:
            raise self._feed_error
        return self._partials.pop(0) if self._partials else None

    def finish(self) -> TranscriptResult:  # the real work is faked in FakeTranscriber.run
        raise AssertionError("FakeTranscriber.run answers finish")

    @property
    def samples(self) -> np.ndarray:
        return np.concatenate(self.fed) if self.fed else np.zeros(0, dtype=np.float32)


class FakeTranscriber:
    """Stands in for TranscriptionService in session tests.

    It speaks the streaming API: `strategies`, `open_stream` and `run`. A turn's final
    transcript is what v0's `transcribe()` was, so the v0 knobs keep their meaning:
    `gate` (an asyncio.Event) holds the FINAL decode so a test can keep a turn in WORKING;
    `error` is raised instead of the final result; `takes_s` moves the fake clock during it;
    `calls` has one (model, all audio fed) entry per final decode.

    Streaming knobs: `partials` are returned by the stream's feeds in order (then None);
    `feed_takes_s` moves the fake clock during each feed (a number, or a list used in order);
    `feed_gate` holds every feed; `feed_error` makes every feed fail; `open_error` makes
    open_stream fail; `supported` maps a model to its ModelStrategies (default: chunked and
    redecode, default redecode). `opened` records (model, strategy) per open_stream and
    `streams` the stream objects.
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
        partials: list[PartialResult] | None = None,
        feed_takes_s: float | list[float] = 0.0,
        feed_gate: asyncio.Event | None = None,
        feed_error: Exception | None = None,
        open_error: Exception | None = None,
        supported: dict[str, ModelStrategies] | None = None,
    ) -> None:
        self.clock = clock
        self.takes_s = takes_s  # how far the final decode moves the fake clock
        self.text = text
        self.asr_ms = asr_ms
        self.model_load_ms = model_load_ms
        self.error = error
        self.gate = gate
        self.partials = partials or []
        self.feed_takes_s = feed_takes_s
        self.feed_gate = feed_gate
        self.feed_error = feed_error
        self.open_error = open_error
        self.supported = supported or {}
        self.calls: list[tuple[str, np.ndarray]] = []
        self.opened: list[tuple[str, str]] = []
        self.streams: list[FakeTurnStream] = []
        self.feeds = 0  # feeds that reached the worker
        self.cancelled = False  # a decode was cancelled while waiting on a gate
        self.shut_down = False

    def strategies(self, model_name: str) -> ModelStrategies:
        if model_name in self.supported:
            return self.supported[model_name]
        if model_name not in MODELS:
            raise UnknownASRModel(f"unknown ASR model: {model_name!r}")
        return ModelStrategies(supported=["chunked", "redecode"], default="redecode")

    async def open_stream(self, model_name: str, strategy: str) -> OpenedStream:
        if self.open_error is not None:
            raise self.open_error
        if strategy not in self.strategies(model_name).supported:
            raise UnsupportedStrategy(f"{model_name!r} does not support {strategy!r}")
        stream = FakeTurnStream(model_name, self.partials, self.feed_error)
        self.opened.append((model_name, strategy))
        self.streams.append(stream)
        return OpenedStream(stream=stream, model_load_ms=self.model_load_ms)

    async def run(self, fn, *args):
        stream = getattr(fn, "__self__", None)
        if isinstance(stream, FakeTurnStream):
            try:
                if fn.__name__ == "feed":
                    self.feeds += 1
                    if self.feed_gate is not None:
                        await self.feed_gate.wait()
                    self._advance(self._next_feed_takes())
                else:
                    return await self._final(stream)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return fn(*args)

    def _next_feed_takes(self) -> float:
        if isinstance(self.feed_takes_s, list):
            return self.feed_takes_s.pop(0) if self.feed_takes_s else 0.0
        return self.feed_takes_s

    def _advance(self, seconds: float) -> None:
        if self.clock is not None:
            self.clock.advance(seconds)

    async def _final(self, stream: FakeTurnStream) -> TranscriptResult:
        audio = stream.samples
        self.calls.append((stream.model, audio))
        self._advance(self.takes_s)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return TranscriptResult(
            text=self.text,
            language="en",
            duration_s=len(audio) / SAMPLE_RATE,
            latency_ms=self.asr_ms,
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
