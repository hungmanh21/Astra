"""Tests for the streaming side of backend/asr/service.py and strategies.py (PLAN_v1 S1.5).

Everything goes through the real registry with the fake adapters from tests/fakes.py, so no
GPU or model download is needed.
"""

import asyncio
import threading

import numpy as np
import pytest

from backend.asr.base import (
    ASRError,
    PartialResult,
    UnknownASRModel,
    UnsupportedStrategy,
)
from backend.asr.registry import ASRModelConfig, ASRRegistry
from backend.asr.service import TranscriptionService
from backend.asr.strategies import (
    BatchStream,
    ChunkedStream,
    RedecodeStream,
    default_strategy,
    supported_strategies,
)
from backend.asr.streaming import StreamingTranscription
from backend.config import StreamingSettings
from tests.fakes import FakeFullASR, FakePlainASR, FakeSegmentedASR, FakeStreamingASR

PLAIN = ["chunked", "redecode"]
SEGMENTED = ["chunked", "redecode", "trimmed"]
STREAMING = ["chunked", "redecode", "native"]
FULL = ["chunked", "redecode", "trimmed", "native"]


@pytest.fixture(autouse=True)
def reset_fakes():
    for fake in (FakePlainASR, FakeStreamingASR, FakeSegmentedASR):
        fake.reset()


def make_registry() -> ASRRegistry:
    def config(cls) -> ASRModelConfig:
        return ASRModelConfig(adapter=f"tests.fakes.{cls.__name__}", model_id="fake/model")

    return ASRRegistry(
        {
            "plain": config(FakePlainASR),
            "segmented": config(FakeSegmentedASR),
            "streaming": config(FakeStreamingASR),
            "full": config(FakeFullASR),
        }
    )


@pytest.fixture
def service():
    svc = TranscriptionService(make_registry())
    yield svc
    svc.shutdown()


def audio(n: int, value: float = 0.1) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


# --- supported_strategies / default_strategy ---


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (FakePlainASR, PLAIN),
        (FakeSegmentedASR, SEGMENTED),
        (FakeStreamingASR, STREAMING),
        (FakeFullASR, FULL),
    ],
)
def test_supported_strategies_follow_the_capabilities(cls, expected):
    assert supported_strategies(cls("m")) == expected


@pytest.mark.parametrize(
    ("cls", "fallback", "expected"),
    [
        (FakePlainASR, "redecode", "redecode"),
        (FakePlainASR, "chunked", "chunked"),  # the fallback is what config says
        (FakeSegmentedASR, "redecode", "redecode"),  # trimmed is opt-in, never the default
        (FakeStreamingASR, "redecode", "native"),
        (FakeFullASR, "chunked", "native"),  # native wins over the fallback
    ],
)
def test_default_strategy_rule(cls, fallback, expected):
    assert default_strategy(cls("m"), fallback) == expected


def test_strategy_choice_does_not_depend_on_the_model_name():
    # The same class under a Whisper-looking name gets the same answer.
    assert supported_strategies(FakePlainASR("whisper-large-v3")) == PLAIN
    assert supported_strategies(FakeStreamingASR("nemotron-streaming")) == STREAMING


# --- BatchStream ---


def test_batch_stream_shows_no_partials_and_transcribes_everything_once():
    model = FakePlainASR("plain")
    stream = BatchStream(model)
    assert stream.feed(audio(1000, 0.1)) is None
    assert stream.feed(audio(500, 0.2)) is None
    assert FakePlainASR.transcribed == []  # feed never decodes

    result = stream.finish()

    assert len(FakePlainASR.transcribed) == 1
    assert (
        FakePlainASR.transcribed[0].tolist() == [np.float32(0.1)] * 1000 + [np.float32(0.2)] * 500
    )
    assert result.text == "plain text"
    assert result.duration_s == pytest.approx(1500 / 16000)
    assert result.latency_ms == 3.0  # the model's own number, like v0's asr_ms


def test_batch_stream_copies_the_fed_audio():
    stream = BatchStream(FakePlainASR("plain"))
    frame = audio(100, 0.5)
    stream.feed(frame)
    frame[:] = 0.0  # the caller reuses its buffer
    stream.finish()
    assert FakePlainASR.transcribed[0].tolist() == [np.float32(0.5)] * 100


def test_batch_streams_do_not_share_audio():
    model = FakePlainASR("plain")
    a, b = BatchStream(model), BatchStream(model)
    a.feed(audio(100))
    b.feed(audio(300))
    a.finish()
    assert [len(x) for x in FakePlainASR.transcribed] == [100]


def test_batch_stream_with_no_audio_does_not_call_the_model():
    result = BatchStream(FakePlainASR("plain")).finish()
    assert FakePlainASR.transcribed == []
    assert (result.text, result.duration_s, result.latency_ms) == ("", 0.0, 0.0)


# --- TranscriptionService.strategies ---


@pytest.mark.parametrize(
    ("name", "supported", "default"),
    [
        ("plain", PLAIN, "redecode"),
        ("segmented", SEGMENTED, "redecode"),
        ("streaming", STREAMING, "native"),
        ("full", FULL, "native"),
    ],
)
def test_service_lists_strategies_per_model(service, name, supported, default):
    info = service.strategies(name)
    assert info.supported == supported
    assert info.default == default
    assert info.default in info.supported


def test_service_default_comes_from_the_streaming_settings():
    svc = TranscriptionService(
        make_registry(), streaming=StreamingSettings(default_strategy="chunked")
    )
    try:
        assert svc.strategies("plain").default == "chunked"
        assert svc.strategies("streaming").default == "native"
    finally:
        svc.shutdown()


def test_listing_strategies_does_not_load_any_model(service):
    for name in ("plain", "segmented", "streaming", "full"):
        service.strategies(name)
    assert FakePlainASR.load_calls == 0
    assert not service._registry.is_loaded("plain")


def test_strategies_of_an_unknown_model(service):
    with pytest.raises(UnknownASRModel):
        service.strategies("nope")


# --- TranscriptionService.open_stream ---


def test_open_stream_native_uses_the_adapters_own_stream(service):
    FakeStreamingASR.partials = [PartialResult("hello", " wor")]

    async def scenario():
        opened = await service.open_stream("streaming", "native")
        partial = await service.run(opened.stream.feed, audio(8000))
        final = await service.run(opened.stream.finish)
        return opened, partial, final

    opened, partial, final = asyncio.run(scenario())
    assert opened.stream is FakeStreamingASR.streams[0]
    assert partial == PartialResult("hello", " wor")
    assert final.text == "final text"


def test_each_native_turn_gets_its_own_stream(service):
    async def scenario():
        first = await service.open_stream("streaming", "native")
        second = await service.open_stream("streaming", "native")
        return first.stream, second.stream

    first, second = asyncio.run(scenario())
    assert first is not second
    assert len(FakeStreamingASR.streams) == 2


@pytest.mark.parametrize(
    ("model", "strategy"),
    [
        ("segmented", "trimmed"),
        ("full", "trimmed"),
    ],
)
def test_open_stream_batch_path_behaves_like_v0(service, model, strategy):
    async def scenario():
        opened = await service.open_stream(model, strategy)
        assert isinstance(opened.stream, BatchStream)
        assert await service.run(opened.stream.feed, audio(16000)) is None
        return await service.run(opened.stream.finish)

    result = asyncio.run(scenario())
    assert result.duration_s == 1.0
    assert result.text  # the whole-turn transcription from the model
    assert not FakeStreamingASR.streams  # the native stream was not touched


@pytest.mark.parametrize("model", ["plain", "streaming"])  # a streaming model also has redecode
def test_open_stream_redecode_uses_the_configured_minimum(model):
    svc = TranscriptionService(
        make_registry(), streaming=StreamingSettings(partial_min_audio_ms=300)
    )
    try:
        opened = asyncio.run(svc.open_stream(model, "redecode"))
    finally:
        svc.shutdown()
    assert isinstance(opened.stream, RedecodeStream)
    assert opened.stream._min_samples == 4800
    assert not FakeStreamingASR.streams  # the native stream was not touched


def test_open_stream_chunked_uses_the_configured_chunk():
    svc = TranscriptionService(make_registry(), streaming=StreamingSettings(chunk_ms=1000))
    try:
        opened = asyncio.run(svc.open_stream("plain", "chunked"))
    finally:
        svc.shutdown()
    assert isinstance(opened.stream, ChunkedStream)
    assert opened.stream._chunk_samples == 16000


@pytest.mark.parametrize(
    ("model", "strategy"),
    [
        ("plain", "native"),
        ("plain", "trimmed"),
        ("segmented", "native"),
        ("streaming", "trimmed"),
        ("plain", "bogus"),
        ("plain", ""),
    ],
)
def test_unsupported_strategy_is_rejected_without_loading(service, model, strategy):
    with pytest.raises(UnsupportedStrategy):
        asyncio.run(service.open_stream(model, strategy))
    assert FakePlainASR.load_calls == 0
    assert not service._registry.is_loaded(model)


def test_unsupported_strategy_is_an_asr_error():
    # The session already handles ASRError; FR-22 rejects "like an unknown model".
    assert issubclass(UnsupportedStrategy, ASRError)


def test_open_stream_unknown_model(service):
    with pytest.raises(UnknownASRModel):
        asyncio.run(service.open_stream("nope", "redecode"))


def test_open_stream_reports_the_load_time_only_once(service):
    FakePlainASR.load_delay_s = 0.1

    async def scenario():
        first = await service.open_stream("plain", "redecode")
        second = await service.open_stream("plain", "redecode")
        return first, second

    first, second = asyncio.run(scenario())
    assert first.model_load_ms >= 100
    assert second.model_load_ms == 0.0
    assert FakePlainASR.load_calls == 1


def test_open_stream_loads_off_the_event_loop():
    FakePlainASR.load_delay_s = 0.3
    svc = TranscriptionService(make_registry())

    async def scenario():
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        await svc.open_stream("plain", "redecode")
        done.set()
        await task
        return ticks

    try:
        assert asyncio.run(scenario()) > 5
    finally:
        svc.shutdown()


def test_open_stream_wraps_unexpected_errors(service):
    def boom(self, sample_rate=16000, language=None):
        raise ValueError("secret internal detail")

    original = FakeStreamingASR.open_stream
    FakeStreamingASR.open_stream = boom
    try:
        with pytest.raises(ASRError) as exc:
            asyncio.run(service.open_stream("streaming", "native"))
    finally:
        FakeStreamingASR.open_stream = original
    assert "secret internal detail" not in str(exc.value)


# --- TranscriptionService.run ---


def test_run_executes_on_the_asr_worker_thread(service):
    main = threading.current_thread().name
    seen = asyncio.run(service.run(lambda x, y: (threading.current_thread().name, x + y), 2, 3))
    assert seen[1] == 5
    assert seen[0] != main
    assert seen[0].startswith("asr")


def test_run_wraps_unexpected_errors_but_keeps_asr_errors(service):
    def secret():
        raise ValueError("secret internal detail")

    def known():
        raise ASRError("decode failed")

    with pytest.raises(ASRError) as exc:
        asyncio.run(service.run(secret))
    assert "secret internal detail" not in str(exc.value)

    with pytest.raises(ASRError, match="decode failed"):
        asyncio.run(service.run(known))


# --- with StreamingTranscription ---


def test_service_stream_drives_a_streaming_transcription(service):
    """The pieces fit: service.run is the Runner and the opened stream is the ASRStream."""
    FakeStreamingASR.partials = [PartialResult("hello", "")]
    seen: list[PartialResult] = []

    async def on_partial(partial, received_at):
        seen.append(partial)

    async def scenario():
        opened = await service.open_stream("streaming", "native")
        turn = StreamingTranscription(opened.stream, service.run, on_partial, interval_ms=500)
        turn.push(audio(8000))  # exactly one interval
        for _ in range(200):  # finish() drops a partial still in flight, so let it arrive
            if seen:
                break
            await asyncio.sleep(0.01)
        return await turn.finish()

    result = asyncio.run(scenario())
    assert seen == [PartialResult("hello", "")]
    assert result.text == "final text"
