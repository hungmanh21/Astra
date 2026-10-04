"""Tests for backend/asr/registry.py and backend/asr/service.py (PLAN T2.2, T2.6).

These use a fake adapter defined in this file so no GPU or model download is needed.
"""

import asyncio
import threading
import time

import numpy as np
import pytest

from backend.asr.base import ASRError, TranscriptResult, UnknownASRModel
from backend.asr.registry import ASRModelConfig, ASRRegistry
from backend.asr.service import TranscriptionService

AUDIO = np.zeros(16000, dtype=np.float32)


class FakeASR:
    load_calls = 0
    last_init: dict = {}
    load_delay = 0.0
    transcribe_delay = 0.0
    error: Exception | None = None

    def __init__(self, name: str, model_id: str, **options) -> None:
        self.name = name
        FakeASR.last_init = {"name": name, "model_id": model_id, **options}

    def load(self) -> None:
        time.sleep(FakeASR.load_delay)
        FakeASR.load_calls += 1

    def transcribe(self, audio, sample_rate=16000, language=None) -> TranscriptResult:
        time.sleep(FakeASR.transcribe_delay)
        if FakeASR.error:
            raise FakeASR.error
        return TranscriptResult(
            text="hello world",
            language="en",
            duration_s=len(audio) / sample_rate,
            latency_ms=FakeASR.transcribe_delay * 1000,
        )


@pytest.fixture(autouse=True)
def reset_fake():
    FakeASR.load_calls = 0
    FakeASR.last_init = {}
    FakeASR.load_delay = 0.0
    FakeASR.transcribe_delay = 0.0
    FakeASR.error = None


def make_registry(*names: str, **options) -> ASRRegistry:
    cfg = ASRModelConfig(
        adapter="tests.test_asr_registry.FakeASR", model_id="fake/model", options=options
    )
    return ASRRegistry({name: cfg for name in names})


def test_available_returns_config_keys_in_order():
    assert make_registry("b", "a", "c").available() == ["b", "a", "c"]


def test_unknown_model_raises_unknown_asr_model():
    with pytest.raises(UnknownASRModel):
        make_registry("a").get("nope")
    assert not make_registry("a").is_loaded("nope")


def test_get_loads_lazily_and_caches():
    reg = make_registry("a")
    assert FakeASR.load_calls == 0
    assert not reg.is_loaded("a")

    FakeASR.load_delay = 0.01
    _, first_ms = reg.get("a")
    model, second_ms = reg.get("a")

    assert FakeASR.load_calls == 1
    assert first_ms > 0
    assert second_ms == 0.0
    assert reg.is_loaded("a")
    assert model.name == "a"


def test_concurrent_get_loads_once():
    FakeASR.load_delay = 0.1
    reg = make_registry("a")
    threads = [threading.Thread(target=reg.get, args=("a",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert FakeASR.load_calls == 1


def test_adapter_receives_name_model_id_and_options():
    make_registry("a", device="cpu", dtype="float32").get("a")
    assert FakeASR.last_init == {
        "name": "a",
        "model_id": "fake/model",
        "device": "cpu",
        "dtype": "float32",
    }


def test_bad_adapter_path_raises_asr_error():
    reg = ASRRegistry({"a": ASRModelConfig(adapter="no.such.Module", model_id="x")})
    with pytest.raises(ASRError):
        reg.get("a")
    reg = ASRRegistry({"a": ASRModelConfig(adapter="tests.test_asr_registry.Nope", model_id="x")})
    with pytest.raises(ASRError):
        reg.get("a")


def test_failed_load_is_not_cached():
    reg = make_registry("a")
    original = FakeASR.load

    def boom(self):
        raise RuntimeError("download failed")

    FakeASR.load = boom
    try:
        with pytest.raises(ASRError, match="download failed"):
            reg.get("a")
        assert not reg.is_loaded("a")
    finally:
        FakeASR.load = original
    reg.get("a")
    assert reg.is_loaded("a")


def test_service_runs_off_the_event_loop():
    FakeASR.transcribe_delay = 0.3
    service = TranscriptionService(make_registry("a"))

    async def scenario():
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        await service.transcribe("a", AUDIO)
        done.set()
        await task
        return ticks

    try:
        assert asyncio.run(scenario()) > 5
    finally:
        service.shutdown()


def test_service_reports_load_time_separately():
    FakeASR.load_delay = 0.1
    service = TranscriptionService(make_registry("a"))
    try:
        first = asyncio.run(service.transcribe("a", AUDIO))
        second = asyncio.run(service.transcribe("a", AUDIO))
    finally:
        service.shutdown()

    assert first.model == "a"
    assert first.text == "hello world"
    assert first.audio_s == 1.0
    assert first.model_load_ms >= 100
    assert first.asr_ms < first.model_load_ms
    assert second.model_load_ms == 0.0


def test_service_wraps_unexpected_errors_in_asr_error():
    service = TranscriptionService(make_registry("a"))
    try:
        FakeASR.error = ValueError("secret internal detail")
        with pytest.raises(ASRError) as exc:
            asyncio.run(service.transcribe("a", AUDIO))
        assert "secret internal detail" not in str(exc.value)

        with pytest.raises(UnknownASRModel):
            asyncio.run(service.transcribe("nope", AUDIO))
    finally:
        service.shutdown()


def test_service_preload_loads_the_model():
    reg = make_registry("a")
    service = TranscriptionService(reg)
    try:
        asyncio.run(service.preload("a"))
    finally:
        service.shutdown()
    assert reg.is_loaded("a")
