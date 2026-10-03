"""Tests for backend/main.py (PLAN T1.3, T2.6, T2.7), through Starlette's TestClient."""

import threading

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.asr.base import TranscriptResult
from backend.asr.service import TranscriptionService
from backend.main import create_app
from tests.helpers import MODELS, FakeTranscriber, chunks, make_settings, pcm
from tests.test_asr_registry import FakeASR


class BlockingASR:
    """Adapter whose transcribe() blocks until the test lets it go."""

    started = threading.Event()
    release = threading.Event()

    def __init__(self, name: str, model_id: str, **options) -> None:
        self.name = name

    def load(self) -> None:
        pass

    def transcribe(self, audio, sample_rate=16000, language=None) -> TranscriptResult:
        BlockingASR.started.set()
        BlockingASR.release.wait(timeout=20)
        return TranscriptResult(text="done", language="en", duration_s=1.0, latency_ms=1.0)


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeASR.load_calls = 0
    FakeASR.load_delay = 0.0
    FakeASR.transcribe_delay = 0.0
    FakeASR.error = None
    BlockingASR.started.clear()
    BlockingASR.release.clear()


def send_turn(ws, data: bytes) -> str:
    ws.send_json({"type": "start_turn", "asr_model": MODELS[0], "review": False})
    turn_id = ws.receive_json()["turn_id"]
    for frame in chunks(data):
        ws.send_bytes(frame)
    ws.send_json({"type": "end_turn"})
    return turn_id


def test_root_serves_the_frontend(tmp_path):
    client = TestClient(create_app(make_settings()))
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="mic"' in response.text


def test_ws_sends_session_message_on_connect():
    client = TestClient(create_app(make_settings()))
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "session"
    assert msg["asr_models"] == MODELS


def test_full_audio_turn_over_websocket(tmp_path):
    data = pcm(0.5)
    client = TestClient(create_app(make_settings(debug_dir=tmp_path)))
    with client.websocket_connect("/ws") as ws:
        session_id = ws.receive_json()["session_id"]
        turn_id = send_turn(ws, data)
        reply = ws.receive_json()  # the WAV is written before the server replies
        assert reply["type"] == "transcript"
        assert reply["turn_id"] == turn_id
        assert reply["text"] == "hello world"  # from FakeASR, through the real registry/service

    samples, rate = sf.read(tmp_path / f"{session_id}-{turn_id}.wav", dtype="int16")
    assert rate == 16000
    np.testing.assert_array_equal(samples, np.frombuffer(data, dtype="<i2"))


def test_asr_failure_reaches_the_browser_as_a_generic_error():
    FakeASR.error = RuntimeError("secret internals")
    client = TestClient(create_app(make_settings()))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        turn_id = send_turn(ws, pcm(0.5))
        reply = ws.receive_json()
    assert reply == {"type": "error", "turn_id": turn_id, "message": "transcription failed"}


def test_create_app_builds_a_lazy_transcriber_from_settings():
    app = create_app(make_settings())
    assert isinstance(app.state.transcriber, TranscriptionService)
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
    assert FakeASR.load_calls == 0  # nothing loads until a turn needs the model


def test_create_app_uses_the_transcriber_it_is_given():
    fake = FakeTranscriber(text="injected")
    client = TestClient(create_app(make_settings(), transcriber=fake))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send_turn(ws, pcm(0.5))
        assert ws.receive_json()["text"] == "injected"
    assert len(fake.calls) == 1


def test_transcriber_is_shut_down_when_the_app_stops():
    fake = FakeTranscriber()
    with TestClient(create_app(make_settings(), transcriber=fake)):
        assert not fake.shut_down
    assert fake.shut_down


def test_second_client_stays_responsive_while_an_asr_call_runs():
    app = create_app(make_settings(adapter="tests.test_main.BlockingASR"))
    # Safety valve: if the event loop is blocked, the ASR gets released after 5 s so the
    # test fails on the check below instead of hanging.
    valve = threading.Timer(5, BlockingASR.release.set)
    valve.start()
    try:
        # `with TestClient` shares one event loop between both connections.
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
                a.receive_json()
                send_turn(a, pcm(0.5))
                assert BlockingASR.started.wait(timeout=3)

                first_message = b.receive_json()  # needs the event loop while A is in the ASR
                still_blocked = not BlockingASR.release.is_set()
                BlockingASR.release.set()

                assert first_message["type"] == "session"
                assert still_blocked, "the second client had to wait for the ASR call"
                assert a.receive_json()["text"] == "done"
    finally:
        valve.cancel()
