"""End-to-end edge cases over a real WebSocket, with fake ASR and LLM (PLAN T5.5).

Each test breaks one rule and checks two things: the browser gets a clear answer, and the
connection (or the server) is still healthy afterwards.
"""

import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.helpers import MODELS, FakeLLM, make_settings, pcm
from tests.test_asr_registry import FakeASR
from tests.test_main import BlockingASR, send_turn


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeASR.load_calls = 0
    FakeASR.load_delay = 0.0
    FakeASR.transcribe_delay = 0.0
    FakeASR.error = None
    BlockingASR.started.clear()
    BlockingASR.release.clear()


def read_until(ws, kind: str) -> list[dict]:
    """Messages up to and including the first one of type `kind`."""
    seen = []
    while True:
        msg = ws.receive_json()
        seen.append(msg)
        if msg["type"] == kind:
            return seen


def turn_lines(caplog) -> list[dict]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "astra.turns"]


def test_turn_over_the_length_cap_is_rejected_and_the_connection_survives(caplog):
    llm = FakeLLM()
    caplog.set_level(logging.INFO)
    client = TestClient(create_app(make_settings(max_turn_seconds=30), llm=llm))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        turn_id = send_turn(ws, pcm(31))  # frames keep arriving after the cap is hit
        error = ws.receive_json()
        assert error["type"] == "error" and error["turn_id"] == turn_id
        assert "30" in error["message"]  # says what the limit is

        # The rejected turn never reached the model, and the next turn works.
        assert llm.calls == []
        send_turn(ws, pcm(0.5))
        done = read_until(ws, "llm_done")[-1]
        assert done["text"] == "Hi there"
    assert [line["status"] for line in turn_lines(caplog)] == ["rejected", "ok"]


def test_start_turn_during_a_turn_is_refused_and_the_first_turn_finishes():
    llm = FakeLLM()
    app = create_app(make_settings(adapter="tests.test_main.BlockingASR"), llm=llm)
    valve = threading.Timer(5, BlockingASR.release.set)  # never hang on a broken server
    valve.start()
    try:
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            ws.receive_json()
            first = send_turn(ws, pcm(0.5))
            assert BlockingASR.started.wait(timeout=3)

            ws.send_json({"type": "start_turn", "asr_model": MODELS[0], "review": False})
            refusal = ws.receive_json()
            assert refusal["type"] == "error" and "in flight" in refusal["message"]

            BlockingASR.release.set()
            messages = read_until(ws, "llm_done")
    finally:
        valve.cancel()
    # The first turn was not disturbed: its transcript and reply arrive under its own id.
    assert {m["turn_id"] for m in messages} == {first}
    assert [m["type"] for m in messages][0] == "transcript"
    assert len(llm.calls) == 1


def test_unknown_asr_model_is_refused_without_starting_a_turn():
    llm = FakeLLM()
    client = TestClient(create_app(make_settings(), llm=llm))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start_turn", "asr_model": "no-such-model", "review": False})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "no-such-model" in error["message"]

        # No turn exists, so audio is ignored; a valid turn can start right away.
        ws.send_bytes(b"\x00\x00" * 800)
        send_turn(ws, pcm(0.5))
        assert read_until(ws, "llm_done")[-1]["text"] == "Hi there"
    assert len(llm.calls) == 1


def test_connection_dropped_mid_turn_is_logged_and_the_server_carries_on(caplog):
    llm = FakeLLM()
    caplog.set_level(logging.INFO)
    app = create_app(make_settings(adapter="tests.test_main.BlockingASR"), llm=llm)
    valve = threading.Timer(5, BlockingASR.release.set)
    valve.start()
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                send_turn(ws, pcm(0.5))
                assert BlockingASR.started.wait(timeout=3)
            # The browser left while the ASR was still running.
            BlockingASR.release.set()

            # The abandoned turn must not reach the LLM, and a new connection works. That the
            # ASR job is cancelled is checked in test_session; here a send to the closed socket
            # would stop the turn even without it.
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                send_turn(ws, pcm(0.5))
                assert read_until(ws, "llm_done")[-1]["text"] == "Hi there"
    finally:
        valve.cancel()
    assert len(llm.calls) == 1  # only the second connection's turn
    assert [line["status"] for line in turn_lines(caplog)] == ["dropped", "ok"]
