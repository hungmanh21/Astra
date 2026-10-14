"""End-to-end edge cases over a real WebSocket, with fake ASR and LLM (PLAN T5.5).

Each test breaks one rule and checks two things: the browser gets a clear answer, and the
connection (or the server) is still healthy afterwards.
"""

import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from backend.asr.base import PartialResult
from backend.main import create_app
from tests.fakes import FakeStreamingASR
from tests.helpers import MODELS, FakeLLM, chunks, make_settings, pcm
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
    FakeStreamingASR.reset()


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


# --- streaming (PLAN_v1 S1.8): a fake model that streams natively, behind the real service ---

STREAMING = "tests.fakes.FakeStreamingASR"
P1 = PartialResult("hello", "")
P2 = PartialResult("hello wor", "ld")


def receive_within(ws, within_s: float = 5.0) -> dict:
    """receive_json() that fails the test instead of hanging when the server stays silent.

    If nothing arrives in time the server gets an invalid message, and its `error` reply
    wakes the receive up, so the assertions that follow report what went wrong.
    """
    nudge = threading.Timer(within_s, ws.send_json, args=[{"type": "nudge"}])
    nudge.start()
    try:
        return ws.receive_json()
    finally:
        nudge.cancel()


def send_frames(ws, data: bytes) -> None:
    for frame in chunks(data):
        ws.send_bytes(frame)


def start_turn(ws, **extra) -> dict:
    ws.send_json({"type": "start_turn", "asr_model": MODELS[0], "review": False, **extra})
    return receive_within(ws)


def test_partials_arrive_while_the_frames_are_still_being_sent():
    FakeStreamingASR.partials = [P1, P2]
    client = TestClient(create_app(make_settings(adapter=STREAMING), llm=FakeLLM()))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        turn_id = start_turn(ws)["turn_id"]

        send_frames(ws, pcm(0.6))
        first = receive_within(ws)  # end_turn has not been sent yet
        assert first == {
            "type": "partial",
            "turn_id": turn_id,
            "committed": "hello",
            "tentative": "",
        }

        send_frames(ws, pcm(0.6))
        second = receive_within(ws)
        assert (second["type"], second["committed"], second["tentative"]) == (
            "partial",
            "hello wor",
            "ld",
        )

        ws.send_json({"type": "end_turn"})
        rest = read_until(ws, "llm_done")
    types = [m["type"] for m in rest]
    assert "partial" not in types  # nothing after end_turn
    assert types[0] == "transcript"
    assert rest[0]["text"] == "final text"
    assert rest[0]["stream_strategy"] == "native"
    assert rest[-1]["timings"]["partial_count"] == 2
    # the model heard all the audio exactly once
    assert len(FakeStreamingASR.streams[0].samples) == 19200


def test_a_failing_partial_still_ends_in_transcript_and_reply(caplog):
    FakeStreamingASR.feed_error = RuntimeError("decode blew up")
    caplog.set_level(logging.INFO)
    client = TestClient(create_app(make_settings(adapter=STREAMING), llm=FakeLLM()))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        send_turn(ws, pcm(1.5))
        messages = read_until(ws, "llm_done")
    types = [m["type"] for m in messages]
    assert "error" not in types and "partial" not in types  # the user sees nothing wrong
    assert types[:2] == ["transcript", "llm_delta"]
    assert messages[0]["text"] == "final text"
    assert messages[-1]["text"] == "Hi there"
    assert [line["status"] for line in turn_lines(caplog)] == ["ok"]


def test_unsupported_strategy_is_refused_and_the_next_turn_starts():
    client = TestClient(create_app(make_settings(adapter=STREAMING), llm=FakeLLM()))
    with client.websocket_connect("/ws") as ws:
        info = ws.receive_json()
        assert info["stream_strategies"][MODELS[0]] == {
            "supported": ["chunked", "redecode", "native"],  # no segments, so no `trimmed`
            "default": "native",
        }

        refusal = start_turn(ws, stream_strategy="trimmed")
        assert refusal["type"] == "error" and refusal["turn_id"] is None
        assert "trimmed" in refusal["message"]
        assert FakeStreamingASR.streams == []  # nothing was opened for the refused turn

        ws.send_bytes(b"\x00\x00" * 800)  # no turn exists, so this is ignored
        assert start_turn(ws, stream_strategy="chunked")["type"] == "turn_started"
        send_frames(ws, pcm(0.5))
        ws.send_json({"type": "end_turn"})
        messages = read_until(ws, "llm_done")
    transcript = next(m for m in messages if m["type"] == "transcript")
    assert transcript["stream_strategy"] == "chunked"
    assert FakeStreamingASR.streams == []  # `chunked` does not use the adapter's own stream


def test_disconnect_mid_stream_leaves_the_server_usable(caplog):
    FakeStreamingASR.feed_gate = threading.Event()  # holds the decode on the ASR thread
    caplog.set_level(logging.INFO)
    app = create_app(make_settings(adapter=STREAMING), llm=FakeLLM())
    valve = threading.Timer(5, FakeStreamingASR.feed_gate.set)
    valve.start()
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                start_turn(ws)
                send_frames(ws, pcm(0.6))  # a decode starts and is held
            # the browser left in the middle of the turn, with a decode in flight
            FakeStreamingASR.feed_gate.set()  # the decode ends
            FakeStreamingASR.feed_gate = None

            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                send_turn(ws, pcm(0.5))
                done = read_until(ws, "llm_done")[-1]
                assert done["text"] == "Hi there"
    finally:
        valve.cancel()
    assert [line["status"] for line in turn_lines(caplog)] == ["dropped", "ok"]
