"""Tests for backend/protocol.py (PLAN T1.4)."""

import json

import pytest

from backend.protocol import (
    ConfirmTurn,
    DiscardTurn,
    EndTurn,
    ErrorMessage,
    LlmDone,
    ProtocolError,
    Reset,
    SessionInfo,
    StartTurn,
    Timings,
    Transcript,
    TurnStarted,
    parse_client_message,
)


def test_parses_each_client_message():
    start = parse_client_message(
        '{"type": "start_turn", "asr_model": "whisper-large-v3", "review": true}'
    )
    assert start == StartTurn(type="start_turn", asr_model="whisper-large-v3", review=True)
    assert isinstance(parse_client_message('{"type": "end_turn"}'), EndTurn)
    assert isinstance(parse_client_message('{"type": "reset"}'), Reset)

    confirm = parse_client_message('{"type": "confirm_turn", "turn_id": "t1", "text": "hi there"}')
    assert isinstance(confirm, ConfirmTurn)
    assert (confirm.turn_id, confirm.text) == ("t1", "hi there")

    discard = parse_client_message('{"type": "discard_turn", "turn_id": "t1"}')
    assert isinstance(discard, DiscardTurn)
    assert discard.turn_id == "t1"


def test_accepts_bytes_input():
    assert isinstance(parse_client_message(b'{"type": "end_turn"}'), EndTurn)


def test_start_turn_review_defaults_to_false():
    msg = parse_client_message('{"type": "start_turn", "asr_model": "m"}')
    assert msg.review is False


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "",
        "[1, 2]",  # valid JSON, not an object
        "{}",  # no type
        '{"type": "nope"}',
        '{"type": "start_turn"}',  # no asr_model
        '{"type": "start_turn", "asr_model": "m", "review": "maybe"}',
        '{"type": "confirm_turn", "text": "hi"}',  # no turn_id
        '{"type": "confirm_turn", "turn_id": "t1"}',  # no text
        '{"type": "discard_turn"}',
    ],
)
def test_rejects_malformed_messages_with_protocol_error(raw):
    with pytest.raises(ProtocolError):
        parse_client_message(raw)


def test_server_messages_serialize_to_the_wire_shape():
    assert ErrorMessage(message="x").model_dump(mode="json") == {
        "type": "error",
        "turn_id": None,
        "message": "x",
    }
    assert SessionInfo(
        session_id="s1", asr_models=["a", "b"], default_asr_model="a", max_turn_seconds=30
    ).model_dump(mode="json") == {
        "type": "session",
        "session_id": "s1",
        "asr_models": ["a", "b"],
        "default_asr_model": "a",
        "max_turn_seconds": 30,
    }
    assert TurnStarted(turn_id="t1").model_dump(mode="json") == {
        "type": "turn_started",
        "turn_id": "t1",
    }
    assert Transcript(turn_id="t1", text="hi", asr_ms=12.5).model_dump(mode="json") == {
        "type": "transcript",
        "turn_id": "t1",
        "text": "hi",
        "asr_ms": 12.5,
    }


def test_llm_done_carries_timings_the_frontend_reads():
    done = LlmDone(
        turn_id="t1",
        text="ok",
        timings=Timings(audio_s=1.5, asr_ms=300, llm_ttft_ms=200, llm_total_ms=900, e2e_ms=1200),
    )
    wire = json.loads(done.model_dump_json())
    assert wire["timings"] == {
        "audio_s": 1.5,
        "asr_ms": 300,
        "llm_ttft_ms": 200,
        "llm_total_ms": 900,
        "e2e_ms": 1200,
    }
