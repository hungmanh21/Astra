"""Tests for backend/protocol.py (PLAN T1.4, PLAN_v1 S1.3)."""

import json
from typing import get_args

import pytest

from backend.protocol import (
    ConfirmTurn,
    DiscardTurn,
    EndTurn,
    ErrorMessage,
    LlmDone,
    Partial,
    ProtocolError,
    Reset,
    ServerMessage,
    SessionInfo,
    StartTurn,
    StrategyInfo,
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
    assert start.stream_strategy is None
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
        "release_tail_ms": 0,
        "stream_strategies": {},
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
        "stream_strategy": None,
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
        "stream_strategy": None,
        "partial_count": None,
        "partial_lag_ms": None,
        "final_ms": None,
    }


# --- v1 streaming (PLAN_v1 S1.3) ---


def test_start_turn_carries_an_optional_stream_strategy():
    msg = parse_client_message(
        '{"type": "start_turn", "asr_model": "m", "stream_strategy": "trimmed"}'
    )
    assert msg.stream_strategy == "trimmed"


def test_unknown_stream_strategy_still_parses():
    # The session rejects it with an `error` and no turn (FR-22); the parser must let it through.
    msg = parse_client_message('{"type": "start_turn", "asr_model": "m", "stream_strategy": "x"}')
    assert msg.stream_strategy == "x"


def test_session_lists_each_models_strategies():
    info = SessionInfo(
        session_id="s1",
        asr_models=["whisper", "nemotron"],
        default_asr_model="whisper",
        max_turn_seconds=30,
        stream_strategies={
            "whisper": StrategyInfo(
                supported=["chunked", "redecode", "trimmed"], default="redecode"
            ),
            "nemotron": StrategyInfo(supported=["chunked", "redecode", "native"], default="native"),
        },
    )
    assert info.model_dump(mode="json")["stream_strategies"] == {
        "whisper": {"supported": ["chunked", "redecode", "trimmed"], "default": "redecode"},
        "nemotron": {"supported": ["chunked", "redecode", "native"], "default": "native"},
    }


def test_partial_wire_shape():
    msg = Partial(turn_id="t1", committed="hello", tentative="wor")
    assert msg.model_dump(mode="json") == {
        "type": "partial",
        "turn_id": "t1",
        "committed": "hello",
        "tentative": "wor",
    }


def test_partial_needs_both_texts():
    with pytest.raises(ValueError):
        Partial(turn_id="t1", committed="hello")


def test_transcript_carries_the_strategy_used():
    msg = Transcript(turn_id="t1", text="hi", asr_ms=5, stream_strategy="redecode")
    assert msg.model_dump(mode="json")["stream_strategy"] == "redecode"


def test_timings_carry_the_streaming_values():
    timings = Timings(
        audio_s=5.0,
        asr_ms=120,
        stream_strategy="native",
        partial_count=9,
        partial_lag_ms=180.5,
        final_ms=95.0,
    )
    wire = timings.model_dump(mode="json")
    assert wire["stream_strategy"] == "native"
    assert wire["partial_count"] == 9
    assert wire["partial_lag_ms"] == 180.5
    assert wire["final_ms"] == 95.0


def test_partial_is_a_server_message():
    assert Partial in get_args(ServerMessage)
