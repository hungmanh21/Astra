"""Tests for backend/session.py (PLAN T1.6, T1.8, T2.7).

Each test drives a Session through a FakeTransport and inspects what it sent.
"""

import asyncio
import json
import logging
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from backend.asr.base import ASRError
from backend.llm import LLMError
from backend.session import Session, TurnStage
from tests.helpers import MODELS, FakeClock, FakeLLM, FakeTranscriber, chunks, make_settings, pcm


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.sent_bytes: list[bytes] = []

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    def types(self) -> list[str]:
        return [m["type"] for m in self.sent]

    def last(self, kind: str) -> dict[str, Any]:
        return next(m for m in reversed(self.sent) if m["type"] == kind)


def start_msg(model: str = MODELS[0], review: bool = False) -> str:
    return json.dumps({"type": "start_turn", "asr_model": model, "review": review})


END = json.dumps({"type": "end_turn"})


def run(
    scenario,
    transcriber: FakeTranscriber | None = None,
    llm: FakeLLM | None = None,
    clock: FakeClock | None = None,
    **settings_kwargs,
):
    """Run `scenario(session, transport)` on a fresh connected session."""
    transport = FakeTransport()
    session = Session(
        transport,
        make_settings(**settings_kwargs),
        transcriber or FakeTranscriber(),
        llm or FakeLLM(),
        **({"clock": clock} if clock else {}),
    )

    async def go():
        await session.on_connect()
        async with asyncio.timeout(5):  # a hung scenario should fail, not stall the suite
            await scenario(session, transport)

    asyncio.run(go())
    return session, transport


async def send_turn(session: Session, data: bytes, model: str = MODELS[0]) -> None:
    await session.on_json(start_msg(model))
    for frame in chunks(data):
        await session.on_audio_frame(frame)
    await session.on_json(END)


async def settle(session: Session, timeout: float = 2.0) -> None:  # noqa: ASYNC109
    """Wait until the background ASR job is over and the session is idle again."""
    async with asyncio.timeout(timeout):
        while session.stage != TurnStage.IDLE:  # noqa: ASYNC110  (polls a plain attribute)
            await asyncio.sleep(0.005)


def test_connect_sends_session_message():
    async def scenario(session, transport):
        pass

    _, transport = run(scenario)
    assert transport.types() == ["session"]
    msg = transport.sent[0]
    assert msg["session_id"]
    assert msg["asr_models"] == MODELS
    assert msg["default_asr_model"] == MODELS[0]


def test_start_turn_acks_with_turn_id():
    async def scenario(session, transport):
        await session.on_json(start_msg())

    _, transport = run(scenario)
    assert transport.types() == ["session", "turn_started"]
    assert transport.last("turn_started")["turn_id"]


def test_start_turn_rejected_while_in_flight_and_turn_unaffected():
    async def scenario(session, transport):
        await session.on_json(start_msg())
        first = transport.last("turn_started")["turn_id"]
        await session.on_json(start_msg())
        error = transport.last("error")
        assert error["turn_id"] is None
        assert transport.types().count("turn_started") == 1

        # the first turn still takes audio and ends normally (no extra error about it)
        for frame in chunks(pcm(0.5)):
            await session.on_audio_frame(frame)
        errors_before = transport.types().count("error")
        await session.on_json(END)
        new_errors = [m for m in transport.sent if m["type"] == "error"][errors_before:]
        assert all(m["turn_id"] == first for m in new_errors)

    run(scenario)


def test_start_turn_rejects_unknown_model():
    async def scenario(session, transport):
        await session.on_json(start_msg(model="nope"))
        assert transport.last("error")["turn_id"] is None
        assert "turn_started" not in transport.types()
        # still idle: a valid start works
        await session.on_json(start_msg())
        assert "turn_started" in transport.types()

    run(scenario)


def test_bad_json_and_unknown_type_send_error_and_keep_going():
    async def scenario(session, transport):
        await session.on_json("garbage")
        await session.on_json('{"type": "nope"}')
        await session.on_json('{"type": "start_turn"}')
        assert transport.types().count("error") == 3
        await session.on_json(start_msg())
        assert "turn_started" in transport.types()

    run(scenario)


def test_end_turn_too_short_sends_error_and_session_stays_usable():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.1))
        turn_id = transport.last("turn_started")["turn_id"]
        assert transport.last("error")["turn_id"] == turn_id
        await session.on_json(start_msg())
        assert transport.types().count("turn_started") == 2

    run(scenario)


def test_end_turn_with_no_audio_sends_error():
    async def scenario(session, transport):
        await session.on_json(start_msg())
        await session.on_json(END)
        assert transport.last("error")["turn_id"] == transport.last("turn_started")["turn_id"]

    run(scenario)


def test_end_turn_with_odd_byte_count_sends_error():
    async def scenario(session, transport):
        await session.on_json(start_msg())
        await session.on_audio_frame(pcm(0.5) + b"\x00")
        await session.on_json(END)
        assert "error" in transport.types()

    run(scenario)


def test_audio_over_the_cap_sends_one_error_and_drops_the_turn():
    async def scenario(session, transport):
        await session.on_json(start_msg())
        for _ in range(4):  # 4 x 0.6 s against a 1 s cap
            await session.on_audio_frame(pcm(0.6))
        assert transport.types().count("error") == 1
        assert transport.last("error")["turn_id"] == transport.last("turn_started")["turn_id"]

        await session.on_json(END)  # nothing to end any more
        assert transport.types().count("error") == 1

    run(scenario, max_turn_seconds=1)


def test_frames_without_a_turn_are_ignored():
    async def scenario(session, transport):
        await session.on_audio_frame(pcm(0.5))

    _, transport = run(scenario)
    assert transport.types() == ["session"]


def test_end_turn_without_a_turn_is_ignored():
    async def scenario(session, transport):
        await session.on_json(END)

    _, transport = run(scenario)
    assert transport.types() == ["session"]


def test_debug_flag_on_writes_matching_wav(tmp_path):
    data = pcm(0.5)

    async def scenario(session, transport):
        await send_turn(session, data)
        turn_id = transport.last("turn_started")["turn_id"]
        wav = tmp_path / f"{session.id}-{turn_id}.wav"
        samples, rate = sf.read(wav, dtype="int16")
        assert rate == 16000
        np.testing.assert_array_equal(samples, np.frombuffer(data, dtype="<i2"))

    run(scenario, debug_dir=tmp_path)


def test_debug_flag_off_writes_no_file(tmp_path):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))

    run(scenario)
    assert list(tmp_path.iterdir()) == []


def test_reset_rejected_while_in_flight_and_ok_when_idle():
    async def scenario(session, transport):
        reset = json.dumps({"type": "reset"})
        await session.on_json(reset)
        assert "error" not in transport.types()

        await session.on_json(start_msg())
        await session.on_json(reset)
        assert transport.last("error")["turn_id"] is None

    run(scenario)


def test_confirm_and_discard_without_review_are_errors():
    async def scenario(session, transport):
        await session.on_json('{"type": "confirm_turn", "turn_id": "t1", "text": "hi"}')
        await session.on_json('{"type": "discard_turn", "turn_id": "t1"}')
        errors = [m for m in transport.sent if m["type"] == "error"]
        assert len(errors) == 2
        assert all(m["turn_id"] == "t1" for m in errors)

    run(scenario)


def test_disconnect_drops_the_in_flight_turn():
    async def scenario(session, transport):
        await session.on_json(start_msg())
        for frame in chunks(pcm(0.5)):
            await session.on_audio_frame(frame)
        sent_before = len(transport.sent)
        await session.on_disconnect()
        assert len(transport.sent) == sent_before
        assert session.turn is None

    run(scenario)


# --- M2: ASR (T2.7) ---


def test_transcriber_gets_the_chosen_model_and_float32_samples():
    fake = FakeTranscriber()

    async def scenario(session, transport):
        await send_turn(session, pcm(0.5), model="other-model")
        await settle(session)

    run(scenario, transcriber=fake)
    assert len(fake.calls) == 1
    model, audio = fake.calls[0]
    assert model == "other-model"
    assert audio.dtype == np.float32
    assert audio.shape == (8000,)
    assert np.abs(audio).max() <= 1.0


def test_end_turn_returns_before_the_asr_finishes_and_the_session_stays_responsive():
    async def scenario(session, transport):
        gate = fake.gate
        await send_turn(session, pcm(0.5))
        # the handler came back while the ASR is still blocked
        assert session.stage == TurnStage.WORKING
        assert "transcript" not in transport.types()

        # while working: a new turn and a reset are rejected, frames and end_turn ignored
        await session.on_json(start_msg())
        assert transport.last("error")["turn_id"] is None
        await session.on_json(json.dumps({"type": "reset"}))
        assert transport.types().count("error") == 2
        await session.on_audio_frame(pcm(0.5))
        await session.on_json(END)
        assert transport.types().count("error") == 2
        assert transport.types().count("turn_started") == 1
        await asyncio.sleep(0.02)  # the job has started and is waiting on the gate
        assert len(fake.calls) == 1

        gate.set()
        await settle(session)
        assert transport.types().count("transcript") == 1

        # and the next turn works
        fake.gate = None
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types().count("transcript") == 2

    fake = FakeTranscriber(gate=asyncio.Event())
    run(scenario, transcriber=fake)


def test_asr_error_sends_a_generic_error_and_the_session_recovers():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        turn_id = transport.last("turn_started")["turn_id"]

        error = transport.last("error")
        assert error["turn_id"] == turn_id
        assert error["message"] == "transcription failed"
        assert "secret" not in error["message"]
        assert "transcript" not in transport.types()
        assert "llm_done" not in transport.types()

        fake.error = None
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types().count("transcript") == 1

    fake = FakeTranscriber(error=ASRError("secret internals"))
    run(scenario, transcriber=fake)


def test_unexpected_exception_in_the_asr_job_does_not_wedge_the_session(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)  # times out if the stage is stuck on WORKING
        error = transport.last("error")
        assert error["message"] == "transcription failed"
        assert error["turn_id"] == transport.last("turn_started")["turn_id"]
        assert session.turn is None

    with caplog.at_level(logging.ERROR):
        run(scenario, transcriber=FakeTranscriber(error=RuntimeError("secret")))
    assert any(r.exc_info for r in caplog.records)  # logged with the traceback


def test_disconnect_while_working_cancels_the_asr_and_sends_nothing():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await asyncio.sleep(0.02)  # let the job reach the transcriber
        assert session.stage == TurnStage.WORKING
        sent_before = len(transport.sent)

        await session.on_disconnect()
        fake.gate.set()
        await asyncio.sleep(0.05)

        assert fake.cancelled
        assert len(transport.sent) == sent_before
        assert session.stage == TurnStage.IDLE
        assert session.turn is None

    fake = FakeTranscriber(gate=asyncio.Event())
    run(scenario, transcriber=fake)


def test_model_load_time_is_logged_only_when_the_model_was_loaded(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    with caplog.at_level(logging.INFO, logger="backend.session"):
        run(scenario, transcriber=FakeTranscriber(model_load_ms=4321.0))
    assert any("4321" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backend.session"):
        run(scenario, transcriber=FakeTranscriber(model_load_ms=0.0))
    assert not any(r.levelno == logging.INFO for r in caplog.records if r.name == "backend.session")


def test_debug_wav_is_still_written_with_asr_on(tmp_path):
    data = pcm(0.5)

    async def scenario(session, transport):
        await send_turn(session, data)
        await settle(session)
        turn_id = transport.last("turn_started")["turn_id"]
        samples, _ = sf.read(tmp_path / f"{session.id}-{turn_id}.wav", dtype="int16")
        np.testing.assert_array_equal(samples, np.frombuffer(data, dtype="<i2"))

    run(scenario, debug_dir=tmp_path)


# --- M3: LLM reply (T3.4 - T3.7) ---

SYSTEM = {"role": "system", "content": "be brief"}


def test_turn_sends_transcript_deltas_then_llm_done_with_timings():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        turn_id = transport.last("turn_started")["turn_id"]

        assert transport.types() == [
            "session",
            "turn_started",
            "transcript",
            "llm_delta",
            "llm_delta",
            "llm_done",
        ]
        assert all(m["turn_id"] == turn_id for m in transport.sent[1:])
        transcript = transport.last("transcript")
        assert transcript["text"] == "hello world"
        assert transcript["asr_ms"] == 12.5

        deltas = [m["text"] for m in transport.sent if m["type"] == "llm_delta"]
        assert deltas == ["Hi", " there"]
        done = transport.last("llm_done")
        assert done["text"] == "Hi there"

        t = done["timings"]
        assert t["audio_s"] == 0.5
        assert t["asr_ms"] == 12.5
        assert t["llm_ttft_ms"] is not None and t["llm_ttft_ms"] >= 0
        assert t["llm_total_ms"] >= t["llm_ttft_ms"]
        assert t["e2e_ms"] >= t["llm_total_ms"]

    run(scenario)


def test_first_request_is_system_prompt_plus_the_user_transcript():
    llm = FakeLLM()

    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    run(scenario, llm=llm)
    assert llm.calls == [[SYSTEM, {"role": "user", "content": "hello world"}]]


def test_history_holds_the_user_and_assistant_messages_after_a_turn():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert session.history == [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]

    run(scenario)


def test_second_turn_sends_the_earlier_turns_too():
    llm = FakeLLM()

    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        await send_turn(session, pcm(0.5))
        await settle(session)

    run(scenario, llm=llm)
    assert [m["role"] for m in llm.calls[1]] == ["system", "user", "assistant", "user"]
    assert llm.calls[1][2]["content"] == "Hi there"


def test_reset_clears_the_history_so_the_next_request_has_no_memory():
    llm = FakeLLM()

    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        await session.on_json(json.dumps({"type": "reset"}))
        assert session.history == []
        await send_turn(session, pcm(0.5))
        await settle(session)

    run(scenario, llm=llm)
    assert [m["role"] for m in llm.calls[1]] == ["system", "user"]


def test_blank_transcript_calls_no_llm_and_leaves_the_history_alone():
    llm = FakeLLM()

    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types() == ["session", "turn_started", "transcript", "llm_done"]
        assert transport.last("transcript")["text"].strip() == ""
        done = transport.last("llm_done")  # the UI waits for it before it unlocks
        assert done["text"] == ""
        assert done["timings"]["audio_s"] == 0.5
        assert "error" not in transport.types()
        assert session.history == []

    run(scenario, transcriber=FakeTranscriber(text="  \n"), llm=llm)
    assert llm.calls == []


def test_asr_failure_adds_nothing_to_the_history_and_calls_no_llm():
    llm = FakeLLM()

    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert session.history == []
        assert "llm_delta" not in transport.types()

    run(scenario, transcriber=FakeTranscriber(error=ASRError("x")), llm=llm)
    assert llm.calls == []


def test_llm_failure_keeps_the_user_message_adds_no_reply_and_sends_a_generic_error():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        turn_id = transport.last("turn_started")["turn_id"]

        assert "transcript" in transport.types()  # ASR succeeded and was shown
        error = transport.last("error")
        assert error["turn_id"] == turn_id
        assert error["message"] == "LLM request failed"
        assert "llm_done" not in transport.types()
        assert session.history == [{"role": "user", "content": "hello world"}]

        # the session is usable and the next request still contains the failed question
        llm.error = None
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types().count("llm_done") == 1
        assert [m["role"] for m in llm.calls[1]] == ["system", "user", "user"]

    llm = FakeLLM(error=RuntimeError("secret internals"))
    run(scenario, llm=llm)


def test_llm_failure_in_the_middle_of_the_stream_adds_no_assistant_message():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types().count("llm_delta") == 1  # the part that was already sent
        assert transport.last("error")["message"] == "LLM request failed"
        assert "llm_done" not in transport.types()
        assert session.history == [{"role": "user", "content": "hello world"}]

    run(scenario, llm=FakeLLM(deltas=("Hi", " there"), error=RuntimeError("x"), error_after=1))


def test_llm_error_text_never_reaches_the_user():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert "secret" not in json.dumps(transport.sent)

    run(scenario, llm=FakeLLM(error=RuntimeError("secret internals")))


def test_llm_error_is_logged_as_a_warning_without_a_second_traceback(caplog):
    # llm.py already logged the details, so the session adds one line, not another traceback
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.last("error")["message"] == "LLM request failed"

    with caplog.at_level(logging.INFO, logger="backend.session"):
        run(scenario, llm=FakeLLM(error=LLMError("LLM request failed")))
    records = [
        r for r in caplog.records if r.name == "backend.session" and r.levelno >= logging.WARNING
    ]
    assert len(records) == 1 and records[0].exc_info is None


def test_unexpected_error_in_the_llm_step_keeps_its_traceback_in_the_log(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    with caplog.at_level(logging.INFO, logger="backend.session"):
        run(scenario, llm=FakeLLM(error=RuntimeError("bug")))
    assert any(r.exc_info for r in caplog.records)


def test_history_is_trimmed_to_the_token_budget_oldest_first():
    llm = FakeLLM(deltas=("x" * 400,))  # a reply of roughly 100 tokens

    async def scenario(session, transport):
        for _ in range(2):
            await send_turn(session, pcm(0.5))
            await settle(session)
        # two full turns are ~210 tokens against a budget of 150: the first turn goes
        assert [m["role"] for m in session.history] == ["user", "assistant"]
        await send_turn(session, pcm(0.5))
        await settle(session)

    run(scenario, llm=llm, max_history_tokens=150)
    assert [m["role"] for m in llm.calls[2]] == ["system", "user", "assistant", "user"]
    assert llm.calls[2][0] == SYSTEM  # the system prompt is never trimmed


def test_start_turn_and_reset_are_rejected_while_the_reply_streams():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await asyncio.sleep(0.05)  # first delta sent, now waiting on the gate
        assert session.stage == TurnStage.WORKING
        assert transport.types().count("llm_delta") == 1

        await session.on_json(start_msg())
        await session.on_json(json.dumps({"type": "reset"}))
        errors = [m for m in transport.sent if m["type"] == "error"]
        assert len(errors) == 2 and all(m["turn_id"] is None for m in errors)
        assert session.history[0] == {"role": "user", "content": "hello world"}

        llm.gate.set()
        await settle(session)
        assert transport.last("llm_done")["text"] == "Hi there"

    llm = FakeLLM(gate=asyncio.Event())
    run(scenario, llm=llm)


def test_disconnect_while_the_reply_streams_cancels_the_llm_and_sends_nothing():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await asyncio.sleep(0.05)
        sent_before = len(transport.sent)

        await session.on_disconnect()
        llm.gate.set()
        await asyncio.sleep(0.05)

        assert llm.cancelled
        assert len(transport.sent) == sent_before
        assert session.stage == TurnStage.IDLE

    llm = FakeLLM(gate=asyncio.Event())
    run(scenario, llm=llm)


# --- M4: review mode (T4.6) ---


def confirm_msg(turn_id: str, text: str) -> str:
    return json.dumps({"type": "confirm_turn", "turn_id": turn_id, "text": text})


def discard_msg(turn_id: str) -> str:
    return json.dumps({"type": "discard_turn", "turn_id": turn_id})


async def review_turn(session: Session, data: bytes | None = None) -> None:
    """A turn with review on, up to the point where it waits for the user."""
    await session.on_json(start_msg(review=True))
    for frame in chunks(data or pcm(0.5)):
        await session.on_audio_frame(frame)
    await session.on_json(END)
    async with asyncio.timeout(2):
        while session.stage != TurnStage.AWAITING_CONFIRM:  # noqa: ASYNC110
            await asyncio.sleep(0.005)


def test_review_turn_stops_after_the_transcript_and_waits():
    llm = FakeLLM()

    async def scenario(session, transport):
        await review_turn(session)
        await asyncio.sleep(0.05)
        assert transport.types() == ["session", "turn_started", "transcript"]
        assert session.stage == TurnStage.AWAITING_CONFIRM
        assert session.history == []

    run(scenario, llm=llm)
    assert llm.calls == []


def test_confirm_sends_the_reply_and_the_history_gets_the_text():
    llm = FakeLLM()

    async def scenario(session, transport):
        await review_turn(session)
        turn_id = transport.last("turn_started")["turn_id"]
        await session.on_json(confirm_msg(turn_id, "hello world"))
        await settle(session)

        assert transport.types()[2:] == ["transcript", "llm_delta", "llm_delta", "llm_done"]
        done = transport.last("llm_done")
        assert done["turn_id"] == turn_id and done["text"] == "Hi there"
        assert done["timings"]["audio_s"] == 0.5  # from the original transcription
        assert done["timings"]["asr_ms"] == 12.5
        assert session.history == [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]

    run(scenario, llm=llm)
    assert llm.calls == [[SYSTEM, {"role": "user", "content": "hello world"}]]


def test_confirm_with_edited_text_sends_the_edit_to_the_llm_and_history():
    llm = FakeLLM()

    async def scenario(session, transport):
        await review_turn(session)
        turn_id = transport.last("turn_started")["turn_id"]
        await session.on_json(confirm_msg(turn_id, "  what is the weather  "))
        await settle(session)
        assert transport.last("transcript")["text"] == "hello world"  # the original, unchanged
        assert session.history[0] == {"role": "user", "content": "what is the weather"}

    run(scenario, llm=llm)
    assert llm.calls[0][1] == {"role": "user", "content": "what is the weather"}


def test_discard_ends_the_turn_adds_nothing_and_sends_nothing():
    llm = FakeLLM()

    async def scenario(session, transport):
        await review_turn(session)
        turn_id = transport.last("turn_started")["turn_id"]
        sent_before = len(transport.sent)
        await session.on_json(discard_msg(turn_id))
        await asyncio.sleep(0.02)

        assert len(transport.sent) == sent_before
        assert session.stage == TurnStage.IDLE
        assert session.turn is None
        assert session.history == []

        # the session is usable again
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types().count("llm_done") == 1

    run(scenario, llm=llm)
    assert len(llm.calls) == 1


def test_wrong_turn_id_is_an_error_and_the_waiting_turn_survives():
    llm = FakeLLM()

    async def scenario(session, transport):
        await review_turn(session)
        turn_id = transport.last("turn_started")["turn_id"]

        await session.on_json(confirm_msg("wrong", "hi"))
        assert transport.last("error")["turn_id"] == "wrong"
        await session.on_json(discard_msg("wrong"))
        assert transport.last("error")["turn_id"] == "wrong"
        assert transport.types().count("error") == 2
        assert session.stage == TurnStage.AWAITING_CONFIRM

        await session.on_json(confirm_msg(turn_id, "hello world"))  # the right id still works
        await settle(session)
        assert transport.types().count("llm_done") == 1

    run(scenario, llm=llm)


def test_blank_confirm_is_an_error_and_ends_the_turn():
    llm = FakeLLM()

    async def scenario(session, transport):
        await review_turn(session)
        turn_id = transport.last("turn_started")["turn_id"]
        await session.on_json(confirm_msg(turn_id, "   "))
        error = transport.last("error")
        assert error["turn_id"] == turn_id
        assert session.stage == TurnStage.IDLE  # the browser ends its turn on an error too
        assert session.history == []

    run(scenario, llm=llm)
    assert llm.calls == []


def test_blank_transcript_in_review_mode_waits_and_can_be_discarded():
    llm = FakeLLM()

    async def scenario(session, transport):
        await review_turn(session)
        assert transport.last("transcript")["text"].strip() == ""
        assert "llm_done" not in transport.types()
        await session.on_json(discard_msg(transport.last("turn_started")["turn_id"]))
        assert session.stage == TurnStage.IDLE

    run(scenario, transcriber=FakeTranscriber(text=""), llm=llm)
    assert llm.calls == []


def test_the_turn_stays_in_flight_while_waiting_for_review():
    async def scenario(session, transport):
        await review_turn(session)
        errors_before = transport.types().count("error")

        await session.on_json(start_msg())
        await session.on_json(json.dumps({"type": "reset"}))
        assert transport.types().count("error") == errors_before + 2
        assert transport.types().count("turn_started") == 1

        await session.on_audio_frame(pcm(0.5))  # ignored
        await session.on_json(END)  # ignored
        assert transport.types().count("error") == errors_before + 2
        assert session.stage == TurnStage.AWAITING_CONFIRM

    run(scenario)


def test_confirm_and_discard_are_rejected_in_every_other_stage():
    gate = asyncio.Event()

    async def scenario(session, transport):
        # idle
        await session.on_json(confirm_msg("t", "hi"))
        await session.on_json(discard_msg("t"))
        # receiving
        await session.on_json(start_msg())
        turn_id = transport.last("turn_started")["turn_id"]
        await session.on_json(confirm_msg(turn_id, "hi"))
        await session.on_json(discard_msg(turn_id))
        assert session.stage == TurnStage.RECEIVING
        for frame in chunks(pcm(0.5)):
            await session.on_audio_frame(frame)
        await session.on_json(END)
        await asyncio.sleep(0.02)
        # working (the ASR is held on the gate)
        await session.on_json(confirm_msg(turn_id, "hi"))
        await session.on_json(discard_msg(turn_id))
        assert session.stage == TurnStage.WORKING
        assert transport.types().count("error") == 6
        assert all(m["turn_id"] in ("t", turn_id) for m in transport.sent if m["type"] == "error")
        gate.set()
        await settle(session)

    run(scenario, transcriber=FakeTranscriber(gate=gate))


def test_a_second_confirm_for_the_same_turn_is_rejected():
    llm = FakeLLM(gate=asyncio.Event())

    async def scenario(session, transport):
        await review_turn(session)
        turn_id = transport.last("turn_started")["turn_id"]
        await session.on_json(confirm_msg(turn_id, "hello world"))
        await asyncio.sleep(0.05)  # the reply is streaming now
        await session.on_json(confirm_msg(turn_id, "hello again"))
        assert transport.last("error")["turn_id"] == turn_id
        llm.gate.set()
        await settle(session)
        assert transport.types().count("llm_done") == 1

    run(scenario, llm=llm)
    assert len(llm.calls) == 1


def test_e2e_time_excludes_the_users_thinking_time():
    async def scenario(session, transport):
        await review_turn(session)
        await asyncio.sleep(0.4)  # the user reads and edits
        await session.on_json(confirm_msg(transport.last("turn_started")["turn_id"], "hi"))
        await settle(session)
        timings = transport.last("llm_done")["timings"]
        assert timings["e2e_ms"] < 300
        assert timings["e2e_ms"] >= timings["llm_total_ms"]

    run(scenario)


def test_llm_failure_after_confirm_keeps_the_confirmed_text_in_history():
    async def scenario(session, transport):
        await review_turn(session)
        turn_id = transport.last("turn_started")["turn_id"]
        await session.on_json(confirm_msg(turn_id, "edited question"))
        await settle(session)
        assert transport.last("error")["message"] == "LLM request failed"
        assert session.history == [{"role": "user", "content": "edited question"}]

    run(scenario, llm=FakeLLM(error=RuntimeError("secret")))


def test_disconnect_while_waiting_for_review_drops_the_turn():
    async def scenario(session, transport):
        await review_turn(session)
        sent_before = len(transport.sent)
        await session.on_disconnect()
        assert len(transport.sent) == sent_before
        assert session.stage == TurnStage.IDLE
        assert session.turn is None
        await session.on_json(confirm_msg("anything", "hi"))  # nothing is waiting any more
        assert transport.last("error")["turn_id"] == "anything"

    run(scenario)


# --- M5: timings with a fake clock (T5.1) ---
# SPEC 6.3.2: asr_ms comes from the transcriber; llm_ttft_ms and llm_total_ms start when the
# request is sent; e2e_ms runs from end_turn to the last delta, minus review think time.


def timing_run(scenario, **fake_kwargs):
    clock = FakeClock()
    transcriber = FakeTranscriber(clock=clock, takes_s=0.3, asr_ms=250.0)
    llm = FakeLLM(deltas=("A", "B"), clock=clock, delays=(0.5, 0.25), **fake_kwargs)

    async def go(session, transport):
        await scenario(session, transport, clock)

    run(go, transcriber=transcriber, llm=llm, clock=clock)


def test_timings_follow_the_spec_definitions():
    async def scenario(session, transport, clock):
        await send_turn(session, pcm(0.5))
        await settle(session)
        t = transport.last("llm_done")["timings"]
        assert t["audio_s"] == 0.5
        assert t["asr_ms"] == 250.0  # the transcriber's own number, not the session clock
        assert t["llm_ttft_ms"] == pytest.approx(500)
        assert t["llm_total_ms"] == pytest.approx(750)
        assert t["e2e_ms"] == pytest.approx(1050)  # 0.3 ASR + 0.75 LLM

    timing_run(scenario)


def test_e2e_starts_at_end_turn_not_at_start_turn():
    async def scenario(session, transport, clock):
        await session.on_json(start_msg())
        clock.advance(5.0)  # the user is still talking
        for frame in chunks(pcm(0.5)):
            await session.on_audio_frame(frame)
        await session.on_json(END)
        await settle(session)
        assert transport.last("llm_done")["timings"]["e2e_ms"] == pytest.approx(1050)

    timing_run(scenario)


def test_review_think_time_is_excluded_from_e2e_but_asr_time_counts():
    async def scenario(session, transport, clock):
        await review_turn(session)
        clock.advance(10.0)  # the user reads and edits
        await session.on_json(confirm_msg(transport.last("turn_started")["turn_id"], "hi"))
        await settle(session)
        t = transport.last("llm_done")["timings"]
        assert t["e2e_ms"] == pytest.approx(1050)  # not 11050
        assert t["llm_total_ms"] == pytest.approx(750)

    timing_run(scenario)


def test_blank_transcript_has_only_audio_and_asr_timings():
    async def scenario(session, transport, clock):
        await send_turn(session, pcm(0.5))
        await settle(session)
        t = transport.last("llm_done")["timings"]
        assert t["audio_s"] == 0.5 and t["asr_ms"] == 250.0
        assert t["llm_ttft_ms"] is None
        assert t["llm_total_ms"] is None
        assert t["e2e_ms"] is None

    clock = FakeClock()
    run(
        lambda session, transport: scenario(session, transport, clock),
        transcriber=FakeTranscriber(text="", clock=clock, takes_s=0.3, asr_ms=250.0),
        clock=clock,
    )


def test_a_reply_with_no_text_has_no_first_token_time():
    async def scenario(session, transport, clock):
        await send_turn(session, pcm(0.5))
        await settle(session)
        t = transport.last("llm_done")["timings"]
        assert t["llm_ttft_ms"] is None
        assert t["llm_total_ms"] == 0
        assert session.history == [{"role": "user", "content": "hello world"}]

    clock = FakeClock()
    run(
        lambda session, transport: scenario(session, transport, clock),
        llm=FakeLLM(deltas=()),
        clock=clock,
    )


# --- M5: one structured log line per turn (T5.3) ---

TURN_LOG = "astra.turns"
TIMING_KEYS = {"audio_s", "asr_ms", "llm_ttft_ms", "llm_total_ms", "e2e_ms"}


def turn_records(caplog):
    """The parsed JSON of every astra.turns record. Fails if a line is not valid JSON."""
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == TURN_LOG]


def run_logged(caplog, scenario, **kwargs):
    with caplog.at_level(logging.INFO, logger=TURN_LOG):
        run(scenario, **kwargs)
    return turn_records(caplog)


def test_a_good_turn_logs_one_ok_record_with_all_fields(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        scenario.ids = (session.id, transport.last("turn_started")["turn_id"])

    records = run_logged(caplog, scenario)
    assert len(records) == 1
    record = records[0]
    session_id, turn_id = scenario.ids
    assert record["event"] == "turn"
    assert record["session_id"] == session_id
    assert record["turn_id"] == turn_id
    assert record["asr_model"] == MODELS[0]
    assert record["llm_model"] == "test/model"
    assert record["status"] == "ok"
    assert record["error"] is None
    assert set(record["timings"]) == TIMING_KEYS
    assert record["timings"]["audio_s"] == 0.5
    assert record["timings"]["asr_ms"] == 12.5
    assert all(isinstance(v, int | float) for v in record["timings"].values())


def test_the_logged_timings_are_the_ones_sent_to_the_browser(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        scenario.sent = transport.last("llm_done")["timings"]

    records = run_logged(caplog, scenario)
    assert records[0]["timings"] == scenario.sent


def test_the_log_never_contains_the_transcript_or_the_reply(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    run_logged(caplog, scenario, transcriber=FakeTranscriber(text="my secret sentence"))
    lines = " ".join(r.getMessage() for r in caplog.records if r.name == TURN_LOG)
    assert "secret" not in lines and "Hi there" not in lines


def test_one_record_per_turn_across_several_turns(caplog):
    async def scenario(session, transport):
        for _ in range(3):
            await send_turn(session, pcm(0.5))
            await settle(session)

    records = run_logged(caplog, scenario)
    assert [r["status"] for r in records] == ["ok", "ok", "ok"]
    assert len({r["turn_id"] for r in records}) == 3


def test_a_blank_transcript_is_logged_as_empty(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    records = run_logged(caplog, scenario, transcriber=FakeTranscriber(text=""))
    assert [r["status"] for r in records] == ["empty"]
    assert records[0]["timings"]["llm_ttft_ms"] is None


def test_an_asr_failure_is_logged_as_asr_error_at_warning(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    records = run_logged(
        caplog, scenario, transcriber=FakeTranscriber(error=ASRError("secret internals"))
    )
    assert len(records) == 1
    assert records[0]["status"] == "asr_error"
    assert records[0]["error"] == "transcription failed"
    assert records[0]["timings"] is None
    turn_lines = [r for r in caplog.records if r.name == TURN_LOG]
    assert turn_lines[0].levelno == logging.WARNING
    assert "secret" not in turn_lines[0].getMessage()


def test_an_llm_failure_is_logged_once_as_llm_error(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    records = run_logged(caplog, scenario, llm=FakeLLM(error=RuntimeError("secret")))
    assert [r["status"] for r in records] == ["llm_error"]
    assert records[0]["error"] == "LLM request failed"


def test_an_llm_failure_after_review_confirm_is_logged_once(caplog):
    async def scenario(session, transport):
        await review_turn(session)
        await session.on_json(confirm_msg(transport.last("turn_started")["turn_id"], "hi"))
        await settle(session)

    records = run_logged(caplog, scenario, llm=FakeLLM(error=RuntimeError("x")))
    assert [r["status"] for r in records] == ["llm_error"]


def test_rejected_audio_is_logged_with_the_error_text(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.1))  # too short
        scenario.shown = transport.last("error")["message"]

    records = run_logged(caplog, scenario)
    assert [r["status"] for r in records] == ["rejected"]
    assert records[0]["error"] == scenario.shown
    assert records[0]["timings"] is None


def test_audio_over_the_cap_is_logged_once(caplog):
    async def scenario(session, transport):
        await session.on_json(start_msg())
        for _ in range(4):
            await session.on_audio_frame(pcm(0.6))
        await session.on_json(END)

    records = run_logged(caplog, scenario, max_turn_seconds=1)
    assert [r["status"] for r in records] == ["rejected"]


def test_discard_is_logged_at_info(caplog):
    async def scenario(session, transport):
        await review_turn(session)
        await session.on_json(discard_msg(transport.last("turn_started")["turn_id"]))

    records = run_logged(caplog, scenario)
    assert [r["status"] for r in records] == ["discarded"]
    assert [r.levelno for r in caplog.records if r.name == TURN_LOG] == [logging.INFO]


def test_blank_confirm_is_logged_as_rejected(caplog):
    async def scenario(session, transport):
        await review_turn(session)
        await session.on_json(confirm_msg(transport.last("turn_started")["turn_id"], "  "))

    records = run_logged(caplog, scenario)
    assert [r["status"] for r in records] == ["rejected"]
    assert records[0]["error"] == "transcript is empty"


def test_a_dropped_connection_is_logged_once_in_every_stage(caplog):
    async def receiving(session, transport):
        await session.on_json(start_msg())
        await session.on_disconnect()

    async def working(session, transport):
        await send_turn(session, pcm(0.5))
        await asyncio.sleep(0.02)
        await session.on_disconnect()

    async def reviewing(session, transport):
        await review_turn(session)
        await session.on_disconnect()

    for scenario, kwargs in (
        (receiving, {}),
        (working, {"transcriber": FakeTranscriber(gate=asyncio.Event())}),
        (reviewing, {}),
    ):
        caplog.clear()
        records = run_logged(caplog, scenario, **kwargs)
        assert [r["status"] for r in records] == ["dropped"], scenario.__name__
        assert records[0]["error"] is None


def test_disconnect_with_no_turn_logs_nothing(caplog):
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        caplog.clear()
        await session.on_disconnect()

    with caplog.at_level(logging.INFO, logger=TURN_LOG):
        run(scenario)
    assert turn_records(caplog) == []


def test_the_turn_log_names_the_model_the_llm_client_is_using(caplog):
    # With `hosted_vllm/auto` the configured string is not the model; the client knows the real one.
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)

    records = run_logged(caplog, scenario, llm=FakeLLM(model="hosted_vllm/Qwen/Qwen2.5-1.5B"))
    assert records[0]["llm_model"] == "hosted_vllm/Qwen/Qwen2.5-1.5B"
