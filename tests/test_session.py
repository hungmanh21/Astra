"""Tests for backend/session.py (PLAN T1.6, T1.8, T2.7).

Each test drives a Session through a FakeTransport and inspects what it sent.
"""

import asyncio
import json
import logging
from typing import Any

import numpy as np
import soundfile as sf

from backend.asr.base import ASRError
from backend.session import Session, TurnStage
from tests.helpers import MODELS, FakeTranscriber, chunks, make_settings, pcm


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


def run(scenario, transcriber: FakeTranscriber | None = None, **settings_kwargs):
    """Run `scenario(session, transport)` on a fresh connected session."""
    transport = FakeTransport()
    session = Session(transport, make_settings(**settings_kwargs), transcriber or FakeTranscriber())

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


def test_end_turn_sends_transcript_then_llm_done_placeholder():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        turn_id = transport.last("turn_started")["turn_id"]

        assert transport.types() == ["session", "turn_started", "transcript", "llm_done"]
        transcript = transport.last("transcript")
        assert transcript["turn_id"] == turn_id
        assert transcript["text"] == "hello world"
        assert transcript["asr_ms"] == 12.5

        # M2 placeholder: the UI waits for llm_done, so it gets one with no text.
        done = transport.last("llm_done")
        assert done["turn_id"] == turn_id
        assert done["text"] == ""
        assert done["timings"]["audio_s"] == 0.5
        assert done["timings"]["asr_ms"] == 12.5

    run(scenario)


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


def test_empty_transcript_is_sent_as_is_not_as_an_error():
    async def scenario(session, transport):
        await send_turn(session, pcm(0.5))
        await settle(session)
        assert transport.last("transcript")["text"] == ""
        assert "error" not in transport.types()

    run(scenario, transcriber=FakeTranscriber(text=""))


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
    assert not any(r.levelno == logging.INFO for r in caplog.records)


def test_debug_wav_is_still_written_with_asr_on(tmp_path):
    data = pcm(0.5)

    async def scenario(session, transport):
        await send_turn(session, data)
        await settle(session)
        turn_id = transport.last("turn_started")["turn_id"]
        samples, _ = sf.read(tmp_path / f"{session.id}-{turn_id}.wav", dtype="int16")
        np.testing.assert_array_equal(samples, np.frombuffer(data, dtype="<i2"))

    run(scenario, debug_dir=tmp_path)
