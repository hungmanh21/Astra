"""Session tests for live partials and streaming strategies (PLAN_v1 S1.6).

Same style as test_session.py: drive a Session through a FakeTransport and inspect what it
sent. The fake transcriber (tests/helpers.py) hands out scripted streams, so no model is needed.
"""

import asyncio
import json
import logging

import numpy as np
import pytest

from backend.asr.base import ASRError, PartialResult
from backend.asr.service import ModelStrategies
from backend.audio import pcm16_to_float32
from backend.session import TurnStage
from tests.helpers import MODELS, FakeClock, FakeLLM, FakeTranscriber, chunks, pcm
from tests.test_session import END, TURN_LOG, run, settle

OTHER = MODELS[1]
P1 = PartialResult("hello", "")
P2 = PartialResult("hello wor", "ld")
P3 = PartialResult("hello world", "")


def start_with(model: str = MODELS[0], strategy: str | None = None, review: bool = False) -> str:
    msg = {"type": "start_turn", "asr_model": model, "review": review}
    if strategy is not None:
        msg["stream_strategy"] = strategy
    return json.dumps(msg)


async def stream_audio(session, data: bytes, size: int = 1600) -> None:
    """Send frames like a browser would, giving the event loop a turn after each one."""
    for frame in chunks(data, size):
        await session.on_audio_frame(frame)
        await asyncio.sleep(0)  # lets a started partial decode run


async def whole_turn(session, data: bytes, **start_kwargs) -> None:
    await session.on_json(start_with(**start_kwargs))
    await stream_audio(session, data)
    await session.on_json(END)


async def wait_for_stage(session, stage: TurnStage, within_s: float = 2.0) -> None:
    async with asyncio.timeout(within_s):
        while session.stage != stage:  # noqa: ASYNC110  (polls a plain attribute)
            await asyncio.sleep(0.005)


def partials(transport) -> list[dict]:
    return [m for m in transport.sent if m["type"] == "partial"]


def no_other_tasks() -> bool:
    return len(asyncio.all_tasks()) == 1  # only the scenario itself


SUPPORTED = {
    MODELS[0]: ModelStrategies(["chunked", "redecode", "trimmed"], "redecode"),
    OTHER: ModelStrategies(["chunked", "redecode", "native"], "native"),
}


# --- the session message and strategy choice (FR-22) ---


def test_session_message_lists_each_models_strategies():
    async def scenario(session, transport):
        pass

    _, transport = run(scenario, transcriber=FakeTranscriber(supported=SUPPORTED))
    assert transport.sent[0]["stream_strategies"] == {
        MODELS[0]: {"supported": ["chunked", "redecode", "trimmed"], "default": "redecode"},
        OTHER: {"supported": ["chunked", "redecode", "native"], "default": "native"},
    }


def test_a_model_that_cannot_be_inspected_is_left_out_and_the_rest_connect():
    class OneBroken(FakeTranscriber):
        def strategies(self, model_name):
            if model_name == OTHER:
                raise ASRError("cannot create the adapter")
            return super().strategies(model_name)

    async def scenario(session, transport):
        pass

    _, transport = run(scenario, transcriber=OneBroken())
    info = transport.sent[0]
    assert info["asr_models"] == MODELS  # still selectable in the list
    assert list(info["stream_strategies"]) == [MODELS[0]]


def test_missing_strategy_means_the_models_default():
    fake = FakeTranscriber(supported=SUPPORTED)

    async def scenario(session, transport):
        await whole_turn(session, pcm(0.5), model=OTHER)
        await settle(session)
        assert transport.last("transcript")["stream_strategy"] == "native"

    run(scenario, transcriber=fake)
    assert fake.opened == [(OTHER, "native")]


def test_the_requested_strategy_is_used():
    fake = FakeTranscriber(supported=SUPPORTED)

    async def scenario(session, transport):
        await whole_turn(session, pcm(0.5), model=OTHER, strategy="chunked")
        await settle(session)
        assert transport.last("transcript")["stream_strategy"] == "chunked"
        assert transport.last("llm_done")["timings"]["stream_strategy"] == "chunked"

    run(scenario, transcriber=fake)
    assert fake.opened == [(OTHER, "chunked")]


def test_unsupported_strategy_is_rejected_and_the_turn_never_starts():
    fake = FakeTranscriber(supported=SUPPORTED)

    async def scenario(session, transport):
        await session.on_json(start_with(MODELS[0], strategy="native"))  # not for this model
        error = transport.last("error")
        assert error["turn_id"] is None
        assert "native" in error["message"] and MODELS[0] in error["message"]
        assert "turn_started" not in transport.types()
        assert session.stage == TurnStage.IDLE
        assert session.turn is None
        assert fake.opened == []  # nothing was opened, so nothing was loaded

        await session.on_json(start_with(MODELS[0], strategy="bogus"))
        assert transport.types().count("error") == 2

        # the session is fine afterwards
        await whole_turn(session, pcm(0.5), strategy="trimmed")
        await settle(session)
        assert transport.types().count("transcript") == 1

    run(scenario, transcriber=fake)
    assert fake.opened == [(MODELS[0], "trimmed")]


def test_unknown_model_opens_nothing():
    fake = FakeTranscriber()

    async def scenario(session, transport):
        await session.on_json(start_with("nope", strategy="redecode"))
        assert transport.last("error")["message"] == "unknown ASR model: nope"
        assert fake.opened == []

    run(scenario, transcriber=fake)


def test_a_stream_that_cannot_open_gives_a_generic_error_and_no_turn(caplog):
    fake = FakeTranscriber(open_error=ASRError("secret internals"))

    async def scenario(session, transport):
        await session.on_json(start_with())
        error = transport.last("error")
        assert error["turn_id"] is None
        assert "secret" not in error["message"]
        assert "turn_started" not in transport.types()
        assert session.stage == TurnStage.IDLE

        fake.open_error = None
        await whole_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types().count("transcript") == 1

    with caplog.at_level(logging.ERROR):
        run(scenario, transcriber=fake)
    assert any(r.exc_info for r in caplog.records)  # the details went to the log


# --- audio reaches the stream ---


@pytest.mark.parametrize("size", [1601, 7, 1599])
def test_every_sample_reaches_the_stream_whatever_the_frame_sizes(size):
    data = pcm(1.0)
    fake = FakeTranscriber()

    async def scenario(session, transport):
        await session.on_json(start_with())
        await stream_audio(session, data, size=size)  # most frames end in the middle of a sample
        await session.on_json(END)
        await settle(session)

    run(scenario, transcriber=fake)
    np.testing.assert_array_equal(fake.streams[0].samples, pcm16_to_float32(data))
    np.testing.assert_array_equal(fake.calls[0][1], pcm16_to_float32(data))


def test_empty_frames_are_harmless():
    data = pcm(0.5)
    fake = FakeTranscriber()

    async def scenario(session, transport):
        await session.on_json(start_with())
        await session.on_audio_frame(b"")
        await stream_audio(session, data)
        await session.on_audio_frame(b"")
        await session.on_json(END)
        await settle(session)
        assert transport.types().count("transcript") == 1

    run(scenario, transcriber=fake)
    np.testing.assert_array_equal(fake.streams[0].samples, pcm16_to_float32(data))


def test_each_turn_gets_its_own_stream():
    fake = FakeTranscriber()

    async def scenario(session, transport):
        await whole_turn(session, pcm(0.5))
        await settle(session)
        await whole_turn(session, pcm(0.75))
        await settle(session)

    run(scenario, transcriber=fake)
    assert len(fake.streams) == 2
    assert [len(s.samples) for s in fake.streams] == [8000, 12000]


# --- partials ---


def test_partials_arrive_in_order_while_recording_and_before_the_transcript():
    fake = FakeTranscriber(partials=[P1, P2, P3])

    async def scenario(session, transport):
        await session.on_json(start_with())
        turn_id = transport.last("turn_started")["turn_id"]
        await stream_audio(session, pcm(1.5))
        # still recording: the partials are already out
        assert [(m["committed"], m["tentative"]) for m in partials(transport)] == [
            ("hello", ""),
            ("hello wor", "ld"),
            ("hello world", ""),
        ]
        assert all(m["turn_id"] == turn_id for m in partials(transport))
        assert "transcript" not in transport.types()

        await session.on_json(END)
        await settle(session)
        types = transport.types()
        assert types.index("turn_started") < types.index("partial")
        assert max(i for i, t in enumerate(types) if t == "partial") < types.index("transcript")
        assert types[-1] == "llm_done"

    run(scenario, transcriber=fake)


def test_no_partial_follows_the_transcript_even_if_a_decode_is_still_running():
    fake = FakeTranscriber(partials=[P1], feed_gate=asyncio.Event())

    async def scenario(session, transport):
        await session.on_json(start_with())
        await stream_audio(session, pcm(0.6))  # one decode starts and is held
        await session.on_json(END)
        await asyncio.sleep(0.02)
        fake.feed_gate.set()  # the decode finishes after end_turn
        await settle(session)
        assert "transcript" in transport.types()
        assert partials(transport) == []

    run(scenario, transcriber=fake)


def test_the_partial_callback_ignores_a_turn_that_is_over_or_not_current():
    fake = FakeTranscriber()
    late = PartialResult("late", "")

    async def scenario(session, transport):
        await session.on_json(start_with())
        first = session.turn
        await stream_audio(session, pcm(0.5))
        await session.on_json(END)
        await settle(session)

        sent = len(transport.sent)
        await session._on_partial(first, late, 0.0)  # its turn is over
        assert len(transport.sent) == sent
        assert first.partial_count == 0

        await session.on_json(start_with())  # a new turn is current now
        sent = len(transport.sent)
        await session._on_partial(first, late, 0.0)  # a stale callback for the old one
        assert len(transport.sent) == sent

        current = session.turn
        await session._on_partial(current, late, 0.0)  # the current one does go out
        assert partials(transport)[-1]["committed"] == "late"
        assert current.partial_count == 1

    run(scenario, transcriber=fake)


# --- timings (SPEC_v1 6.3) ---


def test_timings_have_the_three_streaming_values():
    clock = FakeClock()
    fake = FakeTranscriber(
        partials=[P1, P2, P3],
        clock=clock,
        feed_takes_s=[0.1, 0.2, 0.6],  # each partial is this late behind its audio
        takes_s=0.4,  # the final decode
        asr_ms=12.5,
    )

    async def scenario(session, transport):
        await whole_turn(session, pcm(1.5))
        await settle(session)
        timings = transport.last("llm_done")["timings"]
        assert timings["partial_count"] == 3
        assert timings["partial_lag_ms"] == pytest.approx(
            200
        )  # median of 100, 200, 600 (not the mean or last)
        assert timings["final_ms"] == pytest.approx(400)
        assert timings["stream_strategy"] == "redecode"
        assert timings["asr_ms"] == 12.5  # still the final decode only

    run(scenario, transcriber=fake, clock=clock)


def test_a_turn_without_partials_reports_zero_and_no_lag():
    clock = FakeClock()
    fake = FakeTranscriber(clock=clock, takes_s=0.25)

    async def scenario(session, transport):
        await whole_turn(session, pcm(0.5))
        await settle(session)
        timings = transport.last("llm_done")["timings"]
        assert timings["partial_count"] == 0
        assert timings["partial_lag_ms"] is None
        assert timings["final_ms"] == pytest.approx(250)

    run(scenario, transcriber=fake, clock=clock)


def test_an_empty_transcript_still_reports_the_streaming_values():
    async def scenario(session, transport):
        await whole_turn(session, pcm(0.5))
        await settle(session)
        timings = transport.last("llm_done")["timings"]
        assert transport.last("llm_done")["text"] == ""
        assert timings["partial_count"] == 0
        assert timings["stream_strategy"] == "redecode"
        assert timings["final_ms"] is not None

    run(scenario, transcriber=FakeTranscriber(text="  "))


# --- a failing partial (FR-18) ---


def test_a_failing_partial_decode_does_not_break_the_turn():
    fake = FakeTranscriber(feed_error=RuntimeError("decode blew up"))

    async def scenario(session, transport):
        await whole_turn(session, pcm(1.5))
        await settle(session)
        assert "error" not in transport.types()
        assert partials(transport) == []
        assert transport.last("transcript")["text"] == "hello world"
        assert transport.last("llm_done")["timings"]["partial_count"] == 0

    run(scenario, transcriber=fake)
    # the final decode still saw every sample, even though partials broke early
    assert len(fake.calls[0][1]) == 24000


def test_a_failing_final_is_still_an_asr_error_after_partials():
    fake = FakeTranscriber(partials=[P1], error=ASRError("secret"))

    async def scenario(session, transport):
        await whole_turn(session, pcm(0.6))
        await settle(session)
        assert transport.last("error")["message"] == "transcription failed"
        assert "transcript" not in transport.types()
        assert "llm_done" not in transport.types()

    run(scenario, transcriber=fake)


# --- rejection, disconnect, reset: the stream is closed ---


def test_a_rejected_turn_stops_the_decode_in_flight():
    fake = FakeTranscriber(feed_gate=asyncio.Event())

    async def scenario(session, transport):
        await session.on_json(start_with())
        turn_id = transport.last("turn_started")["turn_id"]
        await stream_audio(session, pcm(1.2))  # a decode starts at 0.5 s; 1.0 s is the cap
        error = transport.last("error")
        assert error["turn_id"] == turn_id
        assert "too long" in error["message"]
        assert session.stage == TurnStage.IDLE
        assert fake.cancelled  # the held decode was stopped, not left behind
        assert no_other_tasks()

        fake.feed_gate = None  # and the next turn works
        await whole_turn(session, pcm(0.5))
        await settle(session)
        assert transport.types().count("transcript") == 1

    run(scenario, transcriber=fake, max_turn_seconds=1)


def test_a_too_short_recording_is_rejected_without_a_final_decode():
    fake = FakeTranscriber()

    async def scenario(session, transport):
        await whole_turn(session, pcm(0.1))
        await settle(session)
        assert "too short" in transport.last("error")["message"]
        assert "transcript" not in transport.types()
        assert fake.calls == []  # finish() was never called
        assert no_other_tasks()

    run(scenario, transcriber=fake)


def test_disconnect_while_recording_stops_the_decode_and_sends_nothing():
    fake = FakeTranscriber(feed_gate=asyncio.Event())

    async def scenario(session, transport):
        await session.on_json(start_with())
        await stream_audio(session, pcm(0.6))  # a decode is in flight
        sent = len(transport.sent)

        await session.on_disconnect()

        assert fake.cancelled
        assert no_other_tasks()
        assert len(transport.sent) == sent
        assert session.turn is None
        assert session.stage == TurnStage.IDLE

    run(scenario, transcriber=fake)


def test_disconnect_while_the_final_waits_for_a_decode_stops_that_decode_too():
    fake = FakeTranscriber(feed_gate=asyncio.Event())

    async def scenario(session, transport):
        await session.on_json(start_with())
        await stream_audio(session, pcm(0.6))
        await session.on_json(END)  # finish() is now waiting for the decode in flight
        await asyncio.sleep(0.02)
        assert session.stage == TurnStage.WORKING
        sent = len(transport.sent)

        await session.on_disconnect()

        assert fake.cancelled
        assert no_other_tasks()
        assert len(transport.sent) == sent

    run(scenario, transcriber=fake)


def test_a_reset_during_recording_is_refused_and_the_stream_carries_on():
    data = pcm(1.0)
    fake = FakeTranscriber(partials=[P1])

    async def scenario(session, transport):
        await session.on_json(start_with())
        await stream_audio(session, data[:16000])
        await session.on_json(json.dumps({"type": "reset"}))
        assert "in flight" in transport.last("error")["message"]
        await stream_audio(session, data[16000:])
        await session.on_json(END)
        await settle(session)
        assert transport.types().count("transcript") == 1

    run(scenario, transcriber=fake)
    assert len(fake.streams) == 1
    np.testing.assert_array_equal(fake.calls[0][1], pcm16_to_float32(data))


def test_nothing_is_left_running_after_a_normal_turn():
    fake = FakeTranscriber(partials=[P1, P2])

    async def scenario(session, transport):
        await whole_turn(session, pcm(1.2))
        await settle(session)
        await asyncio.sleep(0)
        assert no_other_tasks()

    run(scenario, transcriber=fake)


# --- review mode (FR-21) ---


def test_review_mode_shows_partials_then_waits_for_the_edited_final():
    clock = FakeClock()
    fake = FakeTranscriber(partials=[P1, P2], clock=clock, feed_takes_s=0.2, takes_s=0.3)
    llm = FakeLLM()

    async def scenario(session, transport):
        await whole_turn(session, pcm(1.0), review=True)  # two feeds, no remainder
        await wait_for_stage(session, TurnStage.AWAITING_CONFIRM)
        types = transport.types()
        assert types.count("partial") == 2
        assert max(i for i, t in enumerate(types) if t == "partial") < types.index("transcript")
        assert transport.last("transcript")["text"] == "hello world"
        assert "llm_delta" not in types  # waiting for the user

        turn_id = transport.last("turn_started")["turn_id"]
        await session.on_json(
            json.dumps({"type": "confirm_turn", "turn_id": turn_id, "text": "edited"})
        )
        await settle(session)
        timings = transport.last("llm_done")["timings"]
        assert timings["partial_count"] == 2
        assert timings["partial_lag_ms"] == pytest.approx(200)
        assert timings["final_ms"] == pytest.approx(300)  # not stretched by the user's edit time
        assert timings["stream_strategy"] == "redecode"
        assert partials(transport)[-1]["committed"] == "hello wor"

    run(scenario, transcriber=fake, llm=llm, clock=clock)
    assert llm.calls[0][-1] == {"role": "user", "content": "edited"}


# --- the turn log (S1.7) ---


def logged_turns(caplog, scenario, **kwargs) -> list[dict]:
    with caplog.at_level(logging.INFO, logger=TURN_LOG):
        run(scenario, **kwargs)
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == TURN_LOG]


def test_the_log_line_has_the_streaming_values_and_matches_the_browser(caplog):
    clock = FakeClock()
    fake = FakeTranscriber(
        text="my secret sentence",
        partials=[PartialResult("private words", ""), PartialResult("private words more", "")],
        clock=clock,
        feed_takes_s=[0.1, 0.3],
        takes_s=0.2,
    )

    async def scenario(session, transport):
        await whole_turn(session, pcm(1.0))
        await settle(session)
        scenario.sent = transport.last("llm_done")["timings"]

    (record,) = logged_turns(caplog, scenario, transcriber=fake, clock=clock)
    timings = record["timings"]
    assert timings["stream_strategy"] == "redecode"
    assert timings["partial_count"] == 2
    assert timings["partial_lag_ms"] == pytest.approx(200)
    assert timings["final_ms"] == pytest.approx(200)
    assert timings == scenario.sent  # the log and llm_done carry the same numbers

    # still no text: not the transcript, not a partial, not the reply
    lines = " ".join(r.getMessage() for r in caplog.records if r.name == TURN_LOG)
    assert "secret" not in lines and "private" not in lines and "Hi there" not in lines


def test_a_failed_turn_after_partials_logs_no_timings(caplog):
    fake = FakeTranscriber(partials=[P1], error=ASRError("secret"))

    async def scenario(session, transport):
        await whole_turn(session, pcm(0.6))
        await settle(session)

    (record,) = logged_turns(caplog, scenario, transcriber=fake)
    assert record["status"] == "asr_error"
    assert record["timings"] is None
