"""Tests for backend/asr/streaming.py (PLAN_v1 S1.4).

A one-thread executor plays the ASR worker and FakeStream (tests/fakes.py) plays the model.
A "gated" stream blocks every feed on a threading.Event, which holds a feed "in flight" while
the test pushes more audio. Event-loop waits poll with short sleeps, so they never hang long.
"""

import asyncio
import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytest

from backend.asr.base import PartialResult, TranscriptResult
from backend.asr.streaming import StreamingTranscription
from tests.fakes import FakeStream, FakeStreamingASR
from tests.helpers import FakeClock

INTERVAL_MS = 500
FRAME = 800  # 50 ms at 16 kHz
INTERVAL_SAMPLES = 8000


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeStreamingASR.reset()
    yield
    FakeStreamingASR.reset()


@dataclass
class Rig:
    st: StreamingTranscription
    stream: FakeStream
    got: list[tuple[PartialResult, float]]  # (partial, audio_received_at) as delivered
    gate: threading.Event | None
    clock: FakeClock
    feeds_awaiting: list[int]  # how many feeds were awaiting the worker at once, per dispatch


@contextlib.asynccontextmanager
async def rig(
    *,
    partials: tuple[PartialResult, ...] = (),
    final_text: str = "final text",
    feed_error: Exception | None = None,
    gated: bool = False,
    callback_fails_once: bool = False,
    callback_delay_s: float = 0.0,
):
    gate = threading.Event() if gated else None
    FakeStreamingASR.partials = list(partials)
    FakeStreamingASR.final_text = final_text
    FakeStreamingASR.feed_error = feed_error
    FakeStreamingASR.feed_gate = gate
    stream = FakeStreamingASR("fake").open_stream()

    executor = ThreadPoolExecutor(max_workers=1)

    awaiting = 0
    feeds_awaiting: list[int] = []

    async def run(fn, *args):
        # The one-thread executor would queue a second feed quietly, so count the feeds
        # that are awaiting at once here: "at most one decode in flight" is checked on this.
        nonlocal awaiting
        is_feed = fn == stream.feed
        if is_feed:
            awaiting += 1
            feeds_awaiting.append(awaiting)
        try:
            return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)
        finally:
            if is_feed:
                awaiting -= 1

    got: list[tuple[PartialResult, float]] = []

    async def on_partial(partial, audio_received_at):
        await asyncio.sleep(callback_delay_s)
        got.append((partial, audio_received_at))
        if callback_fails_once and len(got) == 1:
            raise RuntimeError("socket closed")

    clock = FakeClock()
    st = StreamingTranscription(
        stream, run, on_partial, interval_ms=INTERVAL_MS, sample_rate=16000, clock=clock
    )
    try:
        yield Rig(st, stream, got, gate, clock, feeds_awaiting)
    finally:
        await st.close()
        if gate is not None:
            gate.set()  # let a blocked worker thread go, or shutdown would hang
        executor.shutdown(wait=True)


def ramp(n: int, start: int = 0) -> np.ndarray:
    """n distinct float32 samples, so a test can check order and that nothing repeats."""
    return np.arange(start, start + n, dtype=np.float32)


async def wait_until(cond, within_s: float = 2.0) -> None:
    deadline = time.monotonic() + within_s
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


async def settle() -> None:
    """Give the event loop and the worker thread time to do anything they were going to do."""
    await asyncio.sleep(0.05)


def push_interval(st: StreamingTranscription, start: int = 0) -> int:
    """Push exactly one interval of audio as ten frames. Returns the next sample number."""
    for i in range(10):
        st.push(ramp(FRAME, start + i * FRAME))
    return start + 10 * FRAME


def sizes(stream: FakeStream) -> list[int]:
    return [len(a) for a in stream.fed]


def finish_size(stream: FakeStream) -> int | None:
    """How many samples finish() was handed, None when it got no audio."""
    return None if stream.finish_audio is None else len(stream.finish_audio)


def test_no_feed_before_the_interval_has_gathered():
    async def scenario():
        async with rig() as r:
            for i in range(4):  # 200 ms of the 500 ms interval
                r.st.push(ramp(FRAME, i * FRAME))
            await settle()
            assert r.stream.fed == []
            assert r.got == []

    asyncio.run(scenario())


def test_a_feed_starts_once_the_interval_has_gathered():
    async def scenario():
        async with rig(partials=(PartialResult("hello", "wor"),)) as r:
            push_interval(r.st)
            await wait_until(lambda: len(r.got) == 1)
            assert sizes(r.stream) == [INTERVAL_SAMPLES]
            assert r.got[0][0] == PartialResult("hello", "wor")

    asyncio.run(scenario())


def test_audio_arriving_during_a_feed_is_coalesced_into_one_next_feed():
    async def scenario():
        async with rig(gated=True) as r:
            nxt = push_interval(r.st)
            await asyncio.sleep(0.02)  # the feed task starts and takes the first interval
            for i in range(20):  # a full second arrives while that feed is still running
                r.st.push(ramp(FRAME, nxt + i * FRAME))
            await settle()
            assert r.stream.fed == []  # the first feed is blocked at the gate

            r.gate.set()
            await wait_until(lambda: len(r.stream.fed) >= 2)
            await r.st.finish()
            # Two feeds, not twenty-one: the second one carried everything that was waiting.
            assert sizes(r.stream) == [INTERVAL_SAMPLES, 20 * FRAME]
            assert max(r.feeds_awaiting) == 1  # never two decodes in flight

    asyncio.run(scenario())


def test_nothing_is_delivered_after_finish_returns():
    # The session sends `transcript` when finish() returns, and a `partial` after it is a bug
    # (SPEC_v1 6.2). A partial that was still being delivered when finish() began may be
    # delivered or dropped, but finish() must not return while it is still on its way.
    async def scenario():
        async with rig(
            gated=True, partials=(PartialResult("late", ""),), callback_delay_s=0.1
        ) as r:
            push_interval(r.st)
            await asyncio.sleep(0.02)  # the feed is in flight
            finishing = asyncio.create_task(r.st.finish())
            r.gate.set()
            await asyncio.wait_for(finishing, timeout=2)

            delivered = len(r.got)
            await asyncio.sleep(0.3)  # longer than the callback delay
            assert len(r.got) == delivered
            assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []

    asyncio.run(scenario())


def test_every_sample_is_fed_exactly_once_and_in_order():
    async def scenario():
        async with rig() as r:
            pushed = []
            n = 0
            for i in range(60):
                size = 300 + (i * 97) % 900  # frames of different, odd sizes
                frame = ramp(size, n)
                n += size
                pushed.append(frame)
                r.st.push(frame)
                await asyncio.sleep(0)  # let the feed tasks run between frames
            await r.st.finish()

            assert np.array_equal(r.stream.samples, np.concatenate(pushed))
            assert len(r.stream.fed) < 60  # fewer feeds than frames
            assert r.stream.finished

    asyncio.run(scenario())


def test_finish_waits_for_the_feed_in_flight_then_hands_it_the_remainder():
    async def scenario():
        async with rig(gated=True, final_text="all done") as r:
            nxt = push_interval(r.st)
            await asyncio.sleep(0.02)  # first feed in flight
            r.st.push(ramp(2 * FRAME, nxt))  # 100 ms: below the interval, stays pending

            finishing = asyncio.create_task(r.st.finish())
            await settle()
            assert not finishing.done()  # still waiting for the feed in flight
            assert not r.stream.finished

            r.gate.set()
            result = await asyncio.wait_for(finishing, timeout=2)
            assert isinstance(result, TranscriptResult)
            assert result.text == "all done"
            # The remainder is not fed: it goes to finish(), which decodes once.
            assert sizes(r.stream) == [INTERVAL_SAMPLES]
            assert finish_size(r.stream) == 2 * FRAME
            assert r.stream.finished

    asyncio.run(scenario())


def test_finish_with_nothing_pending_gets_no_audio():
    async def scenario():
        async with rig() as r:
            result = await r.st.finish()
            assert r.stream.fed == []
            assert r.stream.finish_audio is None
            assert r.stream.finished
            assert result.text == "final text"

        async with rig() as r:
            push_interval(r.st)
            await wait_until(lambda: len(r.stream.fed) == 1)
            await r.st.finish()
            assert sizes(r.stream) == [INTERVAL_SAMPLES]  # no extra feed
            assert r.stream.finish_audio is None  # and no empty array either

    asyncio.run(scenario())


def test_the_remainder_does_not_produce_a_partial():
    async def scenario():
        async with rig(partials=(PartialResult("a", ""), PartialResult("a b", ""))) as r:
            nxt = push_interval(r.st)
            await wait_until(lambda: len(r.got) == 1)
            r.st.push(ramp(FRAME, nxt))  # below the interval: only finish() will see it
            await r.st.finish()
            assert len(r.stream.fed) == 1  # the remainder is no feed
            assert finish_size(r.stream) == FRAME
            assert len(r.got) == 1  # the second scripted partial was never asked for

    asyncio.run(scenario())


def test_partials_arrive_in_order_with_the_time_of_the_last_frame_they_cover():
    async def scenario():
        async with rig(partials=(PartialResult("one", ""), PartialResult("one two", ""))) as r:
            nxt = 0
            last_frame_at = []
            for _ in range(2):
                for i in range(10):
                    r.clock.advance(0.05)
                    r.st.push(ramp(FRAME, nxt + i * FRAME))
                nxt += 10 * FRAME
                last_frame_at.append(r.clock())
                await wait_until(lambda: len(r.got) == len(last_frame_at))
                r.clock.advance(1.0)  # time passes before the next batch

            assert [p for p, _ in r.got] == [PartialResult("one", ""), PartialResult("one two", "")]
            assert [at for _, at in r.got] == last_frame_at

    asyncio.run(scenario())


def test_a_feed_that_returns_none_delivers_nothing():
    async def scenario():
        async with rig(partials=()) as r:  # the script is empty, so feed() returns None
            push_interval(r.st)
            await wait_until(lambda: len(r.stream.fed) == 1)
            await settle()
            assert r.got == []

    asyncio.run(scenario())


def test_committed_text_that_does_not_extend_the_previous_is_demoted_to_tentative(caplog):
    # FR-16: committed only grows. A strategy bug must not make the UI jump backwards.
    script = (
        PartialResult("hello world", ""),
        PartialResult("hullo there", ""),  # does not start with "hello world"
        PartialResult("hello world again", "x"),
    )

    async def scenario():
        async with rig(partials=script) as r:
            nxt = 0
            for k in (1, 2, 3):
                nxt = push_interval(r.st, nxt)
                await wait_until(lambda k=k: len(r.got) == k)

            first, second, third = (p for p, _ in r.got)
            assert first == PartialResult("hello world", "")
            assert second == PartialResult("hello world", "hullo there")
            assert third == PartialResult("hello world again", "x")

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    assert any("committed" in rec.getMessage().lower() for rec in caplog.records)


def test_a_feed_failure_turns_partials_off_but_the_final_transcript_still_works(caplog):
    async def scenario():
        async with rig(feed_error=RuntimeError("boom"), final_text="still fine") as r:
            nxt = push_interval(r.st)
            await wait_until(lambda: len(r.stream.fed) == 1)
            await settle()
            push_interval(r.st, nxt)  # more audio after the failure
            await settle()
            assert len(r.stream.fed) == 1  # no further feed was tried
            assert r.got == []

            result = await r.st.finish()
            assert result.text == "still fine"
            # finish() still handed over the audio that arrived after the failure.
            assert result.duration_s == pytest.approx(1.0)

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(warnings) == 1  # one log line for the whole turn (FR-18)


def test_a_failing_final_transcript_raises_out_of_finish():
    async def scenario():
        async with rig() as r:

            def broken(audio=None):
                raise RuntimeError("no final")

            r.stream.finish = broken
            push_interval(r.st)
            with pytest.raises(RuntimeError, match="no final"):
                await r.st.finish()

    asyncio.run(scenario())


def test_an_error_in_the_partial_callback_does_not_stop_the_stream():
    script = (PartialResult("one", ""), PartialResult("one two", ""))

    async def scenario():
        async with rig(partials=script, callback_fails_once=True) as r:
            nxt = push_interval(r.st)
            await wait_until(lambda: len(r.got) == 1)  # the callback raised on this one
            push_interval(r.st, nxt)
            await wait_until(lambda: len(r.got) == 2)  # partials are still on
            result = await r.st.finish()
            assert result.text == "final text"

    asyncio.run(scenario())


def test_close_cancels_the_feed_and_leaves_no_task_running():
    async def scenario():
        async with rig(gated=True, partials=(PartialResult("late", ""),)) as r:
            nxt = push_interval(r.st)
            await asyncio.sleep(0.02)  # a feed is in flight, blocked at the gate

            await asyncio.wait_for(r.st.close(), timeout=2)  # must not wait for the thread
            others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            assert others == []

            r.st.push(ramp(INTERVAL_SAMPLES, nxt))  # ignored after close
            r.gate.set()  # the thread finishes its call, nobody is listening
            await settle()
            assert r.got == []  # the late partial is never delivered
            assert len(r.stream.fed) <= 1  # and no new feed started
            assert not r.stream.finished  # close() does not produce a final transcript

            await r.st.close()  # a second close is harmless

    asyncio.run(scenario())


def test_pushes_after_finish_are_ignored():
    async def scenario():
        async with rig() as r:
            push_interval(r.st)
            await r.st.finish()
            fed_before = sizes(r.stream)
            push_interval(r.st)
            await settle()
            assert sizes(r.stream) == fed_before

    asyncio.run(scenario())
