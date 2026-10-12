"""Live partials for one turn: coalescing, one decode in flight (SPEC_v1 6.4, PLAN_v1 S1.4).

The session pushes every audio frame into a `StreamingTranscription` and never waits for
it. Decoding is slow (hundreds of ms), frames arrive every 50 ms, so the audio that arrives
during a decode is collected and sent to the model in one piece when the decode returns
(FR-20). There is never a queue of decodes.

This file knows nothing about models or strategies: it drives an `ASRStream`. It also knows
nothing about WebSockets: partials leave through a callback.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from backend.asr.base import ASRStream, PartialResult, TranscriptResult

log = logging.getLogger(__name__)

# `await run(fn, *args)` runs a blocking `fn` on the ASR worker thread and returns its result.
# S1.5 gives the service one; the tests use a one-thread executor.
Runner = Callable[..., Awaitable[Any]]

# `await on_partial(partial, audio_received_at)`: `audio_received_at` is `clock()` at the
# moment the last frame covered by that decode was pushed. The session turns it into
# `partial_lag_ms` (SPEC_v1 6.3).
PartialCallback = Callable[[PartialResult, float], Awaitable[None]]


class StreamingTranscription:
    """One per turn. Created at `start_turn`, ended by `finish()` or `close()`."""

    def __init__(
        self,
        stream: ASRStream,
        run: Runner,
        on_partial: PartialCallback,
        *,
        interval_ms: int,
        sample_rate: int = 16000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream
        self._run = run
        self._on_partial = on_partial
        self._interval_samples = interval_ms * sample_rate // 1000  # compare with sample counts
        self._clock = clock

        self._pending: list[np.ndarray] = []  # pushed but not yet fed, oldest first
        self._pending_samples = 0  # total length of `_pending`, so push() need not re-add it
        self._last_push_at = 0.0  # clock() when the newest pending frame arrived
        self._task: asyncio.Task | None = None  # the feed in flight, or None (at most one)
        self._committed = ""  # committed text delivered so far (FR-16 check)
        self._partials_on = True  # False after the first feed fails
        self._closed = False  # True once finish() or close() has begun

    def push(self, samples: np.ndarray) -> None:
        """Add new audio. Never blocks, never awaits.

        Input:  samples - 1-D float32, the new samples of one frame (any length, even 0).
                Called from the session's receive loop, inside the event loop.
        Output: None. After finish() or close() has begun it does nothing.
        """
        if self._closed:
            return

        self._pending.append(samples)
        self._pending_samples += len(samples)
        self._last_push_at = self._clock()
        self._start_feed_if_allowed()

    async def finish(self) -> TranscriptResult:
        """End the turn and return the final transcript.

        Output: the TranscriptResult from `stream.finish()`.
        Raises: whatever `stream.finish()` raises (the session turns it into an asr_error).
        """
        self._closed = True
        if self._task is not None:
            # _feed_once handles its own errors, so waiting for it cannot raise.
            await asyncio.wait({self._task})

        if self._pending_samples > 0:
            # The strategies need all the audio to build the final, even when partials are off.
            # The partial this returns is dropped: `transcript` goes out right after, and a
            # `partial` after a `transcript` is a bug (SPEC_v1 6.2).
            audio, _ = self._take_pending()
            try:
                await self._run(self._stream.feed, audio)
            except Exception:
                if self._partials_on:  # otherwise the first failure was already logged
                    log.warning("feeding the last audio failed, the final decides", exc_info=True)

        return await self._run(self._stream.finish)

    async def close(self) -> None:
        """Abandon the turn (disconnect, rejected turn, reset). Safe to call twice.

        The blocking call already on the worker thread cannot be stopped; we only stop
        waiting for it and never deliver its result. Does not call stream.finish().
        """
        self._closed = True
        self._pending.clear()
        self._pending_samples = 0
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait({self._task})

    def _take_pending(self) -> tuple[np.ndarray, float]:
        """All pending audio as one array, and when its last frame arrived. Empties the buffer."""
        audio = np.concatenate(self._pending)
        self._pending.clear()
        self._pending_samples = 0
        return audio, self._last_push_at

    def _start_feed_if_allowed(self) -> None:
        if (
            not self._closed
            and self._partials_on
            and self._task is None
            and self._pending_samples >= self._interval_samples
        ):
            self._task = asyncio.create_task(self._feed_once())

    async def _feed_once(self) -> None:
        """The body of the feed task. Runs at most once at a time."""
        try:
            audio, received_at = self._take_pending()
            try:
                partial = await self._run(self._stream.feed, audio)
            except Exception as e:
                # Best-effort partials (FR-18): log once, stop trying, the final still runs.
                log.warning("partial decode failed, partials off for this turn: %s", e)
                self._partials_on = False
                return

            if partial is None or self._closed:
                return

            if partial.committed.startswith(self._committed):
                self._committed = partial.committed
            else:
                log.warning(
                    "FR-16: committed text %r does not extend the previous %r",
                    partial.committed,
                    self._committed,
                )
                partial = PartialResult(
                    committed=self._committed, tentative=partial.committed + partial.tentative
                )

            try:
                await self._on_partial(partial, received_at)
            except Exception as e:
                log.warning("partial callback failed: %s", e)
        finally:
            # Audio that arrived during this decode goes into the next feed (FR-20).
            self._task = None
            self._start_feed_if_allowed()
