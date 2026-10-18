"""The logic behind `compare_asr.py --stream` (PLAN_v1 S2.1).

The script only parses arguments and prints. Everything that could be wrong in a number it
shows lives here, where tests can reach it: replaying a clip as live audio, the word error
rate, and the summary table.
"""

import asyncio
import re
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.asr.base import ASRStream, PartialResult, TranscriptResult
from backend.asr.streaming import Runner, StreamingTranscription

SAMPLE_RATE = 16_000
_NOT_WORD = re.compile(r"[^\w']+|_")  # anything but letters, digits and apostrophes


@dataclass(frozen=True)
class PartialEvent:
    """One partial as the client would have seen it."""

    at_s: float  # seconds after the replay started, when the partial was delivered
    lag_ms: float  # delivery time minus the time the newest audio it covers arrived
    committed: str
    tentative: str


@dataclass(frozen=True)
class ReplayResult:
    partials: list[PartialEvent]  # in delivery order
    final: TranscriptResult  # what stream.finish() returned
    final_ms: float  # last frame pushed -> final transcript returned (the session's final_ms)


def normalize_words(text: str) -> list[str]:
    """Split `text` into comparable words: lowercase, no punctuation.

    Input:  any transcript, e.g. "Hello, World!" or "HE HAS GRAVE DOUBTS" (the references
            are upper-case) or "LEIGHTON'S WORK".
    Output: the words, in order. An apostrophe inside a word stays ("leighton's", "don't"),
            so a possessive is one word. Any other punctuation separates words or vanishes:
            "well-known" -> ["well", "known"]. Empty or all-punctuation text gives [].
    """
    text = text.lower().replace("\u2019", "'")  # models sometimes write a curly apostrophe
    return _NOT_WORD.sub(" ", text).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + deletions + insertions) / words in the reference.

    Input:  both texts are normalized with `normalize_words` first, so case and punctuation
            never count as errors.
    Output: 0.0 for a perfect match. It can be above 1.0 when the hypothesis has many more
            words than the reference.
    Raises: ValueError when the reference has no words (the rate would divide by zero).
    """
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        raise ValueError("the reference has no words")

    # Edit distance, one row at a time: `previous[j]` is the distance between the reference
    # words seen so far minus the newest one, and the first `j` hypothesis words.
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,  # the reference word was dropped
                    current[j - 1] + 1,  # the hypothesis word was added
                    previous[j - 1] + (ref_word != hyp_word),  # match or substitution
                )
            )
        previous = current
    return previous[-1] / len(ref)


async def replay(
    stream: ASRStream,
    run: Runner,
    audio: np.ndarray,
    *,
    interval_ms: int,
    frame_ms: int = 50,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ReplayResult:
    """Play `audio` into `stream` the way a microphone would, and record what comes out.

    Input:  stream - a fresh stream for one turn (from TranscriptionService.open_stream).
            run - the runner that executes the stream's blocking calls (service.run).
            interval_ms - `asr.streaming.partial_interval_ms`, for the coalescing.
            clock, sleep - injectable so tests do not wait. `sleep(s)` must let other tasks run.
    Output: a ReplayResult.

    How it must behave (the tests check each point):
      - The audio is cut into `frame_ms` frames (the last one may be shorter) and pushed into
        a `StreamingTranscription`, the same class the session uses, so the coalescing is the
        real one.
      - Real-time pace: samples cannot be pushed before they were "spoken". Frame i is pushed
        when the clock reaches `start + (end of frame i) / SAMPLE_RATE`, where `start` is the
        clock when replay began. Count the schedule from `start`; sleeping a fixed
        `frame_ms` per frame drifts. If a decode made replay late, push at once (no negative
        sleep, no skipping).
      - Every delivered partial becomes a PartialEvent. The callback gets
        `(partial, audio_received_at)` from StreamingTranscription; `lag_ms` is
        `(clock() - audio_received_at) * 1000` and `at_s` is `clock() - start`.
      - After the last frame, `final_ms` is the time `await transcription.finish()` takes,
        measured on `clock` like the session's `final_ms` (it includes waiting for a decode
        still in flight).
    """
    events: list[PartialEvent] = []
    start = clock()

    async def on_partial(partial: PartialResult, audio_received_at: float) -> None:
        now = clock()
        events.append(
            PartialEvent(
                at_s=now - start,
                lag_ms=(now - audio_received_at) * 1000,
                committed=partial.committed,
                tentative=partial.tentative,
            )
        )

    transcription = StreamingTranscription(
        stream, run, on_partial, interval_ms=interval_ms, sample_rate=SAMPLE_RATE, clock=clock
    )
    frame_samples = SAMPLE_RATE * frame_ms // 1000
    for pos in range(0, len(audio), frame_samples):
        frame = audio[pos : pos + frame_samples]
        # The frame exists once its last sample was spoken. Counting from `start` keeps the
        # pace from drifting when a decode makes us late.
        due = start + (pos + len(frame)) / SAMPLE_RATE
        await sleep(max(0.0, due - clock()))
        transcription.push(frame)

    last_frame_at = clock()
    final = await transcription.finish()
    return ReplayResult(partials=events, final=final, final_ms=(clock() - last_frame_at) * 1000)


@dataclass(frozen=True)
class BenchRow:
    """One line of the summary table: one model with one strategy (or the batch baseline)."""

    model: str
    strategy: str
    partial_count: int
    median_lag_ms: float | None  # None when no partial was shown
    final_ms: float
    wer: float | None  # None when there is no reference text
    text: str
    first_partial_ms: float | None = None  # when the first partial arrived, from the start


def make_row(
    model: str, strategy: str, result: ReplayResult, reference: str | None = None
) -> BenchRow:
    lags = [p.lag_ms for p in result.partials]
    return BenchRow(
        model=model,
        strategy=strategy,
        partial_count=len(result.partials),
        median_lag_ms=statistics.median(lags) if lags else None,
        final_ms=result.final_ms,
        wer=word_error_rate(reference, result.final.text) if reference else None,
        text=result.final.text,
        first_partial_ms=result.partials[0].at_s * 1000 if result.partials else None,
    )


def make_batch_row(model: str, final: TranscriptResult, reference: str | None = None) -> BenchRow:
    """The baseline: one plain `transcribe` of the whole clip, no streaming."""
    return BenchRow(
        model=model,
        strategy="batch",
        partial_count=0,
        median_lag_ms=None,
        final_ms=final.latency_ms,
        wer=word_error_rate(reference, final.text) if reference else None,
        text=final.text,
    )


def _cell(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_table(rows: list[BenchRow]) -> str:
    """The summary: a header, a rule, then one aligned line per row. Missing values show `-`."""
    header = ("model", "strategy", "partials", "first ms", "lag ms", "final ms", "WER %")
    body = [
        (
            r.model,
            r.strategy,
            str(r.partial_count),
            _cell(r.first_partial_ms),
            _cell(r.median_lag_ms),
            _cell(r.final_ms),
            _cell(None if r.wer is None else r.wer * 100, 1),
        )
        for r in rows
    ]
    widths = [max(len(line[i]) for line in [header, *body]) for i in range(len(header))]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()

    return "\n".join([line(header), "  ".join("-" * w for w in widths), *map(line, body)])


def format_partial(event: PartialEvent) -> str:
    """One timeline line: when it arrived, how late, committed text | tentative text."""
    return (
        f"+{event.at_s:5.2f} s  lag {event.lag_ms:4.0f} ms  {event.committed} | {event.tentative}"
    )


def load_reference(wav: Path) -> str | None:
    """The reference transcript that sits next to `wav` (same name, `.txt`), or None."""
    path = wav.with_suffix(".txt")
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None
