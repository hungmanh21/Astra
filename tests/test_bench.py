"""Tests for backend/asr/bench.py (PLAN_v1 S2.1).

`replay` runs on a FakeClock. The fake `sleep` first lets every ready task run at the
current time (a feed starts the moment its frame is pushed) and only then moves the clock,
the way real time passes while a decode runs in the background. A decode costs fake time
through `Rig.decode_s`, so the lags and `final_ms` come out exact.
"""

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from backend.asr.base import PartialResult, TranscriptResult
from backend.asr.bench import (
    SAMPLE_RATE,
    BenchRow,
    PartialEvent,
    ReplayResult,
    format_partial,
    format_table,
    load_reference,
    make_batch_row,
    make_row,
    normalize_words,
    replay,
    word_error_rate,
)
from tests.helpers import FakeClock

# ---------------------------------------------------------------- normalize_words


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("HE HAS GRAVE DOUBTS", ["he", "has", "grave", "doubts"]),
        ("LEIGHTON'S WORK", ["leighton's", "work"]),
        ("don't", ["don't"]),
        ("don’t", ["don't"]),  # curly apostrophe
        ("well-known", ["well", "known"]),
        ("  many   spaces\tand\nlines ", ["many", "spaces", "and", "lines"]),
        ("", []),
        ("... !?", []),
    ],
)
def test_normalize_words(text, words):
    assert normalize_words(text) == words


# ---------------------------------------------------------------- word_error_rate


def test_identical_texts_have_zero_error():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0


def test_case_and_punctuation_are_not_errors():
    assert word_error_rate("HE HAS GRAVE DOUBTS", "He has grave doubts.") == 0.0


def test_one_substitution():
    assert word_error_rate("the cat sat down", "the cat sat up") == pytest.approx(0.25)


def test_one_deletion():
    assert word_error_rate("the cat sat down", "the cat down") == pytest.approx(0.25)


def test_one_insertion():
    assert word_error_rate("the cat sat", "the big cat sat") == pytest.approx(1 / 3)


def test_the_best_alignment_is_used_not_the_first():
    # Delete "a", insert "e" = 2 edits. Matching position by position would be 4 substitutions.
    assert word_error_rate("a b c d", "b c d e") == pytest.approx(0.5)


def test_swapped_words_cost_two_edits():
    assert word_error_rate("a b", "b a") == pytest.approx(1.0)


def test_error_can_exceed_one():
    assert word_error_rate("hi", "oh hi there you") == pytest.approx(3.0)


def test_an_empty_hypothesis_misses_every_word():
    assert word_error_rate("one two three", "") == pytest.approx(1.0)


def test_an_empty_reference_is_refused():
    with pytest.raises(ValueError):
        word_error_rate("", "anything")
    with pytest.raises(ValueError):
        word_error_rate("...", "anything")


def test_the_fixture_reference_against_a_close_transcript():
    reference = (
        "HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS REALLY GREEK AFTER ALL "
        "AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA"
    )
    heard = (
        "He has grave doubts whether Sir Frederick Layton's work is really Greek after all "
        "and can discover in it but little of rocky Ithaca."
    )
    assert word_error_rate(reference, heard) == pytest.approx(
        1 / 24
    )  # "Layton's" only, of 24 words


# ---------------------------------------------------------------- replay

FRAME = 800  # 50 ms


class ScriptedStream:
    """An ASRStream that returns scripted partials and records when it was called."""

    def __init__(self, partials: list[PartialResult], final_text: str = "done"):
        self._partials = list(partials)
        self._final_text = final_text
        self.fed: list[np.ndarray] = []
        self.fed_at: list[float] = []  # clock when each feed started
        self.finish_calls = 0
        self.finish_at: float | None = None

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        self.fed.append(np.array(audio, copy=True))
        return self._partials.pop(0) if self._partials else None

    def finish(self) -> TranscriptResult:
        self.finish_calls += 1
        return TranscriptResult(self._final_text, "en", 0.0, 0.0)


@dataclass
class Rig:
    clock: FakeClock = field(default_factory=FakeClock)
    decode_s: float = 0.0  # fake time a feed takes
    final_s: float = 0.0  # fake time finish takes
    partials: list[PartialResult] = field(default_factory=list)
    stream: ScriptedStream = field(init=False)
    start: float = field(init=False)

    def __post_init__(self) -> None:
        self.stream = ScriptedStream(self.partials)
        self.start = self.clock.now

    async def run(self, fn, *args):
        if fn == self.stream.feed:
            self.stream.fed_at.append(self.clock.now - self.start)
            self.clock.advance(self.decode_s)
        else:
            self.stream.finish_at = self.clock.now - self.start
            self.clock.advance(self.final_s)
        return fn(*args)

    async def sleep(self, seconds: float) -> None:
        target = self.clock.now + seconds
        for _ in range(5):  # let the feed task started by the last push run to its end
            await asyncio.sleep(0)
        self.clock.now = max(self.clock.now, target)  # a decode may already have passed it

    def replay(self, audio: np.ndarray, interval_ms: int = 500) -> ReplayResult:
        return asyncio.run(
            replay(
                self.stream,
                self.run,
                audio,
                interval_ms=interval_ms,
                clock=self.clock,
                sleep=self.sleep,
            )
        )


def audio_of(seconds: float) -> np.ndarray:
    n = round(seconds * SAMPLE_RATE)
    return (np.arange(n) % 1000).astype(
        np.float32
    ) / 1000  # every sample differs from its neighbours


def p(committed: str, tentative: str = "") -> PartialResult:
    return PartialResult(committed=committed, tentative=tentative)


def test_replay_feeds_the_whole_clip_in_order():
    rig = Rig()
    audio = audio_of(2.0)
    rig.replay(audio)
    assert np.array_equal(np.concatenate(rig.stream.fed), audio)


def test_replay_keeps_a_last_frame_that_is_shorter():
    rig = Rig()
    audio = audio_of(0.62)  # 12 full frames and a 20 ms rest
    rig.replay(audio)
    assert np.array_equal(np.concatenate(rig.stream.fed), audio)


def test_replay_runs_at_real_time_pace():
    rig = Rig()
    rig.replay(audio_of(2.0), interval_ms=500)
    # 500 ms of audio exists only when 500 ms have passed, so feeds start at 0.5, 1.0, 1.5, 2.0.
    assert rig.stream.fed_at == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert [len(a) for a in rig.stream.fed] == [8000, 8000, 8000, 8000]


def test_replay_ends_the_clip_when_its_audio_ends():
    rig = Rig()
    rig.replay(audio_of(0.62))
    assert rig.stream.finish_at == pytest.approx(0.62)


def test_replay_pace_does_not_drift_while_decodes_run():
    rig = Rig(decode_s=0.3)
    rig.replay(audio_of(2.5), interval_ms=500)
    # Each decode makes replay 0.3 s late. It must catch up by pushing the late frames at
    # once; sleeping a fixed 50 ms per frame would push every later feed back by 0.3 s.
    assert rig.stream.fed_at == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])


def test_replay_records_each_partial_with_its_text():
    rig = Rig(partials=[p("he has"), p("he has grave", "doubts")])
    result = rig.replay(audio_of(2.0), interval_ms=500)
    assert [(e.committed, e.tentative) for e in result.partials[:2]] == [
        ("he has", ""),
        ("he has grave", "doubts"),
    ]


def test_replay_stamps_partials_with_arrival_time_and_lag():
    rig = Rig(decode_s=0.3, partials=[p("a"), p("a b"), p("a b c")])
    result = rig.replay(audio_of(3.0), interval_ms=500)
    first = result.partials[0]
    assert first.at_s == pytest.approx(0.8)  # audio of the first 0.5 s, plus a 0.3 s decode
    assert first.lag_ms == pytest.approx(300)
    assert [e.lag_ms for e in result.partials] == pytest.approx([300] * len(result.partials))
    assert [e.at_s for e in result.partials] == sorted(e.at_s for e in result.partials)


def test_replay_drops_the_partial_that_finishes_after_the_last_frame():
    rig = Rig(partials=[p("a"), p("a b"), p("a b c"), p("a b c d")])
    result = rig.replay(audio_of(2.0), interval_ms=500)
    # The fourth feed starts at 2.0 s, with the last frame; its partial would follow the
    # transcript, so StreamingTranscription drops it (SPEC_v1 6.2).
    assert len(rig.stream.fed) == 4
    assert len(result.partials) == 3


def test_final_ms_is_the_time_finish_takes():
    rig = Rig(final_s=0.7)
    result = rig.replay(audio_of(0.3), interval_ms=500)  # never reaches a feed
    assert result.final_ms == pytest.approx(700)
    assert rig.stream.finish_calls == 1


def test_final_ms_includes_a_decode_still_in_flight():
    rig = Rig(decode_s=0.3, final_s=0.7)
    result = rig.replay(audio_of(1.0), interval_ms=1000)  # the feed starts with the last frame
    assert result.final_ms == pytest.approx(1000)  # 300 for the decode in flight, 700 to finish


def test_replay_returns_what_finish_returned():
    rig = Rig()
    result = rig.replay(audio_of(0.3))
    assert result.final.text == "done"


def test_replay_of_an_empty_clip_still_finishes():
    rig = Rig()
    result = rig.replay(np.zeros(0, dtype=np.float32))
    assert result.partials == []
    assert rig.stream.finish_calls == 1


def test_replay_with_the_real_clock_takes_as_long_as_the_audio():
    stream = ScriptedStream([])

    async def run(fn, *args):
        return fn(*args)

    started = time.monotonic()
    asyncio.run(replay(stream, run, audio_of(0.3), interval_ms=500))
    assert 0.29 <= time.monotonic() - started < 1.0


# ---------------------------------------------------------------- rows and table


def result_with_lags(lags: list[float], text: str = "x", final_ms: float = 400) -> ReplayResult:
    events = [
        PartialEvent(at_s=i + 1.0, lag_ms=lag, committed="c", tentative="")
        for i, lag in enumerate(lags)
    ]
    return ReplayResult(
        partials=events, final=TranscriptResult(text, "en", 1.0, 0.0), final_ms=final_ms
    )


def test_make_row_uses_the_median_lag():
    row = make_row("m", "redecode", result_with_lags([100, 200, 600]))
    assert row.median_lag_ms == 200  # not the mean (300) and not the last (600)
    assert row.partial_count == 3


def test_make_row_without_partials_has_no_lag():
    row = make_row("m", "chunked", result_with_lags([]))
    assert row.partial_count == 0
    assert row.median_lag_ms is None


def test_make_row_scores_the_final_text_against_the_reference():
    row = make_row("m", "redecode", result_with_lags([], text="the cat sat up"), "the cat sat down")
    assert row.wer == pytest.approx(0.25)
    assert row.text == "the cat sat up"
    assert row.final_ms == 400


def test_make_row_without_a_reference_has_no_wer():
    assert make_row("m", "redecode", result_with_lags([]), None).wer is None


def test_make_batch_row_is_the_plain_transcribe_result():
    final = TranscriptResult("the cat sat down", "en", 3.0, 420.0)
    row = make_batch_row("m", final, "the cat sat down")
    assert (row.strategy, row.partial_count, row.median_lag_ms) == ("batch", 0, None)
    assert row.final_ms == 420.0
    assert row.wer == 0.0


def test_format_table_has_a_header_a_rule_and_one_line_per_row():
    rows = [
        BenchRow("whisper-large-v3-turbo", "redecode", 5, 312.4, 431.2, 0.0456, "t"),
        BenchRow("whisper-large-v3-turbo", "chunked", 0, None, 440.0, None, "t"),
    ]
    lines = format_table(rows).splitlines()
    assert len(lines) == 4
    assert all(
        word in lines[0] for word in ("model", "strategy", "partials", "lag", "final", "WER")
    )
    assert set(lines[1]) <= {"-", " "}
    assert "redecode" in lines[2] and "312" in lines[2] and "431" in lines[2] and "4.6" in lines[2]
    assert "chunked" in lines[3]
    assert lines[3].split()[-3:] == ["-", "440", "-"]  # lag and WER are missing, final is not


def test_format_partial_shows_time_lag_and_both_texts():
    line = format_partial(
        PartialEvent(at_s=1.55, lag_ms=312.0, committed="he has", tentative="grave")
    )
    assert "1.55" in line and "312" in line
    assert "he has" in line and "grave" in line
    assert line.index("he has") < line.index("|") < line.index("grave")


def test_load_reference_reads_the_txt_next_to_the_wav(tmp_path):
    (tmp_path / "clip.wav").write_bytes(b"")
    (tmp_path / "clip.txt").write_text("  HELLO THERE\n", encoding="utf-8")
    assert load_reference(tmp_path / "clip.wav") == "HELLO THERE"


def test_load_reference_is_none_without_a_usable_file(tmp_path):
    assert load_reference(tmp_path / "missing.wav") is None
    (tmp_path / "blank.txt").write_text(" \n", encoding="utf-8")
    assert load_reference(tmp_path / "blank.wav") is None
