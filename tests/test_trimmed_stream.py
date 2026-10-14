"""Tests for trim_point and TrimmedStream (PLAN_v1 S2.6).

A scripted segmented model hands out one list of segments per decode and records the audio and
prompt of each call, so "the decoded audio gets shorter" and "the prompt carries the settled
text" are checked on the samples and strings themselves.

The main scenario (600 ms steps, minimum 1 s, so the first decode is after the 2nd feed):

    feed 2  decode 1  "Hello there."                  nothing agreed yet
    feed 3  decode 2  "Hello there." "How are"        sentence 1 agreed: cut at 1.0 s
    feed 4  decode 3  "How are you today?"            "How are" agreed
    feed 5  decode 4  "How are you today?"            sentence 2 agreed: cut 1.3 s into the window
    feed 6  decode 5  "I am fine"                     nothing agreed yet
"""

import numpy as np
import pytest

from backend.asr.base import ASRError, ASRStream, PartialResult, Segment
from backend.asr.strategies import TrimmedStream, trim_point
from tests.helpers import RecordingSegmentedModel

SR = 16000
STEP = 9600  # 600 ms
MIN_MS = 1000


def seg(text: str, start: float, end: float) -> Segment:
    return Segment(text, start, end)


def audio_of(samples: int, start: int = 0) -> np.ndarray:
    """Samples that are all different, so a misplaced slice shows up in an equality check."""
    return ((np.arange(start, start + samples) % 30011) / 30011).astype(np.float32)


MASTER = audio_of(SR * 10)  # the "microphone": step i is MASTER[i*STEP:(i+1)*STEP]

SCENARIO = [
    [seg("Hello there.", 0.0, 1.0)],
    [seg("Hello there.", 0.0, 1.0), seg("How are", 1.0, 1.7)],
    [seg("How are you today?", 0.0, 1.2)],
    [seg("How are you today?", 0.0, 1.3)],
    [seg("I am fine", 0.0, 0.9)],
    [seg("I am fine thanks.", 0.0, 1.2)],  # the decode of finish()
]


def stream_for(model: RecordingSegmentedModel, min_ms: int = MIN_MS) -> TrimmedStream:
    return TrimmedStream(model, min_audio_ms=min_ms)


def feed_steps(stream: TrimmedStream, count: int, first: int = 0) -> list[PartialResult | None]:
    return [stream.feed(MASTER[i * STEP : (i + 1) * STEP]) for i in range(first, first + count)]


def pair(partial: PartialResult | None) -> tuple[str, str] | None:
    return None if partial is None else (partial.committed, partial.tentative)


def scenario_stream(steps: int = 6) -> tuple[TrimmedStream, RecordingSegmentedModel, list]:
    model = RecordingSegmentedModel(SCENARIO)
    stream = stream_for(model)
    return stream, model, [pair(p) for p in feed_steps(stream, steps)]


# --- trim_point ---


def test_nothing_to_trim_without_segments():
    assert trim_point([], 5) == 0


def test_nothing_to_trim_without_committed_words():
    assert trim_point([seg("Hello there.", 0, 1)], 0) == 0


@pytest.mark.parametrize(
    ("committed_words", "expected"),
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 3)],
)
def test_trim_point_counts_whole_committed_sentence_segments(committed_words, expected):
    segments = [seg("Hello there.", 0, 1), seg("How are", 1, 2), seg("you today?", 2, 3)]
    assert trim_point(segments, committed_words) == expected


def test_a_partly_committed_segment_is_never_cut():
    assert trim_point([seg("Hello there.", 0, 1)], 1) == 0


def test_trim_point_takes_the_last_sentence_end_that_is_fully_committed():
    segments = [seg("One.", 0, 1), seg("Two.", 1, 2), seg("Three and", 2, 3)]
    assert trim_point(segments, 6) == 2


def test_a_segment_without_a_sentence_end_is_not_a_cut_point_but_later_ones_still_are():
    segments = [seg("one two", 0, 1), seg("three four.", 1, 2)]
    assert trim_point(segments[:1], 2) == 0
    assert trim_point(segments, 4) == 2  # the whole run up to the full stop goes


@pytest.mark.parametrize(
    "text", ["Is it?", "Stop!", "Wait...", "Done.  ", ' He said "go."', "(Done.)"]
)
def test_sentence_ends(text):
    assert trim_point([seg(text, 0, 1)], len(text.split())) == 1


@pytest.mark.parametrize("text", ["Hello,", "and then:", "a b;", "no punctuation", "the 3.5"])
def test_not_sentence_ends(text):
    assert trim_point([seg(text, 0, 1)], len(text.split())) == 0


def test_curly_closing_quotes_after_the_full_stop_still_count():
    assert trim_point([seg("He said “go.”", 0, 1)], 3) == 1


# --- feed: the main scenario ---


def test_it_is_an_asr_stream():
    assert isinstance(stream_for(RecordingSegmentedModel([])), ASRStream)


def test_partials_follow_the_scenario():
    _, _, partials = scenario_stream()
    assert partials == [
        None,
        ("", "Hello there."),
        ("Hello there.", "How are"),  # sentence 1 is committed, and cut in the same feed
        ("Hello there. How are", "you today?"),
        ("Hello there. How are you today?", ""),
        ("Hello there. How are you today?", "I am fine"),
    ]


def test_committed_never_shrinks_across_the_cuts():
    stream = stream_for(RecordingSegmentedModel(SCENARIO))
    committed = [p.committed for p in feed_steps(stream, 6) if p is not None]
    for before, after in zip(committed, committed[1:], strict=False):
        assert after.startswith(before)


def test_no_decode_before_the_minimum_and_it_counts_the_whole_turn():
    model = RecordingSegmentedModel(SCENARIO)
    stream = stream_for(model)
    assert stream.feed(MASTER[: SR - 1]) is None
    assert model.calls == []
    assert pair(stream.feed(MASTER[SR - 1 : SR])) == ("", "Hello there.")  # exactly 1 s
    assert len(model.calls) == 1


def test_the_decoded_audio_grows_until_a_cut_and_is_shorter_after_it():
    _, model, _ = scenario_stream()
    assert [len(c) for c in model.calls] == [
        19200,  # feed 2: the whole turn
        28800,  # feed 3: the whole turn; the cut happens after this decode
        22400,  # feed 4: 38400 samples fed, 16000 cut
        32000,  # feed 5: 48000 fed, 16000 cut; the second cut (20800) happens after it
        20800,  # feed 6: 57600 fed, 36800 cut
    ]


def test_the_window_is_the_exact_slice_after_the_cut_point():
    _, model, _ = scenario_stream()
    assert np.array_equal(model.calls[0], MASTER[:19200])
    assert np.array_equal(model.calls[2], MASTER[16000:38400])  # cut at 1.0 s
    # The second cut is 1.3 s into the window that began at 1.0 s: 36800 samples in all.
    assert np.array_equal(model.calls[4], MASTER[36800:57600])


def test_the_prompt_is_the_settled_text_only():
    _, model, _ = scenario_stream()
    assert model.prompts == [
        None,  # nothing settled yet: no prompt at all, not ""
        None,
        "Hello there.",  # words committed but still inside the window are not in the prompt
        "Hello there.",
        "Hello there. How are you today?",
    ]


def test_the_words_kept_after_a_cut_agree_without_waiting_for_another_decode():
    # After sentence 1 was cut, decode 3 only has to agree with "How are", which decode 2 had
    # already produced. If `_previous` were reset, "How are" would show up one decode later.
    _, _, partials = scenario_stream(4)
    assert partials[3] == ("Hello there. How are", "you today?")


def test_a_cut_is_not_made_on_a_sentence_end_that_is_not_committed_yet():
    _, model, _ = scenario_stream(3)
    assert len(model.calls[1]) == 28800  # decode 1 saw "Hello there." but nothing agreed yet


def test_a_short_window_after_a_cut_is_still_decoded():
    # The minimum gates the first partial of the turn. A window that a cut made short again
    # (here 700 ms) must not silence the stream.
    sentence = [seg("Hello there.", 0.0, 1.7)]
    model = RecordingSegmentedModel([sentence, sentence, [seg("More", 0.0, 0.5)]])
    stream = stream_for(model)
    partials = [pair(p) for p in feed_steps(stream, 4)]
    assert np.array_equal(model.calls[2], MASTER[27200 : 4 * STEP])  # shorter than MIN_MS
    assert partials[3] == ("Hello there.", "More")


def test_no_cut_without_sentence_punctuation():
    model = RecordingSegmentedModel([[seg("he has grave doubts", 0.0, 1.5)]] * 4)
    stream = stream_for(model)
    feed_steps(stream, 5)
    assert [len(c) for c in model.calls] == [19200, 28800, 38400, 48000]
    assert set(model.prompts) == {None}


def test_one_cut_can_take_several_segments():
    two = [seg("A b.", 0.0, 1.0), seg("C d.", 1.0, 2.0)]
    model = RecordingSegmentedModel([two, two, [seg("E f", 0.0, 0.5)]])
    stream = stream_for(model, min_ms=500)
    for start in (0, 40000):
        stream.feed(MASTER[start : start + 40000])
    stream.feed(MASTER[80000:84000])
    assert model.prompts == [None, None, "A b. C d."]
    assert np.array_equal(model.calls[2], MASTER[32000:84000])  # cut at 2.0 s


def test_a_cut_past_the_end_of_the_window_empties_it():
    model = RecordingSegmentedModel([[seg("Hello there.", 0.0, 5.0)]] * 2)
    stream = stream_for(model, min_ms=500)
    feed_steps(stream, 2)
    result = stream.finish()
    assert len(model.calls) == 2  # nothing left to decode
    assert result.text == "Hello there."


def test_segment_text_is_stripped_and_joined_by_single_spaces():
    padded = [seg(" Hello there. ", 0.0, 1.0), seg(" How are you? ", 1.0, 2.0)]
    model = RecordingSegmentedModel([padded])
    partial = stream_for(model, min_ms=500).feed(MASTER[:STEP])
    assert pair(partial) == ("", "Hello there. How are you?")


def test_a_steady_text_is_not_sent_twice():
    same = [seg("hello there", 0.0, 1.0)]
    model = RecordingSegmentedModel([same, same, same, [seg("hello there again", 0.0, 1.5)]])
    stream = stream_for(model, min_ms=500)
    partials = [pair(p) for p in feed_steps(stream, 4)]
    assert partials == [
        ("", "hello there"),
        ("hello there", ""),
        None,
        ("hello there", "again"),
    ]
    assert len(model.calls) == 4  # the silent decodes still ran and still count


def test_a_decode_that_disagrees_after_a_cut_cannot_shrink_the_committed_text():
    scripts = [*SCENARIO[:3], [seg("Who is we today?", 0.0, 1.0)]]
    stream = stream_for(RecordingSegmentedModel(scripts))
    partials = [pair(p) for p in feed_steps(stream, 5)]
    assert partials[3] == ("Hello there. How are", "you today?")
    # Decode 4 disagrees from its first word on. "How are" stays as it was.
    assert partials[4] == ("Hello there. How are", "we today?")


def test_the_settled_text_keeps_the_spelling_the_user_already_saw():
    # The words were committed as "hello, there"; the decode that finishes the sentence spells
    # them "Hello there.". Cutting must not rewrite what was already shown as committed.
    scripts = [
        [seg("hello, there", 0.0, 1.0)],
        [seg("hello, there", 0.0, 1.0)],
        [seg("Hello there.", 0.0, 1.0)],
        [seg("How are you", 0.0, 0.5)],
        [seg("How are you today", 0.0, 0.8)],  # the decode of finish()
    ]
    model = RecordingSegmentedModel(scripts)
    stream = stream_for(model)
    shown = [pair(p) for p in feed_steps(stream, 5)]
    assert shown[2] == ("hello, there", "")
    assert shown[3] is None  # decode 3 cuts the sentence; the text on screen does not change
    assert shown[4] == ("hello, there", "How are you")
    assert model.prompts[3] == "hello, there"  # the prompt is the settled text as shown
    assert stream.finish().text == "hello, there How are you today"


def test_fed_audio_is_copied():
    model = RecordingSegmentedModel(SCENARIO)
    stream = stream_for(model)
    first = MASTER[:STEP].copy()
    stream.feed(first)
    first[:] = 0.0  # the caller reuses its buffer
    stream.feed(MASTER[STEP : 2 * STEP])
    assert np.array_equal(model.calls[0], MASTER[: 2 * STEP])


# --- feed: a decode that fails ---


def test_a_failed_decode_loses_no_audio_and_changes_no_state():
    model = RecordingSegmentedModel(SCENARIO, fail_on={3})  # the first decode after the cut
    stream = stream_for(model)
    assert [pair(p) for p in feed_steps(stream, 3)][1:] == [
        ("", "Hello there."),
        ("Hello there.", "How are"),
    ]
    with pytest.raises(ASRError):
        stream.feed(MASTER[3 * STEP : 4 * STEP])

    # The next feed decodes the window again, with the lost audio in it.
    partial = stream.feed(MASTER[4 * STEP : 5 * STEP])
    assert np.array_equal(model.calls[2], MASTER[16000 : 5 * STEP])
    assert model.prompts[2] == "Hello there."
    assert pair(partial) == ("Hello there. How are", "you today?")


def test_a_failed_decode_before_any_cut_keeps_the_audio():
    model = RecordingSegmentedModel(SCENARIO, fail_on={1})
    stream = stream_for(model)
    stream.feed(MASTER[:STEP])
    with pytest.raises(ASRError):
        stream.feed(MASTER[STEP : 2 * STEP])
    stream.feed(MASTER[2 * STEP : 3 * STEP])
    assert np.array_equal(model.calls[0], MASTER[: 3 * STEP])


# --- finish ---


def test_finish_decodes_only_the_window_and_adds_it_to_the_settled_text():
    stream, model, _ = scenario_stream()
    result = stream.finish()
    assert len(model.calls) == 6
    assert np.array_equal(model.calls[5], MASTER[36800:57600])  # same window as feed 6
    assert model.prompts[5] == "Hello there. How are you today?"
    assert result.text == "Hello there. How are you today? I am fine thanks."
    assert result.language == "en"
    assert result.duration_s == 57600 / SR  # all the audio, cut or not
    assert result.latency_ms == 7.0  # the one decode


def test_finish_before_any_cut_is_one_decode_of_everything_without_a_prompt():
    two = [seg("Hello there.", 0.0, 1.2), seg(" How are you?", 1.2, 2.5)]
    model = RecordingSegmentedModel([two])
    stream = stream_for(model)
    feed_steps(stream, 1)  # below the minimum: no decode yet
    result = stream.finish()
    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0], MASTER[:STEP])
    assert model.prompts == [None]
    assert result.text == "Hello there. How are you?"
    assert result.duration_s == STEP / SR


def test_finish_with_an_empty_tail_returns_the_settled_text_alone():
    model = RecordingSegmentedModel([*SCENARIO[:4], []])
    stream = stream_for(model)
    feed_steps(stream, 5)  # two sentences settled; the window still holds 11200 samples
    result = stream.finish()
    assert len(model.calls) == 5
    assert result.text == "Hello there. How are you today?"  # no stray space


def test_finish_with_nothing_fed_does_not_call_the_model():
    model = RecordingSegmentedModel([])
    result = stream_for(model).finish()
    assert model.attempts == 0
    assert (result.text, result.duration_s, result.latency_ms) == ("", 0.0, 0.0)


def test_finish_decodes_a_turn_shorter_than_the_minimum():
    model = RecordingSegmentedModel([[seg("Hi.", 0.0, 0.2)]])
    stream = stream_for(model)
    assert stream.feed(MASTER[:4800]) is None
    assert stream.finish().text == "Hi."
    assert len(model.calls) == 1


def test_finish_with_no_speech_in_the_window_returns_an_empty_text():
    model = RecordingSegmentedModel([[]])
    stream = stream_for(model)
    stream.feed(MASTER[:STEP])
    assert stream.finish().text == ""


def test_finish_raises_what_the_model_raises():
    model = RecordingSegmentedModel([], fail_on={1})
    stream = stream_for(model)
    stream.feed(MASTER[:STEP])
    with pytest.raises(ASRError):
        stream.finish()
