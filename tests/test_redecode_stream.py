"""Tests for RedecodeStream (PLAN_v1 S2.4).

A scripted batch model hands out one text per decode and records the audio it was asked to
decode, so "the whole turn every time" is checked on the samples themselves.
"""

import numpy as np
import pytest

from backend.asr.base import ASRError, ASRStream, PartialResult
from backend.asr.strategies import RedecodeStream
from tests.helpers import RecordingModel

SR = 16000
MIN_MS = 1000
MIN = 16000  # samples in MIN_MS
STEP = 9600  # 600 ms: two steps pass the minimum


def stream_for(model: RecordingModel, min_ms: int = MIN_MS) -> RedecodeStream:
    return RedecodeStream(model, min_audio_ms=min_ms)


def audio_of(samples: int, start: int = 0) -> np.ndarray:
    """Samples that are all different, so a misplaced slice shows up in an equality check."""
    return ((np.arange(start, start + samples) % 30011) / 30011).astype(np.float32)


def pair(partial: PartialResult | None) -> tuple[str, str] | None:
    return None if partial is None else (partial.committed, partial.tentative)


def test_it_is_an_asr_stream():
    assert isinstance(stream_for(RecordingModel()), ASRStream)


def test_no_partial_and_no_decode_before_the_minimum_audio():
    model = RecordingModel()
    assert stream_for(model).feed(audio_of(MIN - 1)) is None
    assert model.calls == []


def test_the_first_decode_is_all_tentative():
    model = RecordingModel(["he has"])
    partial = stream_for(model).feed(audio_of(MIN))  # exactly the minimum
    assert pair(partial) == ("", "he has")
    assert len(model.calls) == 1


def test_the_minimum_counts_the_audio_of_the_whole_turn_not_of_one_feed():
    model = RecordingModel(["hi"])
    stream = stream_for(model)
    assert stream.feed(audio_of(STEP)) is None
    assert pair(stream.feed(audio_of(STEP, start=STEP))) == ("", "hi")  # 1200 ms so far


def test_every_decode_gets_all_the_audio_so_far():
    model = RecordingModel()
    stream = stream_for(model)
    audio = audio_of(4 * STEP)
    for i in range(4):
        stream.feed(audio[i * STEP : (i + 1) * STEP])
    # Feeds 2, 3 and 4 decode; feed 1 is under the minimum.
    assert [len(c) for c in model.calls] == [2 * STEP, 3 * STEP, 4 * STEP]
    assert all(np.array_equal(c, audio[: len(c)]) for c in model.calls)


def test_the_model_gets_float32_16k_english():
    model = RecordingModel()
    stream_for(model).feed(audio_of(MIN))
    assert model.calls[0].dtype == np.float32
    assert model.kwargs[0] == {"sample_rate": 16000, "language": "en"}


def test_committed_grows_as_decodes_agree():
    model = RecordingModel(["he has", "he has grave doubts", "he has grave doubts whether sir"])
    stream = stream_for(model)
    seen = [pair(stream.feed(audio_of(MIN, start=i * MIN))) for i in range(3)]
    assert seen == [
        ("", "he has"),
        ("he has", "grave doubts"),
        ("he has grave doubts", "whether sir"),
    ]


def test_committed_never_shrinks_when_a_later_decode_disagrees():
    model = RecordingModel(["he has grave doubts", "he has grave doubts", "he has gray", "he"])
    stream = stream_for(model)
    committed = ""
    for i in range(4):
        partial = stream.feed(audio_of(MIN, start=i * MIN))
        if partial is not None:
            assert partial.committed.startswith(committed)
            committed = partial.committed
    assert committed == "he has grave doubts"


def test_an_unchanged_result_is_not_shown_again():
    model = RecordingModel(["a b", "a b", "a b"])
    stream = stream_for(model)
    seen = [pair(stream.feed(audio_of(MIN, start=i * MIN))) for i in range(3)]
    assert seen == [("", "a b"), ("a b", ""), None]
    assert len(model.calls) == 3  # it still decoded every time


def test_a_decode_that_shows_nothing_new_still_counts_as_the_previous_decode():
    # The third decode revises a committed word but leaves the shown pair unchanged, so no
    # partial is returned. The fourth must still be compared with the third, not the second.
    model = RecordingModel(["a b", "a b c", "x b c", "x b c d"])
    stream = stream_for(model)
    seen = [pair(stream.feed(audio_of(MIN, start=i * MIN))) for i in range(4)]
    assert seen == [("", "a b"), ("a b", "c"), None, ("a b c", "d")]


def test_silence_shows_nothing():
    model = RecordingModel(["", ""])
    stream = stream_for(model)
    assert stream.feed(audio_of(MIN)) is None
    assert stream.feed(audio_of(MIN, start=MIN)) is None


def test_feed_keeps_its_own_copy_of_the_audio():
    model = RecordingModel()
    stream = stream_for(model)
    first = audio_of(STEP)
    stream.feed(first)
    first[:] = 0.0  # the caller reuses its buffer
    stream.feed(audio_of(STEP, start=STEP))
    assert np.array_equal(model.calls[0], audio_of(2 * STEP))


def test_a_failed_decode_loses_no_audio_and_no_state():
    model = RecordingModel(["he has", "he has grave"], fail_on={2})
    stream = stream_for(model)
    assert pair(stream.feed(audio_of(MIN))) == ("", "he has")
    with pytest.raises(ASRError):
        stream.feed(audio_of(MIN, start=MIN))
    partial = stream.feed(audio_of(MIN, start=2 * MIN))  # decodes all three seconds
    assert pair(partial) == ("he has", "grave")  # agreed with the first decode, not a lost one
    assert np.array_equal(model.calls[-1], audio_of(3 * MIN))


# --- finish ---


def test_finish_is_one_plain_decode_of_the_whole_turn():
    model = RecordingModel(["partial text", "the final text"])
    stream = stream_for(model)
    audio = audio_of(3 * STEP)
    stream.feed(audio[:STEP])
    stream.feed(audio[STEP : 2 * STEP])  # decodes: "partial text"
    result = stream.finish()
    assert result.text == "the final text"
    assert len(model.calls) == 2
    assert np.array_equal(model.calls[1], audio[: 2 * STEP])
    assert result.latency_ms == 5.0  # the model's own number, unchanged


def test_finish_returns_the_plain_transcribe_result_whatever_the_partials_said():
    plain = RecordingModel(["same text"])
    stream_model = RecordingModel(["wrong guess", "same text"])
    audio = audio_of(2 * MIN)

    expected = plain.transcribe(audio)
    stream = stream_for(stream_model)
    stream.feed(audio)
    assert stream.finish() == expected


def test_a_clip_below_the_minimum_is_still_decoded_at_finish():
    model = RecordingModel(["short clip"])
    stream = stream_for(model)
    assert stream.feed(audio_of(4000)) is None
    assert stream.finish().text == "short clip"
    assert len(model.calls) == 1


def test_finish_with_nothing_fed_is_empty_and_does_not_call_the_model():
    model = RecordingModel()
    result = stream_for(model).finish()
    assert (result.text, result.language, result.duration_s, result.latency_ms) == (
        "",
        "en",
        0.0,
        0.0,
    )
    assert model.calls == []


def test_finish_after_a_failed_decode_still_has_all_the_audio():
    model = RecordingModel(["final"], fail_on={1})
    stream = stream_for(model)
    with pytest.raises(ASRError):
        stream.feed(audio_of(MIN))
    assert stream.finish().text == "final"
    assert np.array_equal(model.calls[0], audio_of(MIN))


def test_each_stream_has_its_own_state():
    model = RecordingModel()
    a, b = stream_for(model), stream_for(model)
    a.feed(audio_of(MIN - 100))
    assert b.feed(audio_of(200)) is None  # a's audio did not count for b
    assert model.calls == []
