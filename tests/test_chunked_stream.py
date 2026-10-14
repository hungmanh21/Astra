"""Tests for ChunkedStream (PLAN_v1 S2.2).

A scripted batch model hands out one text per decode and records exactly which samples it
was asked to decode, so "every sample once, cut at chunk_ms" is checked on the audio itself.
"""

import numpy as np
import pytest

from backend.asr.base import ASRError, ASRStream
from backend.asr.strategies import ChunkedStream
from tests.helpers import RecordingModel

SR = 16000
CHUNK_MS = 500
CHUNK = 8000  # samples in 500 ms
REST = 4000  # a remainder: less than a chunk


def stream_for(model: RecordingModel, chunk_ms: int = CHUNK_MS):
    return ChunkedStream(model, chunk_ms=chunk_ms)


def audio_of(samples: int, start: int = 0) -> np.ndarray:
    """Samples that are all different, so a misplaced cut shows up in an equality check."""
    return ((np.arange(start, start + samples) % 30011) / 30011).astype(np.float32)


def test_it_is_an_asr_stream():
    assert isinstance(stream_for(RecordingModel()), ASRStream)


def test_less_than_a_chunk_shows_nothing_and_decodes_nothing():
    model = RecordingModel()
    assert stream_for(model).feed(audio_of(CHUNK - 1)) is None
    assert model.calls == []


def test_a_full_chunk_is_decoded_once_and_committed():
    model = RecordingModel(["hello"])
    partial = stream_for(model).feed(audio_of(CHUNK))
    assert partial is not None
    assert (partial.committed, partial.tentative) == ("hello", "")
    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0], audio_of(CHUNK))


def test_the_model_gets_float32_16k_english():
    model = RecordingModel()
    stream_for(model).feed(audio_of(CHUNK))
    assert model.calls[0].dtype == np.float32
    assert model.kwargs[0] == {"sample_rate": 16000, "language": "en"}


def test_a_chunk_can_span_two_feeds():
    model = RecordingModel(["one"])
    stream = stream_for(model)
    assert stream.feed(audio_of(4800)) is None
    partial = stream.feed(audio_of(4800, start=4800))  # 9600 samples so far
    assert partial is not None and partial.committed == "one"
    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0], audio_of(CHUNK))  # the first 8000 of the 9600


def test_the_leftover_of_a_feed_starts_the_next_chunk():
    model = RecordingModel(["one", "two"])
    stream = stream_for(model)
    stream.feed(audio_of(12000))  # one chunk, 4000 left over
    stream.feed(audio_of(4000, start=12000))  # 4000 + 4000 = a second chunk
    assert len(model.calls) == 2
    assert np.array_equal(model.calls[1], audio_of(CHUNK, start=CHUNK))


def test_a_coalesced_feed_decodes_every_chunk_it_holds():
    model = RecordingModel(["a", "b", "c"])
    partial = stream_for(model).feed(audio_of(3 * CHUNK + 1000))
    assert len(model.calls) == 3
    assert [len(c) for c in model.calls] == [CHUNK] * 3
    assert partial is not None and partial.committed == "a b c"  # one partial for the feed


def test_every_sample_is_decoded_exactly_once_and_in_order():
    model = RecordingModel()
    stream = stream_for(model)
    audio = audio_of(5 * CHUNK + 6000)
    for start in range(0, len(audio), 1300):  # feeds that do not line up with chunks
        stream.feed(audio[start : start + 1300])
    stream.finish()  # decodes the 6000-sample remainder
    assert np.array_equal(np.concatenate(model.calls), audio)


def test_committed_only_grows_and_joins_with_one_space():
    model = RecordingModel(["he has", "grave", "doubts"])
    stream = stream_for(model)
    seen = [stream.feed(audio_of(CHUNK, start=i * CHUNK)) for i in range(3)]
    assert [p.committed for p in seen if p] == ["he has", "he has grave", "he has grave doubts"]


def test_empty_chunks_add_no_text_and_no_extra_space():
    model = RecordingModel(["hello", "", "world"])
    stream = stream_for(model)
    first = stream.feed(audio_of(CHUNK))
    silent = stream.feed(audio_of(CHUNK, start=CHUNK))
    last = stream.feed(audio_of(CHUNK, start=2 * CHUNK))
    assert first is not None and first.committed == "hello"
    assert silent is None  # decoded, but nothing new to show
    assert last is not None and last.committed == "hello world"


def test_the_text_of_a_chunk_is_stripped():
    model = RecordingModel([" hello ", " world"])
    stream = stream_for(model)
    stream.feed(audio_of(CHUNK))
    partial = stream.feed(audio_of(CHUNK, start=CHUNK))
    assert partial is not None and partial.committed == "hello world"


def test_feed_keeps_its_own_copy_of_the_audio():
    model = RecordingModel()
    stream = stream_for(model)
    first = audio_of(4000)
    stream.feed(first)
    first[:] = 0.0  # the caller reuses its buffer
    stream.feed(audio_of(4000, start=4000))
    assert np.array_equal(model.calls[0], audio_of(CHUNK))


def test_a_failed_decode_loses_no_audio():
    model = RecordingModel(["one", "two"], fail_on={1})
    stream = stream_for(model)
    with pytest.raises(ASRError):
        stream.feed(audio_of(CHUNK))
    partial = stream.feed(audio_of(CHUNK, start=CHUNK))  # retries the first chunk, then the next
    assert partial is not None and partial.committed == "one two"
    assert np.array_equal(np.concatenate(model.calls), audio_of(2 * CHUNK))


def test_a_failure_in_the_second_chunk_keeps_the_first_chunks_text():
    model = RecordingModel(["one", "two"], fail_on={2})
    stream = stream_for(model)
    with pytest.raises(ASRError):
        stream.feed(audio_of(2 * CHUNK))
    result = stream.finish()  # the second chunk is still pending
    assert result.text == "one two"
    assert np.array_equal(np.concatenate(model.calls), audio_of(2 * CHUNK))


# --- finish ---


def test_finish_decodes_the_remainder():
    model = RecordingModel(["one", "tail"])
    stream = stream_for(model)
    stream.feed(audio_of(CHUNK + REST))
    result = stream.finish()
    assert result.text == "one tail"
    assert np.array_equal(model.calls[1], audio_of(REST, start=CHUNK))
    assert result.latency_ms == 5.0  # the remainder decode, not the chunks decoded earlier


def test_finish_decodes_even_a_very_short_remainder():
    model = RecordingModel(["one", "end"])
    stream = stream_for(model)
    stream.feed(audio_of(CHUNK + 800))  # 50 ms
    assert stream.finish().text == "one end"
    assert len(model.calls[1]) == 800


def test_finish_without_a_remainder_does_not_call_the_model_again():
    model = RecordingModel(["one"])
    stream = stream_for(model)
    stream.feed(audio_of(CHUNK))
    result = stream.finish()
    assert result.text == "one"
    assert len(model.calls) == 1
    assert result.latency_ms == 0.0


def test_finish_reports_all_the_audio_fed():
    stream = stream_for(RecordingModel())
    stream.feed(audio_of(CHUNK + 1000))
    result = stream.finish()
    assert result.duration_s == pytest.approx((CHUNK + 1000) / SR)
    assert result.language == "en"


def test_finish_with_nothing_fed_is_empty_and_does_not_call_the_model():
    model = RecordingModel()
    result = stream_for(model).finish()
    assert (result.text, result.duration_s, result.latency_ms) == ("", 0.0, 0.0)
    assert model.calls == []


def test_a_clip_shorter_than_a_chunk_is_decoded_once_at_finish():
    model = RecordingModel(["just this"])
    stream = stream_for(model)
    assert stream.feed(audio_of(REST)) is None
    assert stream.finish().text == "just this"
    assert len(model.calls) == 1


def test_each_stream_has_its_own_buffer():
    model = RecordingModel()
    a, b = stream_for(model), stream_for(model)
    a.feed(audio_of(CHUNK - 100))
    assert b.feed(audio_of(200)) is None  # a's audio did not leak into b
    assert model.calls == []


def test_other_chunk_lengths_cut_where_they_say():
    model = RecordingModel()
    stream = stream_for(model, chunk_ms=1000)
    stream.feed(audio_of(2 * SR + 5000))
    assert [len(c) for c in model.calls] == [SR, SR]
