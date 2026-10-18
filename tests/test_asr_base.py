"""Tests for the streaming types in backend/asr/base.py (PLAN_v1 S1.1).

The capability checks (isinstance against the protocols) are what S1.5 uses to decide
which strategies a model supports, so they are pinned here.
"""

import dataclasses
import threading

import numpy as np
import pytest

from backend.asr.base import (
    GENERIC_STRATEGIES,
    STRATEGIES,
    ASRStream,
    PartialResult,
    Segment,
    SegmentedASRModel,
    StreamingASRModel,
    TranscriptResult,
)
from tests.fakes import FakeSegmentedASR, FakeStreamingASR
from tests.test_asr_registry import FakeASR


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeStreamingASR.reset()
    FakeSegmentedASR.reset()


def test_strategy_names():
    assert STRATEGIES == ("chunked", "redecode", "trimmed", "native")
    assert GENERIC_STRATEGIES == ("chunked", "redecode")
    assert set(GENERIC_STRATEGIES) <= set(STRATEGIES)


def test_partial_result_is_a_frozen_value():
    p = PartialResult(committed="hello", tentative="wor")
    assert (p.committed, p.tentative) == ("hello", "wor")
    assert p == PartialResult("hello", "wor")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.committed = "changed"


def test_segment_is_a_frozen_value():
    s = Segment(text="Hi.", start_s=0.5, end_s=1.25)
    assert (s.text, s.start_s, s.end_s) == ("Hi.", 0.5, 1.25)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.end_s = 2.0


def test_capabilities_are_detected_by_isinstance():
    streaming = FakeStreamingASR("s")
    segmented = FakeSegmentedASR("g")
    batch = FakeASR("b", "fake/model")

    assert isinstance(streaming, StreamingASRModel)
    assert not isinstance(streaming, SegmentedASRModel)

    assert isinstance(segmented, SegmentedASRModel)
    assert not isinstance(segmented, StreamingASRModel)

    # A v0 batch adapter has neither capability: it only gets the generic strategies.
    assert not isinstance(batch, StreamingASRModel)
    assert not isinstance(batch, SegmentedASRModel)


def test_open_stream_returns_an_asr_stream():
    stream = FakeStreamingASR("s").open_stream()
    assert isinstance(stream, ASRStream)


def test_fake_stream_plays_its_script_and_finishes_over_all_fed_audio():
    FakeStreamingASR.partials = [PartialResult("hel", "lo"), PartialResult("hello", "")]
    FakeStreamingASR.final_text = "hello"
    stream = FakeStreamingASR("s").open_stream()

    assert stream.feed(np.ones(8000, dtype=np.float32)) == PartialResult("hel", "lo")
    assert stream.feed(np.ones(4000, dtype=np.float32)) == PartialResult("hello", "")
    assert stream.feed(np.ones(4000, dtype=np.float32)) is None  # script ran out

    result = stream.finish()
    assert isinstance(result, TranscriptResult)
    assert result.text == "hello"
    assert result.duration_s == 1.0
    assert len(stream.samples) == 16000
    assert FakeStreamingASR.streams == [stream]


def test_fake_stream_can_fail_and_block():
    FakeStreamingASR.feed_error = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        FakeStreamingASR("s").open_stream().feed(np.zeros(10, dtype=np.float32))

    FakeStreamingASR.reset()
    gate = threading.Event()
    FakeStreamingASR.feed_gate = gate
    stream = FakeStreamingASR("s").open_stream()
    worker = threading.Thread(target=stream.feed, args=(np.zeros(10, dtype=np.float32),))
    worker.start()
    worker.join(timeout=0.1)
    assert worker.is_alive()  # held at the gate
    gate.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert len(stream.fed) == 1


def test_fake_segmented_model_records_prompts():
    segments, latency_ms = FakeSegmentedASR("g").transcribe_segments(
        np.zeros(16000, dtype=np.float32), prompt="Hello there."
    )
    assert [s.text for s in segments] == ["Hello there.", " How are you?"]
    assert latency_ms == 0.0
    assert FakeSegmentedASR.prompts == ["Hello there."]
