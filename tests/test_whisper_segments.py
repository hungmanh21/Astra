"""Tests for the segment side of backend/asr/whisper.py (PLAN_v1 S2.5).

The pipeline and its tokenizer are stubs, so no weights are loaded. What the real pipeline
returns was checked by hand on the fixture (see PLAN_v1 Notes).
"""

import time

import numpy as np
import pytest
import torch

from backend.asr.base import ASRError, Segment, SegmentedASRModel
from backend.asr.strategies import supported_strategies
from backend.asr.whisper import MAX_PROMPT_TOKENS, WhisperASR, segments_from_chunks

AUDIO = np.zeros(2 * 16000, dtype=np.float32)  # 2.0 s


def chunk(text: str, start, end) -> dict:
    return {"text": text, "timestamp": (start, end)}


# --- segments_from_chunks ---


def test_chunks_become_segments_with_stripped_text():
    chunks = [chunk(" He has grave doubts ", 0.0, 3.2), chunk(" whether Sir", 3.2, 4.5)]
    assert segments_from_chunks(chunks, duration_s=5.0) == [
        Segment("He has grave doubts", 0.0, 3.2),
        Segment("whether Sir", 3.2, 4.5),
    ]


def test_no_chunks_give_no_segments():
    assert segments_from_chunks([], duration_s=2.0) == []


def test_chunks_without_text_are_dropped():
    chunks = [chunk(" ", 0.0, 1.0), chunk("", 1.0, 2.0), chunk(" hello", 2.0, 3.0)]
    assert segments_from_chunks(chunks, duration_s=3.0) == [Segment("hello", 2.0, 3.0)]


def test_a_missing_end_is_the_end_of_the_audio():
    assert segments_from_chunks([chunk("cut off", 1.0, None)], duration_s=2.5) == [
        Segment("cut off", 1.0, 2.5)
    ]


def test_a_missing_start_is_the_end_of_the_previous_segment():
    chunks = [chunk("one", 0.5, 1.5), chunk("two", None, 2.0)]
    assert segments_from_chunks(chunks, duration_s=3.0)[1] == Segment("two", 1.5, 2.0)


def test_a_missing_start_on_the_first_segment_is_zero():
    assert segments_from_chunks([chunk("one", None, 1.0)], duration_s=3.0) == [
        Segment("one", 0.0, 1.0)
    ]


def test_times_are_clamped_into_the_audio():
    chunks = [chunk("a", -0.2, 1.0), chunk("b", 1.0, 31.0)]
    assert segments_from_chunks(chunks, duration_s=9.9) == [
        Segment("a", 0.0, 1.0),
        Segment("b", 1.0, 9.9),
    ]


def test_a_segment_never_ends_before_it_starts():
    assert segments_from_chunks([chunk("odd", 2.0, 1.0)], duration_s=3.0) == [
        Segment("odd", 2.0, 2.0)
    ]


def test_a_list_pair_works_like_a_tuple():
    assert segments_from_chunks([{"text": "x", "timestamp": [0.5, 1.5]}], duration_s=2.0) == [
        Segment("x", 0.5, 1.5)
    ]


def test_times_are_plain_floats():
    segment = segments_from_chunks([chunk("x", np.float32(0.5), np.float32(1.5))], 2.0)[0]
    assert type(segment.start_s) is float and type(segment.end_s) is float


# --- transcribe_segments ---


class StubTokenizer:
    """Every word is one token, plus one leading token that stands for <|startofprev|>."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def get_prompt_ids(self, text: str, return_tensors: str = "pt"):
        self.prompts.append(text)
        return torch.arange(1000, 1000 + 1 + len(text.split()))


class StubPipe:
    def __init__(self, chunks=None, error: Exception | None = None, takes_s: float = 0.0):
        self.chunks = chunks if chunks is not None else [chunk(" hello there", 0.0, 1.5)]
        self.error = error
        self.takes_s = takes_s
        self.tokenizer = StubTokenizer()
        self.calls: list[dict] = []

    def __call__(self, inputs, generate_kwargs=None, **kwargs):
        self.calls.append({"inputs": inputs, "generate_kwargs": generate_kwargs, **kwargs})
        time.sleep(self.takes_s)
        if self.error is not None:
            raise self.error
        text = "".join(c["text"] for c in self.chunks)
        return {"text": text, "chunks": self.chunks}


def adapter(pipe: StubPipe | None = None) -> WhisperASR:
    model = WhisperASR("whisper-test", "fake/whisper", device="cpu")
    model._pipe = pipe
    return model


def test_whisper_is_a_segmented_model_with_the_trimmed_strategy():
    model = adapter()
    assert isinstance(model, SegmentedASRModel)
    assert supported_strategies(model) == ["chunked", "redecode", "trimmed"]


def test_transcribe_segments_returns_segments_and_latency():
    pipe = StubPipe(
        chunks=[chunk(" He has grave doubts", 0.0, 1.2), chunk(" whether", 1.2, None)],
        takes_s=0.03,
    )
    segments, latency_ms = adapter(pipe).transcribe_segments(AUDIO)
    assert segments == [Segment("He has grave doubts", 0.0, 1.2), Segment("whether", 1.2, 2.0)]
    assert 25 <= latency_ms < 1000


def test_transcribe_segments_asks_for_timestamps_and_the_audio_as_given():
    pipe = StubPipe()
    adapter(pipe).transcribe_segments(AUDIO, sample_rate=16000)
    call = pipe.calls[0]
    assert call["return_timestamps"] is True
    assert call["inputs"]["sampling_rate"] == 16000
    assert call["inputs"]["raw"] is AUDIO
    assert call["generate_kwargs"]["language"] == "en"
    assert call["generate_kwargs"]["task"] == "transcribe"


def test_transcribe_segments_without_a_prompt_sends_no_prompt_ids():
    pipe = StubPipe()
    adapter(pipe).transcribe_segments(AUDIO)
    assert "prompt_ids" not in pipe.calls[0]["generate_kwargs"]
    assert pipe.tokenizer.prompts == []


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_prompt_counts_as_none(blank):
    pipe = StubPipe()
    adapter(pipe).transcribe_segments(AUDIO, prompt=blank)
    assert "prompt_ids" not in pipe.calls[0]["generate_kwargs"]


def test_a_prompt_goes_to_the_model_as_prompt_ids():
    pipe = StubPipe()
    adapter(pipe).transcribe_segments(AUDIO, prompt="  He has grave doubts ")
    assert pipe.tokenizer.prompts == ["He has grave doubts"]
    ids = pipe.calls[0]["generate_kwargs"]["prompt_ids"]
    assert ids.tolist() == [1000, 1001, 1002, 1003, 1004]


def test_a_long_prompt_is_cut_to_the_limit_keeping_the_start_token_and_the_newest_words():
    pipe = StubPipe()
    prompt = " ".join(f"w{i}" for i in range(400))  # 401 tokens with the start token
    adapter(pipe).transcribe_segments(AUDIO, prompt=prompt)
    ids = pipe.calls[0]["generate_kwargs"]["prompt_ids"].tolist()
    assert len(ids) == MAX_PROMPT_TOKENS
    assert ids[0] == 1000  # the start token stays first
    assert ids[-1] == 1400  # and the newest word stays last


def test_silence_gives_no_segments():
    segments, latency_ms = adapter(StubPipe(chunks=[])).transcribe_segments(AUDIO)
    assert segments == [] and latency_ms >= 0


def test_transcribe_segments_before_load_is_an_asr_error():
    with pytest.raises(ASRError, match="whisper-test is not loaded"):
        adapter(None).transcribe_segments(AUDIO)


def test_a_pipeline_failure_is_an_asr_error():
    pipe = StubPipe(error=RuntimeError("cuda out of memory"))
    with pytest.raises(ASRError, match="whisper-test inference failed: cuda out of memory"):
        adapter(pipe).transcribe_segments(AUDIO)


def test_plain_transcribe_does_not_ask_for_timestamps():
    pipe = StubPipe()
    result = adapter(pipe).transcribe(AUDIO)
    assert "return_timestamps" not in pipe.calls[0]
    assert "prompt_ids" not in pipe.calls[0]["generate_kwargs"]
    assert result.text == "hello there"
