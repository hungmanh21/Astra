"""Tests for the token cap in backend/asr/whisper.py (PLAN_v1 S2.7 follow-up).

A Whisper decode that loops ("I'm sorry, I'm sorry, ...") runs until the 448-token context is
full. The adapter limits `max_new_tokens` by the length of the audio.
"""

import numpy as np
import pytest

from backend.asr.whisper import (
    MAX_PROMPT_TOKENS,
    MAX_TARGET_POSITIONS,
    RESERVED_TOKENS,
    max_new_tokens,
)
from tests.test_whisper_segments import StubPipe, adapter

SR = 16000


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


# --- max_new_tokens ---


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.5, 22),  # a 500 ms chunk: 6 + 16
        (1.0, 28),
        (9.9, 135),
        (30.0, 376),
    ],
)
def test_the_budget_grows_with_the_audio(seconds, expected):
    assert max_new_tokens(seconds) == expected


def test_a_tiny_clip_still_gets_a_few_tokens():
    assert max_new_tokens(0.0) >= 1
    assert max_new_tokens(0.05) >= 10


def test_the_budget_never_exceeds_the_context():
    assert max_new_tokens(30.0) < MAX_TARGET_POSITIONS
    assert max_new_tokens(1000.0) == MAX_TARGET_POSITIONS - RESERVED_TOKENS


def test_the_prompt_takes_its_tokens_out_of_the_budget():
    assert max_new_tokens(30.0, prompt_tokens=100) == 340  # room is 448 - 8 - 100
    full = max_new_tokens(30.0, prompt_tokens=MAX_PROMPT_TOKENS)
    assert full == MAX_TARGET_POSITIONS - RESERVED_TOKENS - MAX_PROMPT_TOKENS
    assert full + MAX_PROMPT_TOKENS < MAX_TARGET_POSITIONS  # Whisper refuses anything more


def test_a_huge_prompt_still_leaves_one_token():
    assert max_new_tokens(1.0, prompt_tokens=1000) == 1


# --- the pipeline gets the cap ---


def test_plain_transcribe_caps_the_tokens_by_the_audio_length():
    pipe = StubPipe()
    adapter(pipe).transcribe(audio(0.5))
    assert pipe.calls[0]["generate_kwargs"]["max_new_tokens"] == 22


def test_transcribe_segments_caps_the_tokens_too():
    pipe = StubPipe()
    adapter(pipe).transcribe_segments(audio(9.9))
    assert pipe.calls[0]["generate_kwargs"]["max_new_tokens"] == 135


def test_a_prompt_lowers_the_cap():
    pipe = StubPipe()
    prompt = " ".join(f"w{i}" for i in range(99))  # 100 tokens with the start token
    adapter(pipe).transcribe_segments(audio(30.0), prompt=prompt)
    assert pipe.calls[0]["generate_kwargs"]["max_new_tokens"] == 340


def test_a_cut_prompt_is_counted_after_the_cut():
    pipe = StubPipe()
    prompt = " ".join(f"w{i}" for i in range(400))  # cut to MAX_PROMPT_TOKENS
    adapter(pipe).transcribe_segments(audio(30.0), prompt=prompt)
    kwargs = pipe.calls[0]["generate_kwargs"]
    assert len(kwargs["prompt_ids"]) + kwargs["max_new_tokens"] <= MAX_TARGET_POSITIONS - 8
