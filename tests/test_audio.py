"""Tests for backend/audio.py (PLAN T1.7). Replace each skip with a real test."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO: implement backend/audio.py first")


def test_pcm16_to_float32_range_and_values():
    """Samples [0, 16384, -16384, 32767, -32768] -> ~[0, 0.5, -0.5, ~1.0, -1.0], float32."""


def test_pcm16_to_float32_rejects_odd_length():
    """AudioError on an odd byte count."""


def test_buffer_accumulates_frames_in_order():
    """Appending frames then to_float32() equals converting the concatenated bytes once."""


def test_buffer_rejects_empty_turn():
    """to_float32() on an empty buffer raises AudioError."""


def test_buffer_rejects_too_short_turn():
    """Under 0.25 s raises AudioError; exactly 0.25 s is accepted."""


def test_buffer_rejects_too_long_turn_and_stays_unchanged():
    """append() that would pass the max raises AudioError and does not change num_samples."""


def test_buffer_accepts_exactly_30_seconds():
    """480000 samples is allowed (the browser stops at exactly this length)."""


def test_buffer_rejects_odd_total_bytes():
    """Odd total byte count raises AudioError in to_float32()."""


def test_save_wav_round_trip(tmp_path):
    """Write a WAV, read it back with soundfile, samples and rate (16000) match."""
