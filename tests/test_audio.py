"""Tests for backend/audio.py (PLAN T1.7)."""

import numpy as np
import pytest
import soundfile as sf

from backend.audio import (
    MAX_TURN_SECONDS,
    MIN_TURN_SECONDS,
    SAMPLE_RATE,
    AudioError,
    TurnAudioBuffer,
    pcm16_to_float32,
)


def pcm(samples: list[int] | np.ndarray) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def silence(seconds: float) -> bytes:
    return pcm(np.zeros(int(seconds * SAMPLE_RATE)))


def test_pcm16_to_float32_range_and_values():
    out = pcm16_to_float32(pcm([0, 16384, -16384, 32767, -32768]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0, 0.5, -0.5, 32767 / 32768, -1.0])


def test_pcm16_to_float32_empty():
    assert len(pcm16_to_float32(b"")) == 0


def test_pcm16_to_float32_rejects_odd_length():
    with pytest.raises(AudioError):
        pcm16_to_float32(b"\x00\x01\x02")


def test_buffer_accumulates_frames_in_order():
    data = pcm(np.arange(8000))
    buf = TurnAudioBuffer()
    for i in range(0, len(data), 1600):
        buf.append(data[i : i + 1600])
    np.testing.assert_array_equal(buf.to_float32(), pcm16_to_float32(data))
    assert buf.num_samples == 8000
    assert buf.duration_s == 0.5


def test_buffer_rejects_empty_turn():
    with pytest.raises(AudioError):
        TurnAudioBuffer().to_float32()


def test_buffer_rejects_too_short_turn():
    buf = TurnAudioBuffer()
    buf.append(silence(MIN_TURN_SECONDS - 0.01))
    with pytest.raises(AudioError, match="too short"):
        buf.to_float32()


def test_buffer_accepts_exactly_min_turn():
    buf = TurnAudioBuffer()
    buf.append(silence(MIN_TURN_SECONDS))
    assert len(buf.to_float32()) == int(MIN_TURN_SECONDS * SAMPLE_RATE)


def test_buffer_rejects_too_long_turn_and_stays_unchanged():
    buf = TurnAudioBuffer(max_seconds=1)
    buf.append(silence(0.9))
    before = buf.num_samples
    with pytest.raises(AudioError, match="too long"):
        buf.append(silence(0.2))
    assert buf.num_samples == before


def test_buffer_accepts_exactly_30_seconds():
    buf = TurnAudioBuffer()
    buf.append(silence(MAX_TURN_SECONDS))
    assert len(buf.to_float32()) == 480_000


def test_buffer_rejects_odd_total_bytes():
    buf = TurnAudioBuffer()
    buf.append(silence(0.5))
    buf.append(b"\x00")  # odd chunks are fine on their own
    with pytest.raises(AudioError, match="corrupted"):
        buf.to_float32()


def test_odd_chunks_that_sum_to_even_are_accepted():
    data = silence(0.5)
    buf = TurnAudioBuffer()
    buf.append(data[:1])
    buf.append(data[1:])
    assert buf.num_samples == len(data) // 2
    assert len(buf.to_float32()) == len(data) // 2


def test_clear_empties_the_buffer():
    buf = TurnAudioBuffer()
    buf.append(silence(0.5))
    buf.clear()
    assert buf.num_samples == 0


def test_save_wav_round_trip(tmp_path):
    samples = np.arange(-4000, 4000, dtype="<i2")
    buf = TurnAudioBuffer()
    buf.append(samples.tobytes())
    path = tmp_path / "out" / "turn.wav"
    buf.save_wav(path)

    data, rate = sf.read(path, dtype="int16")
    assert rate == SAMPLE_RATE
    np.testing.assert_array_equal(data, samples)
