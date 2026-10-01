"""PCM buffering and validation for one turn (SPEC 6.3.1, PLAN T1.7).

Wire format from the browser: 16 kHz, mono, signed 16-bit little-endian PCM.
"""

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
MAX_TURN_SECONDS = 30
MIN_TURN_SECONDS = 0.25


class AudioError(ValueError):
    """Bad turn audio. The message is shown to the user in an `error`, so keep it readable."""


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Raw PCM16 little-endian bytes -> float32 samples in [-1, 1]."""
    if len(data) % BYTES_PER_SAMPLE:
        raise AudioError("audio data is corrupted (odd number of bytes)")
    samples = np.frombuffer(data, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


class TurnAudioBuffer:
    """Collects the binary frames of one turn until `end_turn`."""

    def __init__(self, max_seconds: float = MAX_TURN_SECONDS) -> None:
        self._max_seconds = max_seconds
        self._max_bytes = int(max_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        self._data = bytearray()

    @property
    def num_samples(self) -> int:
        return len(self._data) // BYTES_PER_SAMPLE

    @property
    def duration_s(self) -> float:
        return self.num_samples / SAMPLE_RATE

    def append(self, chunk: bytes) -> None:
        if len(self._data) + len(chunk) > self._max_bytes:
            raise AudioError(f"recording is too long (max {self._max_seconds:g} s)")
        self._data.extend(chunk)

    def to_float32(self) -> np.ndarray:
        """Validate the whole turn (including the odd-byte rule) and convert it."""
        if not self._data:
            raise AudioError("no audio received")
        if len(self._data) % BYTES_PER_SAMPLE:
            raise AudioError("audio data is corrupted (odd number of bytes)")
        if self.duration_s < MIN_TURN_SECONDS:
            raise AudioError("recording is too short")
        if len(self._data) > self._max_bytes:
            raise AudioError(f"recording is too long (max {self._max_seconds:g} s)")
        return pcm16_to_float32(bytes(self._data))

    def save_wav(self, path: Path) -> None:
        """Debug only (SPEC Privacy NFR)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(BYTES_PER_SAMPLE)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(bytes(self._data))

    def clear(self) -> None:
        self._data.clear()
