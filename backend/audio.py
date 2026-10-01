"""PCM buffering and validation for one turn (SPEC 6.3.1, PLAN T1.7).

Wire format from the browser: 16 kHz, mono, signed 16-bit little-endian PCM.
"""

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
MAX_TURN_SECONDS = 30
MIN_TURN_SECONDS = 0.25


class AudioError(ValueError):
    """Bad turn audio. The message is shown to the user in an `error`, so keep it readable."""


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """TODO(you): convert raw PCM16 bytes to float32 samples.

    Input:  data - raw bytes, signed 16-bit little-endian PCM, any length (can be empty).
    Output: 1-D np.ndarray, dtype float32, values in [-1, 1]. Length = len(data) // 2.
    Raises: AudioError if len(data) is odd.

    Steps:
      1. Check the byte count is even.
      2. np.frombuffer(data, dtype="<i2"). The "<" makes it little-endian on any machine.
      3. Convert to float32 and divide by 32768.0.
    """
    raise NotImplementedError


class TurnAudioBuffer:
    """Collects the binary frames of one turn until `end_turn`."""

    def __init__(self, max_seconds: float = MAX_TURN_SECONDS) -> None:
        """TODO(you): set up an empty buffer.

        Input:  max_seconds - the longest turn the buffer accepts.
        Output: None.
        State to keep: the max size in bytes (max_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
                       and a bytearray for the audio so far.
        """
        raise NotImplementedError

    @property
    def num_samples(self) -> int:
        """TODO(you): how many samples are in the buffer.

        Input:  none.
        Output: int, bytes held // BYTES_PER_SAMPLE.
        """
        raise NotImplementedError

    @property
    def duration_s(self) -> float:
        """TODO(you): how long the buffered audio is.

        Input:  none.
        Output: float, num_samples / SAMPLE_RATE.
        """
        raise NotImplementedError

    def append(self, chunk: bytes) -> None:
        """TODO(you): add one binary frame from the browser (about 20-100 ms of audio).

        Input:  chunk - raw PCM16 bytes of one WebSocket frame.
        Output: None.
        Raises: AudioError if adding the chunk would pass max_seconds.
                The buffer must be left unchanged in that case (no partial frame).

        Steps:
          1. If len(buffer) + len(chunk) > max bytes, raise AudioError.
          2. Otherwise extend the bytearray. Don't concatenate bytes objects in a loop.

        Note: do NOT reject an odd-sized chunk here. The odd-byte rule is checked on the
        whole turn in to_float32().
        """
        raise NotImplementedError

    def to_float32(self) -> np.ndarray:
        """TODO(you): validate the whole turn and return it ready for the ASR.

        Input:  none (uses the buffered bytes).
        Output: 1-D np.ndarray, float32, values in [-1, 1] (use pcm16_to_float32).
        Raises: AudioError if the turn is
                  - empty,
                  - shorter than MIN_TURN_SECONDS (0.25 s),
                  - longer than max_seconds,
                  - or has an odd total byte count.
                Exactly MIN_TURN_SECONDS and exactly max_seconds are allowed.

        Note: the error messages go to the user, e.g. "recording is too short".
        """
        raise NotImplementedError

    def save_wav(self, path: Path) -> None:
        """TODO(you): debug only (SPEC Privacy NFR). Write the buffered turn as a WAV file.

        Input:  path - where to write, e.g. scripts/output/<turn_id>.wav.
        Output: None (writes the file).
        Format: 16 kHz, mono, 16-bit PCM.

        Steps: the stdlib `wave` module (setnchannels/setsampwidth/setframerate/writeframes),
               or `soundfile`, which is already a dependency.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """TODO(you): drop the buffered audio so the buffer can be reused.

        Input:  none.
        Output: None. After this, num_samples == 0.
        """
        raise NotImplementedError
