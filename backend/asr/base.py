"""ASR adapter contract (SPEC 6.4, PLAN T2.1).

Every ASR model is one adapter class that satisfies `ASRModel`. The registry and
the transcription service only ever talk to this interface, never to a specific model.

Nothing to implement in this file: it defines the shapes the other files use.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class ASRError(RuntimeError):
    """Anything that goes wrong while loading or running an ASR model."""


class UnknownASRModel(ASRError):
    """The requested registry key does not exist."""


class UnsupportedStrategy(ASRError):
    """The model exists but does not support the requested streaming strategy (FR-22)."""


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str | None
    duration_s: float  # audio length, from the sample count
    latency_ms: float  # time inside transcribe() only; excludes model loading (SPEC 6.3.2)


class ASRModel(Protocol):
    name: str  # the registry key, e.g. "whisper-large-v3"

    def load(self) -> None:
        """Load weights onto the device. Called once, lazily, from a worker thread.

        Input:  none.
        Output: None. After this returns, transcribe() must work.
        Raises: ASRError if loading fails.
        """
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptResult:
        """Blocking transcription of one whole turn.

        Input:  audio - 1-D float32 array in [-1, 1], mono, at `sample_rate`.
                sample_rate - 16000 in v0.
                language - "en" in v0 (adapters may also just ignore it and force English).
        Output: TranscriptResult. latency_ms covers inference only.
        Raises: ASRError if inference fails.
        """
        ...


# --- Streaming (SPEC_v1 6.4, PLAN_v1 S1.1) ---

STRATEGIES = ("chunked", "redecode", "trimmed", "native")
GENERIC_STRATEGIES = ("chunked", "redecode")  # work with any ASRModel


@dataclass(frozen=True)
class PartialResult:
    committed: str  # settled text, only ever grows within one stream
    tentative: str  # may change or vanish on the next partial


@runtime_checkable
class ASRStream(Protocol):
    """One turn's stream. Every call blocks and runs on the ASR worker thread."""

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        """Add audio and maybe produce a partial.

        Input:  audio - only the NEW samples since the last feed (float32, mono, 16 kHz).
        Output: PartialResult, or None when there is nothing new to show yet.
        Raises: ASRError if inference fails.
        """
        ...

    def finish(self, audio: np.ndarray | None = None) -> TranscriptResult:
        """Final transcript of the whole turn. Called once, after the last feed.

        Input:  audio - the last samples, which were never fed (None when there are none).
                The stream treats them as one more feed whose partial nobody wants, so a
                strategy that decodes on every feed decodes once instead of twice.
        Output: TranscriptResult. latency_ms covers this call only (the v0 `asr_ms`).
        Raises: ASRError if inference fails.
        """
        ...


@runtime_checkable
class StreamingASRModel(ASRModel, Protocol):
    """Optional capability: the model streams natively (the `native` strategy)."""

    def open_stream(self, sample_rate: int = 16000, language: str | None = None) -> ASRStream:
        """Start a stream for one turn, with its own state (nothing shared with other streams).

        Input:  sample_rate - 16000 in v1.
                language - "en" in v1 (adapters may ignore it).
        Output: an ASRStream.
        Raises: ASRError if the stream cannot be started.
        """
        ...


@dataclass(frozen=True)
class Segment:
    text: str
    start_s: float  # seconds from the start of the audio passed to transcribe_segments
    end_s: float


@runtime_checkable
class SegmentedASRModel(ASRModel, Protocol):
    """Optional capability: the model returns timestamped segments (the `trimmed` strategy)."""

    def transcribe_segments(
        self, audio: np.ndarray, sample_rate: int = 16000, prompt: str | None = None
    ) -> tuple[list[Segment], float]:
        """Transcribe into segments with timestamps.

        Input:  audio - 1-D float32 array in [-1, 1], mono, at `sample_rate`.
                prompt - text the model should continue from (the committed text), or None.
        Output: (segments in time order, latency_ms). latency_ms covers inference only.
        Raises: ASRError if inference fails.
        """
        ...
