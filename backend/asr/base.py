"""ASR adapter contract (SPEC 6.4, PLAN T2.1).

Every ASR model is one adapter class that satisfies `ASRModel`. The registry and
the transcription service only ever talk to this interface, never to a specific model.

Nothing to implement in this file: it defines the shapes the other files use.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ASRError(RuntimeError):
    """Anything that goes wrong while loading or running an ASR model."""


class UnknownASRModel(ASRError):
    """The requested registry key does not exist."""


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
