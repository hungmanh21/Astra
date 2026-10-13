"""Which streaming strategies a model supports, and the placeholder stream (PLAN_v1 S1.5).

Strategies come from what an adapter can do, never from its name (SPEC_v1 6.4). S2 adds the
real `ChunkedStream`, `RedecodeStream` and `TrimmedStream` to this file.
"""

import numpy as np

from backend.asr.base import (
    ASRModel,
    PartialResult,
    SegmentedASRModel,
    StreamingASRModel,
    TranscriptResult,
)


def supported_strategies(model: ASRModel) -> list[str]:
    """The strategies this model can run, in `STRATEGIES` order.

    Input:  model - an adapter instance. It need not be loaded.
    Output: always "chunked" and "redecode"; plus "trimmed" if the model is a
            SegmentedASRModel; plus "native" if it is a StreamingASRModel.
            Example: a plain model gives ["chunked", "redecode"]; a model with both
            capabilities gives ["chunked", "redecode", "trimmed", "native"].
    """
    supported = ["chunked", "redecode"]
    if isinstance(model, SegmentedASRModel):
        supported.append("trimmed")
    if isinstance(model, StreamingASRModel):
        supported.append("native")
    return supported


def default_strategy(model: ASRModel, fallback: str) -> str:
    """The strategy a turn uses when the client does not choose one (SPEC_v1 6.4).

    Input:  model - an adapter instance.
            fallback - `asr.streaming.default_strategy` (a generic strategy).
    Output: "native" when the model supports it, else `fallback`. A model that is only
            segmented does not default to "trimmed": that one is opt-in.
    """
    if isinstance(model, StreamingASRModel):
        return "native"
    return fallback


class BatchStream:
    """Stands in for every non-native strategy until S2: it behaves exactly like v0.

    `feed` keeps the audio and shows no partial. `finish` transcribes the whole turn in one
    go. Each stream has its own buffer, even for two streams on the same model.
    """

    def __init__(self, model: ASRModel, sample_rate: int = 16000, language: str = "en") -> None:
        self._audio: list[np.ndarray] = []  # joined once, in finish()
        self._model = model
        self._sample_rate = sample_rate
        self._language = language

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        """Store `audio` (copy it: the caller may reuse its array) and return None."""
        self._audio.append(np.array(audio, dtype=np.float32, copy=True))
        return None

    def finish(self) -> TranscriptResult:
        """Transcribe everything fed, in order, with one `model.transcribe` call.

        Output: the model's TranscriptResult unchanged. If nothing was fed, return an empty
                result (text "", language "en", duration 0.0, latency 0.0) without calling
                the model: a real model may fail on empty audio.
        """
        if not self._audio:
            return TranscriptResult(text="", language="en", duration_s=0.0, latency_ms=0.0)
        audio = np.concatenate(self._audio)
        return self._model.transcribe(audio, sample_rate=self._sample_rate, language=self._language)
