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


class ChunkedStream:
    """The `chunked` strategy: every `chunk_ms` of audio is transcribed on its own, once.

    Each chunk's text is appended to `committed`; nothing is ever tentative and nothing is
    decoded twice. Words cut at a chunk edge and the missing context are the point of the
    experiment (SPEC_v1 6.4). Works with any ASRModel.
    """

    def __init__(
        self,
        model: ASRModel,
        *,
        chunk_ms: int,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> None:
        self._model = model
        self._chunk_samples = chunk_ms * sample_rate // 1000
        self._sample_rate = sample_rate
        self._language = language
        self._pending: list[np.ndarray] = []  # audio not decoded yet, oldest first
        self._pending_samples = 0  # total length of `_pending`
        self._texts: list[str] = []  # the non-empty text of every decoded chunk, in order
        self._total_samples = 0  # everything ever fed, including a dropped remainder

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        """Add `audio`, then decode every complete chunk now available, oldest first.

        Input:  audio - the NEW samples (float32). Copy what you keep: the caller may reuse
                the array. A feed may hold less than one chunk, or several (coalescing).
        Output: PartialResult(committed=all chunk texts so far joined by one space,
                tentative="") when this call added text; None when no chunk was decoded or
                the chunks it decoded were empty. Chunks are exactly `chunk_ms` long and
                follow on from each other with no gap and no overlap, even across feeds.
        Raises: whatever the model raises. A chunk whose decode failed must NOT be lost:
                leave its audio pending, so the next feed decodes it again.
        """
        self._pending.append(np.array(audio, dtype=np.float32, copy=True))
        self._pending_samples += len(audio)
        self._total_samples += len(audio)

        texts_before = len(self._texts)
        while self._pending_samples >= self._chunk_samples:
            joined = np.concatenate(self._pending)
            chunk, rest = joined[: self._chunk_samples], joined[self._chunk_samples :]
            # The chunk leaves `_pending` only after the decode worked: if transcribe raises,
            # the audio is still here for the next feed.
            text = self._model.transcribe(
                chunk, sample_rate=self._sample_rate, language=self._language
            ).text.strip()
            self._pending = [rest] if len(rest) else []
            self._pending_samples = len(rest)
            if text:
                self._texts.append(text)

        if len(self._texts) == texts_before:
            return None
        return PartialResult(committed=" ".join(self._texts), tentative="")

    def finish(self) -> TranscriptResult:
        """Decode the remainder, however short, and return the whole turn's text.

        Output: TranscriptResult(text=all chunk texts joined by one space, language="en",
                duration_s=all audio fed, in seconds,
                latency_ms=the model's own latency_ms for the remainder decode, 0.0 when
                nothing was decoded here).
                Nothing fed gives an empty result without calling the model. A very short
                remainder can make Whisper invent text; the bench shows whether it does.
        """
        latency_ms = 0.0
        if self._pending_samples > 0:
            result = self._model.transcribe(
                np.concatenate(self._pending),
                sample_rate=self._sample_rate,
                language=self._language,
            )
            latency_ms = result.latency_ms
            if result.text.strip():
                self._texts.append(result.text.strip())

        return TranscriptResult(
            text=" ".join(self._texts),
            language="en",
            duration_s=self._total_samples / self._sample_rate,
            latency_ms=latency_ms,
        )
