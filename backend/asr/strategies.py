"""Which streaming strategies a model supports, and the placeholder stream (PLAN_v1 S1.5).

Strategies come from what an adapter can do, never from its name (SPEC_v1 6.4). S2 adds the
real `ChunkedStream`, `RedecodeStream` and `TrimmedStream` to this file.
"""

import numpy as np

from backend.asr.agreement import local_agreement
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


class RedecodeStream:
    """The `redecode` strategy: every feed transcribes the whole turn again.

    The text is what the model hears with all the audio so far, so it has full context.
    LocalAgreement-2 (`local_agreement`) splits each decode into committed words, which
    two decodes in a row agree on, and a tentative tail. `finish` runs one plain full
    decode, so the final transcript equals v0's. Works with any ASRModel.
    """

    def __init__(
        self,
        model: ASRModel,
        *,
        min_audio_ms: int,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> None:
        self._model = model
        self._min_samples = min_audio_ms * sample_rate // 1000  # no partial before this much audio
        self._sample_rate = sample_rate
        self._language = language
        self._audio: list[np.ndarray] = []  # the whole turn, oldest first
        self._total_samples = 0
        self._committed = ""  # committed text so far
        self._previous = ""  # text of the last successful decode
        self._shown = ("", "")  # (committed, tentative) of the last partial returned

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        """Add `audio`, then (once enough audio exists) decode the whole turn again.

        Input:  audio - the NEW samples (float32). Copy what you keep: the caller may reuse
                the array. Keep it BEFORE decoding, so a failed decode loses nothing.
        Output: None while the turn holds less than `min_audio_ms`. After that, decode ALL
                the audio so far with one `model.transcribe` call and return
                PartialResult(committed, tentative) from
                `local_agreement(self._committed, self._previous, text)`, updating
                `_committed` and `_previous`. Return None instead when that (committed,
                tentative) pair equals the last one returned (`_shown`), so silence or a
                steady text does not send the same partial again.
        Raises: whatever the model raises; `_committed` and `_previous` stay as they were.
        """
        self._audio.append(np.array(audio, dtype=np.float32, copy=True))
        self._total_samples += len(audio)
        if self._total_samples < self._min_samples:
            return None

        samples = np.concatenate(self._audio)
        result = self._model.transcribe(
            samples, sample_rate=self._sample_rate, language=self._language
        )
        new_committed, tentative = local_agreement(self._committed, self._previous, result.text)
        # The decode counts as "the previous one" even when it shows nothing new.
        self._committed = new_committed
        self._previous = result.text
        if (new_committed, tentative) == self._shown:
            return None

        self._shown = (new_committed, tentative)
        return PartialResult(new_committed, tentative)

    def finish(self) -> TranscriptResult:
        """One plain decode of everything fed: the final transcript.

        Output: the model's TranscriptResult, unchanged, so the text equals what v0's
                single `transcribe` gives. A clip shorter than `min_audio_ms` is decoded
                too: the minimum only gates partials. Nothing fed gives an empty result
                (text "", language "en", duration 0.0, latency 0.0) without calling the model.
        """
        if self._total_samples == 0:
            return TranscriptResult(text="", language="en", duration_s=0.0, latency_ms=0.0)

        samples = np.concatenate(self._audio)
        return self._model.transcribe(
            samples, sample_rate=self._sample_rate, language=self._language
        )
