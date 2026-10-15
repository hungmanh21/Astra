"""Which streaming strategies a model supports, and the Whisper-side streams (PLAN_v1 S1.5, S2).

Strategies come from what an adapter can do, never from its name (SPEC_v1 6.4). The generic
`ChunkedStream` and `RedecodeStream` and the segment-based `TrimmedStream` live here; the
`native` stream belongs to the adapter.
"""

import re

import numpy as np

from backend.asr.agreement import local_agreement
from backend.asr.base import (
    ASRModel,
    PartialResult,
    Segment,
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


_SENTENCE_END = re.compile(r"[.?!][\"')\]\u201d\u2019]*$")
_ELLIPSIS_END = re.compile(r"(\.\.\.|\u2026)[\"')\]\u201d\u2019]*$")


def _ends_sentence(text: str) -> bool:
    text = text.rstrip()
    return bool(_SENTENCE_END.search(text)) and not _ELLIPSIS_END.search(text)


def trim_point(
    segments: list[Segment], committed_words: int, latest_end_s: float | None = None
) -> int:
    """How many leading segments are settled enough to cut their audio away.

    Input:  segments - the segments of the latest decode of the window, in time order.
            committed_words - how many leading words of that decode are committed
            (LocalAgreement). A segment's words are what `segment.text.split()` gives.
            latest_end_s - when given, a segment that ends after this many seconds is not a
            cut point, so the cut leaves enough audio in the window (see TrimmedStream).
    Output: the largest k such that BOTH hold, or 0 when no k does:
              - the words of segments[:k] together are at most `committed_words`: a segment
                that is only partly committed is never cut, and
              - segments[k-1] ends a sentence: its text, after trailing whitespace and closing
                quotes or brackets (`"`, `'`, `)`, `]`, and the curly `\u201d`, `\u2019`)
                are removed, ends with ".", "?" or "!" and is not a trailing-off ellipsis
                ("..." or "\u2026"): the speaker usually goes on after those.
            Example: segments "Hello there." / "How are" / "you today?" with 4 committed
            words -> 1 (the second segment is not a sentence end, so the cut stops before it);
            with 6 committed words -> 3.
    """
    words = 0
    best = 0
    for k, segment in enumerate(segments, start=1):
        words += len(segment.text.split())
        if words > committed_words:
            break
        if _ends_sentence(segment.text) and (latest_end_s is None or segment.end_s <= latest_end_s):
            best = k
    return best


def _join(segments: list[Segment]) -> str:
    """The segments' texts, stripped, joined by single spaces."""
    return " ".join(text for seg in segments if (text := seg.text.strip()))


class TrimmedStream:
    """The `trimmed` strategy: `redecode`, but the decoded audio never grows past one sentence.

    Once a whole sentence is committed, its audio is cut away and its text becomes the
    prompt of the next decodes. So a decode covers only the audio after the last cut (the
    "window"), however long the turn gets. Needs a SegmentedASRModel (SPEC_v1 6.4).

    The text of a turn has two parts: `_settled`, the text of the audio already cut away
    (never decoded again), and the window's own text, which LocalAgreement splits into
    committed words and a tentative tail like in `RedecodeStream`.
    """

    def __init__(
        self,
        model: SegmentedASRModel,
        *,
        min_audio_ms: int,
        min_tail_ms: int | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self._model = model
        self._min_samples = min_audio_ms * sample_rate // 1000  # no partial before this much audio
        # A cut must leave at least this much audio in the window: Whisper invents text ("Thank
        # you.") on a window of a second or less. The same reason as `min_audio_ms`, so it is
        # the default.
        tail_ms = min_audio_ms if min_tail_ms is None else min_tail_ms
        self._min_tail_samples = tail_ms * sample_rate // 1000
        self._sample_rate = sample_rate
        self._window: list[np.ndarray] = []  # audio after the last cut, oldest first
        self._total_samples = 0  # everything ever fed, cut or not
        self._settled = ""  # text of the audio that was cut away
        self._committed = ""  # committed text of the window (not counting `_settled`)
        self._previous = ""  # text of the window at the last successful decode
        self._shown = ("", "")  # (committed, tentative) of the last partial returned

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        """Add `audio`, decode the window (once enough audio exists), maybe cut a sentence.

        Input:  audio - the NEW samples (float32). Copy what you keep, and keep it BEFORE
                decoding, so a failed decode loses nothing.
        Output: None while the whole turn (cut audio included) is shorter than `min_audio_ms`.
                After that, one `model.transcribe_segments(window, sample_rate,
                prompt=self._settled or None)` call, then:
                  1. text = the segments' texts, stripped, joined by single spaces.
                  2. committed, tentative = `local_agreement(self._committed, self._previous,
                     text)`.
                  3. k = `trim_point(segments, number of words in committed, latest_end_s)`,
                     where `latest_end_s` is the window's length in seconds minus
                     `_min_tail_samples`. If k > 0:
                     drop the window audio up to `segments[k-1].end_s` (seconds counted from
                     the START OF THE WINDOW, not of the turn; clamp to the window length),
                     append the cut segments' text to `_settled`, and take their words off
                     the front of both the committed text and the text kept as `_previous`
                     (the window now starts after them, and the next decode will not contain
                     them).
                  4. update `_committed` and `_previous`.
                The partial is PartialResult(settled + committed, tentative) with one space
                between the two parts (no stray space when `_settled` or the committed text is
                empty). Return None instead when that pair equals `_shown`, so a steady text is
                not sent twice; the decode still counts as the previous one.
        Raises: whatever the model raises; the state (`_settled`, `_committed`, `_previous`,
                the cut point) stays as it was, and the audio just fed stays in the window.
        """
        self._window.append(np.array(audio, dtype=np.float32, copy=True))
        self._total_samples += len(audio)
        if self._total_samples < self._min_samples:
            return None

        window = np.concatenate(self._window)
        segments, _ = self._model.transcribe_segments(
            window, self._sample_rate, prompt=self._settled or None
        )
        text = _join(segments)
        committed, tentative = local_agreement(self._committed, self._previous, text)
        committed_words = committed.split()

        # From here on nothing can raise: the decode worked, so the state may change.
        latest_end_s = (len(window) - self._min_tail_samples) / self._sample_rate
        cut = trim_point(segments, len(committed_words), latest_end_s)
        if cut:
            dropped = len(_join(segments[:cut]).split())
            # The committed spelling, not this decode's: what the user saw must not change.
            self._settled = " ".join(filter(None, [self._settled, *committed_words[:dropped]]))
            committed_words = committed_words[dropped:]
            text = " ".join(text.split()[dropped:])
            # Seconds count from the start of the window, so cuts add up by themselves.
            end = min(int(segments[cut - 1].end_s * self._sample_rate), len(window))
            self._window = [window[end:]] if end < len(window) else []

        self._committed = " ".join(committed_words)
        self._previous = text
        shown = (" ".join(filter(None, [self._settled, self._committed])), tentative)
        if shown == self._shown:
            return None
        self._shown = shown
        return PartialResult(*shown)

    def finish(self) -> TranscriptResult:
        """Decode only the window and return the whole turn's text: settled + the tail.

        Output: TranscriptResult(text=`_settled` and the window's decode (segments joined
                like in `feed`, prompted with `_settled`), joined by one space, language="en",
                duration_s=ALL the audio fed, cut or not, latency_ms=the decode's latency).
                No model call, and the settled text alone, when the window holds no audio
                (the cut was exactly at the end). Nothing fed gives an empty result
                (text "", duration 0.0, latency 0.0) without a model call. A turn shorter
                than `min_audio_ms` is decoded too: the minimum only gates partials.
        Raises: whatever the model raises.
        """
        if self._total_samples == 0:
            return TranscriptResult(text="", language="en", duration_s=0.0, latency_ms=0.0)

        tail, latency_ms = "", 0.0
        if self._window:  # a cut at the very end leaves no audio to decode
            segments, latency_ms = self._model.transcribe_segments(
                np.concatenate(self._window), self._sample_rate, prompt=self._settled or None
            )
            tail = _join(segments)
        return TranscriptResult(
            text=" ".join(filter(None, [self._settled, tail])),
            language="en",
            duration_s=self._total_samples / self._sample_rate,
            latency_ms=latency_ms,
        )
