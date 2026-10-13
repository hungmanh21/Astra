"""Fake streaming ASR adapters for the v1 tests (PLAN_v1 S1.1).

Both follow the registry contract `Adapter(name=..., model_id=..., **options)`, so a
test can name them in an ASRModelConfig ("tests.fakes.FakeStreamingASR"). Their
behaviour is scripted through class attributes, like FakeASR in test_asr_registry.py;
call `reset()` in a fixture so tests do not leak scripts into each other.
"""

import threading
import time

import numpy as np

from backend.asr.base import PartialResult, Segment, TranscriptResult


class FakeStream:
    """One scripted turn: each feed returns the next scripted partial, then None."""

    def __init__(
        self,
        partials: list[PartialResult],
        final_text: str,
        sample_rate: int,
        feed_error: Exception | None,
        feed_gate: threading.Event | None,
    ) -> None:
        self._partials = list(partials)
        self._final_text = final_text
        self._sample_rate = sample_rate
        self._feed_error = feed_error
        self._feed_gate = feed_gate
        self.fed: list[np.ndarray] = []  # every feed's audio, in order
        self.finished = False

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        # A gate lets a test hold a feed "in flight" while more frames arrive (S1.4).
        if self._feed_gate is not None and not self._feed_gate.wait(timeout=5):
            raise TimeoutError("feed gate was never opened")
        self.fed.append(np.array(audio, dtype=np.float32, copy=True))
        if self._feed_error is not None:
            raise self._feed_error
        return self._partials.pop(0) if self._partials else None

    def finish(self) -> TranscriptResult:
        self.finished = True
        samples = sum(len(a) for a in self.fed)
        return TranscriptResult(
            text=self._final_text,
            language="en",
            duration_s=samples / self._sample_rate,
            latency_ms=0.0,
        )

    @property
    def samples(self) -> np.ndarray:
        """Everything fed so far, joined."""
        return np.concatenate(self.fed) if self.fed else np.zeros(0, dtype=np.float32)


class FakePlainASR:
    """A batch-only model: `ASRModel` and nothing else, so only the generic strategies."""

    load_calls = 0
    load_delay_s = 0.0
    transcribed: list[np.ndarray] = []  # the audio of every transcribe call

    def __init__(self, name: str, model_id: str = "fake/plain", **options) -> None:
        self.name = name

    @classmethod
    def reset(cls) -> None:
        cls.load_calls = 0
        cls.load_delay_s = 0.0
        cls.transcribed = []

    def load(self) -> None:
        time.sleep(FakePlainASR.load_delay_s)
        FakePlainASR.load_calls += 1

    def transcribe(self, audio, sample_rate=16000, language=None) -> TranscriptResult:
        FakePlainASR.transcribed.append(np.array(audio, dtype=np.float32, copy=True))
        return TranscriptResult(
            text="plain text", language="en", duration_s=len(audio) / sample_rate, latency_ms=3.0
        )


class FakeStreamingASR:
    """A model with native streaming (`StreamingASRModel`) and batch `transcribe`."""

    partials: list[PartialResult] = []
    final_text = "final text"
    feed_error: Exception | None = None
    feed_gate: threading.Event | None = None
    streams: list[FakeStream] = []  # every stream opened, oldest first

    def __init__(self, name: str, model_id: str = "fake/streaming", **options) -> None:
        self.name = name
        self.loaded = False

    @classmethod
    def reset(cls) -> None:
        cls.partials = []
        cls.final_text = "final text"
        cls.feed_error = None
        cls.feed_gate = None
        cls.streams = []

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio, sample_rate=16000, language=None) -> TranscriptResult:
        return TranscriptResult(
            text=self.final_text,
            language="en",
            duration_s=len(audio) / sample_rate,
            latency_ms=0.0,
        )

    def open_stream(self, sample_rate: int = 16000, language: str | None = None) -> FakeStream:
        stream = FakeStream(
            self.partials, self.final_text, sample_rate, self.feed_error, self.feed_gate
        )
        FakeStreamingASR.streams.append(stream)
        return stream


class FakeSegmentedASR:
    """A batch model that also returns segments (`SegmentedASRModel`), like Whisper in S2.5."""

    segments: list[Segment] = [
        Segment("Hello there.", 0.0, 1.2),
        Segment(" How are you?", 1.2, 2.5),
    ]
    prompts: list[str | None] = []  # the prompt of every transcribe_segments call

    def __init__(self, name: str, model_id: str = "fake/segmented", **options) -> None:
        self.name = name

    @classmethod
    def reset(cls) -> None:
        cls.segments = [Segment("Hello there.", 0.0, 1.2), Segment(" How are you?", 1.2, 2.5)]
        cls.prompts = []

    def load(self) -> None:
        pass

    def transcribe(self, audio, sample_rate=16000, language=None) -> TranscriptResult:
        return TranscriptResult(
            text="".join(s.text for s in self.segments).strip(),
            language="en",
            duration_s=len(audio) / sample_rate,
            latency_ms=0.0,
        )

    def transcribe_segments(
        self, audio, sample_rate: int = 16000, prompt: str | None = None
    ) -> tuple[list[Segment], float]:
        FakeSegmentedASR.prompts.append(prompt)
        return list(self.segments), 0.0


class FakeFullASR(FakeStreamingASR, FakeSegmentedASR):
    """Both optional capabilities at once: all four strategies."""

    def __init__(self, name: str, model_id: str = "fake/full", **options) -> None:
        FakeStreamingASR.__init__(self, name, model_id, **options)
