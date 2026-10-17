"""The Nemotron adapter with a scripted fake processor and model (PLAN_v1 S4.4).

No weights and no GPU: `FakeModel.generate` behaves like transformers' streaming generate (it
asks the audio generator for the next chunk only after it has handled the previous one, and
puts token ids on the streamer). The real model is in test_nemotron_real.py.
"""

import gc
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from backend.asr import nemotron
from backend.asr.base import ASRError, StreamingASRModel
from backend.asr.nemotron import NemotronASR, committed_and_tentative
from backend.asr.strategies import default_strategy, supported_strategies

# Small sizes with the real shape: a window starts `n_fft // 2` before its first mel frame and
# the windows overlap.
FIRST_SAMPLES, CHUNK_SAMPLES = 35, 60
FIRST_MEL, CHUNK_MEL = 3, 4
HOP, N_FFT = 10, 40

VOCAB = {0: "", 1: "He", 2: "has", 3: "grave", 4: "doubts", 5: "whether"}


def windows(total: int) -> list[tuple[int, int]]:
    """The (start, end) of every full window after the first one, for `total` samples."""
    out, mel = [], FIRST_MEL
    while (end := mel * HOP - N_FFT // 2 + CHUNK_SAMPLES) <= total:
        out.append((mel * HOP - N_FFT // 2, end))
        mel += CHUNK_MEL
    return out


class FakeBatch(dict):
    """What the processor returns: a dict with `.to()` and `.input_features`."""

    def __init__(self, features: np.ndarray) -> None:
        super().__init__(input_features=features, marker="first")
        self.input_features = features

    def to(self, *args, **kwargs) -> "FakeBatch":
        return self


class FakeProcessor:
    num_samples_first_audio_chunk = FIRST_SAMPLES
    num_samples_per_audio_chunk = CHUNK_SAMPLES
    num_mel_frames_first_audio_chunk = FIRST_MEL
    num_mel_frames_per_audio_chunk = CHUNK_MEL
    feature_extractor = SimpleNamespace(hop_length=HOP, n_fft=N_FFT)
    tokenizer = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: " ".join(VOCAB[i] for i in ids if VOCAB[i])
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []  # one entry per streaming call, with the audio it got

    def __call__(self, audio, sampling_rate=16000, return_tensors=None, **kwargs) -> FakeBatch:
        self.calls.append({"audio": np.array(audio, copy=True), **kwargs})
        if kwargs.get("is_streaming") and kwargs.get("is_first_audio_chunk"):
            return FakeBatch(np.zeros((1, FIRST_MEL + 2, 1)))  # more than the first chunk needs
        return FakeBatch(np.zeros((1, CHUNK_MEL, 1)))

    def decode(self, sequences, skip_special_tokens=True) -> list[str]:
        return [sequences]


class FakeModel:
    """`script[i]` is the list of token ids it decodes after handling chunk i."""

    device = "cpu"
    dtype = None

    def __init__(self, script: list[list[int]], fail_on_chunk: int | None = None) -> None:
        self.script = script
        self.fail_on_chunk = fail_on_chunk
        self.streams = 0  # generate() calls with a streamer
        self.chunk_shapes: list[tuple] = []
        self.batch_text = "batch text"

    def generate(self, **kwargs):
        if "streamer" not in kwargs:
            return SimpleNamespace(sequences=self.batch_text)
        self.streams += 1
        streamer = kwargs["streamer"]
        streamer.put(np.array([[0]]))  # the decoder prompt comes first
        for i, features in enumerate(kwargs["input_features"]):
            self.chunk_shapes.append(features.shape)
            if i == self.fail_on_chunk:
                raise RuntimeError("boom")
            for token in self.script[i] if i < len(self.script) else []:
                streamer.put(np.array([token]))
        streamer.end()


def make_adapter(script: list[list[int]], **model_kwargs) -> NemotronASR:
    adapter = NemotronASR("nemo", "fake/model")
    adapter._processor = FakeProcessor()
    adapter._model = FakeModel(script, **model_kwargs)
    return adapter


@pytest.fixture(autouse=True)
def plain_split(monkeypatch):
    """Adapter tests do not depend on how words are split into committed and tentative."""
    monkeypatch.setattr(nemotron, "committed_and_tentative", lambda text, final: (text, ""))


def audio(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float32)


def stream_calls(adapter: NemotronASR) -> list[np.ndarray]:
    return [c["audio"] for c in adapter._processor.calls if c.get("is_streaming")]


# --- feeding ---


def test_audio_shorter_than_the_first_window_starts_nothing() -> None:
    adapter = make_adapter([[1]])
    stream = adapter.open_stream()
    assert stream.feed(audio(FIRST_SAMPLES - 1)) is None
    assert adapter._model.streams == 0
    assert adapter._processor.calls == []


def test_the_first_window_starts_the_stream_and_gives_a_partial() -> None:
    adapter = make_adapter([[1, 2]])
    stream = adapter.open_stream()
    partial = stream.feed(audio(FIRST_SAMPLES))
    assert partial is not None
    assert (partial.committed, partial.tentative) == ("He has", "")
    first = stream_calls(adapter)[0]
    assert np.array_equal(first, audio(FIRST_SAMPLES))
    stream.finish()


def test_first_chunk_is_cut_to_its_mel_frames() -> None:
    adapter = make_adapter([[]])
    stream = adapter.open_stream()
    stream.feed(audio(FIRST_SAMPLES))
    stream.finish()
    assert adapter._model.chunk_shapes[0] == (1, FIRST_MEL, 1)


@pytest.mark.parametrize("piece", [1, 7, 13, 50, 200])
def test_windows_do_not_depend_on_how_the_audio_arrives(piece: int) -> None:
    adapter = make_adapter([[1]])
    stream = adapter.open_stream()
    data = audio(200)
    for i in range(0, 200, piece):
        stream.feed(data[i : i + piece])
    expected = [data[:FIRST_SAMPLES]] + [data[s:e] for s, e in windows(200)]
    got = stream_calls(adapter)
    assert len(got) == len(expected) == 5
    for a, b in zip(got, expected, strict=True):
        assert np.array_equal(a, b)
    stream.finish()


def test_a_partial_comes_only_when_the_text_changed() -> None:
    adapter = make_adapter([[1], [], [2]])
    stream = adapter.open_stream()
    data = audio(200)
    first = stream.feed(data[:FIRST_SAMPLES])
    assert first is not None and first.committed == "He"
    assert stream.feed(data[FIRST_SAMPLES:70]) is None  # one window, no new token
    third = stream.feed(data[70:110])
    assert third is not None and third.committed == "He has"
    stream.finish()


# --- finishing ---


def test_finish_pads_the_rest_to_a_full_window_and_returns_the_text() -> None:
    adapter = make_adapter([[1], [2], [3], [3], [4], [5]])
    stream = adapter.open_stream()
    data = audio(200)
    stream.feed(data)
    result = stream.finish()
    last = stream_calls(adapter)[-1]
    start = 170  # the next window after the four full ones: (19 mel frames x 10) - 20
    assert len(last) == CHUNK_SAMPLES
    assert np.array_equal(last[: 200 - start], data[start:])
    assert not last[200 - start :].any()
    assert result.text == "He has grave grave doubts whether"
    assert result.language == "en"
    assert result.duration_s == pytest.approx(200 / 16000)
    assert result.latency_ms >= 0


def test_finish_with_audio_equals_feed_then_finish() -> None:
    a = make_adapter([[1], [2], [3], [4], [5], [1]])
    sa = a.open_stream()
    sa.feed(audio(120))
    ra = sa.finish(audio(200)[120:])
    b = make_adapter([[1], [2], [3], [4], [5], [1]])
    sb = b.open_stream()
    sb.feed(audio(120))
    sb.feed(audio(200)[120:])
    rb = sb.finish()
    assert ra.text == rb.text
    for x, y in zip(stream_calls(a), stream_calls(b), strict=True):
        assert np.array_equal(x, y)


def test_finish_with_less_than_a_first_window_pads_the_first_window() -> None:
    adapter = make_adapter([[1, 2]])
    stream = adapter.open_stream()
    result = stream.finish(audio(20))
    calls = stream_calls(adapter)
    assert len(calls) == 1 and len(calls[0]) == FIRST_SAMPLES
    assert np.array_equal(calls[0][:20], audio(20)) and not calls[0][20:].any()
    assert result.text == "He has"


def test_finish_with_no_audio_is_empty_and_never_runs_the_model() -> None:
    adapter = make_adapter([[1]])
    result = adapter.open_stream().finish()
    assert result.text == "" and result.duration_s == 0
    assert adapter._model.streams == 0


def test_finish_twice_is_an_error() -> None:
    adapter = make_adapter([[1]])
    stream = adapter.open_stream()
    stream.finish(audio(10))
    with pytest.raises(ASRError, match="already finished"):
        stream.finish()


# --- state ---


def test_two_streams_in_a_row_share_nothing() -> None:
    adapter = make_adapter([[1, 2, 3]])
    first = adapter.open_stream()
    first.feed(audio(100))
    assert first.finish().text == "He has grave"
    second = adapter.open_stream()
    assert second.feed(audio(FIRST_SAMPLES)).committed == "He has grave"
    assert second.finish().text == "He has grave"
    assert adapter._model.streams == 2


def test_a_second_stream_or_a_batch_decode_waits_for_no_one_and_fails() -> None:
    adapter = make_adapter([[1]])
    running = adapter.open_stream()
    running.feed(audio(FIRST_SAMPLES))
    other = adapter.open_stream()
    with pytest.raises(ASRError, match="busy"):
        other.feed(audio(FIRST_SAMPLES))
    with pytest.raises(ASRError, match="busy"):
        adapter.transcribe(audio(100))
    running.finish()
    assert adapter.transcribe(audio(100)).text == "batch text"
    assert adapter.open_stream().feed(audio(FIRST_SAMPLES)) is not None


def test_a_failing_decode_raises_and_frees_the_model() -> None:
    adapter = make_adapter([[1]], fail_on_chunk=1)
    stream = adapter.open_stream()
    stream.feed(audio(FIRST_SAMPLES))
    with pytest.raises(ASRError, match="boom"):
        stream.feed(audio(100))
    assert wait_for(lambda: not adapter._lock.locked())
    assert adapter.transcribe(audio(100)).text == "batch text"


def test_a_stream_dropped_without_finish_frees_the_model() -> None:
    adapter = make_adapter([[1]])
    stream = adapter.open_stream()
    stream.feed(audio(FIRST_SAMPLES))
    assert adapter._lock.locked()
    del stream
    gc.collect()
    assert wait_for(lambda: not adapter._lock.locked())


def test_a_stream_nobody_feeds_gives_up_and_the_next_feed_says_so(monkeypatch) -> None:
    monkeypatch.setattr(nemotron, "CHUNK_TIMEOUT_S", 1.0)
    adapter = make_adapter([[1]])
    stream = adapter.open_stream()
    stream.feed(audio(FIRST_SAMPLES))
    assert wait_for(lambda: not adapter._lock.locked())
    with pytest.raises(ASRError, match="ended early"):
        stream.feed(audio(100))


def wait_for(condition, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


# --- the model object ---


def test_open_stream_checks_the_sample_rate_and_that_the_model_is_loaded() -> None:
    with pytest.raises(ASRError, match="not loaded"):
        NemotronASR("nemo", "fake/model").open_stream()
    with pytest.raises(ASRError, match="16000"):
        make_adapter([]).open_stream(sample_rate=8000)


def test_the_adapter_is_a_streaming_model_so_native_is_its_default() -> None:
    adapter = NemotronASR("nemo", "fake/model")  # not loaded: asking costs nothing
    assert isinstance(adapter, StreamingASRModel)
    assert supported_strategies(adapter) == ["chunked", "redecode", "native"]
    assert default_strategy(adapter, "redecode") == "native"


def test_batch_transcribe_returns_the_decoded_text() -> None:
    result = make_adapter([]).transcribe(audio(16000))
    assert result.text == "batch text" and result.language == "en"
    assert result.duration_s == pytest.approx(1.0)


def test_a_language_is_passed_to_the_processor_only_when_configured() -> None:
    plain = make_adapter([[1]])
    plain.open_stream().finish(audio(10))
    assert all("language" not in c for c in plain._processor.calls)
    tagged = make_adapter([[1]])
    tagged.language = "en-US"
    tagged.open_stream().finish(audio(10))
    assert all(c["language"] == "en-US" for c in tagged._processor.calls)


def test_nothing_runs_on_a_thread_after_finish() -> None:
    before = {t for t in threading.enumerate() if t.name == "nemotron-stream"}
    adapter = make_adapter([[1]])
    adapter.open_stream().finish(audio(100))
    after = {t for t in threading.enumerate() if t.name == "nemotron-stream"}
    assert after <= before


# --- which words are committed (the human part) ---


def test_the_newest_word_is_tentative_until_the_text_is_final() -> None:
    assert committed_and_tentative("He has grave", False) == ("He has", "grave")
    assert committed_and_tentative("He has grave", True) == ("He has grave", "")


def test_one_word_and_no_words() -> None:
    assert committed_and_tentative("He", False) == ("", "He")
    assert committed_and_tentative("He", True) == ("He", "")
    assert committed_and_tentative("", False) == ("", "")
    assert committed_and_tentative("", True) == ("", "")


def test_words_are_joined_by_single_spaces() -> None:
    assert committed_and_tentative("He  has   grave ", False) == ("He has", "grave")
    assert committed_and_tentative("He  has   grave ", True) == ("He has grave", "")


def test_committed_only_grows_as_the_text_grows() -> None:
    text = "He has grave doubts, whether Sir Fred"
    words = text.split()
    previous = ""
    for n in range(1, len(words) + 1):
        committed, tentative = committed_and_tentative(" ".join(words[:n]), False)
        assert committed.startswith(previous)
        assert f"{committed} {tentative}".strip() == " ".join(words[:n])
        previous = committed
