"""Nemotron streaming ASR on Hugging Face transformers (PLAN_v1 S4.4, SPEC_v1 6.4).

Models: nvidia/nemotron-speech-streaming-en-0.6b (default) and the multilingual
nvidia/nemotron-3.5-asr-streaming-0.6b. Both are cache-aware streaming RNNT models, about
2.6 GB on the GPU. Besides plain `transcribe`, the adapter can open a native stream
(`open_stream`), which makes it a StreamingASRModel and unlocks the `native` strategy.

How a stream works. transformers streams through `generate(input_features=<generator>)`: the
loop asks the generator for the next chunk of mel features, encodes it with the model's cache
and decodes the new tokens, then asks again. That is a pull API; the session pushes audio. So a
stream runs `generate` on a thread of its own and `feed` hands it chunks through a queue. A
chunk is done when `generate` asks for the next one. Tokens reach us through a streamer
object, so the text is read from token ids and does not wait for a word boundary.
"""

import logging
import queue
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.asr.base import ASRError, PartialResult, TranscriptResult

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_TIMEOUT_S = 30.0  # longest the worker waits for audio, or `feed` waits for the worker


def committed_and_tentative(text: str, final: bool) -> tuple[str, str]:
    """Split the text decoded so far into settled words and a tail that may still change.

    Native tokens never change once decoded, but the newest word may still be growing
    ("Sir Fred" now, "Sir Frederick" a chunk later), and a half word looks wrong in the UI.

    Input:  text  - everything decoded so far, as the tokenizer returns it (single spaces,
                    no leading or trailing space, may be "").
            final - True from `finish`, when no more audio will come.
    Output: (committed, tentative). Words are whatever `str.split()` gives, joined by single
            spaces. committed + tentative are all the words of `text`, in order.
            Examples:
              ("He has grave", False) -> ("He has", "grave")
              ("He has grave", True)  -> ("He has grave", "")
              ("He", False)           -> ("", "He")
              ("", False)             -> ("", "")
    """
    words = text.split()
    if not words:
        return "", ""
    if final:
        return " ".join(words), ""
    return " ".join(words[:-1]), words[-1]


@dataclass(frozen=True)
class ChunkPlan:
    """How the model wants its audio cut (all from the processor, for one lookahead setting)."""

    first_samples: int  # audio in the first window
    chunk_samples: int  # audio in every later window
    first_mel_frames: int  # mel frames the first window yields
    chunk_mel_frames: int  # mel frames every later window yields
    hop: int  # samples between two mel frames
    n_fft: int  # windows overlap by this much on the left (the STFT looks around each frame)


class _Worker:
    """One `generate()` call on a thread, fed through a queue.

    It knows nothing about the processor's chunk sizes: `feed_features` gets ready mel features
    from the caller's `to_features` and the worker only moves them. Everything that touches the
    GPU runs on this thread; callers only enqueue numpy audio.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        first_inputs: Any,
        to_features: Any,
        first_features: Any,
        release: Any,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._first_inputs = first_inputs
        self._to_features = to_features
        self._first_features = first_features
        self._release = release  # called once when the thread ends, to free the model lock

        self._chunks: queue.Queue[np.ndarray | None] = queue.Queue()
        self._waiting = threading.Event()  # set while generate() waits for the next chunk
        self._ended = threading.Event()  # set when the thread is done
        self._error: BaseException | None = None
        self._ids: list[int] = []
        self._prompt_seen = False  # generate() first puts its decoder prompt; it is not text
        self._thread = threading.Thread(target=self._run, name="nemotron-stream", daemon=True)

    # --- the thread ---

    def _run(self) -> None:
        import torch

        try:
            with torch.inference_mode():
                self._model.generate(
                    **{**self._first_inputs, "input_features": self._generator()},
                    streamer=self,
                )
        except BaseException as exc:  # noqa: BLE001 - handed to the caller in `wait`
            self._error = exc
        finally:
            self._ended.set()
            self._waiting.set()  # wake a `wait` that is blocked
            self._release()

    def _generator(self):
        yield self._first_features
        while True:
            self._waiting.set()
            try:
                audio = self._chunks.get(timeout=CHUNK_TIMEOUT_S)
            except queue.Empty:
                return  # nobody feeds us any more: end the stream
            if audio is None:
                return
            yield self._to_features(audio)

    # the streamer interface generate() calls
    def put(self, value: Any) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        self._ids.extend(value.flatten().tolist())

    def end(self) -> None:
        pass

    # --- the caller ---

    def start(self) -> None:
        self._thread.start()
        self.wait()  # the first chunk is processed

    def send(self, audio: np.ndarray) -> None:
        """Hand one chunk of audio to the thread and wait until it has been processed."""
        self._waiting.clear()
        self._chunks.put(audio)
        self.wait()

    def wait(self) -> None:
        if not self._waiting.wait(CHUNK_TIMEOUT_S) and not self._ended.is_set():
            raise ASRError("Nemotron stream timed out")
        if self._ended.is_set():
            if self._error is not None:
                raise ASRError(f"Nemotron inference failed: {self._error}") from self._error
            raise ASRError("Nemotron stream ended early")

    def text(self) -> str:
        return self._tokenizer.decode(self._ids, skip_special_tokens=True).strip()

    def close(self) -> str:
        """End the stream after the last chunk. Returns the full text."""
        self._chunks.put(None)
        self._thread.join(CHUNK_TIMEOUT_S)
        if self._error is not None:
            raise ASRError(f"Nemotron inference failed: {self._error}") from self._error
        if self._thread.is_alive():
            raise ASRError("Nemotron stream did not stop")
        return self.text()

    def abort(self) -> None:
        """Let the thread end by itself (the stream was dropped without finish)."""
        self._chunks.put(None)


class NemotronStream:
    """One turn. Implements ASRStream: `feed` adds audio, `finish` flushes and returns the text."""

    def __init__(self, adapter: "NemotronASR") -> None:
        self._adapter = adapter
        self._audio = np.zeros(0, dtype=np.float32)  # every sample fed so far
        self._worker: _Worker | None = None
        self._mel_done = 0  # mel frames handed to the model so far
        self._sent_end = 0  # end (in samples) of the last window handed over
        self._last_text = ""
        self._finished = False

    def feed(self, audio: np.ndarray) -> PartialResult | None:
        self._audio = np.concatenate([self._audio, audio.astype(np.float32, copy=False)])
        if self._pump(flush=False) == 0:
            return None
        return self._partial(final=False)

    def finish(self, audio: np.ndarray | None = None) -> TranscriptResult:
        if self._finished:
            raise ASRError("this Nemotron stream is already finished")
        self._finished = True
        start = time.perf_counter()
        if audio is not None and len(audio):
            self._audio = np.concatenate([self._audio, audio.astype(np.float32, copy=False)])
        if len(self._audio) == 0:
            text = ""
        else:
            self._pump(flush=True)
            assert self._worker is not None
            text = self._worker.close()
        return TranscriptResult(
            text=text,
            language="en",
            duration_s=len(self._audio) / SAMPLE_RATE,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _pump(self, flush: bool) -> int:
        """Hand every full window to the worker. With `flush`, pad the rest to a window too.

        Returns how many windows were handed over.
        """
        plan = self._adapter.plan()
        sent = 0
        if self._worker is None:
            if len(self._audio) < plan.first_samples:
                if not flush:
                    return 0
            first = _padded(self._audio[: plan.first_samples], plan.first_samples)
            self._worker = self._adapter.start_worker(first)
            weakref.finalize(self, self._worker.abort)
            self._mel_done = plan.first_mel_frames
            self._sent_end = plan.first_samples
            sent += 1
        while True:
            start = self._mel_done * plan.hop - plan.n_fft // 2
            end = start + plan.chunk_samples
            if end > len(self._audio):
                break
            self._worker.send(self._audio[start:end])
            self._mel_done += plan.chunk_mel_frames
            self._sent_end = end
            sent += 1
        if flush and len(self._audio) > self._sent_end:
            start = self._mel_done * plan.hop - plan.n_fft // 2
            self._worker.send(_padded(self._audio[start:], plan.chunk_samples))
            self._mel_done += plan.chunk_mel_frames
            self._sent_end = len(self._audio)
            sent += 1
        return sent

    def _partial(self, final: bool) -> PartialResult | None:
        assert self._worker is not None
        text = self._worker.text()
        if text == self._last_text:
            return None
        self._last_text = text
        committed, tentative = committed_and_tentative(text, final)
        return PartialResult(committed, tentative)


def _padded(audio: np.ndarray, length: int) -> np.ndarray:
    """`audio` followed by silence up to `length` samples."""
    out = np.zeros(length, dtype=np.float32)
    out[: len(audio)] = audio[:length]
    return out


class NemotronASR:
    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "auto",  # "auto" | "cuda" | "cpu"
        dtype: str = "float16",
        lookahead_tokens: int = 6,  # chunk = (n + 1) x 80 ms; English 0/1/6/13, 3.5 0/3/6/13
        language: str | None = None,  # the 3.5 checkpoint needs a locale such as "en-US"
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.lookahead_tokens = lookahead_tokens
        self.language = language
        self._model: Any = None
        self._processor: Any = None
        # The model object keeps its streaming state on itself, so a stream or a batch decode
        # must never overlap another one. Taken without waiting: waiting would deadlock the
        # single ASR worker thread, which is also the one that would release it.
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForRNNT, AutoProcessor

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(self.model_id)
            processor.set_num_lookahead_tokens(self.lookahead_tokens)
            model = AutoModelForRNNT.from_pretrained(
                self.model_id, dtype=getattr(torch, self.dtype)
            )
            self._processor, self._model = processor, model.to(device).eval()
            # The first CUDA call is much slower; keep it out of the first real turn's asr_ms.
            self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except Exception as exc:
            self._model = self._processor = None
            raise ASRError(f"failed to load {self.model_id}: {exc}") from exc

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
    ) -> TranscriptResult:
        import torch

        self._require_loaded()
        if not self._lock.acquire(blocking=False):
            raise ASRError(f"{self.name} is busy with another stream")
        try:
            start = time.perf_counter()
            inputs = self._processor(
                audio, sampling_rate=sample_rate, return_tensors="pt", **self._language()
            ).to(self._model.device, dtype=self._model.dtype)
            with torch.inference_mode():
                out = self._model.generate(**inputs, return_dict_in_generate=True)
            text = self._processor.decode(out.sequences, skip_special_tokens=True)[0]
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            raise ASRError(f"{self.name} inference failed: {exc}") from exc
        finally:
            self._lock.release()
        return TranscriptResult(
            text=text.strip(),
            language="en",
            duration_s=len(audio) / sample_rate,
            latency_ms=latency_ms,
        )

    def open_stream(self, sample_rate: int = SAMPLE_RATE, language: str | None = None):
        """A native stream for one turn (StreamingASRModel). Nothing runs until audio arrives."""
        self._require_loaded()
        if sample_rate != SAMPLE_RATE:
            raise ASRError(f"{self.name} streams {SAMPLE_RATE} Hz audio, not {sample_rate}")
        return NemotronStream(self)

    # --- used by NemotronStream ---

    def plan(self) -> ChunkPlan:
        p = self._processor
        return ChunkPlan(
            first_samples=p.num_samples_first_audio_chunk,
            chunk_samples=p.num_samples_per_audio_chunk,
            first_mel_frames=p.num_mel_frames_first_audio_chunk,
            chunk_mel_frames=p.num_mel_frames_per_audio_chunk,
            hop=p.feature_extractor.hop_length,
            n_fft=p.feature_extractor.n_fft,
        )

    def start_worker(self, first_audio: np.ndarray) -> _Worker:
        """Start the generate() thread on the first window; the thread holds the model lock."""
        if not self._lock.acquire(blocking=False):
            raise ASRError(f"{self.name} is busy with another stream")
        try:
            first = self._features(first_audio, first=True)
            first_features = first.input_features[
                :, : self._processor.num_mel_frames_first_audio_chunk, :
            ]
        except Exception as exc:
            self._lock.release()
            raise ASRError(f"{self.name} inference failed: {exc}") from exc
        worker = _Worker(
            self._model,
            self._processor.tokenizer,
            first,
            lambda audio: self._features(audio, first=False).input_features,
            first_features,
            release=self._lock.release,
        )
        worker.start()
        return worker

    def _features(self, audio: np.ndarray, *, first: bool):
        return self._processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=first,
            return_tensors="pt",
            **self._language(),
        ).to(self._model.device, dtype=self._model.dtype)

    def _language(self) -> dict[str, str]:
        return {"language": self.language} if self.language else {}

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ASRError(f"{self.name} is not loaded")
