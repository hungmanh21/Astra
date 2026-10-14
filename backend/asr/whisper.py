"""Whisper adapter on Hugging Face transformers (PLAN T2.3, PLAN_v1 S2.5).

Model: openai/whisper-large-v3. English only. In fp16 it needs roughly 3-4 GB, so it fits on
the 12 GB RTX 3060. Besides plain `transcribe`, the adapter can return timestamped segments
(`transcribe_segments`), which makes it a SegmentedASRModel and unlocks the `trimmed`
strategy.
"""

import time
from typing import Any

import numpy as np

from backend.asr.base import ASRError, Segment, TranscriptResult

MAX_PROMPT_TOKENS = 224  # Whisper reads at most half of its 448-token context as prompt


def segments_from_chunks(chunks: list[dict[str, Any]], duration_s: float) -> list[Segment]:
    """Turn the pipeline's `chunks` into Segments.

    Input:  chunks - what `pipe(..., return_timestamps=True)["chunks"]` holds, e.g.
            [{"text": " He has grave doubts", "timestamp": (0.0, 3.2)}, ...]. The pair is
            (start, end) in seconds from the start of the audio; the last end can be None
            when the audio ended mid-segment. A pair may be a tuple or a list.
            duration_s - length of the audio, in seconds.
    Output: Segments in the same order. Text is stripped. A chunk with no text is dropped.
            A missing end becomes `duration_s`, a missing start becomes the previous
            segment's end (0.0 for the first). Every time is clamped into [0, duration_s],
            and a segment never ends before it starts.
    """
    segments: list[Segment] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        start, end = chunk.get("timestamp", (None, None))
        if start is None:
            start = segments[-1].end_s if segments else 0.0
        if end is None:
            end = duration_s

        # Clamp the times into [0, duration_s] and keep start <= end.
        start = max(0.0, min(float(start), duration_s))
        end = max(start, min(float(end), duration_s))
        segments.append(Segment(text=text, start_s=start, end_s=end))

    return segments


class WhisperASR:
    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "auto",  # "auto" | "cuda" | "cpu"
        dtype: str = "float16",
    ) -> None:
        import torch

        self.name = name
        self.model_id = model_id
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # fp16 is not supported on CPU.
        self.dtype = "float32" if self.device == "cpu" else dtype
        self._pipe = None

    def load(self) -> None:
        try:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                dtype=getattr(torch, self.dtype),
            )
            # The first CUDA call is much slower; keep it out of the first real turn's asr_ms.
            self._run(np.zeros(16000, dtype=np.float32), 16000)
        except Exception as exc:
            self._pipe = None
            raise ASRError(f"failed to load {self.model_id}: {exc}") from exc

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptResult:
        if self._pipe is None:
            raise ASRError(f"{self.name} is not loaded")
        try:
            start = time.perf_counter()
            out = self._run(audio, sample_rate)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            raise ASRError(f"{self.name} inference failed: {exc}") from exc
        return TranscriptResult(
            text=out["text"].strip(),
            language="en",
            duration_s=len(audio) / sample_rate,
            latency_ms=latency_ms,
        )

    def transcribe_segments(
        self, audio: np.ndarray, sample_rate: int = 16000, prompt: str | None = None
    ) -> tuple[list[Segment], float]:
        """Transcribe `audio` into timestamped segments (SegmentedASRModel, SPEC_v1 6.4).

        Input:  audio, sample_rate as in `transcribe`. prompt - text the model should
                continue from (the committed text), or None. A blank prompt counts as None.
        Output: (segments, latency_ms). Times are seconds from the start of `audio`.
                latency_ms covers the pipeline call only, like `transcribe`. Silence can
                give no segments at all.
        Raises: ASRError("<name> is not loaded") before `load()`; ASRError("<name> inference
                failed: ...") when the pipeline raises (same wording as `transcribe`).
        """
        if self._pipe is None:
            raise ASRError(f"{self.name} is not loaded")
        try:
            start = time.perf_counter()
            out = self._run(audio, sample_rate, timestamps=True, prompt=prompt)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            raise ASRError(f"{self.name} inference failed: {exc}") from exc
        return segments_from_chunks(out["chunks"], len(audio) / sample_rate), latency_ms

    def _run(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        timestamps: bool = False,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        import torch

        generate_kwargs: dict[str, Any] = {"language": "en", "task": "transcribe"}
        if prompt and prompt.strip():
            generate_kwargs["prompt_ids"] = self._prompt_ids(prompt.strip())
        # return_timestamps is only passed when asked for, so plain transcribe is unchanged.
        extra = {"return_timestamps": True} if timestamps else {}
        with torch.inference_mode():
            return self._pipe(
                {"raw": audio, "sampling_rate": sample_rate},
                generate_kwargs=generate_kwargs,
                **extra,
            )

    def _prompt_ids(self, text: str):
        """Token ids for `generate_kwargs["prompt_ids"]`, cut to Whisper's prompt limit."""
        import torch

        ids = self._pipe.tokenizer.get_prompt_ids(text, return_tensors="pt")
        if len(ids) > MAX_PROMPT_TOKENS:
            # Keep the leading <|startofprev|> token and the newest words.
            ids = torch.cat([ids[:1], ids[-(MAX_PROMPT_TOKENS - 1) :]])
        return ids.to(self.device)
