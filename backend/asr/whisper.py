"""Whisper adapter on Hugging Face transformers (PLAN T2.3).

Model: openai/whisper-large-v3. English only in v0. In fp16 it needs roughly
3-4 GB, so it fits on the 12 GB RTX 3060.
"""

from typing import Any

import numpy as np

from backend.asr.base import TranscriptResult


class WhisperASR:
    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "auto",  # "auto" | "cuda" | "cpu"
        dtype: str = "float16",
    ) -> None:
        """TODO(you): store the settings only. No heavy imports, no weight loading.

        Input:  name - registry key (also the `name` attribute of the ASRModel protocol).
                model_id - Hugging Face id, "openai/whisper-large-v3".
                device - "auto", "cuda" or "cpu".
                dtype - "float16" or "float32" (config value, as a string).
        Output: None.
        State to keep: name, model_id, resolved device, resolved dtype, and a
                       `self._pipe = None` that load() fills in.

        Steps:
          1. "auto" -> "cuda" if torch.cuda.is_available() else "cpu".
          2. fp16 is not supported on CPU, so use float32 there.
        """
        raise NotImplementedError

    def load(self) -> None:
        """TODO(you): build the transformers ASR pipeline and warm it up.

        Input:  none.
        Output: None. Afterwards self._pipe is ready.
        Raises: ASRError if the download or the load fails.

        Steps:
          1. Import torch/transformers inside this method so importing the module is cheap.
          2. transformers.pipeline("automatic-speech-recognition", model=self.model_id,
             device=..., dtype=...). Recent transformers versions use `dtype=`;
             `torch_dtype=` is the deprecated spelling. Check the version you have.
          3. Run one dummy transcription (about 1 s of silence, np.zeros(16000, np.float32)).
             The first CUDA call is much slower than later ones and would otherwise be
             counted in the first real turn's asr_ms.
        """
        raise NotImplementedError

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptResult:
        """TODO(you): transcribe one whole turn.

        Input:  audio - 1-D float32 array in [-1, 1], mono, up to 30 s.
                sample_rate - 16000.
                language - ignored in v0; always force English.
        Output: TranscriptResult(text=..., language="en", duration_s=len(audio)/sample_rate,
                latency_ms=<inference time only>).
        Raises: ASRError if the model was not loaded or inference fails (don't let a bare
                RuntimeError or CUDA error escape).

        Steps:
          1. Start time.perf_counter() just before inference, stop right after (SPEC 6.3.2).
          2. The pipeline takes {"raw": audio, "sampling_rate": sample_rate}.
          3. Force English: generate_kwargs={"language": "en", "task": "transcribe"}.
          4. Wrap inference in torch.inference_mode().
          5. Strip whitespace from the text.
          6. 30 s is Whisper's window and the turn cap, so no chunking is needed.

        Gotcha: Whisper tends to invent text on silence or noise (for example "Thank you.").
        Don't fix it yet, but write down what you see for the M2 notes in PLAN.md.
        """
        raise NotImplementedError

    def _run(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """TODO(you), optional: the raw pipeline call, shared by warm-up and transcribe().

        Input:  audio - float32 array; sample_rate - int.
        Output: whatever the pipeline returns, a dict like {"text": "..."}.
        """
        raise NotImplementedError
