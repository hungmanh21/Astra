"""The Nemotron adapter with the real checkpoint (PLAN_v1 S4.4).

Skipped when there is no GPU or the weights are not in the Hugging Face cache (run
`scripts/compare_asr.py` once to download them). Takes a few seconds with the weights cached.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.asr.bench import word_error_rate

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
FIXTURE = Path(__file__).parent / "fixtures" / "librispeech_1272-128104-0003"


def _weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(MODEL_ID, "model.safetensors"), str)
    except Exception:
        return False


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_has_gpu() and _weights_cached()), reason="needs a GPU and the Nemotron weights"
)


@pytest.fixture(scope="module", params=[1, 6, 13])
def adapter(request):
    from backend.asr.nemotron import NemotronASR

    model = NemotronASR("nemotron", MODEL_ID, lookahead_tokens=request.param)
    model.load()
    return model


@pytest.fixture(scope="module")
def clip() -> tuple[np.ndarray, str]:
    audio, rate = sf.read(f"{FIXTURE}.wav", dtype="float32")
    assert rate == 16000
    return audio, Path(f"{FIXTURE}.txt").read_text().strip()


def stream_text(adapter, audio: np.ndarray) -> tuple[str, list]:
    stream = adapter.open_stream()
    partials = []
    for i in range(0, len(audio), 1600):  # 100 ms, like the session's coalesced feeds
        partial = stream.feed(audio[i : i + 1600])
        if partial is not None:
            partials.append(partial)
    return stream.finish().text, partials


def test_batch_text_matches_the_reference(adapter, clip) -> None:
    audio, reference = clip
    assert word_error_rate(reference, adapter.transcribe(audio).text) < 0.1


def test_streamed_text_is_complete_and_close_to_the_reference(adapter, clip) -> None:
    audio, reference = clip
    text, partials = stream_text(adapter, audio)
    assert word_error_rate(reference, text) < 0.15  # the last words must not be dropped
    assert len(partials) >= 5  # words came while the clip was playing (13: 8 chunks of 1.1 s)


def test_committed_text_only_grows(adapter, clip) -> None:
    _, partials = stream_text(adapter, clip[0])
    previous = ""
    for partial in partials:
        assert partial.committed.startswith(previous)
        previous = partial.committed


def test_two_streams_in_a_row_give_the_same_text(adapter, clip) -> None:
    first, _ = stream_text(adapter, clip[0])
    second, _ = stream_text(adapter, clip[0])
    assert first == second


def test_batch_still_works_after_streaming(adapter, clip) -> None:
    audio, reference = clip
    stream_text(adapter, audio)
    assert word_error_rate(reference, adapter.transcribe(audio).text) < 0.1


def test_the_last_word_survives_a_turn_that_ends_at_the_last_sound(adapter, clip) -> None:
    """Released right after the last word: no trailing silence in the audio at all."""
    audio, _ = clip
    window = 320  # 20 ms
    quiet = 0.02 * np.abs(audio).max()
    loud = [
        i
        for i in range(0, len(audio) - window, window)
        if np.abs(audio[i : i + window]).max() > quiet
    ]
    cut = audio[: loud[-1] + window]
    assert len(audio) - len(cut) > 1600  # the fixture does have silence after the word
    text, _ = stream_text(adapter, cut)
    assert text.lower().rstrip(".").endswith("ithaca")
    assert adapter.transcribe(cut).text.lower().rstrip(".").endswith("ithaca")
