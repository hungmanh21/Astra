"""Run every registered ASR adapter on one WAV file and print transcript + timing (PLAN T2.5).

Run with:
    uv run python scripts/compare_asr.py tests/fixtures/<clip>.wav
    uv run python scripts/compare_asr.py <clip>.wav --models whisper-large-v3

No browser and no server needed. A model that fails to load or run prints its
error and the script moves on to the next one.
"""

from pathlib import Path

import numpy as np


def load_wav(path: Path) -> np.ndarray:
    """TODO(you): read a WAV file for the ASR.

    Input:  path - a WAV file. The fixture is 16 kHz.
    Output: 1-D np.ndarray, float32, values in [-1, 1], mono, 16 kHz.
    Raises: ValueError if the sample rate is not 16 kHz (don't resample silently).

    Steps:
      1. soundfile.read(path, dtype="float32") returns (data, sample_rate).
      2. If data is 2-D (stereo), average the channels.
      3. Check the sample rate.
    """
    raise NotImplementedError


def load_registry():
    """TODO(you): build the ASRRegistry from config.yaml.

    Input:  none (reads config.yaml in the repo root).
    Output: ASRRegistry.

    Until the config loader (PLAN T1.2) exists: parse config.yaml with PyYAML here and
    build an ASRModelConfig per entry under `asr.models`. PyYAML is not a direct
    dependency yet; `uv add pyyaml` when you get here.
    """
    raise NotImplementedError


def main() -> None:
    """TODO(you): the command-line entry point.

    Input:  command line: a positional WAV path and an optional --models list
            (default: every registry key).
    Output: None. Prints to stdout:
              - the audio length once,
              - one row per model: name, load time, asr_ms, transcript.
    Raises: nothing per model; catch errors around each model so one failure
            (for example a failed download) does not stop the others.

    Steps:
      1. argparse the arguments.
      2. audio = load_wav(path); registry = load_registry().
      3. For each model: model, load_ms = registry.get(name); result = model.transcribe(audio).
         Print the load time separately from asr_ms (SPEC 6.3.2).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
