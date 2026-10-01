"""Run every registered ASR adapter on one WAV file and print transcript + timing (PLAN T2.5).

Run with:
    uv run python scripts/compare_asr.py tests/fixtures/<clip>.wav
    uv run python scripts/compare_asr.py <clip>.wav --models whisper-large-v3

No browser and no server needed. A model that fails to load or run prints its
error and the script moves on to the next one.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.asr.registry import ASRModelConfig, ASRRegistry  # noqa: E402

SAMPLE_RATE = 16_000


def load_wav(path: Path) -> np.ndarray:
    """Read a 16 kHz WAV as mono float32. Does not resample."""
    data, sample_rate = sf.read(path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz, expected {SAMPLE_RATE} Hz")
    return data


def load_registry() -> ASRRegistry:
    # Replace with the config loader once PLAN T1.2 exists.
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    models = {
        name: ASRModelConfig(
            adapter=entry["adapter"],
            model_id=entry["model_id"],
            options=entry.get("options") or {},
        )
        for name, entry in config["asr"]["models"].items()
    }
    return ASRRegistry(models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--models", nargs="+", help="registry keys (default: all)")
    args = parser.parse_args()

    audio = load_wav(args.wav)
    registry = load_registry()
    print(f"audio: {len(audio) / SAMPLE_RATE:.2f} s ({args.wav})")

    for name in args.models or registry.available():
        try:
            model, load_ms = registry.get(name)
            result = model.transcribe(audio)
        except Exception as exc:
            print(f"{name}: ERROR {exc}")
            continue
        print(f"{name}: load {load_ms:.0f} ms | asr {result.latency_ms:.0f} ms | {result.text}")


if __name__ == "__main__":
    main()
