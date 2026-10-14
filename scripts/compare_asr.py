"""Run every registered ASR adapter on one WAV file and print transcript + timing (PLAN T2.5).

Run with:
    uv run python scripts/compare_asr.py tests/fixtures/<clip>.wav
    uv run python scripts/compare_asr.py <clip>.wav --models whisper-large-v3

With --stream the clip is played into each streaming strategy at real-time pace (PLAN_v1
S2.1): it prints every partial with its time, then a table of partial count, median lag,
final_ms and word error rate per model and strategy. The WER needs the reference text in
a .txt file next to the WAV.
    uv run python scripts/compare_asr.py <clip>.wav --stream --models whisper-large-v3-turbo
    uv run python scripts/compare_asr.py <clip>.wav --stream --strategy chunked --chunk-ms 1000

No browser and no server needed. A model that fails to load or run prints its
error and the script moves on to the next one.
"""

import argparse
import asyncio
import dataclasses
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.asr.base import STRATEGIES  # noqa: E402
from backend.asr.bench import (  # noqa: E402
    BenchRow,
    format_partial,
    format_table,
    load_reference,
    make_batch_row,
    make_row,
    replay,
)
from backend.asr.registry import ASRRegistry  # noqa: E402
from backend.asr.service import TranscriptionService  # noqa: E402
from backend.config import load_settings  # noqa: E402

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
    return ASRRegistry(load_settings().asr_models)


async def stream_model(
    service: TranscriptionService,
    registry: ASRRegistry,
    name: str,
    audio: np.ndarray,
    reference: str | None,
    strategy: str,
    interval_ms: int,
) -> list[BenchRow]:
    """A batch baseline, then each wanted strategy of one model. Errors print, never raise."""
    supported = service.strategies(name).supported
    wanted = supported if strategy == "all" else [strategy]
    rows: list[BenchRow] = []

    try:
        # The baseline goes first: it loads the model and absorbs the slow first inference,
        # so no strategy pays for the warm-up.
        model, load_ms = await service.run(registry.get, name)
        final = await service.run(model.transcribe, audio)
    except Exception as exc:
        print(f"{name}: ERROR {exc}")
        return rows
    print(f"\n{name}: load {load_ms:.0f} ms | batch asr {final.latency_ms:.0f} ms | {final.text}")
    rows.append(make_batch_row(name, final, reference))

    for chosen in wanted:
        if chosen not in supported:
            print(f"{name} / {chosen}: skipped, supports {', '.join(supported)}")
            continue
        try:
            opened = await service.open_stream(name, chosen)
            result = await replay(opened.stream, service.run, audio, interval_ms=interval_ms)
        except Exception as exc:
            print(f"{name} / {chosen}: ERROR {exc}")
            continue
        print(f"\n{name} / {chosen}: {len(result.partials)} partials")
        for event in result.partials:
            print("  " + format_partial(event))
        print(f"  final ({result.final_ms:.0f} ms after the last frame): {result.final.text}")
        rows.append(make_row(name, chosen, result, reference))
    return rows


async def stream_main(args: argparse.Namespace, audio: np.ndarray, registry: ASRRegistry) -> None:
    streaming = load_settings().asr_streaming
    if args.chunk_ms:
        streaming = dataclasses.replace(streaming, chunk_ms=args.chunk_ms)
    interval_ms = args.interval_ms or streaming.partial_interval_ms
    service = TranscriptionService(registry, streaming=streaming)
    reference = load_reference(args.wav)
    print(f"partial interval: {interval_ms} ms | reference: {'yes' if reference else 'none'}")

    rows: list[BenchRow] = []
    try:
        for name in args.models or registry.available():
            rows += await stream_model(
                service, registry, name, audio, reference, args.strategy, interval_ms
            )
    finally:
        service.shutdown()
    if rows:
        print("\n" + format_table(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--models", nargs="+", help="registry keys (default: all)")
    parser.add_argument("--stream", action="store_true", help="play the clip into each strategy")
    parser.add_argument(
        "--strategy",
        choices=["all", *STRATEGIES],
        default="all",
        help="with --stream: one strategy, or all the model supports (default)",
    )
    parser.add_argument(
        "--interval-ms", type=int, help="with --stream: override asr.streaming.partial_interval_ms"
    )
    parser.add_argument(
        "--chunk-ms", type=int, help="with --stream: override asr.streaming.chunk_ms (chunked)"
    )
    args = parser.parse_args()

    audio = load_wav(args.wav)
    registry = load_registry()
    print(f"audio: {len(audio) / SAMPLE_RATE:.2f} s ({args.wav})")

    if args.stream:
        asyncio.run(stream_main(args, audio, registry))
        return

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
