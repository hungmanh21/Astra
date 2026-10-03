"""Stream one reply from the configured LLM, or list the models the API key can use (PLAN T3.2).

Run with:
    uv run python scripts/try_llm.py                       # config.yaml model, default prompt
    uv run python scripts/try_llm.py "Say hi"               # your own prompt
    uv run python scripts/try_llm.py --model gemini/gemini-2.5-flash
    uv run python scripts/try_llm.py --list-models          # Gemini model ids for your key

Uses the same `load_settings` and `LLMClient` as the server, so a pass here means the model
name, the key and LiteLLM work together.
"""

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from backend.config import load_settings  # noqa: E402
from backend.llm import LLMClient, LLMError  # noqa: E402

DEFAULT_PROMPT = "Say hello in one short sentence."


def list_models() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 1
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
    request = urllib.request.Request(url, headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            models = json.load(response).get("models", [])
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: the API rejected the request (is the key valid?)", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"network error: {exc.reason}", file=sys.stderr)
        return 1

    print("Use these with a `gemini/` prefix in config.yaml (llm.model):")
    for model in models:
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(f"  gemini/{model['name'].removeprefix('models/')}")
    return 0


async def stream_reply(model: str | None, prompt: str) -> int:
    settings = load_settings()
    llm_settings = replace(settings.llm, model=model) if model else settings.llm
    client = LLMClient(llm_settings)
    messages = [
        {"role": "system", "content": llm_settings.system_prompt},
        {"role": "user", "content": prompt},
    ]

    print(f"model: {llm_settings.model}\nprompt: {prompt}\n")
    start = time.perf_counter()
    first: float | None = None
    try:
        async for delta in client.stream(messages):
            if first is None:
                first = time.perf_counter()
            print(delta, end="", flush=True)
    except LLMError as exc:
        print(f"\n\nFAILED: {exc}", file=sys.stderr)
        print(f"cause: {exc.__cause__!r}", file=sys.stderr)
        return 1

    total = time.perf_counter() - start
    ttft = "n/a (no text came back)" if first is None else f"{(first - start) * 1000:.0f} ms"
    print(f"\n\nfirst token: {ttft}, total: {total * 1000:.0f} ms")
    return 0 if first is not None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", help="LiteLLM model name, overrides config.yaml")
    parser.add_argument("--list-models", action="store_true", help="list Gemini models for the key")
    args = parser.parse_args()

    if args.list_models:
        from dotenv import load_dotenv

        load_dotenv()
        return list_models()
    return asyncio.run(stream_reply(args.model, args.prompt))


if __name__ == "__main__":
    sys.exit(main())
