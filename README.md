# Astra

Real-time speech recognition (and, later, translation) with open-source Hugging Face models,
aimed at live conversations, meetings and calls.

The long-term vision is streaming ASR with translation. What exists today is a stepping stone:
a push-to-talk voice agent. You hold a button and speak, the audio goes to the backend over a
WebSocket, a pluggable ASR model transcribes it, an LLM answers, and the reply streams into a
chat page. It is English only and handles one turn at a time.

- **v0** (`SPEC_v0.md`) is turn-based: the transcript appears after you let go.
- **v1** (`SPEC_v1.md`) adds **live partial transcripts** while you are still speaking, and a
  bench for comparing four streaming strategies on speed and accuracy (see
  [Live partials and streaming strategies](#live-partials-and-streaming-strategies)).

```
browser (push-to-talk) -> WebSocket -> ASR (Whisper or Nemotron) -> LiteLLM (Gemini or vLLM) -> chat UI
                                          \-> partial transcripts back to the page while you speak
```

## Requirements

- Linux, or WSL2 on Windows, with one NVIDIA GPU. Developed on an RTX 3060 (12 GB). Whisper
  large-v3 (fp16), Whisper large-v3-turbo and Nemotron (fp16) together use about 10 GB, so all
  three can stay loaded.
- [uv](https://docs.astral.sh/uv/) and Python 3.13 (uv installs it from `.python-version`).
- A Chrome-based browser on the same machine. The microphone only works on `localhost` or HTTPS.
- A Gemini API key for the default LLM (free at <https://aistudio.google.com/apikey>), or a local
  vLLM server (see below).

## Setup

```bash
uv sync                  # installs the dependencies (torch alone is several GB)
cp .env.example .env     # then put your GEMINI_API_KEY in .env
```

`.env` holds secrets and is git-ignored. `config.yaml` holds model names and defaults.

**WSL2: keep the venv off `/mnt/d`.** Installing thousands of small files on a Windows drive is
very slow. The Makefile sets `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/astra`, so `make` targets use
a venv on the Linux filesystem. For plain `uv` commands, export it once in your shell profile:

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/astra
```

Then `uv sync` and `uv run` use the same env. (`rm -rf .venv` once the new env works.) No NeMo
install is needed: Nemotron runs through `transformers`.

## Run

```bash
make run                 # http://localhost:8000
```

Open <http://localhost:8000>, allow the microphone, then hold the mic button while you speak.

Models load lazily, when a turn first uses them, and stay loaded. The **first turn with a model
is slow** because it imports the libraries and loads the weights (about two minutes on a slow
disk, plus a download of 1 to 3 GB the very first time). Later turns are fast. On the dev machine, a 4.6 s utterance on warm Whisper large-v3 took
about 1.2 s to transcribe, with the first LLM token 0.85 s later and 2.3 s from end of speech to
the last token.

While you hold the button, the transcript grows in your bubble: settled words in normal text,
the newest words dimmed because they may still change. When you let go, the final transcript
replaces them. Next to the model dropdown, a **Streaming** dropdown picks the strategy (below);
it lists what the chosen model supports and remembers your choice per model.

The UI shows these timings under each turn: audio length, the strategy with its partial count and
median lag, the final transcript wait (`final`), ASR, first token, LLM total and end to end. The server writes one JSON line per turn to its log (`astra.turns`: session, turn,
model names, status, timings). It never logs the transcript, the reply or any audio.

**Review mode** (the "Review transcript" checkbox) shows the transcript first and lets you edit it or discard
it before it goes to the LLM. Time spent editing is not counted in the end-to-end timing.

## Configuration

`config.yaml`:

| Section | What it sets |
|---|---|
| `asr.models` | The ASR registry: one entry per model (`adapter`, `model_id`, `options`) |
| `asr.default_model` | The model selected in the UI at start |
| `asr.streaming` | Live partials: `default_strategy`, `partial_interval_ms`, `partial_min_audio_ms`, `chunk_ms` (see below) |
| `llm` | LiteLLM `model`, optional `api_base`, `system_prompt`, `max_history_tokens` |
| `limits.max_turn_seconds` | Longest accepted recording (30 s) |
| `limits.release_tail_ms` | How long the UI keeps recording after you let go (400 ms), so the end of the last word is not cut off. 0 switches it off |

Environment variables (in `.env` or the shell):

| Variable | Effect |
|---|---|
| `GEMINI_API_KEY` | Key for `gemini/...` models. LiteLLM reads provider keys from the environment |
| `LLM_MODEL` | Overrides `llm.model` |
| `LLM_API_BASE` | Overrides `llm.api_base` |
| `LLM_API_KEY` | Optional key for an endpoint that wants one (sent as a bearer token) |
| `DEBUG_SAVE_AUDIO=1` | Saves every received turn as a WAV in `debug_audio/` (off by default, for checking what the browser sent) |

Example: `make run DEBUG_SAVE_AUDIO=1`.

### Switching the LLM

All LLM calls go through [LiteLLM](https://docs.litellm.ai/), so the backend changes by config
only.

- **Gemini (default):** `llm.model: gemini/gemini-3.5-flash-lite` plus `GEMINI_API_KEY`.
- **vLLM (or any OpenAI-compatible server):** vLLM runs wherever you started it; only its URL
  matters. Set `LLM_MODEL=hosted_vllm/auto` and `LLM_API_BASE=http://<host>:<port>/v1`. With
  `auto` the app asks the endpoint (`GET /models`) which model it serves on the first request and
  uses the first one it lists, so the config never names it. To pin a model instead, write its
  id: `LLM_MODEL=hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct`. If the server was started with an API
  key, set `LLM_API_KEY`. Not yet tried against a real server (PLAN T5.4).

To check a model name and key before starting the server:

```bash
uv run python scripts/try_llm.py "Say hello in five words"
uv run python scripts/try_llm.py --list-models    # Gemini models your key can use
```

### Adding an ASR model

One adapter file plus one config entry. Write a class in `backend/asr/` that follows the
`ASRModel` protocol in `backend/asr/base.py` (`load()` and `transcribe(audio, ...)`), then add it under
`asr.models` in `config.yaml` with its `adapter` path, `model_id` and `options`. The session code
never special-cases a model.

Available today: `whisper-large-v3`, `whisper-large-v3-turbo` and `nemotron-streaming-en-0.6b`
(NVIDIA's cache-aware streaming model, English, run through `transformers` in fp16, about 1.2 GB
on the GPU). A Parakeet adapter is planned.

An adapter also unlocks strategies by what it can do, never by its name: any model gets `chunked`
and `redecode`; a model with `transcribe_segments` (timestamped segments, Whisper) also gets
`trimmed`; a model with `open_stream` (Nemotron) also gets `native`. Nemotron's chunk size is the
config option `lookahead_tokens` (English checkpoint: 1, 6 or 13, chunk = (n + 1) x 80 ms; 6 is the
default). The multilingual `nvidia/nemotron-3.5-asr-streaming-0.6b` works with the same adapter
(`lookahead_tokens` 3, 6 or 13 and `language: en-US`).

## Live partials and streaming strategies

While you speak, the server sends `partial` messages: `committed` text that will not change
and `tentative` text that may. The final transcript, sent after you let go, is what goes to the
LLM. How the partials are made is the **strategy**, chosen in the UI per turn:

| Strategy | What it does | Use it when |
|---|---|---|
| `native` | The model streams itself (Nemotron): each chunk is processed once, carrying the model's own cache. Lowest lag, flat `final` wait | The model supports it. It is Nemotron's default |
| `redecode` | Re-decodes the whole turn on every update and commits the words two decodes in a row agree on. Same accuracy as a plain decode, but lag and `final` grow with the turn | Any model. It is the default for Whisper (use turbo) |
| `trimmed` | Like `redecode`, but only decodes the audio after the last finished sentence, with the settled text as the prompt. Shorter decodes | Whisper turbo, to cut the `final` wait; opt-in, not reliable on large-v3 |
| `chunked` | Decodes fixed 2 s chunks one by one, each once. Cheap, but it cuts words at chunk edges | A baseline for comparison; worst accuracy on Whisper |

On the dev machine (RTX 3060), a 28-30 s turn gives: Nemotron `native` lag about 100 ms and `final` 83-121 ms;
turbo `redecode` lag 260-490 ms and `final` 0.5-1.1 s; large-v3 `redecode` lag 0.6-1.4 s and `final` 3-5 s
(too slow for live use). The full comparison is in `SPEC_v1.md` section 8.

`asr.streaming` in `config.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `default_strategy` | Strategy for models without `native` | `redecode` |
| `partial_interval_ms` | Least new audio between two partial decodes (`redecode`, `trimmed`) | 500 |
| `partial_min_audio_ms` | No partial before this much audio (Whisper invents text on tiny clips) | 1000 |
| `chunk_ms` | Chunk length of `chunked` (500 and 1000 garble Whisper) | 2000 |

The microphone keeps recording `limits.release_tail_ms` (400 ms) after you let go, and Nemotron gets
0.5 s of silence after your last sound; without them the last word was sometimes cut off.

A failed partial decode on a Whisper strategy never fails the turn: partials stop and the
final transcript still arrives. With `native`, a failed chunk ends the model's stream, so that turn
ends with an ASR error. Only one native stream (or Nemotron batch decode) runs at a time per model; a second
session streaming at once gets an error instead of waiting.

## Compare ASR models on a clip

Runs every registered model on one 16 kHz WAV and prints the transcript and timing. No server
and no browser needed.

```bash
uv run python scripts/compare_asr.py tests/fixtures/<clip>.wav
uv run python scripts/compare_asr.py <clip>.wav --models whisper-large-v3
```

With `--stream` the clip is played into each strategy at real-time pace. It prints every partial
with its time and lag, then a table per model and strategy: partial count, time of the first
partial, median lag, `final_ms` and word error rate. The WER needs the reference text in a `.txt`
file next to the WAV (the fixture has one).

```bash
uv run python scripts/compare_asr.py tests/fixtures/<clip>.wav --stream --strategy all
uv run python scripts/compare_asr.py <clip>.wav --stream --models nemotron-streaming-en-0.6b --strategy native
uv run python scripts/compare_asr.py <clip>.wav --stream --strategy chunked --chunk-ms 1000 --interval-ms 250
```

## Development

```bash
make check               # Ruff lint, Ruff format check, then pytest
make fmt                 # auto-fix lint issues and format
```

Tests use fake ASR and LLM, so they need no GPU, no model downloads and no API key.

## Layout

| Path | What is there |
|---|---|
| `backend/` | FastAPI app (`main.py`), turn state machine (`session.py`), transport seam, ASR adapters (`whisper.py`, `nemotron.py`) and registry, streaming (`streaming.py`, `strategies.py`, `agreement.py`), bench (`bench.py`), LLM wrapper, history trimming, per-turn log |
| `frontend/` | Plain HTML and JS chat page, push-to-talk recorder (AudioWorklet to 16 kHz PCM16) |
| `scripts/` | `compare_asr.py`, `try_llm.py` |
| `tests/` | pytest suite and a small LibriSpeech fixture clip (CC BY 4.0) |
| `theory/` | Learning notes on audio and ASR fundamentals. Not part of the app |

Project documents: `SPEC_v0.md` / `SPEC_v1.md` (requirements and acceptance), `PLAN_v0.md` / `PLAN_v1.md` (tasks), `DECISIONS.md`
(design decisions and why), `PROGRESS.md` (current state).

## Troubleshooting

- **The first turn takes minutes.** Expected: lazy imports plus the Whisper load and, once, the
  download. A repo on a slow disk (such as `/mnt/d` under WSL2) makes it worse.
- **"LLM request failed" in the chat.** Check `GEMINI_API_KEY` and the model name with
  `scripts/try_llm.py`. The real cause is in the server log; the browser only gets a generic message.
- **No partials, or a model says "busy".** A native stream or Nemotron batch decode is already
  running on that model (another tab or session). Wait for it to finish, or pick another strategy.
- **Nemotron `lookahead_tokens: 0` hangs.** Use 1, 6 or 13 (3 on the multilingual checkpoint).
- **No microphone prompt.** Open the page through `http://localhost:8000`, not an IP address.
