# Astra

Real-time speech recognition (and, later, translation) with open-source Hugging Face models,
aimed at live conversations, meetings and calls.

The long-term vision is streaming ASR with translation. What exists today is **v0**, a stepping
stone: a turn-based voice agent. You hold a button and speak, the audio goes to the backend over
a WebSocket, a pluggable ASR model transcribes it, an LLM answers, and the reply streams into a
chat page. v0 is English only, one turn at a time, and not real-time on purpose. It builds the
seams (transport, ASR interface, LLM interface) that streaming will reuse. See `SPEC_v0.md` for
what v0 is and is not.

```
browser (push-to-talk) -> WebSocket -> ASR (Whisper) -> LiteLLM (Gemini or vLLM) -> chat UI
```

## Requirements

- Linux, or WSL2 on Windows, with one NVIDIA GPU. Developed on an RTX 3060 (12 GB), which fits
  Whisper large-v3 in fp16.
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

## Run

```bash
make run                 # http://localhost:8000
```

Open <http://localhost:8000>, allow the microphone, then hold the mic button while you speak.

Models load lazily. The **first turn is slow** because it imports the libraries and loads the
Whisper weights (about two minutes on a slow disk, plus a download of about 3 GB the very first
time). Later turns are fast. On the dev machine, a 4.6 s utterance on warm Whisper large-v3 took
about 1.2 s to transcribe, with the first LLM token 0.85 s later and 2.3 s from end of speech to
the last token.

The UI shows five timings under each turn: audio length, ASR, first token, LLM total and
end to end. The server writes one JSON line per turn to its log (`astra.turns`: session, turn,
model names, status, timings). It never logs the transcript, the reply or any audio.

**Review mode** (the "Review transcript" checkbox) shows the transcript first and lets you edit it or discard
it before it goes to the LLM. Time spent editing is not counted in the end-to-end timing.

## Configuration

`config.yaml`:

| Section | What it sets |
|---|---|
| `asr.models` | The ASR registry: one entry per model (`adapter`, `model_id`, `options`) |
| `asr.default_model` | The model selected in the UI at start |
| `llm` | LiteLLM `model`, optional `api_base`, `system_prompt`, `max_history_tokens` |
| `limits.max_turn_seconds` | Longest accepted recording (30 s) |

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

Available today: `whisper-large-v3` and `whisper-large-v3-turbo`. Parakeet and Nemotron adapters (NeMo) are planned but not
written yet.

## Compare ASR models on a clip

Runs every registered model on one 16 kHz WAV and prints the transcript and timing. No server
and no browser needed.

```bash
uv run python scripts/compare_asr.py tests/fixtures/<clip>.wav
uv run python scripts/compare_asr.py <clip>.wav --models whisper-large-v3
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
| `backend/` | FastAPI app (`main.py`), turn state machine (`session.py`), transport seam, ASR adapters and registry, LLM wrapper, history trimming, per-turn log |
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
- **No microphone prompt.** Open the page through `http://localhost:8000`, not an IP address.
