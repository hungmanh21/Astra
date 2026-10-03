# Voice Agent v0: Offline Transcript Pipeline

**Status:** Draft
**Author:** hhm
**Last updated:** 2026-09-30

## 1. Overview

v0 of a voice agent that works in **turns, not streams**. The user records a spoken message in a simple chat UI, the audio is sent to the backend over a WebSocket, transcribed by a pluggable ASR model, and the transcript is sent to an LLM. The LLM's reply is shown in the chat.

v0 is deliberately *not* real-time. Its purpose is to get the end-to-end loop working and to build the seams (transport, ASR interface, LLM interface) that v1 will reuse for true streaming.

**Relation to the project vision:** the long-term aim of this repo (see `README.md`) is real-time ASR, including translation, for live conversations, meetings and calls, using open-source models. v0 is a stepping stone toward that. Translation and meeting/call use cases are out of scope here (see Non-Goals), and the README will be updated to match once v0 lands.

## 2. Goals

- End-to-end loop: **record → transcribe → LLM → reply in chat**.
- Transport layer built by us (WebSocket in v0; WebRTC is a later swap-in, not a rewrite).
- Transcription pipeline built by us, with **swappable ASR backends** behind one interface.
- LLM access through **LiteLLM** so we can switch between Google (Gemini API) and a local vLLM server by changing config only.
- Per-stage latency measurement, so we know where time goes before optimizing in v1.

## 3. Non-Goals (v0)

- Real-time / streaming ASR partial results.
- Text-to-speech (reply is text only).
- Voice activity detection, barge-in / interruption handling, turn-taking logic.
- Auth, multi-user accounts, persistent history across sessions.
- Production deployment, horizontal scaling.
- Mobile-native clients.
- Multilingual support (v0 is English only).
- Translation, and meeting/call scenarios (multi-speaker audio, diarization).
- Concurrent or queued turns within a session (one turn at a time).
- Recordings longer than 30 s (no chunking / long-form decoding).
- Remote/network deployment (browser and backend run on the same machine, `localhost`).

## 4. Users & Use Cases

| User | Need | Example scenario |
|------|------|------------------|
| Developer (us) | Compare ASR models on the same audio | Record a sentence, switch model from the dropdown, record again, compare transcript and latency |
| Developer (us) | Swap the LLM backend | Change `LLM_MODEL` from Gemini to a local vLLM model with no code changes |
| Tester | Talk to the agent | Hold the mic button, speak, release, and read the transcript and the reply |

## 5. Requirements

### Functional

- [ ] FR-1: Chat UI shows a scrolling conversation of user turns (transcript) and assistant turns (LLM reply).
- [ ] FR-2: Push-to-talk recording: press to start, release to stop. One recording = one turn. Recording auto-stops at 30 s (FR-12).
- [ ] FR-3: Browser captures mic audio and sends it to the backend over a WebSocket.
- [ ] FR-4: Backend receives audio already in the ASR format (16 kHz, mono, PCM16; see 6.3.1), validates it, and runs the selected ASR model.
- [ ] FR-5: Transcript is returned to the UI as soon as ASR finishes (before the LLM reply).
- [ ] FR-6: Transcript is appended to the conversation history and sent to the LLM via LiteLLM. By default this happens automatically (auto-send); see FR-9 for review mode.
- [ ] FR-7: LLM reply is streamed to the UI token by token (`llm_delta`), followed by a final `llm_done`.
- [ ] FR-8: User can select the ASR model from the UI without restarting the server. The server advertises the available models (from the registry) to the client on connect. The selection is sent with each `start_turn`.
- [ ] FR-9: UI has a "review transcript" toggle (default off). When on, the transcript is shown editable after ASR and is only sent to the LLM when the user confirms (`confirm_turn`, with the possibly edited text) or discarded (`discard_turn`). When off, the transcript is auto-sent.
- [ ] FR-10: Each turn reports timings: audio duration, ASR time, LLM time to first token, LLM total time, and end-to-end time (see 6.3.2 for exact definitions).
- [ ] FR-11: Conversation history lives in memory for the duration of the WebSocket session.
- [ ] FR-12: Recordings are capped at 30 s. The UI auto-stops at 30 s; the server rejects longer turns with an `error`.
- [ ] FR-13: One turn at a time per session. The UI disables the mic while a turn is in flight; the server rejects a `start_turn` received during an in-flight turn with an `error` (the in-flight turn is not affected). In review mode, a turn stays in flight until it is confirmed or discarded.
- [ ] FR-14: Failure handling: if ASR fails, nothing is added to the history and an error is shown in the chat. If the LLM fails, the user's transcript stays in the history, no assistant message is added, and an error is shown in the chat. The session stays usable after either failure.

### Non-Functional

- **Latency (target, not a hard gate):** ASR + LLM first token under ~3 s for a 5 s utterance on a single GPU.
- **Modularity:** adding a new ASR model means adding one adapter file and one config entry.
- **Observability:** structured logs per turn (session id, turn id, model names, timings, errors).
- **Compatibility:** latest Chrome and Safari for the UI; backend on Linux (WSL2 is fine) with an NVIDIA GPU (CPU fallback for Whisper acceptable for dev). Browser and backend run on the same machine, so the page is served from `localhost` and no TLS is needed.
- **Privacy:** audio is processed in memory; saving audio to disk is off by default and enabled by a debug flag.

## 6. Technical Approach

### 6.1 Architecture

```
Browser (chat UI)
  mic → AudioWorklet → PCM16 16kHz frames
        │  WebSocket (JSON control + binary audio)
        ▼
Backend (Python, FastAPI)
  Session manager ──► Audio buffer (per turn)
        │
        ▼
  ASR adapter (selected at runtime)
    ├─ whisper-large-v3
    ├─ parakeet-tdt-0.6b-v3
    └─ nemotron-3.5-asr-streaming-0.6b
        │ transcript
        ▼
  LLM client (LiteLLM)
    ├─ Google Gemini API
    └─ vLLM (OpenAI-compatible endpoint)
        │ reply tokens
        ▼
  WebSocket → UI
```

### 6.2 Stack (proposed)

- **Frontend:** small vanilla JS web app with no build step, served as static files by the FastAPI backend. Audio capture with `getUserMedia` + `AudioWorklet`, downsampled to 16 kHz mono PCM16 in the browser so the server does not need to decode Opus/WebM.
- **Backend:** Python (exact version set by NeMo compatibility, see section 7), FastAPI (WebSocket support), `uvicorn`. One process, one `uv`-managed environment; all ASR adapters run in-process on a thread pool.
- **ASR runtimes:** Hugging Face `transformers` for Whisper; NVIDIA NeMo for Parakeet and Nemotron.
- **LLM:** `litellm` as the single client interface.
- **Config:** `.env` + a `config.yaml` (model registry, defaults).

### 6.3 Transport: WebSocket protocol (v0)

Single endpoint: `/ws`

Client → server:
- JSON `{"type": "start_turn", "asr_model": "parakeet-tdt-0.6b-v3", "review": false}` (`language` is fixed to `en` in v0 and not sent)
- Binary frames: raw PCM16 audio chunks
- JSON `{"type": "end_turn"}`
- JSON `{"type": "confirm_turn", "turn_id": "...", "text": "..."}` (review mode only; `text` is the possibly edited transcript)
- JSON `{"type": "discard_turn", "turn_id": "..."}` (review mode only)
- JSON `{"type": "reset"}` (clears history; rejected while a turn is in flight)

Server → client:
- JSON `{"type": "session", "session_id": "...", "asr_models": ["..."], "default_asr_model": "..."}` (sent once on connect)
- JSON `{"type": "turn_started", "turn_id": "..."}` (ack of `start_turn`; the id used by every later message of this turn)
- JSON `{"type": "transcript", "turn_id": "...", "text": "...", "asr_ms": 0}`
- JSON `{"type": "llm_delta", "turn_id": "...", "text": "..."}` (repeated)
- JSON `{"type": "llm_done", "turn_id": "...", "text": "...", "timings": {...}}`
- JSON `{"type": "error", "turn_id": "...", "message": "..."}` (`turn_id` is null for errors not tied to a turn, such as a rejected `start_turn`)

Turn lifecycle: `start_turn` → audio frames → `end_turn` → `transcript` → (review mode: `confirm_turn` / `discard_turn`) → `llm_delta`* → `llm_done`. A discarded turn adds nothing to the history. The connection closing mid-turn drops the in-flight turn.

**WebRTC seam:** keep the transport behind a small interface (`send_json`, `send_bytes`, `on_audio_frame`) so a WebRTC data/audio channel can replace the WebSocket in a later version.

### 6.3.1 Audio format contract

16 kHz, mono, signed 16-bit little-endian PCM. Frames of 20-100 ms. The server concatenates frames until `end_turn`, then hands a single float32 array (values in [-1, 1]) to the ASR. The server rejects a turn with an `error` if it is shorter than 0.25 s (this includes empty, and covers an accidental tap of the mic button), has an odd byte count, or exceeds 30 s.

The browser does the resampling. `AudioContext` is created at 16 kHz where the browser honors it; where it does not (Safari may run at the device rate), the worklet resamples to 16 kHz itself.

### 6.3.2 Timing definitions

All durations in milliseconds, measured on the server with a monotonic clock, except audio duration.

- `audio_s`: audio duration, computed from the sample count.
- `asr_ms`: time inside `transcribe()`, excluding queueing for the worker thread and any first-use model load. A first-use load is logged separately as `model_load_ms`.
- `llm_ttft_ms`: from sending the request to LiteLLM to the first non-empty content delta.
- `llm_total_ms`: from sending the request to the last delta.
- `e2e_ms`: from receiving `end_turn` to the last delta (in review mode, user think time between `transcript` and `confirm_turn` is excluded).

### 6.4 ASR adapter interface

```python
class ASRModel(Protocol):
    name: str

    def load(self) -> None: ...
    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> TranscriptResult: ...


@dataclass
class TranscriptResult:
    text: str
    language: str | None
    duration_s: float
    latency_ms: float
```

- A registry maps a config key to an adapter class (`ASR_REGISTRY["whisper-large-v3"]`).
- Models are **lazy-loaded** on first use and then kept resident. All three fit on the H100 together (see GPU budget), so there is no eviction logic in v0. The first turn on a model pays the load time; this is reported separately (`model_load_ms`) and not counted in `asr_ms`.
- Inference runs in a worker thread (thread pool) so it does not block the event loop.
- Adapters always run English; `language` is fixed to `en`.

Candidate models:

| Key | Model | Runtime | Notes |
|-----|-------|---------|-------|
| `whisper-large-v3` | `openai/whisper-large-v3` | transformers | Broad multilingual coverage; offline (non-streaming) model; 30 s window matches the turn cap |
| `parakeet-tdt-0.6b-v3` | `nvidia/parakeet-tdt-0.6b-v3` | NeMo | Fast, 0.6B params; English is covered |
| `nemotron-3.5-asr-streaming-0.6b` | `nvidia/nemotron-3.5-asr-streaming-0.6b` (fallback: `nvidia/nemotron-speech-streaming-en-0.6b`) | NeMo | Cache-aware streaming RNNT with configurable chunk sizes; in v0 used in batch mode, and it is the natural v1 streaming candidate |

The Nemotron registry key stays `nemotron-3.5-asr-streaming-0.6b` in the UI and config, while the underlying checkpoint id is a config value, so switching to the fallback is a config change only.

### 6.5 LLM layer (LiteLLM)

- Use `litellm.acompletion(..., stream=True)`.
- Model and endpoint come from config, for example:
  - Google: `model="gemini/<model-name>"`, key from `GEMINI_API_KEY`
  - vLLM: `model="hosted_vllm/<model-name>"`, `api_base="http://localhost:8000/v1"`
- Streaming is required (FR-7); the first content delta marks `llm_ttft_ms`.
- The exact Gemini and vLLM model names are config values chosen during planning; the SPEC only fixes the config shape (`LLM_MODEL`, optional `api_base`, keys from `.env`).
- System prompt lives in config (short, voice-friendly answers).
- History is a list of `{role, content}` messages, trimmed to a max token budget (oldest turns dropped first; system prompt always kept).

### 6.6 Suggested repo layout

Inside the existing `astra` uv project (root `pyproject.toml`; the `theory/` folder is learning material and stays untouched):

```
astra/
  SPEC.md
  PLAN.md
  pyproject.toml
  backend/
    main.py            # FastAPI app + /ws + static file serving
    session.py         # per-connection state, history, turn state machine
    audio.py           # PCM validation, int16 -> float32 helpers
    asr/
      base.py          # ASRModel protocol, TranscriptResult
      registry.py
      whisper.py
      parakeet.py
      nemotron.py
    llm.py             # LiteLLM wrapper
    config.py
  frontend/            # served as static files, no build step; no shared code with backend/
    index.html
    style.css
    app.js             # chat UI, ws client, turn state
    recorder.js        # mic + AudioWorklet wrapper (main thread)
    recorder-worklet.js  # resample to 16 kHz, PCM16 framing (audio thread)
  scripts/
    compare_asr.py     # runs every registered adapter on a WAV fixture
  tests/
    fixtures/          # 10 s English clip + reference transcript
  config.yaml
  .env.example
```

### 6.7 Technology decisions and reasons

| Decision | Choice | Why | Revisit when |
|----------|--------|-----|--------------|
| Language | Python | NeMo, transformers, LiteLLM and vLLM are all Python-native, so no glue code or FFI | Never for v0 |
| Backend framework | FastAPI + uvicorn | First-class async WebSockets, easy to push blocking ASR calls into a thread pool, Pydantic models for the message protocol | WebRTC work: `aiortc` runs on asyncio and works alongside it |
| Transport | WebSocket | Simplest for turn-based audio: binary frames, no signaling, no STUN/TURN, easy to debug | v1 real-time: WebRTC gives UDP transport, jitter handling, echo cancellation |
| Audio capture | `AudioWorklet` to 16 kHz PCM16 | Deterministic format, no server-side decoding, same format streaming ASR will need in v1 | Never, unless a device can't support it (then `MediaRecorder` + ffmpeg) |
| Turn control | Push-to-talk | Avoids VAD and endpointing complexity in v0 | v1 (add VAD) |
| ASR abstraction | Adapter + registry | Comparing models is a main goal; new model = one file + one config entry | If models need very different call patterns (streaming) |
| Parakeet / Nemotron runtime | NeMo | Official runtime for both models | If a lighter export (ONNX / TensorRT) is needed for speed |
| Whisper runtime | `transformers` | Fewest moving parts to get a baseline; same HF ecosystem as the learning material | If speed matters: `faster-whisper` |
| Process layout | Single process, single uv env | Least plumbing; one interface (the adapter) instead of two | If NeMo's PyTorch/CUDA pins clash with the other packages and cannot be resolved: move ASR to a worker subprocess or its own service |
| LLM client | LiteLLM | One streaming API for Gemini and vLLM (OpenAI-compatible); switch by config only | If you only ever use one provider, or LiteLLM overhead shows up in latency measurements |
| Local LLM server | vLLM | High-throughput serving, OpenAI-compatible endpoint, fits an H100 | If you want simpler ops (e.g. Ollama) |
| Frontend | Vanilla JS, no build step | The hard part is audio and WebSocket, not UI; a framework adds little in v0, and static files can be served by FastAPI | If the UI grows |
| State | In-memory history per session | No persistence needed for v0 | When sessions must survive reconnects |
| Config | `.env` for secrets, `config.yaml` for models | Keeps keys out of the repo, makes model swaps trivial | Never |

**GPU budget (approximate):** Whisper large-v3 (~1.5B params) and the two 0.6B models each need only a few GB in fp16, so all three fit on the H100 at once, with headroom for activations. When running a local vLLM server, cap `--gpu-memory-utilization` so the ASR models don't run out of memory. Start at about 0.6 and tune it after measuring real ASR memory use; this is a starting value, not a requirement.

## 7. Constraints & Assumptions

- Single developer, Python end to end. Current dev machine: 1x NVIDIA RTX 3060 (12 GB), WSL2. Whisper large-v3 is the focus for now (fp16, roughly 3-4 GB). The earlier plan of keeping all three ASR models plus a local vLLM server resident assumed an H100 (80 GB) and does not hold on 12 GB; revisit when the NeMo adapters (M4) land. Until then the Gemini API is the practical LLM.
- English only for v0: language is fixed to `en` in all ASR adapters.
- v0 prioritizes development speed over efficiency.
- The user's browser can access the mic (`localhost`).
- API keys are provided through environment variables, never committed.
- **Python version risk:** the uv scaffold currently pins Python 3.13, but NeMo (and its PyTorch/CUDA pins) may not support it. Verify early (first task in PLAN) and lower `.python-version` / `requires-python` if needed. If NeMo cannot coexist with the other dependencies in one environment, fall back to a worker subprocess for the NeMo adapters (see 6.7).

## 8. Acceptance Criteria

- [ ] I can open the UI on `localhost`, hold the mic button, speak, release, and see my transcript followed by a streamed LLM reply.
- [ ] Switching the ASR model in the UI changes which model produces the transcript, with no server restart.
- [ ] `scripts/compare_asr.py` runs every registered ASR adapter on the committed 10 s English fixture clip and prints, per model, the transcript and `asr_ms`. Every model returns non-empty text. (No WER threshold in v0; the reference transcript is there for manual comparison.)
- [ ] Changing only config switches the LLM between Gemini and a local vLLM server.
- [ ] Each turn displays audio duration, ASR time, LLM time to first token, LLM total time and end-to-end time.
- [ ] Review mode: with the toggle on, I can edit the transcript before it reaches the LLM, or discard it and nothing is added to the history.
- [ ] A failed ASR or LLM call shows an error in the chat and does not crash the session; history follows FR-14.
- [ ] A recording over 30 s is stopped by the UI; a longer turn sent directly to the server is rejected with an error.
- [ ] A `start_turn` sent while another turn is in flight is rejected and does not disturb the in-flight turn.
- [ ] `README` steps let a fresh machine run the project in under 15 minutes (excluding model downloads).

## 9. Milestones

0. **M0: Environment check.** Confirm NeMo + transformers + FastAPI + LiteLLM install and import together in the uv env on the target Python version, and that the Nemotron checkpoint (3.5 or the fallback) downloads and loads.
1. **M1: Skeleton.** FastAPI `/ws` echo, chat UI, mic capture to PCM16, audio arrives intact on the server (verify by saving a WAV in debug mode).
2. **M2: One ASR end to end.** ASR interface + one adapter (start with Whisper) returning transcripts to the UI, plus the `compare_asr.py` script and fixture clip.
3. **M3: LLM loop.** LiteLLM integration with Gemini, streaming reply in the chat, history handling, turn state machine (one turn at a time, failure handling).
4. **M4: Plug and play.** Remaining ASR adapters, model selector, registry, lazy loading, review-transcript mode.
5. **M5: Metrics and polish.** Per-turn timings, structured logs, error handling, vLLM config path, README.

Stretch, not part of v0 acceptance: WebRTC transport behind the same interface, or a first streaming experiment with Nemotron.

## 10. Open Questions

Resolved during spec review (2026-09-30):

- Whisper runtime: `transformers`.
- Auto-send vs confirmation: auto-send by default, with a review toggle (FR-9).
- Nemotron: try 3.5, fall back to `nvidia/nemotron-speech-streaming-en-0.6b` (config-only switch).
- vLLM GPU memory: start at ~0.6 and tune (see GPU budget).
- Process layout: single process, single env.

Still open:

- [ ] Nemotron 3.5 access and license: availability notes on the Hugging Face pages conflict. Confirm in M0 that `nvidia/nemotron-3.5-asr-streaming-0.6b` downloads and that its license fits our use.
- [ ] Python version: can NeMo run on the scaffold's Python 3.13, or must the project be pinned lower? Settled in M0.
- [ ] Default Gemini model name and vLLM model name (config values, to be chosen in PLAN.md).

## 11. Notes / References

- Hugging Face model cards: `openai/whisper-large-v3`, `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/nemotron-3.5-asr-streaming-0.6b`
- LiteLLM docs (Gemini and `hosted_vllm` providers)
- NVIDIA NeMo ASR docs
- v1 direction: streaming ASR (Nemotron chunked inference) + VAD + TTS + WebRTC + barge-in
- Long-term project vision: see `README.md` (real-time translation ASR for live conversations, meetings and calls)