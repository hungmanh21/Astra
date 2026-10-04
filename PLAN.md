# Voice Agent v0: Implementation Plan

**Spec:** [SPEC.md](SPEC.md)
**Status:** Draft
**Last updated:** 2026-09-30

How to read this: milestones (M0-M5) match SPEC section 9. Each task is small enough to finish and check in one sitting. Every task has a **Done when** line, and that line is how you know it is finished. `FR-n` and `AC` refer to SPEC section 5 (requirements) and section 8 (acceptance criteria). Tasks within a milestone run in order unless a task says otherwise.

**Current order:** frontend first (T1.1, T1.9, T1.10 plus a protocol mock), then the M1 backend tasks. Whisper is the only ASR for now; NeMo adapters (Parakeet, Nemotron) and their environment work (T0.3 for those models, T0.4, T4.1, T4.2) are deferred. Dev hardware is an RTX 3060 (12 GB), not the H100 the SPEC first assumed.

## Decisions to make along the way

| Decision | Made in | Notes |
|----------|---------|-------|
| Python version (3.13 vs lower) | T0.1 | Depends on NeMo support. |
| Nemotron checkpoint (3.5 vs fallback) | T0.3 | Also check the license. |
| Default Gemini model name | T3.1 | Config value only. |
| vLLM model name | T5.4 | Config value only. |

---

## M0: Environment check

Goal: prove the dependency stack works together before writing any app code. This is the biggest technical risk in v0.

- [ ] **T0.1 Pick the Python version.** Try installing NeMo (ASR extras) with `uv` on the scaffold's Python 3.13. If it fails, lower `.python-version` and `requires-python` to the newest version NeMo supports (likely 3.12) and record the choice in SPEC section 7.
  Done when: `uv sync` succeeds and `import nemo.collections.asr` works.
- [ ] **T0.2 Add the backend dependencies.** Add `fastapi`, `uvicorn[standard]`, `litellm`, `transformers`, `torch` (CUDA build), `python-dotenv` or `pydantic-settings`, `pyyaml`, and `pytest` (dev). Keep the existing `librosa`/`soundfile` deps.
  Done when: one command imports FastAPI, LiteLLM, transformers and NeMo in the same env with no version conflicts, and `torch.cuda.is_available()` is `True`.
- [ ] **T0.3 Confirm model downloads and loading.** In a throwaway script under `scripts/`, load and transcribe a short clip with each of: `openai/whisper-large-v3`, `nvidia/parakeet-tdt-0.6b-v3`, and `nvidia/nemotron-3.5-asr-streaming-0.6b`. If the Nemotron 3.5 download fails or the license does not fit, use `nvidia/nemotron-speech-streaming-en-0.6b`. Note how Nemotron behaves in batch (non-streaming) mode.
  Done when: all three models produce text on the GPU, and the Nemotron checkpoint decision plus its license note are written into SPEC section 10.
- [ ] **T0.4 Fallback trigger.** If T0.2 shows an unresolvable clash between NeMo and the other packages, decide now: move the NeMo adapters into a worker subprocess (SPEC 6.7). Otherwise skip this task.
  Done when: the env is either clean or the subprocess design is written down and PLAN is updated.
- [ ] **T0.5 Update the SPEC.** Close the Python version and Nemotron open questions in SPEC section 10.
  Done when: section 10 lists only the LLM model names as still open.

---

## M1: Skeleton (transport and audio capture)

Goal: audio recorded in the browser reaches the server intact. No ASR or LLM yet. Covers FR-2, FR-3, FR-12 (UI side).

- [x] **T1.1 Project scaffold.** Create `backend/`, `frontend/`, `scripts/`, `tests/fixtures/`, `config.yaml`, `.env.example`, matching SPEC 6.6. Add `.env` to `.gitignore`.
  Done when: the folders exist and `.env` is ignored.
- [x] **T1.2 Config loader.** `backend/config.py`: load `.env` and `config.yaml` into a typed settings object (ASR model registry entries, default ASR model, LLM settings, system prompt, `DEBUG_SAVE_AUDIO` flag, max turn seconds = 30).
  Done when: a unit test loads a sample config and reads each field.
- [x] **T1.3 FastAPI app and static serving.** `backend/main.py` with `/ws` and static serving of `frontend/`. Run with `uvicorn`.
  Done when: opening `http://localhost:8000` serves the page and `/ws` accepts a connection.
- [x] **T1.4 Protocol models.** Pydantic models for every message in SPEC 6.3 (client and server), with a `type` discriminator.
  Done when: unit tests parse valid messages and reject malformed ones.
- [x] **T1.5 Transport interface.** Define the small interface (`send_json`, `send_bytes`, `on_audio_frame`) and a WebSocket implementation of it (SPEC 6.3, WebRTC seam). Session code must only use the interface.
  Done when: the session layer imports no FastAPI types.
- [x] **T1.6 Session skeleton.** (Verified in Chrome: the page handles the `session` message from the real backend and the recording turn went through.) `backend/session.py`: per-connection state (session id, history list, current turn or none). On connect, send the `session` message (session id, available models, default model).
  Done when: the browser console shows the `session` message after connecting.
- [x] **T1.7 Audio buffer and validation.** `backend/audio.py`: collect binary frames per turn, then produce a float32 array in [-1, 1]. Reject an empty turn, odd byte count, or over 30 s with an `error` (FR-12, SPEC 6.3.1).
  Done when: unit tests cover a good buffer, empty, odd length, and too long.
- [x] **T1.8 Turn handling for audio only.** Handle `start_turn` (send `turn_started`), audio frames, and `end_turn`. With `DEBUG_SAVE_AUDIO` on, write the turn to a WAV file. Otherwise audio stays in memory (privacy NFR).
  Done when: a test client sending a known PCM16 buffer produces a matching WAV in debug mode and no file when the flag is off.
- [x] **T1.9 Recorder worklet.** (Verified in a browser against `scripts/mock_server.py`; logic also covered by stubbed Node tests.) `frontend/recorder-worklet.js`: capture the mic through `AudioWorklet`, resample to 16 kHz mono where the `AudioContext` does not run at 16 kHz, convert to PCM16, and post 20-100 ms frames.
  Done when: logged frame sizes and sample rate are correct in Chrome.
- [x] **T1.10 Chat UI shell and push-to-talk.** (Verified in a browser against the mock. Also implements the transcript review UI (T4.7), streamed reply, timings display (T5.2) and reconnect ahead of schedule, tested against `scripts/mock_server.py`.) `frontend/index.html` and `app.js`: scrolling conversation, hold-to-record mic button (press to start, release to stop), WebSocket client, auto-stop at 30 s (FR-1, FR-2, FR-12).
  Done when: holding the button sends `start_turn`, frames, `end_turn`, and a 30 s hold stops on its own.
- [x] **T1.11 M1 end-to-end check.** (Chrome passed against the real backend with `DEBUG_SAVE_AUDIO=1`: the WAV sounds correct. Safari skipped on purpose: no Safari available on the dev machine. `scripts/mock_server.py` was deleted.) In Chrome, record a sentence with debug saving on and play back the WAV. Repeat once in Safari, since Safari's sample-rate handling differs.
  Done when: the WAV sounds correct at the right speed and pitch in both browsers (Chrome only, see above).

---

## M2: One ASR end to end

Goal: recorded speech comes back as a transcript in the chat, using Whisper. Covers FR-4, FR-5, and the compare script.

- [x] **T2.1 ASR interface.** `backend/asr/base.py`: `ASRModel` protocol and `TranscriptResult` from SPEC 6.4.
  Done when: type checks pass and a fake adapter satisfies the protocol in a test.
- [x] **T2.2 Registry.** `backend/asr/registry.py`: map config keys to adapter classes, lazy-load and cache instances, expose the list of available keys. Fixed language `en`.
  Done when: unit tests with a fake adapter show one `load()` call across two `get()` calls.
- [x] **T2.3 Whisper adapter.** (Verified on the T2.4 clip through `compare_asr.py`: matches the reference apart from case and punctuation; 1.6 s for 9.9 s of audio on the RTX 3060, fp16.) `backend/asr/whisper.py`: `transformers` pipeline for `openai/whisper-large-v3`, English forced, fp16 on GPU (CPU fallback for dev).
  Done when: it transcribes the test clip through the adapter interface.
- [x] **T2.4 Fixture clip and reference.** (LibriSpeech dev-clean utterance 1272-128104-0003, 9.9 s, CC BY 4.0, attribution in `tests/fixtures/README.md`.) Add a 10 s English WAV (16 kHz mono) and its reference transcript to `tests/fixtures/`. Use a clip you own or recorded yourself, or one with a license that allows committing it.
  Done when: both files are committed.
- [x] **T2.5 `scripts/compare_asr.py`.** Run every registered adapter on the fixture, print transcript and `asr_ms` per model. A model that fails to load prints its error and the script continues.
  Done when: the script runs and prints a row for Whisper (the others are added in M4).
- [x] **T2.6 Run ASR off the event loop.** (Verified by `test_second_client_stays_responsive_while_an_asr_call_runs`.) Run `transcribe()` in a thread pool from the session. Time it per SPEC 6.3.2 (`asr_ms` excludes queueing and first-use load; log `model_load_ms` separately).
  Done when: a second WebSocket client stays responsive while an ASR call runs.
- [x] **T2.7 Wire ASR into the turn.** (Browser check passed: the transcript renders in the UI.) (Until M3 the turn ends with an empty `llm_done` carrying `audio_s` and `asr_ms`, because the UI waits for `llm_done` before it unlocks.) On `end_turn`, transcribe and send `transcript` (with `asr_ms`). Show the user's transcript in the chat.
  Done when: hold, speak, release shows the transcript in the UI.
- [x] **T2.8 M2 check.** (Chrome: 4.6 s of audio gave `asr_ms` 1.16 s, warm `whisper-large-v3`, fp16, RTX 3060.) Record 3 different sentences and confirm sensible transcripts. Note the `asr_ms` for a 5 s utterance.
  Done when: transcripts show correctly and the timing is recorded in the M2 notes.

---

## M3: LLM loop

Goal: the transcript goes to the LLM and the reply streams into the chat, with a proper turn state machine. Covers FR-6, FR-7, FR-11, FR-13, FR-14.

- [x] **T3.1 LLM config and Gemini model choice.** (`gemini/gemini-3.5-flash-lite`, see DECISIONS; `litellm` added; the key goes in `.env`, which is git-ignored.) Choose the default Gemini model, put it in `config.yaml` as `LLM_MODEL`, keep `GEMINI_API_KEY` in `.env` (documented in `.env.example`), and add the system prompt (short, voice-friendly answers).
  Done when: the model name and system prompt are in config and the key is not committed.
- [x] **T3.2 LLM wrapper.** (Real Gemini stream confirmed with `scripts/try_llm.py` and a full Whisper + Gemini run; model `gemini/gemini-3.5-flash-lite` exists.) `backend/llm.py`: async generator over `litellm.acompletion(..., stream=True)` that yields text deltas, with `api_base` optional from config.
  Done when: a script streams a reply for a hard-coded prompt from Gemini.
- [x] **T3.3 History with token budget.** Store `{role, content}` messages in the session. Trim oldest turns when over the budget and always keep the system prompt.
  Done when: unit tests show trimming order and that the system prompt stays.
- [x] **T3.4 Turn state machine.** Explicit states: idle, receiving audio, transcribing, (awaiting confirm in review mode), streaming LLM, done or failed. Reject `start_turn` while a turn is in flight with an `error` (`turn_id` null) and leave the in-flight turn alone. Reject `reset` while a turn is in flight (FR-13). A closing connection drops the in-flight turn.
  Done when: unit tests cover every allowed and rejected transition.
- [x] **T3.5 Auto-send flow.** After ASR, append the transcript to history and stream `llm_delta` messages, then `llm_done` with the full text and timings (FR-6, FR-7).
  Done when: a test with a fake ASR and fake LLM produces the exact message sequence `transcript`, `llm_delta`*, `llm_done`.
- [x] **T3.6 Failure handling.** ASR failure: add nothing to history and send `error`. LLM failure: keep the user transcript in history, add no assistant message, send `error`. The session stays usable in both cases (FR-14).
  Done when: tests with a failing fake ASR and a failing fake LLM check history contents and that the next turn works.
- [x] **T3.7 `reset` message.** Clear history when idle (FR-11).
  Done when: after a reset, the next reply shows no memory of earlier turns.
- [x] **T3.8 UI: streamed reply, disabled mic, errors.** (Checked in Chrome by the user, including the bad-key case. The server log for that case was too noisy (about 80 lines), now one line.) Append `llm_delta` text live, disable the mic while a turn is in flight (FR-13), show errors as chat messages, add a reset button.
  Done when: in the browser the reply streams in, the mic is disabled until `llm_done` or `error`, and an invalid API key shows an error and the mic recovers.
- [x] **T3.9 M3 check.** (Checked in Chrome by the user.) Hold a short multi-turn conversation and confirm the model uses earlier turns.
  Done when: a follow-up question such as "what did I just ask?" is answered correctly.

---

## M4: Plug and play

Goal: all three ASR models are selectable at runtime, and transcripts can be reviewed before sending. Covers FR-8, FR-9.

- [ ] **T4.1 Parakeet adapter.** `backend/asr/parakeet.py` using NeMo, plus its config entry.
  Done when: `compare_asr.py` prints a Parakeet row.
- [ ] **T4.2 Nemotron adapter.** `backend/asr/nemotron.py` using NeMo in batch mode, with the checkpoint id read from config (3.5 or fallback, as decided in T0.3). Registry key stays `nemotron-3.5-asr-streaming-0.6b`.
  Done when: `compare_asr.py` prints a Nemotron row, and switching the checkpoint is a config-only edit.
- [ ] **T4.3 Compare script over all models.** Confirm `compare_asr.py` covers all three, prints a small table, and returns non-empty text for each (AC).
  Done when: one run prints three rows with non-empty transcripts.
- [x] **T4.4 Runtime model selection.** (Code and tests done; the switch between `whisper-large-v3` and `whisper-large-v3-turbo` checked in Chrome.) Honor `asr_model` in `start_turn` (reject unknown keys with an `error`). Keep loaded models resident, with no eviction.
  Done when: two consecutive turns with different models use different adapters with no restart, and the second turn on an already-used model has no load delay.
- [x] **T4.5 UI: model dropdown.** (Dropdown and the switch checked in Chrome with two Whisper models.) Fill the dropdown from the `session` message and send the selection with every `start_turn`.
  Done when: changing the dropdown changes which model transcribes the next turn.
- [x] **T4.6 Review mode (server).** (Blank `confirm_turn` text is an error that ends the turn, `discard_turn` sends nothing back, think time is excluded from `e2e_ms`.) With `review: true`, stop after `transcript` and wait for `confirm_turn` (edited text is what goes into history and to the LLM) or `discard_turn` (nothing added, turn ends). The turn stays in flight while waiting (FR-13). Reject a `confirm_turn` or `discard_turn` with a wrong `turn_id`.
  Done when: state machine and flow tests cover confirm, edited confirm, discard, and wrong id.
- [x] **T4.7 Review mode (UI).** (Checked in Chrome by the user.) Add the "review transcript" toggle (default off), an editable transcript box with Send and Discard buttons, and keep the mic disabled while reviewing.
  Done when: with the toggle on I can edit a transcript before it reaches the LLM, or discard it and see nothing added to the conversation.
- [ ] **T4.8 M4 check.** Record the same sentence with all three models and compare transcripts and `asr_ms`.
  Done when: results are noted in the M4 notes below.

---

## M5: Metrics and polish

Goal: timings visible, logs structured, vLLM path working, README done. Covers FR-10 and the remaining acceptance criteria.

- [x] **T5.1 Timings.** (Fake-clock tests pass; `Session` takes an injectable `clock`.) Compute `audio_s`, `asr_ms`, `llm_ttft_ms`, `llm_total_ms`, `e2e_ms` as defined in SPEC 6.3.2 with a monotonic clock, and include them in `llm_done`.
  Done when: a unit test with fake clocks checks each definition, including that review-mode think time is excluded from `e2e_ms`.
- [x] **T5.2 Timings in the UI.** (Checked in Chrome: audio, ASR, first token, LLM and end-to-end all show under the turn.) Show all five values under each turn (FR-10).
  Done when: every finished turn shows the five values.
- [x] **T5.3 Structured logs.** (One JSON line per turn from `backend/turnlog.py`, never containing the transcript or reply.) One structured log record per turn with session id, turn id, model names, timings, and any error.
  Done when: a run produces parseable log lines that include failed turns.
- [ ] **T5.4 vLLM path.** Choose the vLLM model, start a local vLLM server with a capped `--gpu-memory-utilization` (start near 0.6), and point the config at it with `hosted_vllm/<model>` and `api_base`. Measure real ASR memory use and adjust the cap if needed.
  Done when: changing only the config (no code changes) switches the reply source between Gemini and vLLM, and all three ASR models plus vLLM run at once without an out-of-memory error.
- [x] **T5.5 Edge-case tests.** (`tests/test_edge_cases.py`.) Add end-to-end tests over a real WebSocket with fake ASR and LLM for: over-30 s turn rejected, `start_turn` during an in-flight turn, unknown ASR model, connection dropped mid-turn.
  Done when: the tests pass in CI-style `pytest` with no GPU needed.
- [ ] **T5.6 Latency check.** Measure ASR + LLM first token for a 5 s utterance on each ASR model (target under about 3 s; not a hard gate). Record results and note any obvious hotspot for v1.
  Done when: the numbers are written down in a short notes section in this file or the README.
- [~] **T5.7 README.** (Written; the clean-checkout run is still to do.) Write setup and run steps: uv sync, `.env` and `config.yaml`, running the server, running `compare_asr.py`, switching LLM backends. Update the project vision line so it matches the SPEC's "stepping stone" note.
  Done when: someone can go from a fresh clone to a working chat in under 15 minutes, excluding model downloads (try it on a clean checkout).
- [ ] **T5.8 Final acceptance pass.** Walk through every item in SPEC section 8 and tick it. Update SPEC status to Accepted, or list what is left.
  Done when: every acceptance box is ticked or has a noted follow-up.

---

## Stretch (not part of v0 acceptance)

- [ ] **S1 WebRTC transport** implementing the T1.5 interface.
- [ ] **S2 Nemotron streaming experiment** using chunked inference.

## Notes

Fill in as you go: measured latencies, Python version outcome, Nemotron checkpoint outcome, anything that surprised you.

- M0 result:
- M2 `asr_ms` (5 s utterance, Whisper):
- M4 model comparison:
- M5 latency check:
