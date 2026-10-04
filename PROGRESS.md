# Project Progress

## Current State
- Latest commit: 521f75e (Fix review-mode stale buttons and blank transcript, add WebSocket origin check, send the turn cap to the UI)
- Test status: 221 pytest tests passing (audio, history, LLM wrapper, turn log, ASR registry/service with a fake adapter, config, protocol, transport, session, main). Whisper itself was run on real weights through `compare_asr.py`, not in pytest. JS logic was checked with throwaway Node scripts, not committed
- Lint: Ruff (lint + format), passing. `make check` exists and passes
- Hardware: RTX 3060 (12 GB) on WSL2, `torch.cuda.is_available()` is True.
- Deps installed: `torch`, `transformers`, `fastapi`, `uvicorn[standard]`, `pyyaml`, `pydantic`, `python-dotenv` (plus the original audio libs); dev: `pytest`, `ruff`, `httpx`

## Completed
- [x] Pre-PR review of v0 and fixes: review mode no longer pre-fills a blank transcript with "(no speech detected)"; a review turn that fails (disconnect or error) loses its Send/Discard, so stale buttons cannot confirm or discard the next turn; `/ws` refuses a cross-origin browser (HTTP 403) and the UI's `?ws=` override is gone; the `session` message carries `max_turn_seconds` and the UI auto-stops there (SPEC 6.3 updated); a turn is never logged twice when the client leaves while an error is being sent; blank `LLM_MODEL` / `LLM_API_BASE` in `.env` fall back to `config.yaml`. Frontend fixes checked with a throwaway Node harness (fails on the old code, passes on the new); origin check also tried against the real server
- [x] Project scaffold: README, uv project, `.gitignore`
- [x] Theory notes: `signal_basics.md`, `audio_representations.md` with runnable example
- [x] `SPEC.md`, `PLAN.md`, `CLAUDE.md`, `PROGRESS.md`, `DECISIONS.md`
- [x] Frontend (T1.9, T1.10): chat UI, push-to-talk, AudioWorklet to 16 kHz PCM16, WebSocket client, review mode, timings display, reconnect. Tried in a browser against the mock and working
- [x] ASR pipeline code and tests: `backend/audio.py` (T1.7), `backend/asr/{base,registry,whisper,service}.py` (T2.1, T2.2, T2.3 code, T2.6 code), `scripts/compare_asr.py`
- [x] M1 backend scaffold (stubs with TODO blocks for you to implement): `backend/{config,protocol,transport,session,main}.py`, `config.yaml` llm/limits/debug sections, `.env.example`, `make run`, test stubs. `.env` and `debug_audio/` are git-ignored
- [x] M1 end-to-end (T1.11): `make run` with `DEBUG_SAVE_AUDIO=1`, Chrome recording saved as a WAV that sounds correct; `scripts/mock_server.py` deleted
- [x] T2.3, T2.4, T2.5 verified: LibriSpeech clip in `tests/fixtures/` (CC BY 4.0, see its README); `compare_asr.py` on `whisper-large-v3` gave the reference text apart from case and punctuation, `asr_ms` 1603 for 9.9 s of audio (fp16, RTX 3060, warm model). The weights (about 3 GB) are now in `~/.cache/huggingface`
- [x] T3.8, T3.9, T4.5, T4.7 checked in Chrome by you. LLM failure log is now one line (`llm.py` logs a short summary, full traceback at DEBUG; `litellm.suppress_debug_info` hides its banner); `session.py` adds one warning for an `LLMError` and keeps `log.exception` for unexpected errors
- [x] T5.1 and T5.3: `Session(clock=...)` is used for every timing (fake-clock tests); `backend/turnlog.py` writes one JSON line per turn (`ok`, `empty`, `asr_error`, `llm_error`, `rejected`, `discarded`, `dropped`; never the transcript or reply) and `configure_logging()` makes INFO logs visible under uvicorn and quiets httpx/httpcore/LiteLLM. Turn status is recorded exactly once per turn
- [x] T5.5: `tests/test_edge_cases.py` runs four edge cases over a real WebSocket with fake ASR and LLM (turn over the cap, `start_turn` mid-turn, unknown model, connection dropped mid-turn); each also checks the connection or server still works afterwards. 4 of 5 mutations caught here; the fifth (no task cancel on disconnect) is caught in `test_session.py`
- [x] T2.8 and T5.2 checked in Chrome by you: a 4.6 s utterance on warm `whisper-large-v3` showed audio 4.6 s, ASR 1.16 s, first token 847 ms, LLM 1.11 s, end to end 2.27 s. All five timings show under the turn. ASR plus first token is about 2.0 s, under the 3 s target (T5.6, Whisper only so far)
- [x] T5.7: `README.md` rewritten: vision plus v0 stepping stone, setup, run, configuration and env vars, switching the LLM (vLLM path marked untried), adding an ASR model, `compare_asr.py`, development, layout, troubleshooting
- [x] You checked in Chrome: the 30 s UI auto-stop works, and model switching works with a second model (`whisper-large-v3-turbo` added to `config.yaml`, config entry only)
- [x] T5.4 code, T5.7 and T5.8: vLLM support (config only, model discovered from the endpoint, `LLM_API_KEY` optional) is built and was run through real LiteLLM against a local fake endpoint; the real-server test is deferred, Gemini stays the v0 LLM. README tried on a clean clone (offline `uv sync`, `make check`, server answers). `compare_asr.py` run on both Whisper models. SPEC section 8: 9 of 10 boxes ticked, the vLLM one carries a follow-up; SPEC status is now Accepted with follow-ups
- [x] T3.2 verified: `scripts/try_llm.py` streams a reply (model `gemini/gemini-3.5-flash-lite` exists); full Whisper + Gemini run works live. Cold first turn took 116 s (lazy imports and Whisper load from the slow `/mnt/d` venv), warm turns about 3 s end to end (Whisper `asr_ms` about 1.7 s for 9.9 s of audio, Gemini first token about 0.9 s)
- [x] T4.6 review mode on the server: `TurnStage.AWAITING_CONFIRM`, confirm (edited text goes to the LLM and the history), discard, wrong id and blank text handling, think time excluded from `e2e_ms`. Frontend side (T4.7) was already built
- [x] M3 (T3.1 to T3.7): `gemini/gemini-3.5-flash-lite` in config, `backend/llm.py` (LiteLLM streaming, lazy import), `backend/history.py` (token-budget trimming), auto-send flow with five timings, failure handling per FR-14, `reset`; `tests/conftest.py` fails a hung test after 20 s.
- [x] M2 wiring (T2.6, T2.7): `create_app` builds the registry and `TranscriptionService` (lazy models, shut down with the app); `Session` transcribes on `end_turn` in a background task (`TurnStage.WORKING`), sends `transcript` then an empty `llm_done` placeholder until M3, cancels the job on disconnect; generic "transcription failed" error
- [x] Worklet and recorder logic checked with stubbed Node tests (resampling at 48k/44.1k/16k, 30 s cap, stale-message handling)

## In Progress
- [~] T4.8 M4 check: waits for the NeMo adapters (Parakeet, Nemotron); Whisper is the only ASR for now. T4.4 and T4.5 are now seen with two Whisper models; Parakeet and Nemotron still wait for NeMo

## Known Issues
- Review-mode frontend fixes (blank transcript, stale buttons after a failed turn) not yet re-checked in Chrome
- Firefox unverified: it may refuse a mic stream in a 16 kHz `AudioContext`, and that error is thrown in `createMediaStreamSource`, outside the fallback in `recorder.js`
- Review nits not done: `log_turn` silently ignores an unknown status; `parse_client_message` builds a `TypeAdapter` per message; a missing `config.yaml` key raises `KeyError` instead of `ConfigError`; `librosa`, `matplotlib`, `scipy` are app dependencies but only `theory/` uses them; `pyproject.toml` description is a placeholder
- PyPI wheels download slowly from the dev machine (about 20-35 KB/s; Hugging Face is about 8 MB/s). `litellm` is installed now. Plan for extra time when a new dependency like NeMo is needed
- `.venv` lives on `/mnt/d` (9p mount, about 65x slower for small files). Setting `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/astra` would fix that; not done yet (tried, reverted while bandwidth is bad)
- `transformers` prints two warnings on Whisper load (a deprecated `generation_config` plus `begin_suppress_tokens` mix, and `clean_up_tokenization_spaces` for BPE). They come from the library defaults, not our kwargs; the output is correct. Revisit only if they get noisy
- Whisper load time was about 400 s on the first run because it included the download; measure a warm load for `model_load_ms` expectations
- M0 checked: Python 3.13, torch 2.14 (CUDA 13.0, RTX 3060 visible), transformers 5.17, fastapi, uvicorn, pydantic all import together and `uv pip check` is clean. Missing: NeMo (deferred, T0.1/T0.3/T0.4)
- Safari 16 kHz handling never checked (no Safari on the dev machine; T1.11 passed on Chrome only)
- `ProtocolError` messages include pydantic's raw text (multi-line, with a URL); shorten before they are shown in the chat
- The mic stays open (browser mic indicator on) after first enabling it, to avoid clipping the start of speech
- Python 3.13 vs NeMo compatibility unchecked (deferred; Whisper only for now)
- Whisper may emit text such as "Thank you." on silence or noise; not handled in v0, note what you see in M2

## Next Steps
1. Merge the v0 PR into main (after a Chrome check of review mode)
2. Decide what comes next: NeMo adapters and the M0 questions (T0.1, T0.3 to T0.5, T4.1 to T4.3, T4.8), or v1 streaming ASR (Nemotron chunked inference, VAD)
3. T5.6 latency notes (skipped for now; the Whisper numbers are in Completed, plus turbo 416 ms vs large-v3 1507 ms ASR on the 9.9 s fixture clip)
4. Optional: preload Whisper and import `litellm` at startup so the first turn is fast (needs a `config.yaml` flag; SPEC says lazy)
5. Switch `compare_asr.py` to `load_settings` (T1.2) and shorten `ProtocolError` messages
