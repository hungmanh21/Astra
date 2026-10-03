# Project Progress

## Current State
- Latest commit: 81bb7cf (Add M1 backend: config, protocol, transport, session, app)
- Test status: 82 pytest tests passing (audio, ASR registry/service with a fake adapter, config, protocol, transport, session, main). Whisper itself has not been run on real weights yet. JS logic was checked with throwaway Node scripts, not committed
- Lint: Ruff (lint + format), passing. `make check` exists and passes
- Hardware: RTX 3060 (12 GB) on WSL2, `torch.cuda.is_available()` is True.
- Deps installed: `torch`, `transformers`, `fastapi`, `uvicorn[standard]`, `pyyaml`, `pydantic`, `python-dotenv` (plus the original audio libs); dev: `pytest`, `ruff`, `httpx`

## Completed
- [x] Project scaffold: README, uv project, `.gitignore`
- [x] Theory notes: `signal_basics.md`, `audio_representations.md` with runnable example
- [x] `SPEC.md`, `PLAN.md`, `CLAUDE.md`, `PROGRESS.md`, `DECISIONS.md`
- [x] Frontend (T1.9, T1.10): chat UI, push-to-talk, AudioWorklet to 16 kHz PCM16, WebSocket client, review mode, timings display, reconnect. Tried in a browser against the mock and working
- [x] ASR pipeline code and tests: `backend/audio.py` (T1.7), `backend/asr/{base,registry,whisper,service}.py` (T2.1, T2.2, T2.3 code, T2.6 code), `scripts/compare_asr.py`
- [x] M1 backend scaffold (stubs with TODO blocks for you to implement): `backend/{config,protocol,transport,session,main}.py`, `config.yaml` llm/limits/debug sections, `.env.example`, `make run`, test stubs. `.env` and `debug_audio/` are git-ignored
- [x] M1 end-to-end (T1.11): `make run` with `DEBUG_SAVE_AUDIO=1`, Chrome recording saved as a WAV that sounds correct; `scripts/mock_server.py` deleted
- [x] Worklet and recorder logic checked with stubbed Node tests (resampling at 48k/44.1k/16k, 30 s cap, stale-message handling)

## In Progress
- [~] T2.3 Whisper adapter: written, not run on real weights; needs the model download and a clip (T2.4)
- [~] T2.5 `compare_asr.py`: written, not run; needs the T2.4 fixture clip
- [~] T2.6 service: unit tested with a fake adapter; the "second WebSocket client stays responsive" check needs the session code

## Known Issues
- Safari 16 kHz handling never checked (no Safari on the dev machine; T1.11 passed on Chrome only)
- `ProtocolError` messages include pydantic's raw text (multi-line, with a URL); shorten before they are shown in the chat
- The mic stays open (browser mic indicator on) after first enabling it, to avoid clipping the start of speech
- Python 3.13 vs NeMo compatibility unchecked (deferred; Whisper only for now)
- Default Gemini model name not chosen yet (T3.1)
- Whisper may emit text such as "Thank you." on silence or noise; not handled in v0, note what you see in M2

## Next Steps
1. T2.4: add a 10 s English clip (your own voice or licensed) and its reference transcript to `tests/fixtures/`
2. Run `uv run python scripts/compare_asr.py tests/fixtures/<clip>.wav` to check Whisper end to end (downloads about 3 GB), then tick T2.3 and T2.5; switch `compare_asr.py` to `load_settings` (T1.2)
3. M2 wiring: build the registry and `TranscriptionService` in `create_app`, transcribe on `end_turn` (T2.7), then the M2 check (T2.8)
