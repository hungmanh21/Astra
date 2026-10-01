# Project Progress

## Current State
- Latest commit: 88c03b4 (Add push-to-talk frontend and protocol mock server), before the backend stubs and lint commit that follows it; see `git log -1` for the newest
- Test status: 17 pytest stubs, all skipped (nothing implemented yet); JS logic was checked with throwaway Node scripts, not committed
- Lint: Ruff (lint + format), passing. `make check` exists and passes
- Hardware: RTX 3060 (12 GB) on WSL2, GPU visible via `nvidia-smi`. Torch CUDA not yet verified from Python.
- Deps installed: `torch`, `transformers`, `fastapi`, `uvicorn[standard]` (plus the original audio libs); dev: `pytest`, `ruff`

## Completed
- [x] Project scaffold: README, uv project, `.gitignore`
- [x] Theory notes: `signal_basics.md`, `audio_representations.md` with runnable example
- [x] `SPEC.md`, `PLAN.md`, `CLAUDE.md`, `PROGRESS.md`, `DECISIONS.md`
- [x] Frontend (T1.9, T1.10): chat UI, push-to-talk, AudioWorklet to 16 kHz PCM16, WebSocket client, review mode, timings display, reconnect. Tried in a browser against the mock and working
- [x] `scripts/mock_server.py`: protocol mock that saves received audio as WAV
- [x] Worklet and recorder logic checked with stubbed Node tests (resampling at 48k/44.1k/16k, 30 s cap, stale-message handling)

## In Progress
- [~] T1.11 M1 end-to-end check: working against the mock; the Safari result has not been recorded
- [~] T1.1 scaffold: `frontend/` and `scripts/` exist; `backend/`, `tests/`, `config.yaml`, `.env.example` come with the backend tasks

## Known Issues
- Safari 16 kHz handling not confirmed
- The mic stays open (browser mic indicator on) after first enabling it, to avoid clipping the start of speech
- Python 3.13 vs NeMo compatibility unchecked (deferred; Whisper only for now)
- Default Gemini model name not chosen yet (T3.1)
- `CLAUDE.md` still says the target hardware is an H100; the dev machine is an RTX 3060 (see DECISIONS.md)

## Next Steps
1. Record the Safari result for T1.11 (or note it as skipped)
2. Verify torch sees the GPU: `uv run python -c "import torch; print(torch.cuda.is_available())"`
3. M1 backend: config loader (T1.2), protocol models (T1.4), transport interface (T1.5), session and audio validation (T1.6-T1.8), replacing the mock with `backend/main.py` (T1.3)
4. Then M2: ASR interface, registry and the Whisper adapter
