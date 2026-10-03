# Project Progress

## Current State
- Latest commit: 283248f (Add LibriSpeech fixture clip; verify Whisper end to end)
- Test status: 96 pytest tests passing (82 at the last commit) (audio, ASR registry/service with a fake adapter, config, protocol, transport, session, main). Whisper itself was run on real weights through `compare_asr.py`, not in pytest. JS logic was checked with throwaway Node scripts, not committed
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
- [x] T2.3, T2.4, T2.5 verified: LibriSpeech clip in `tests/fixtures/` (CC BY 4.0, see its README); `compare_asr.py` on `whisper-large-v3` gave the reference text apart from case and punctuation, `asr_ms` 1603 for 9.9 s of audio (fp16, RTX 3060, warm model). The weights (about 3 GB) are now in `~/.cache/huggingface`
- [x] Worklet and recorder logic checked with stubbed Node tests (resampling at 48k/44.1k/16k, 30 s cap, stale-message handling)

## In Progress
- [~] T2.7 ASR wired into the turn (about 90%): code and tests pass (96); still to do is the browser check (`make run`, hold, speak, release shows the transcript), which is T2.8

## Known Issues
- `transformers` prints two warnings on Whisper load (a deprecated `generation_config` plus `begin_suppress_tokens` mix, and `clean_up_tokenization_spaces` for BPE). They come from the library defaults, not our kwargs; the output is correct. Revisit only if they get noisy
- Whisper load time was about 400 s on the first run because it included the download; measure a warm load for `model_load_ms` expectations
- M0 checked: Python 3.13, torch 2.14 (CUDA 13.0, RTX 3060 visible), transformers 5.17, fastapi, uvicorn, pydantic all import together and `uv pip check` is clean. Missing: `litellm` (T0.2; a dry-run install adds only new packages, no conflicts, so add it at T3.1) and NeMo (deferred, T0.1/T0.3/T0.4)
- Safari 16 kHz handling never checked (no Safari on the dev machine; T1.11 passed on Chrome only)
- `ProtocolError` messages include pydantic's raw text (multi-line, with a URL); shorten before they are shown in the chat
- The mic stays open (browser mic indicator on) after first enabling it, to avoid clipping the start of speech
- Python 3.13 vs NeMo compatibility unchecked (deferred; Whisper only for now)
- Default Gemini model name not chosen yet (T3.1)
- Whisper may emit text such as "Thank you." on silence or noise; not handled in v0, note what you see in M2

## Next Steps
1. T2.8 M2 check (`make run`, Chrome; the first turn loads Whisper, so expect a few seconds for `model_load_ms`): record 3 sentences in Chrome, note `asr_ms` for a 5 s utterance
2. Switch `compare_asr.py` to `load_settings` (T1.2) and shorten `ProtocolError` messages
3. M3: add `litellm` (T0.2), pick the Gemini model name (T3.1), then the LLM streaming tasks
