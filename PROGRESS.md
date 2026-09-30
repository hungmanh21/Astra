# Project Progress

## Current State
- Latest commit: b7cc57f (Finalize audio_representations.md and align spectrum plot with HF course)
- Test status: no tests yet
- Lint: not configured yet
- Note: `make check` does not exist yet (no Makefile)

## Completed
- [x] Project scaffold: README, uv project, `.gitignore`
- [x] Theory notes: `signal_basics.md`, `audio_representations.md` with runnable example
- [x] `SPEC.md` for v0 (draft, reviewed 2026-09-30)
- [x] `PLAN.md` for v0 (draft)
- [x] `CLAUDE.md`

## In Progress
- [ ] M0: Environment check (not started)

## Known Issues
- Python version: the scaffold pins 3.13 and NeMo may not support it (settled in T0.1)
- Nemotron 3.5 access and license unconfirmed (settled in T0.3)
- Default Gemini and vLLM model names not chosen yet (T3.1, T5.4)
- `make check` referenced in `CLAUDE.md` does not exist yet

## Next Steps
1. T0.1: try installing NeMo on the current Python version, lower it if needed
2. T0.2: add backend dependencies and confirm they import together on the GPU
3. T0.3: confirm the three ASR models download and transcribe a clip
4. Create a Makefile with a `check` target (lint + tests) early in M1
