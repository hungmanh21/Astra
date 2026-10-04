# CLAUDE.md

## Project

Astra: real-time ASR (including translation) with open-source Hugging Face models, aimed at live conversations, meetings and calls (see `README.md`).

v0 (done, merged) is a turn-based voice agent: browser push-to-talk → WebSocket → pluggable ASR → LiteLLM → streamed reply in a chat UI. The current target is **v1**: the same agent with live partial transcripts while the user speaks, and a bench for comparing streaming strategies (`chunked`, `redecode`, `trimmed` on Whisper first, then Nemotron `native`).

- `SPEC_v1.md`: what v1 adds and changes. Source of truth for v1 requirements (FR-15 on) and acceptance criteria. Everything it does not change is still defined by `SPEC_v0.md`.
- `PLAN_v1.md`: v1 broken into tasks (S0.1 ... S5.4) with "Done when" checks. Tick tasks off as they finish.
- `SPEC_v0.md`, `PLAN_v0.md`: the accepted v0 spec and plan. Section and task references in the existing code (`SPEC 6.3`, `PLAN T1.7`) point here; v1 code cites `SPEC_v1` sections and `S` task ids.
- `PROGRESS.md`: current state of the code (see workflow below).
- `DECISIONS.md`: important design decisions and why (see workflow below).
- `theory/`: learning notes and runnable examples on audio and ASR fundamentals. Not part of the v0 app; leave it alone unless asked.

## Environment and commands

Python project managed with `uv` (`pyproject.toml`, `uv.lock`, `.python-version`). The Python version may be lowered in task T0.1 if NeMo does not support the current one.

- Install: `uv sync`
- Run a script: `uv run <path/to/script.py>` (for example `uv run theory/fundamentals/examples/audio_representations.py`)
- Consistency check: `make check` (Ruff lint, Ruff format check, then pytest). `make fmt` auto-fixes lint issues and formats. Ruff is configured in `pyproject.toml` (line length 100, `theory/` excluded) and covers Python only, not `frontend/`.

Target hardware for v0 is Linux (WSL2 is fine) with one NVIDIA GPU. The dev machine is an RTX 3060 (12 GB), which fits Whisper large-v3 in fp16. Browser and backend run on the same machine (`localhost`).

## Conventions

- Follow `SPEC_v1.md` (and `SPEC_v0.md` where v1 changes nothing). If the code needs to deviate from the SPEC, or the SPEC is ambiguous, ask before deviating and update the SPEC once decided.
- Work through `PLAN_v1.md` in order and treat each task's "Done when" line as the definition of finished. Do not mark a task done without verifying that line.
- v1 is English only, push-to-talk, one turn at a time per session, 30 s cap. Do not add VAD, long-form audio, TTS, WebRTC, translation or early (speculative) LLM calls to v1 (these are listed as non-goals).
- Every ASR model streams. Strategies come from adapter capabilities (`open_stream` for `native`, `transcribe_segments` for `trimmed`; `chunked` and `redecode` work for any model), never from the model's name. The session never checks which kind of stream it has.
- Session and turn logic must depend on the transport interface (`send_json`, `send_bytes`, `on_audio_frame`), not on FastAPI types directly, so WebRTC can replace the WebSocket later.
- Adding an ASR model means one adapter file plus one config entry. Do not special-case models in the session code.
- LLM access goes through LiteLLM only. Provider and model are config values, never hard-coded.
- Secrets live in `.env` (never committed). Model names and defaults live in `config.yaml`.
- Audio stays in memory. Saving audio to disk only happens behind the debug flag.
- Match the style of surrounding code. Keep comments sparse and explain why, not what.

## Session workflow

### At session start (clock in)
1. Read `PROGRESS.md` for the current state.
2. Read `DECISIONS.md` for important decisions.
3. Run `make check` to confirm the repo is in a consistent state. A modified `PROGRESS.md` in `git status` is expected (see clock out).
4. Continue from the "Next Steps" section of `PROGRESS.md`.

### Before session end (clock out)
1. Run `make check` to confirm a consistent state.
2. Commit the completed work (not `PROGRESS.md`).
3. Update `PROGRESS.md` to describe the repo as of that commit: set "Latest commit" to it (`git log -1 --oneline`), move finished items to Completed, refresh In Progress, Known Issues and Next Steps. Leave the file modified and uncommitted.

`PROGRESS.md` therefore shows up in `git status` at the start of the next session. That is intended: it is the only thing changed since the last commit, and it goes into the next work commit. It is never committed on its own.

### PROGRESS.md
Tracks the progress of the **repo code**, not the learning notes. Keep this structure:

- **Current State**: latest commit (hash and message), test status, lint status.
- **Completed**: checked-off list of finished work.
- **In Progress**: unfinished work, with a rough percentage and what is blocking it.
- **Known Issues**: bugs and open questions.
- **Next Steps**: numbered list of what to do next, in order.

### DECISIONS.md
Records important design decisions in a few lines each: what was decided, why, and when. No detailed design documents. One entry per decision, newest at the bottom, in this shape:

```
## YYYY-MM-DD: Decision title
- Reason: ...
- Rejected alternative: ...   (if any)
- Constraint: ...             (if any)
```

Add an entry whenever a choice affects structure, dependencies, or the SPEC (for example the Python version, the Nemotron checkpoint, the Gemini or vLLM model names).

## Git

- Commit messages are short and in the imperative, matching the existing history (for example "Add uv project scaffold and audio representations fundamentals").
- Commit completed work only. Do not commit `.env`, model weights, or generated output.
