# Voice Agent v1: Implementation Plan

**Spec:** [SPEC_v1.md](SPEC_v1.md) (builds on [SPEC_v0.md](SPEC_v0.md))
**Status:** Approved, in progress
**Last updated:** 2026-10-04

How to read this: milestones S0-S5 match SPEC_v1 section 9. Task ids start with `S` (streaming) so they never clash with the v0 `T` ids that the existing code cites. Each task has a **Done when** line, and that line is how you know it is finished. `FR-n` refers to SPEC_v1 section 5, or to SPEC_v0 for FR-1 to FR-14. Tasks within a milestone run in order unless a task says otherwise.

**Order and why:** build the whole streaming path first (protocol, session, coalescing, UI) on models that already run: a fake in tests, then the Whisper strategies (`chunked`, `redecode`, `trimmed`, see SPEC_v1 section 1). NeMo comes after (S4), because installing it is the biggest risk: slow PyPI downloads, the Python version, and the Nemotron license. That way a NeMo problem cannot block the seam, and Nemotron plugs into an interface that is already tested.

**Carried over from v0:** NeMo setup (v0 T0.1-T0.5) is now S4.1-S4.3. The Parakeet adapter (T4.1) is S4.5 and the Nemotron adapter (T4.2) is S4.4. The latency check (T5.6) is S0.2 and S5.1. Still open in PLAN_v0 and not part of v1: the real vLLM server test (T5.4).

## Decisions to make along the way

| Decision | Made in | Notes |
|----------|---------|-------|
| `partial_interval_ms`, `partial_min_audio_ms` defaults | S2.7 | Starting values 500 / 1000; tune from `compare_asr.py --stream` numbers. |
| `chunk_ms` for `chunked` | S2.7 | Try 500, 1000, 2000. |
| Default strategy for batch models | S2.7 | `redecode` until the numbers say otherwise. |
| Are `whisper-large-v3` partials usable on the RTX 3060? | S2.7 | Or recommend `trimmed` or turbo for it. |
| Python version vs NeMo | S4.2 | Carried over from v0 T0.1. |
| Nemotron checkpoint (3.5 vs fallback), license | S4.3 | Carried over from v0 T0.3. |
| Nemotron chunk size | S4.3 | Latency vs accuracy; a config option. |
| Which models stay resident on 12 GB | S4.6 | No eviction logic in v0. Ask before adding any. |

---

## S0: Docs and v0 baseline

Goal: v1 documents in place, plus the numbers v1 must beat.

- [x] **S0.1 Split the docs.** Rename `SPEC.md` to `SPEC_v0.md` and `PLAN.md` to `PLAN_v0.md`, add `SPEC_v1.md` and `PLAN_v1.md`, and update every link to them (CLAUDE.md, README, PROGRESS). Add a DECISIONS entry for the v1 scope.
  Done when: no link or instruction outside `.venv` and `theory/` points at `SPEC.md` or `PLAN.md` (mentions of the rename itself are fine), and `make check` passes.
- [ ] **S0.2 Baseline and `compare_asr.py` cleanup.** Switch `compare_asr.py` to `load_settings` (v0 Next Step). Record the v0 numbers: `asr_ms` on the fixture clip and on a ~5 s live utterance (Chrome timings line) for both Whisper models.
  Done when: `compare_asr.py` reads models through `load_settings`, and the baseline table is in the Notes below.

---

## S1: Streaming seam with a fake model

Goal: the server sends `partial` messages for a fake streaming model, end to end over a real WebSocket, with no real ASR involved. Batch models behave exactly as in v0 (no partials yet).

- [ ] **S1.1 Interface types.** In `backend/asr/base.py`, add `PartialResult`, `ASRStream`, `StreamingASRModel`, `Segment` and `SegmentedASRModel` as in SPEC_v1 6.4, plus a `FakeStreamingASR` in the tests (scripted partials, can be told to fail or block).
  Done when: the fake satisfies the protocols (checked in a test) and `make check` passes.
- [ ] **S1.2 Config.** Add the `asr.streaming` section (`default_strategy`, `partial_interval_ms`, `partial_min_audio_ms`, `chunk_ms`) to `config.yaml` and `Settings`. Reject non-positive values and an unknown or non-generic default strategy (`native` and `trimmed` depend on the model) with a `ConfigError`.
  Done when: config tests cover the defaults, an override, and each bad value.
- [ ] **S1.3 Protocol.** Add the `partial` server message; the optional `stream_strategy` on `start_turn`; `stream_strategies` on `session`; `stream_strategy` on `transcript`; and `partial_count`, `partial_lag_ms` and `final_ms` on `Timings` (all optional) (SPEC_v1 6.2).
  Done when: protocol tests pin the wire shape of each changed message, an old-style `start_turn` (no strategy) still parses, and every v0 test still passes.
- [ ] **S1.4 `StreamingTranscription`.** `backend/asr/streaming.py`, one per turn. It takes audio frames (non-blocking push into a pending buffer) and keeps at most one `feed` in flight on the ASR worker. When a feed returns, it starts the next one with all pending audio once `partial_interval_ms` worth has gathered (FR-20). Each result goes to a callback. `finish()` waits for the feed in flight, feeds the remainder, then calls `ASRStream.finish()`. It enforces FR-16: a `committed` that does not extend the previous one is kept as tentative and logged. The first feed failure turns partials off for the turn (one log line, FR-18), and `finish` is still tried. `close()` cancels everything, for a disconnect or a rejected turn.
  Done when: tests with a blocking fake show no backlog (fewer feeds than frames, all samples fed exactly once, in order), finish runs after the feed in flight, a feed failure still yields a final result, and close leaves no task running.
- [ ] **S1.5 Service and strategy lookup.** `TranscriptionService.open_stream(model_name, strategy)` and `strategies(model_name)` (supported list and default, from the adapter's capabilities, SPEC_v1 6.4). `native` calls the adapter's `open_stream`. Until S2, every generic strategy maps to a `BatchStream` (`feed` returns None, `finish` transcribes the whole turn, so it behaves exactly like v0). The first use loads the model as in v0, and the load time is logged as `model_load_ms`. S2 replaces `BatchStream` with the real strategies one by one.
  Done when: service tests cover the native and batch paths through the real registry with fake adapters, the supported list per capability, and the default rule.
- [ ] **S1.6 Session wiring.** The `session` message lists each model's strategies. `start_turn` checks `stream_strategy` (missing = the model's default; unsupported = `error`, turn not started, FR-22) and opens the turn's stream. `on_audio_frame` still fills the turn buffer (validation, debug WAV) and also pushes whole samples to the stream, carrying a trailing odd byte over to the next frame. Partials go out as `partial` messages for the current turn, and never after its `transcript`. `end_turn` validates as in v0, then calls `finish` for the final `transcript`. A rejected turn, `reset` rules and a disconnect close the stream. Fill `final_ms`, `partial_count` and `partial_lag_ms` (SPEC_v1 6.3) with the injectable clock. Review mode is untouched (FR-21).
  Done when: session tests (fake clock and fake streaming model) cover partials in order, no partial after `transcript`, the three timings, a partial failure still ending in `ok`, rejection and disconnect mid-stream, and review mode with partials. Every v0 session test still passes.
- [ ] **S1.7 Turn log.** Add `stream_strategy`, `partial_count`, `partial_lag_ms` and `final_ms` to the per-turn log line.
  Done when: turn-log tests show the new fields and still find no text in the line.
- [ ] **S1.8 Edge cases over a real WebSocket.** Extend `tests/test_edge_cases.py` with a fake streaming model: partials arrive while frames are still being sent, a forced partial failure still gives `transcript` and `llm_done`, an unsupported strategy is rejected, and a disconnect mid-stream leaves the server usable.
  Done when: those tests pass, and each one fails when its fix is reverted (mutation check, as in v0 T5.5).

---

## S2: Whisper strategies

Goal: real partials from the Whisper models we already have, in three ways that can be compared (SPEC_v1 section 1). Each strategy task replaces `BatchStream` for that strategy only, so the others keep working throughout.

- [ ] **S2.1 `compare_asr.py --stream`.** First, so every strategy is measured the moment it exists. Feed a WAV in 50 ms frames at real-time pace through `open_stream` and `StreamingTranscription` (the real coalescing). Print each partial with its time offset (committed | tentative), then `partial_count`, median `partial_lag_ms`, `final_ms`, the final text and its word error rate (WER) against the reference `.txt` when one sits next to the WAV. `--strategy` picks one or `all`, and a summary table (model × strategy) closes the run.
  Done when: it runs on the fixture for both Whisper models with every strategy (all still `BatchStream`, so no partials yet), and the WER of the final text matches the batch run.
- [ ] **S2.2 `chunked` (`ChunkedStream`).** Every `chunk_ms` of new audio is transcribed on its own, once, and appended to `committed`. A coalesced feed holding several chunks decodes each of them. `finish` transcribes the remainder (dropped if under `partial_min_audio_ms`) and returns the joined text (SPEC_v1 6.4).
  Done when: unit tests with a scripted fake batch model show each sample decoded exactly once, chunks cut at `chunk_ms`, and the remainder rule; `compare_asr.py --stream --strategy chunked` runs on the fixture with `chunk_ms` 500, 1000 and 2000, with the numbers in the Notes.
- [ ] **S2.3 LocalAgreement-2.** `backend/asr/agreement.py`, a pure function: given the committed words and the last two decodes, return the new committed text and the tentative tail. Compare words ignoring case and punctuation, but output the decode's own spelling. Committed text never shrinks.
  Done when: unit tests cover agreement growth, a disagreeing middle word, a decode that shortens, punctuation-only changes, and an empty decode.
- [ ] **S2.4 `redecode` (`RedecodeStream`).** Keeps the turn's audio, re-transcribes everything on `feed` once `partial_min_audio_ms` is reached, and applies S2.3. `finish` runs one full decode.
  Done when: tests with a scripted fake show committed growth across feeds, no partial before the minimum audio, and a `finish` result equal to the plain `transcribe` result; `compare_asr.py --stream --strategy redecode` on the fixture is in the Notes.
- [ ] **S2.5 Whisper segments and prompt.** Make `WhisperASR` a `SegmentedASRModel`: `transcribe_segments` returns segments with start and end times (pipeline timestamps) and takes an optional text prompt (Whisper `prompt_ids`). This is a capability of the adapter, not a special case in the session.
  Done when: on the fixture, the segments' text joined equals the plain transcript (apart from spacing), the timestamps increase and lie inside the clip, and a prompt run still gives sensible text (checked with `compare_asr.py`; the adapter's unit test uses a stubbed pipeline).
- [ ] **S2.6 `trimmed` (`TrimmedStream`).** Like `redecode`, but it decodes only the audio after the trim point, prompted with the committed text. It moves the trim point to the end of a committed segment that ends a sentence (SPEC_v1 6.4). `finish` decodes only the tail and returns committed + tail.
  Done when: tests with a scripted fake segmented model show the trim point moving only on a committed sentence end, the decoded audio getting shorter after a trim, the prompt carrying the committed text, and committed text never shrinking across a trim; `compare_asr.py --stream --strategy trimmed` on the fixture is in the Notes.
- [ ] **S2.7 Compare, tune and decide.** Run `compare_asr.py --stream --strategy all` on the fixture for both Whisper models (and on a longer clip near 30 s if one is at hand, where `trimmed` should pull ahead). Set the `partial_interval_ms`, `partial_min_audio_ms`, `chunk_ms` and default-strategy values, and decide what to recommend for large-v3. Close those questions in SPEC_v1 section 10.
  Done when: the comparison table is in the Notes, the defaults are in `config.yaml`, and there is a DECISIONS entry with the numbers behind them.

---

## S3: UI partials

Goal: the live text in the browser.

- [ ] **S3.1 Render partials.** While recording (and while the final transcript is pending), show `committed` normally and `tentative` dimmed in the user's bubble, instead of "Recording…". Ignore a `partial` for another turn, or one that arrives after the turn's `transcript`. The `transcript` replaces the bubble text (FR-17). Review mode starts from the final transcript (FR-21).
  Done when: a throwaway Node harness (as in the v0 review) shows partials rendering, being replaced by the final transcript, and stale partials being ignored.
- [ ] **S3.2 Timings line.** Add the strategy, `final_ms` and the partial lag to the timings shown under each turn (FR-10).
  Done when: the line shows them when they are present and skips them when null.
- [ ] **S3.3 Strategy dropdown.** Next to the model dropdown, list the selected model's strategies from the `session` message, preselect its default, remember the choice per model (like the model choice), and send it with every `start_turn` (FR-22). Disabled while a turn is in flight.
  Done when: the Node harness shows the list changing with the model, the default chosen, and the choice sent in `start_turn`.
- [ ] **S3.4 Chrome check.** Use `whisper-large-v3-turbo` and `whisper-large-v3`, each with `chunked`, `redecode` and `trimmed`, with review mode off and on, plus a turn that hits the 30 s auto-stop.
  Done when: you have seen words appear while speaking with every strategy, committed text never jumping back, and the final transcript and reply as before.

---

## S4: NeMo environment and Nemotron

Goal: native streaming. Mind the slow PyPI downloads (PROGRESS Known Issues).

- [ ] **S4.1 Move the venv off `/mnt/d` (recommended first).** Set `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/astra` (about 65x faster for small files), re-sync, and note it in the README.
  Done when: `make check` passes from the new env and the README says how to set it.
- [ ] **S4.2 Install NeMo and settle the Python version.** (v0 T0.1, T0.2, T0.4.) Add NeMo with the ASR extras. If it does not support 3.13, lower `.python-version` and `requires-python`. If its pins clash with the rest, decide on the worker-subprocess fallback (SPEC_v0 6.7) and ask before building it.
  Done when: `uv sync` succeeds, one process imports `nemo.collections.asr`, transformers and litellm, `torch.cuda.is_available()` is True, `make check` passes, and the Python version is in DECISIONS.
- [ ] **S4.3 Nemotron spike.** (v0 T0.3.) In a throwaway script, download the checkpoint (3.5, else the fallback), check the license, transcribe the fixture in batch, and run NeMo's cache-aware streaming loop over it with each supported chunk size. Note text, time per chunk and GPU memory.
  Done when: SPEC_v1 section 10 questions 1-3 are closed (checkpoint, license, Python version, chunk sizes and default), with DECISIONS entries.
- [ ] **S4.4 Nemotron adapter.** `backend/asr/nemotron.py`: `load`, batch `transcribe`, and `open_stream` returning a native `ASRStream` (chunk buffering, cache state per stream, flush in `finish`). Add the config entry with `chunk_ms`. Keep NeMo imports lazy, as for transformers.
  Done when: `compare_asr.py` (batch and `--stream`) gives non-empty fixture text close to the batch run, `native` is Nemotron's default strategy while `chunked` and `redecode` also run on it (through its batch `transcribe`), `final_ms` is reported, and two streams in a row do not share state (unit test with the real model, skipped when NeMo or the weights are missing).
- [ ] **S4.5 Parakeet adapter (optional).** (v0 T4.1.) `backend/asr/parakeet.py`, batch only; it streams through `chunked` and `redecode` with no extra code (and `trimmed` if it returns timestamps).
  Done when: `compare_asr.py` batch and `--stream --strategy all` both work on it.
- [ ] **S4.6 GPU memory.** Load every configured model the way the server would (lazily, all resident) and record peak memory with `nvidia-smi`.
  Done when: the numbers are in the Notes. If everything does not fit in 12 GB, propose a default model set or eviction and ask before building it.

---

## S5: Measurements and acceptance

- [ ] **S5.1 Comparison table.** For each model × strategy, record first partial, median `partial_lag_ms`, `partial_count`, `final_ms`, `asr_ms`, final WER and LLM first token, on the fixture (`--stream --strategy all`) and on a ~5 s live utterance in Chrome. Compare with the S0.2 baseline.
  Done when: the table is in SPEC_v1 section 8 (with the S0.2 baseline), and the winner per use (fastest final, most accurate, steadiest partials) plus any hotspot for the next version are noted.
- [ ] **S5.2 Chrome check on all models.** Nemotron (and Parakeet if added) in the UI with each of their strategies, with review mode on and off, a forced LLM failure, and model and strategy switching without a restart.
  Done when: you have checked each item in Chrome.
- [ ] **S5.3 README.** Describe streaming (what partials mean, the four strategies and when to use which), `compare_asr.py --stream`, NeMo setup and the venv location, and add Nemotron to the model list.
  Done when: the README steps work on a clean clone (as in v0 T5.7).
- [ ] **S5.4 Final acceptance pass.** Walk through SPEC_v1 section 8 and tick each item, or list what is left. Set the SPEC_v1 status.
  Done when: every box is ticked or carries a follow-up, and the status is updated.

---

## Notes

- **Baseline (S0.2):** _to fill in._ Known so far: on the 9.9 s fixture, `asr_ms` is 1507 ms for large-v3 and 416 ms for turbo; a 4.6 s live utterance gave large-v3 `asr_ms` 1.16 s and first token 847 ms.
- **Why chunks are not free for Whisper:** Whisper pads every input to a 30 s window, so its encoder costs about the same for 0.5 s as for 20 s. A decode's cost grows mostly with the number of output tokens (the decoder). So `chunked` saves less time than it seems and loses context, while `redecode` costs more per partial as the text grows (on large-v3, a long turn may take well over 1 s per decode). `trimmed` keeps the decoded text short. S2.7 measures all three.
- **Strategy results (S2.2-S2.7):** _to fill in._
- **Testing:** real model weights stay out of pytest, as in v0. A test that needs NeMo or the weights is skipped when they are missing.
