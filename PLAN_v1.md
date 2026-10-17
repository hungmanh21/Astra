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
| `partial_interval_ms`, `partial_min_audio_ms` defaults | S2.7 | **Kept 500 / 1000** (sweep in the Notes). |
| `chunk_ms` for `chunked` | S2.7 | **2000.** |
| Default strategy for batch models | S2.7 | **`redecode`.** |
| Are `whisper-large-v3` partials usable on the RTX 3060? | S2.7 | **No for live use; recommend turbo.** See DECISIONS 2026-10-15. |
| Python version vs NeMo | S4.2 (settled: stays 3.13) | Carried over from v0 T0.1. |
| Nemotron checkpoint (3.5 vs fallback), license | S4.3 | Carried over from v0 T0.3. |
| Nemotron chunk size | S4.3 | Latency vs accuracy; a config option. |
| Which models stay resident on 12 GB | S4.6 | No eviction logic in v0. Ask before adding any. |

---

## S0: Docs and v0 baseline

Goal: v1 documents in place, plus the numbers v1 must beat.

- [x] **S0.1 Split the docs.** Rename `SPEC.md` to `SPEC_v0.md` and `PLAN.md` to `PLAN_v0.md`, add `SPEC_v1.md` and `PLAN_v1.md`, and update every link to them (CLAUDE.md, README, PROGRESS). Add a DECISIONS entry for the v1 scope.
  Done when: no link or instruction outside `.venv` and `theory/` points at `SPEC.md` or `PLAN.md` (mentions of the rename itself are fine), and `make check` passes.
- [x] **S0.2 Baseline and `compare_asr.py` cleanup.** Switch `compare_asr.py` to `load_settings` (v0 Next Step). Record the v0 numbers: `asr_ms` on the fixture clip and on a ~5 s live utterance (Chrome timings line) for both Whisper models.
  Done when: `compare_asr.py` reads models through `load_settings`, and the baseline table is in the Notes below.

---

## S1: Streaming seam with a fake model

Goal: the server sends `partial` messages for a fake streaming model, end to end over a real WebSocket, with no real ASR involved. Batch models behave exactly as in v0 (no partials yet).

- [x] **S1.1 Interface types.** In `backend/asr/base.py`, add `PartialResult`, `ASRStream`, `StreamingASRModel`, `Segment` and `SegmentedASRModel` as in SPEC_v1 6.4, plus a `FakeStreamingASR` in the tests (scripted partials, can be told to fail or block).
  Done when: the fake satisfies the protocols (checked in a test) and `make check` passes.
- [x] **S1.2 Config.** Add the `asr.streaming` section (`default_strategy`, `partial_interval_ms`, `partial_min_audio_ms`, `chunk_ms`) to `config.yaml` and `Settings`. Reject non-positive values and an unknown or non-generic default strategy (`native` and `trimmed` depend on the model) with a `ConfigError`.
  Done when: config tests cover the defaults, an override, and each bad value.
- [x] **S1.3 Protocol.** Add the `partial` server message; the optional `stream_strategy` on `start_turn`; `stream_strategies` on `session`; `stream_strategy` on `transcript`; and `partial_count`, `partial_lag_ms` and `final_ms` on `Timings` (all optional) (SPEC_v1 6.2).
  Done when: protocol tests pin the wire shape of each changed message, an old-style `start_turn` (no strategy) still parses, and every v0 test still passes.
- [x] **S1.4 `StreamingTranscription`.** `backend/asr/streaming.py`, one per turn. It takes audio frames (non-blocking push into a pending buffer) and keeps at most one `feed` in flight on the ASR worker. When a feed returns, it starts the next one with all pending audio once `partial_interval_ms` worth has gathered (FR-20). Each result goes to a callback. `finish()` waits for the feed in flight, feeds the remainder, then calls `ASRStream.finish()`. It enforces FR-16: a `committed` that does not extend the previous one is kept as tentative and logged. The first feed failure turns partials off for the turn (one log line, FR-18), and `finish` is still tried. `close()` cancels everything, for a disconnect or a rejected turn.
  Done when: tests with a blocking fake show no backlog (fewer feeds than frames, all samples fed exactly once, in order), finish runs after the feed in flight, a feed failure still yields a final result, and close leaves no task running.
- [x] **S1.5 Service and strategy lookup.** `TranscriptionService.open_stream(model_name, strategy)` and `strategies(model_name)` (supported list and default, from the adapter's capabilities, SPEC_v1 6.4). `native` calls the adapter's `open_stream`. Until S2, every generic strategy maps to a `BatchStream` (`feed` returns None, `finish` transcribes the whole turn, so it behaves exactly like v0). The first use loads the model as in v0, and the load time is logged as `model_load_ms`. S2 replaces `BatchStream` with the real strategies one by one.
  Done when: service tests cover the native and batch paths through the real registry with fake adapters, the supported list per capability, and the default rule.
- [x] **S1.6 Session wiring.** The `session` message lists each model's strategies. `start_turn` checks `stream_strategy` (missing = the model's default; unsupported = `error`, turn not started, FR-22) and opens the turn's stream. `on_audio_frame` still fills the turn buffer (validation, debug WAV) and also pushes whole samples to the stream, carrying a trailing odd byte over to the next frame. Partials go out as `partial` messages for the current turn, and never after its `transcript`. `end_turn` validates as in v0, then calls `finish` for the final `transcript`. A rejected turn, `reset` rules and a disconnect close the stream. Fill `final_ms`, `partial_count` and `partial_lag_ms` (SPEC_v1 6.3) with the injectable clock. Review mode is untouched (FR-21).
  Done when: session tests (fake clock and fake streaming model) cover partials in order, no partial after `transcript`, the three timings, a partial failure still ending in `ok`, rejection and disconnect mid-stream, and review mode with partials. Every v0 session test still passes.
- [x] **S1.7 Turn log.** Add `stream_strategy`, `partial_count`, `partial_lag_ms` and `final_ms` to the per-turn log line.
  Done when: turn-log tests show the new fields and still find no text in the line.
- [x] **S1.8 Edge cases over a real WebSocket.** Extend `tests/test_edge_cases.py` with a fake streaming model: partials arrive while frames are still being sent, a forced partial failure still gives `transcript` and `llm_done`, an unsupported strategy is rejected, and a disconnect mid-stream leaves the server usable.
  Done when: those tests pass, and each one fails when its fix is reverted (mutation check, as in v0 T5.5).

---

## S2: Whisper strategies

Goal: real partials from the Whisper models we already have, in three ways that can be compared (SPEC_v1 section 1). Each strategy task replaces `BatchStream` for that strategy only, so the others keep working throughout.

- [x] **S2.1 `compare_asr.py --stream`.** First, so every strategy is measured the moment it exists. Feed a WAV in 50 ms frames at real-time pace through `open_stream` and `StreamingTranscription` (the real coalescing). Print each partial with its time offset (committed | tentative), then `partial_count`, median `partial_lag_ms`, `final_ms`, the final text and its word error rate (WER) against the reference `.txt` when one sits next to the WAV. `--strategy` picks one or `all`, and a summary table (model × strategy) closes the run.
  Done when: it runs on the fixture for both Whisper models with every strategy (all still `BatchStream`, so no partials yet), and the WER of the final text matches the batch run.
- [x] **S2.2 `chunked` (`ChunkedStream`).** Every `chunk_ms` of new audio is transcribed on its own, once, and appended to `committed`. A coalesced feed holding several chunks decodes each of them. `finish` transcribes the remainder (dropped if under `partial_min_audio_ms`) and returns the joined text (SPEC_v1 6.4).
  Done when: unit tests with a scripted fake batch model show each sample decoded exactly once, chunks cut at `chunk_ms`, and the remainder rule; `compare_asr.py --stream --strategy chunked` runs on the fixture with `chunk_ms` 500, 1000 and 2000, with the numbers in the Notes.
- [x] **S2.3 LocalAgreement-2.** `backend/asr/agreement.py`, a pure function: given the committed words and the last two decodes, return the new committed text and the tentative tail. Compare words ignoring case and punctuation, but output the decode's own spelling. Committed text never shrinks.
  Done when: unit tests cover agreement growth, a disagreeing middle word, a decode that shortens, punctuation-only changes, and an empty decode.
- [x] **S2.4 `redecode` (`RedecodeStream`).** Keeps the turn's audio, re-transcribes everything on `feed` once `partial_min_audio_ms` is reached, and applies S2.3. `finish` runs one full decode.
  Done when: tests with a scripted fake show committed growth across feeds, no partial before the minimum audio, and a `finish` result equal to the plain `transcribe` result; `compare_asr.py --stream --strategy redecode` on the fixture is in the Notes.
- [x] **S2.5 Whisper segments and prompt.** Make `WhisperASR` a `SegmentedASRModel`: `transcribe_segments` returns segments with start and end times (pipeline timestamps) and takes an optional text prompt (Whisper `prompt_ids`). This is a capability of the adapter, not a special case in the session.
  Done when: on the fixture, the segments' text joined equals the plain transcript (apart from spacing), the timestamps increase and lie inside the clip, and a prompt run still gives sensible text (checked with `compare_asr.py`; the adapter's unit test uses a stubbed pipeline).
- [x] **S2.6 `trimmed` (`TrimmedStream`).** Like `redecode`, but it decodes only the audio after the trim point, prompted with the committed text. It moves the trim point to the end of a committed segment that ends a sentence (SPEC_v1 6.4). `finish` decodes only the tail and returns committed + tail.
  Done when: tests with a scripted fake segmented model show the trim point moving only on a committed sentence end, the decoded audio getting shorter after a trim, the prompt carrying the committed text, and committed text never shrinking across a trim; `compare_asr.py --stream --strategy trimmed` on the fixture is in the Notes.
- [x] **S2.7 Compare, tune and decide.** Run `compare_asr.py --stream --strategy all` on the fixture for both Whisper models (and on a longer clip near 30 s if one is at hand, where `trimmed` should pull ahead). Set the `partial_interval_ms`, `partial_min_audio_ms`, `chunk_ms` and default-strategy values, and decide what to recommend for large-v3. Close those questions in SPEC_v1 section 10.
  Done when: the comparison table is in the Notes, the defaults are in `config.yaml`, and there is a DECISIONS entry with the numbers behind them.

---

## S3: UI partials

Goal: the live text in the browser.

- [x] **S3.1 Render partials.** While recording (and while the final transcript is pending), show `committed` normally and `tentative` dimmed in the user's bubble, instead of "Recording…". Ignore a `partial` for another turn, or one that arrives after the turn's `transcript`. The `transcript` replaces the bubble text (FR-17). Review mode starts from the final transcript (FR-21).
  Done when: a throwaway Node harness (as in the v0 review) shows partials rendering, being replaced by the final transcript, and stale partials being ignored.
- [x] **S3.2 Timings line.** Add the strategy, `final_ms` and the partial lag to the timings shown under each turn (FR-10).
  Done when: the line shows them when they are present and skips them when null.
- [x] **S3.3 Strategy dropdown.** Next to the model dropdown, list the selected model's strategies from the `session` message, preselect its default, remember the choice per model (like the model choice), and send it with every `start_turn` (FR-22). Disabled while a turn is in flight.
  Done when: the Node harness shows the list changing with the model, the default chosen, and the choice sent in `start_turn`.
- [x] **S3.4 Chrome check.** Use `whisper-large-v3-turbo` and `whisper-large-v3`, each with `chunked`, `redecode` and `trimmed`, with review mode off and on, plus a turn that hits the 30 s auto-stop.
  Done when: you have seen words appear while speaking with every strategy, committed text never jumping back, and the final transcript and reply as before.

---

## S4: NeMo environment and Nemotron

Goal: native streaming. Mind the slow PyPI downloads (PROGRESS Known Issues).

- [x] **S4.1 Move the venv off `/mnt/d` (recommended first).** Set `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/astra` (about 65x faster for small files), re-sync, and note it in the README.
  Done when: `make check` passes from the new env and the README says how to set it.
- [x] **S4.2 Install NeMo and settle the Python version.** (v0 T0.1, T0.2, T0.4.) Add NeMo with the ASR extras. If it does not support 3.13, lower `.python-version` and `requires-python`. If its pins clash with the rest, decide on the worker-subprocess fallback (SPEC_v0 6.7) and ask before building it.
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

- **Baseline (S0.2),** v0 batch `transcribe` on the 9.9 s fixture (`librispeech_1272-128104-0003.wav`), fp16, RTX 3060, via `compare_asr.py` (2026-10-12):

  | Model | `asr_ms` | Cold load | Text |
  |---|---|---|---|
  | `whisper-large-v3` | 1499 | 112 s (first load, includes reading the weights from `/mnt/d`) | correct, "Leighton" |
  | `whisper-large-v3-turbo` | 385 | 6.2 s | one word differs: "Layton" for "Leighton" |

  Earlier runs on the same clip gave 1507 ms (large-v3) and 416 ms (turbo), so the numbers are stable to within about 10%. Live in Chrome, one 4.6 s utterance on large-v3 gave `asr_ms` 1.16 s and first token 847 ms. Turbo has no live number yet; S5.1 records one.
- **Why chunks are not free for Whisper:** Whisper pads every input to a 30 s window, so its encoder costs about the same for 0.5 s as for 20 s. A decode's cost grows mostly with the number of output tokens (the decoder). So `chunked` saves less time than it seems and loses context, while `redecode` costs more per partial as the text grows (on large-v3, a long turn may take well over 1 s per decode). `trimmed` keeps the decoded text short. S2.7 measures all three.
- **Real-time bench (S2.1),** `compare_asr.py --stream` on the fixture, interval 500 ms, 2026-10-14. The final WER of `chunked` and `redecode` equals the batch run while both are still `BatchStream` (large-v3 0.0 %, turbo 4.2 %).
- **`chunked` (S2.2),** turbo, one run each:

  | `chunk_ms` | partials | median lag | `final_ms` | WER, remainder dropped (min 1000) | WER, remainder decoded |
  |---|---|---|---|---|---|
  | 500 | 19 | 195 ms | 193 ms | 66.7 % | 70.8 % ("Yeah." from trailing silence) |
  | 1000 | 9 | 227 ms | 181 ms | 29.2 % | 37.5 % ("Thank you." appended) |
  | 2000 | 4 | 247 ms | 262 ms | 16.7 % | 16.7 % |

  Batch is 4.2 % at about 420 ms. Longer chunks read better, but the first text appears later. Even 2000 ms garbles words at the cuts ("originally Greek", "of of rocky"). See DECISIONS 2026-10-14 for the remainder rule.
- **`redecode` (S2.4),** fixture, interval 500 ms, minimum 1000 ms, one run each (2026-10-14):

  | Model | partials | median lag | `final_ms` | batch `asr_ms` | WER (both) |
  |---|---|---|---|---|---|
  | `whisper-large-v3-turbo` | 17 | 304 ms (222 first, 401 last) | 839 ms | 400 ms | 4.2 % |
  | `whisper-large-v3` | 10 | 784 ms (434 first, 1254 last) | 3199 ms | 1562 ms | 0.0 % |

  The final text equals batch on both. Settled words trail the newest by about one decode ("He has grasped." became "He has grave doubts" before "He has" was committed), and the tentative tail shows mistakes the next decode removes ("rocky Ithanae." then "Ithaca."). The lag grows with the turn because every decode covers all the audio; large-v3 takes 1.5 s per decode, so it cannot keep the 500 ms pace (10 partials instead of 17). **`final_ms` is about twice a full decode**, not one as SPEC_v1 6.3 expects: at `end_turn` the remaining audio is fed (a full decode whose partial is dropped) and `finish` then decodes again. Not yet separated from a decode already in flight.
- **Whisper segments (S2.5),** `compare_asr.py --segments` on the fixture (2026-10-14): both models return one segment, 0.00-9.90 s (a single sentence; a 25 s clip with three repeats split into segments in a hand probe), and its text joined equals the plain transcript. Times increase and lie inside the clip. A prompted run on the audio after 3 s (prompt "He has grave doubts whether Sir") gives sensible text and no echo of the prompt. **Watch out for S2.6:** on `whisper-large-v3` the timestamped text came back **lower case without punctuation** ("he has grave doubts ... rocky ithaca"), while plain `transcribe` and turbo give "He has grave doubts ... Ithaca." The "segment ends in sentence punctuation" rule cannot fire on that text. The prompted large-v3 run was punctuated again ("Newton's work is really Greek after all, and ..."), so a prompt may restore the style; S2.6/S2.7 should test it.
- **`trimmed` (S2.6),** `compare_asr.py --stream --strategy trimmed`, same fixture (2026-10-14):

  | model | partials | median lag | `final_ms` | WER |
  |---|---|---|---|---|
  | `whisper-large-v3-turbo` | 17 | 344 ms | 621 ms | 4.2 % |
  | `whisper-large-v3` | 9 | 879 ms | 4163 ms | 0.0 % |

  The fixture is one sentence, so **no cut ever happens on it** and `trimmed` behaves like `redecode` (turbo `final_ms` is lower, 621 against 839, only because the timestamped decode is a little cheaper there). On `whisper-large-v3` the final text is lower case without punctuation (the timestamp-mode finding above) and `final_ms` is higher than `redecode`'s 3199 ms.

  To exercise the cut I made a 29.7 s clip from the fixture three times in a row (scratch file, not committed; the 30 s cap rules out a longer one). Turbo, all strategies, real-time pace:

  | strategy | partials | median lag | `final_ms` | WER |
  |---|---|---|---|---|
  | batch | - | - | 955 ms | 4.2 % |
  | `chunked` (500 ms) | 59 | 193 ms | 200 ms | 69.4 % |
  | `redecode` | 46 | 478 ms | 2231 ms | 4.2 % |
  | `trimmed` | 55 | 348 ms | 519 ms | 8.3 % |

  `trimmed` pays off where it should: `final_ms` stays under 0.6 s instead of growing with the turn, and the lag is lower than `redecode`'s. The price is accuracy: sentence 2 and 3 get extra "Ithaca." repeats at their ends and a few punctuation changes, so the WER doubles. A 200 ms margin on the cut did not change it, so it is not the timestamps; the likely cause is that this clip repeats one sentence, which makes the prompt (the sentence just settled) the next thing Whisper "expects". S2.7 must repeat this on a clip with different sentences before judging `trimmed` on accuracy.
- **Found by the long run (S2.6):** the first version built the settled text from the current decode's spelling, so a cut could turn "doubts," into "doubts" after it was shown (the FR-16 check in `compare_asr.py` flagged it). The settled text now uses the committed words as they were shown; `test_the_settled_text_keeps_the_spelling_the_user_already_saw` covers it.
- **Strategy results (S2.7),** `compare_asr.py --stream`, interval 500 ms unless noted, 2026-10-15. Two scratch clips, not committed, joined from the LibriSpeech dummy set (`hf-internal-testing/librispeech_asr_dummy`) with 0.2 s gaps and their reference text: **A** = utterances 1272-128104-0005..0008 (29.6 s, 60 words), **B** = 1272-135031-0000, -0001, -0003 (27.2 s).

  | model | clip | strategy | partials | median lag | `final_ms` | WER |
  |---|---|---|---|---|---|---|
  | turbo | A | batch | - | - | 782 ms | 3.3 % |
  | turbo | A | `chunked` 500 | 50 | 196 ms | 440 ms | 445 % |
  | turbo | A | `chunked` 1000 | 29 | 198 ms | 198 ms | 30.0 % |
  | turbo | A | `chunked` 2000 | 14 | 248 ms | 184 ms | 13.3 % |
  | turbo | A | `redecode` | 45 | 433 ms | 2229 ms | 3.3 % |
  | turbo | A | `trimmed` | 57 | 290 ms | 536 ms | 8.3 % |
  | turbo | B | batch | - | - | 860 ms | 1.4 % |
  | turbo | B | `chunked` 500 | 45 | 194 ms | 331 ms | 78.3 % |
  | turbo | B | `redecode` | 41 | 480 ms | 1803 ms | 1.4 % |
  | turbo | B | `trimmed` | 35 | 368 ms | 476 ms | 5.8 % |
  | large-v3 | A | batch | - | - | 2887 ms | 1.7 % |
  | large-v3 | A | `chunked` 500 | 58 | 390 ms | 934 ms | 90.0 % |
  | large-v3 | A | `redecode` | 22 | 1024 ms | 7588 ms | 1.7 % |
  | large-v3 | A | `trimmed` | 33 | 844 ms | 16970 ms | 21.7 % |
  | large-v3 | A | `trimmed`, second run | 19 | 1463 ms | 4537 ms | 3.3 % (lower case, never cut) |
  | large-v3 | B | batch | - | - | 3543 ms | 1.4 % |
  | large-v3 | B | `chunked` 500 | 53 | 386 ms | 639 ms | 82.6 % |
  | large-v3 | B | `redecode` | 17 | 1196 ms | 7499 ms | 1.4 % |
  | large-v3 | B | `trimmed` | 25 | 894 ms | 2646 ms | 26.1 % |

  Interval sweep, turbo `redecode`, clip A: 250 ms gives 60 partials, lag 374 ms, `final_ms` 1672; 500 ms 45, 433 ms, 2229; 1000 ms 28, 453 ms, 1623 (WER 3.3 % each). `trimmed` at 1000 ms: 29 partials, 370 ms, 454 ms, 8.3 %. No FR-16 violation (committed text always extends) in any run after the S2.6 spelling fix.

  `trimmed` after the minimum-window rule (DECISIONS 2026-10-15, same clips, turbo): A 48 partials, lag 382 ms, `final_ms` 1209 ms, WER 5.0 % (before: 290 ms, 536 ms, 8.3 %); B 45 partials, 392 ms, 1557 ms, 0.0 % (before: 368 ms, 476 ms, 5.8 %). The "Thank you." is gone.

  After the two follow-ups (`finish(audio)` and the token cap; DECISIONS 2026-10-15), turbo, `chunk_ms` 2000: clip A `redecode` 45 partials, lag 446 ms, `final_ms` 1217 ms, WER 3.3 %; `trimmed` 48, 383 ms, 691 ms, 5.0 %; `chunked` 14, 281 ms, 194 ms, 13.3 %. Clip B `redecode` 44, 457 ms, 1469 ms, 1.4 %; `trimmed` 43, 429 ms, 892 ms, 1.4 %; `chunked` 13, 241 ms, 200 ms, 14.5 %. `trimmed` on clip B gave 0.0 % in the run before and 1.4 % now: the cuts depend on decode timing, so the WER of `trimmed` moves by a word between runs.

  What the runs showed (before that rule):
  - `chunked` 500 ms triggered a Whisper repetition loop ("I'm sorry" dozens of times in one chunk) on clip A; Whisper tokens are not capped by audio length in `WhisperASR._run`. Even 2000 ms garbles words at the cuts.
  - `redecode` is as accurate as batch, but `final_ms` grows with the turn (about 2x a decode: the flush feed plus the final decode; 1.6-2.2 s on turbo, 7.5 s on large-v3 for 30 s).
  - `trimmed` on turbo keeps `final_ms` near 0.5 s on a 30 s turn and the lag below `redecode`'s, for 4-5 WER points more. The final decode on the short window left after the last cut once hallucinated "Thank you."
  - `trimmed` on large-v3: timestamped text sometimes loses its punctuation (then it never cuts and the lag grows); with a prompt it dropped a sentence ("On the general principles of art ...") on A and repeated the last sentence on B. The 17 s `final_ms` did not reproduce on a rerun.
  - Whisper does emit "..." mid-sentence on natural speech ("Painting, he tells us, is... of a different quality"), so the no-cut-after-ellipsis rule was needed.
- **UI (S3.1-S3.3),** checked with a throwaway Node harness (fake DOM, the real `app.js`, scripted server messages; kept out of the repo like the v0 one): 17 checks pass. They cover the strategy list following the model and its default, the choice remembered per model and sent in `start_turn`, both dropdowns disabled while a turn runs, committed text normal and tentative text dimmed (`<span class="committed">` / `<span class="tentative">`), a partial for another turn or before `turn_started` or after `transcript` ignored, "Transcribing…" not replacing a shown partial, the transcript replacing the partials, review mode starting from the final transcript, markup shown as text, the timings line with strategy / partial count / lag / `final_ms` (null skipped, a zero count shown), and an older server without `stream_strategies`. Four deliberate bugs in `app.js` were each caught. Against the real server (turbo, `trimmed`, the fixture): the page serves the dropdown, `session` lists the strategies, 17 `partial` messages arrive with the expected keys and turn id, and the transcript carries `stream_strategy: trimmed`. S3.4 (the Chrome check with a microphone) was done by hand and passed.
- **Testing:** real model weights stay out of pytest, as in v0. A test that needs NeMo or the weights is skipped when they are missing.
