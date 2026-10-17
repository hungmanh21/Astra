# Design Decisions

## 2026-09-30: v0 is a turn-based voice agent, not the final product
- Reason: get the record → ASR → LLM → reply loop working and build the seams (transport, ASR interface, LLM interface) that v1 reuses for streaming
- Constraint: translation and meeting/call use cases are out of scope for v0; README to be updated once v0 lands

## 2026-09-30: Single process, single uv environment
- Reason: least plumbing; one interface (the ASR adapter) instead of two
- Rejected alternative: separate ASR service or per-model subprocess workers (extra interface to build in v0)
- Constraint: if NeMo's PyTorch/CUDA pins cannot coexist with the other packages, move the NeMo adapters into a worker subprocess (decided in M0)

## 2026-09-30: Transcript auto-sends to the LLM, with a review toggle
- Reason: fastest default flow, while still allowing ASR debugging by editing or discarding a transcript
- Rejected alternative: always require confirmation, or drop review mode entirely

## 2026-09-30: Whisper via transformers, vanilla JS frontend
- Reason: fewest moving parts for a baseline; the hard part is audio and WebSocket, not UI
- Rejected alternative: faster-whisper, React/Vite
- Constraint: revisit faster-whisper if Whisper speed matters

## 2026-09-30: 30 s recording cap, one turn at a time, LLM streaming required
- Reason: matches Whisper's 30 s window so no chunking logic is needed; avoids cancellation plumbing (that belongs to v1 barge-in); time to first token needs streaming
- Rejected alternative: longer or uncapped recordings, cancel/queue turns, optional streaming

## 2026-09-30: On failure, keep the user transcript, drop the failed reply
- Reason: an ASR failure adds nothing; an LLM failure keeps the transcript so context is not lost and the user just speaks again
- Rejected alternative: roll back the whole turn, retry button

## 2026-09-30: Nemotron checkpoint id is a config value, with an English fallback
- Reason: Nemotron 3.5 availability and license are unconfirmed
- Constraint: fallback is `nvidia/nemotron-speech-streaming-en-0.6b`; registry key stays `nemotron-3.5-asr-streaming-0.6b`

## 2026-09-30: Acceptance test for ASR is an offline script with a fixture clip
- Reason: repeatable and needs no browser; the reference transcript allows manual comparison
- Rejected alternative: UI-only checks, WER pass/fail threshold in v0

## 2026-09-30: Whisper only for now; NeMo adapters deferred
- Reason: dev hardware is now an RTX 3060 (12 GB), not the H100 the SPEC assumed; NeMo compatibility work (Python version, PyTorch pins) is not needed until the Parakeet and Nemotron adapters (M4)
- Constraint: the "all three models plus local vLLM resident" plan does not fit 12 GB; revisit at M4. Gemini is the practical LLM until then

## 2026-09-30: Frontend built first, against a protocol mock
- Reason: the WebSocket protocol in SPEC 6.3 is the contract, so the UI (and the browser-specific audio risk) can be built and verified before the backend exists
- Rejected alternative: mock inside the frontend JS (cannot verify audio bytes arrive intact), separate dev server or Vite build (contradicts the no-build-step decision)
- Constraint: `frontend/` shares no code with `backend/`; the WebSocket URL defaults to same-origin and can be overridden with `?ws=`. `scripts/mock_server.py` is throwaway and is deleted once `backend/main.py` replaces it

## 2026-09-30: Server rejects turns shorter than 0.25 s
- Reason: push-to-talk taps would otherwise send empty or near-empty audio to ASR; the client has already sent `start_turn` by then, so the server is the one to reject
- Constraint: added to SPEC 6.3.1

## 2026-10-01: Ruff for lint and format, wired into `make check`
- Reason: one fast tool for both linting and formatting; `make check` is what the session workflow in CLAUDE.md runs
- Constraint: rules E, F, I, UP, B, ASYNC; line length 100; `theory/` excluded (learning notes). Python only, so `frontend/` has no linter yet

## 2026-10-01: PROGRESS.md is committed together with the work
- Reason: PROGRESS.md should describe the repo as of the commit that contains it (done, in progress, next), so it is updated before committing and goes in the same commit.
- Constraint: a commit cannot contain its own hash, so "Latest commit" names the previous commit.

## 2026-10-01: Transport seam is a send-side Protocol plus a handler the transport drives
- Reason: `Transport` (send_json, send_bytes) is what the session sends through; the transport's receive loop calls `Session.on_json` / `on_audio_frame`. Session code imports no FastAPI types, so WebRTC can replace the WebSocket later (SPEC 6.3 seam).
- Constraint: in M1 the session answers a valid `end_turn` with an "ASR is not connected yet" error so the frontend unlocks; T2.7 replaces that with the transcription.

## 2026-10-03: PROGRESS.md is updated after committing and left uncommitted
- Reason: after each work commit, PROGRESS.md is updated to name that commit and the new state, then left modified. The uncommitted diff shows what is new since the last commit, and PROGRESS.md always names a real hash. It is committed together with the next batch of work.
- Rejected alternative: committing PROGRESS.md in the same commit (it could only name the previous commit) or in a separate "Update PROGRESS.md" commit (noise in the history).
- Constraint: supersedes the 2026-10-01 "PROGRESS.md is committed together with the work" entry.

## 2026-10-03: Default LLM is gemini/gemini-3.5-flash-lite
- Reason: cheapest and fastest of the current Gemini line, which suits short voice answers. It is a config value (`llm.model`, overridable with `LLM_MODEL`), never hard-coded. The key is `GEMINI_API_KEY` in `.env`.
- Rejected alternative: `gemini-2.5-flash` (Google lists the 2.5 series for shutdown on 2026-10-16); `gemini-3.5-flash` (slower, costlier, answers are one to three sentences anyway).
- Constraint: the exact model ID is checked for the first time in T3.2 against the real API. If it is wrong, change `config.yaml` only.

## 2026-10-03: M3 history is trimmed by an estimate, and one WORKING stage covers ASR and LLM
- Reason: the history budget is a safety net, so `estimate_tokens` (about 4 characters per token) is enough and avoids a tokenizer download. The session keeps one `WORKING` stage from `end_turn` until `llm_done` or `error`; the PLAN T3.4 list of finer states (transcribing, streaming) adds nothing the client can observe.
- Rejected alternative: `litellm.token_counter` (model-specific and may fetch tokenizer files).
- Constraint: trimming drops the oldest messages first, never the newest, and the history always starts with a user message. A failed LLM turn keeps its user message, so two user messages in a row are normal.

## 2026-10-04: Discover the vLLM model from the endpoint (`hosted_vllm/auto`)
- Reason: vLLM runs on another machine and the user gives only its URL. The served model is whatever that server was started with, so naming it in our config would go stale. With `llm.model: hosted_vllm/auto` (any provider prefix, name part exactly `auto`) the client asks `GET <api_base>/models` on the first request and uses the first model listed, keeping the prefix. The result is remembered, a failed lookup is retried next turn, and the turn log shows the real model name.
- Rejected alternative: a separate `discover_model` flag, because one `LLM_MODEL` value is already the whole switch.
- Constraint: `auto` needs `llm.api_base` (checked when settings load). The optional `LLM_API_KEY` is sent as a bearer token to the lookup and to the completion. This relaxes SPEC 6.5 ("model names are config values") in one way: the name may be resolved at run time. Nothing else in the code names a provider or model.

## 2026-10-04: Accept v0 on Gemini; defer the real vLLM test
- Reason: the repo's real contribution is real-time ASR, so v0 stays on the Google API. The vLLM path is built and checked against a fake endpoint, only the real-server run is left.
- Constraint: SPEC section 8 keeps the vLLM box open with this follow-up. NeMo adapters (Parakeet, Nemotron) and the M0 questions (Python version, Nemotron license) are also still open, and v1 streaming ASR is where the project goes next.


## 2026-10-04: Same-origin WebSocket only; the server tells the UI the turn cap
- Reason: browsers do not apply the same-origin policy to WebSockets, so any page the user visits could open `ws://localhost:8000/ws` and, through review mode, send any text to the LLM on the user's key. `/ws` now refuses a connection whose `Origin` does not match its `Host` (no `Origin` = not a browser, allowed). The UI's `?ws=` override is gone, since a crafted link could send the microphone audio to another server. The `session` message now carries `max_turn_seconds`, so the UI auto-stops at the server's cap instead of a hard-coded 30 s.
- Rejected alternative: a configurable list of allowed origins, not needed while browser and backend run on the same machine.

## 2026-10-04: v1 scope: live partials for the same agent, Whisper re-decode before Nemotron
- Reason: smallest real-time step that reuses every v0 seam. Partials show while the button is held; the final transcript still goes to the LLM on release. Whisper re-decode (LocalAgreement-2) proves the protocol, session, coalescing and UI with models that already run, so the NeMo install risk (slow PyPI, Python version, Nemotron license) cannot block the seam. Nemotron native streaming then plugs into a tested interface.
- Rejected alternative: a live transcription mode with no LLM and long-form audio (closer to the meetings vision, but needs long-form decoding; later); Nemotron first (NeMo blocks everything); VAD endpointing (push-to-talk stays for v1).
- Constraint: docs split into `SPEC_v0.md` / `PLAN_v0.md` (accepted, cited by v0 code) and `SPEC_v1.md` / `PLAN_v1.md`. v1 FRs continue at FR-15, v1 tasks use `S` ids.

## 2026-10-04: v1 compares streaming strategies, chosen per turn
- Reason: v1 doubles as an experiment bench. The same model can stream as `chunked` (each chunk decoded once, independently), `redecode` (whole turn re-decoded, LocalAgreement-2), `trimmed` (re-decode since the last settled sentence, with the settled text as prompt) or `native` (model's own streaming, Nemotron). `compare_asr.py --stream --strategy all` and a UI dropdown compare them on latency and WER.
- Rejected alternative: one strategy per config entry (e.g. `whisper-turbo-chunked`), because the registry would load the same weights twice.
- Constraint: strategies come from adapter capabilities (`open_stream` → `native`, `transcribe_segments` → `trimmed`), never from model names. `chunked` is expected to lose accuracy (cut words, no context, Whisper pads every input to 30 s) and is measured, not gated.

## 2026-10-12: Strategy support is read from a constructed, not loaded, adapter
- Reason: the `session` message lists every model's strategies at connect time, and loading weights for that (112 s for large-v3) is not acceptable. `ASRRegistry.peek(name)` builds the adapter without `load()` and `get()` reuses that instance, so the service can ask `isinstance(model, StreamingASRModel)` for free.
- Rejected alternative: a `capabilities` field in config.yaml (a second source of truth that can disagree with the adapter); `issubclass` on the adapter class (not allowed for Protocols with data members like `name`).
- Constraint: adapter constructors must stay cheap (no weights, no GPU). `load()` does the heavy work. `peek` uses its own lock so it never waits behind a model load.

## 2026-10-13: The stream is opened at start_turn, so a cold model loads before the ack
- Reason: partials need a stream from the first frame, and the stream needs the loaded model. Opening it in `start_turn` and acking after keeps the session simple: no buffering of frames while a model loads, and an unsupported strategy or a load failure becomes an `error` with the turn never started (FR-22).
- Rejected alternative: ack at once and open the stream in the background while frames queue up (the first turn on a cold model would feel faster, but the session would need a "stream not ready" state).
- Constraint: the first turn on a cold model now waits at `start_turn`, not after `end_turn` as in v0 (large-v3 loads in ~112 s). Preloading the default model at startup or a "loading" hint in the UI (S3) fixes the feel; until then, warm the model first.

## 2026-10-14: `chunked` decodes the last remainder however short
- Reason: SPEC_v1 dropped a remainder under `partial_min_audio_ms`, but with the shipped values (`chunk_ms` 500, minimum 1000) every remainder is shorter than a chunk, so speech in the last <500 ms of a push-to-talk turn would always be lost. `partial_min_audio_ms` now only gates the first partial of `redecode` and `trimmed`.
- Measured cost (fixture, turbo): the fixture ends in silence, so decoding the remainder makes Whisper invent text: `chunk_ms` 500 appended "Yeah." (WER 66.7 -> 70.8 %), 1000 appended "Thank you." (29.2 -> 37.5 %); 2000 was unchanged (16.7 %, its 1.9 s remainder was kept before too).
- Rejected alternative: keep the minimum for `chunked` and lower its default. The final transcript should not depend on how the last chunk happens to fall.
- Decided to leave it as measured: trailing-silence text is a job for a client-side VAD later (only chunks with real sound would be sent), not for a rule in `chunked`.

## 2026-10-15: S2 defaults: `chunk_ms` 2000, `redecode` stays the default, turbo is the live model
- Reason: `compare_asr.py --stream` at real-time pace on two natural clips (LibriSpeech utterances joined, 29.6 s and 27.2 s) plus the fixture. Turbo, clip A: batch WER 3.3 %; `chunked` 500 / 1000 / 2000 ms: 445 % (one repetition loop) / 30.0 % / 13.3 %; `redecode` 3.3 %, lag 433 ms, `final_ms` 2229; `trimmed` 8.3 %, lag 290 ms, `final_ms` 536. Clip B: batch 1.4 %; `redecode` 1.4 %, `final_ms` 1803; `trimmed` 5.8 %, `final_ms` 476. So `chunk_ms` becomes 2000 and `redecode` stays the default (same final text as batch).
- `partial_interval_ms` 500 and `partial_min_audio_ms` 1000 stay. Interval 250 / 500 / 1000 on turbo `redecode`: 60 / 45 / 28 partials, lag 374 / 433 / 453 ms, same WER. 250 gives 33% more partials for 14% less lag at twice the GPU load; 1000 feels sluggish. Nothing measured argues for a change.
- large-v3 is not recommended for live partials on the RTX 3060: on 30 s turns `redecode` has a lag of 1.0-1.2 s and a `final_ms` of 7.5 s, and `trimmed` is unreliable (its timestamped text sometimes loses punctuation, so it never cuts and the lag grows to 3 s; with a prompt it dropped a whole sentence on clip A and repeated the last sentence on clip B). Turbo with `redecode` is the live setup; large-v3 remains the accuracy option.
- Constraint: `trimmed` costs 4-5 WER points on turbo on natural speech (SPEC_v1 asks for "a small margin"). The causes seen: a hallucinated "Thank you." from the short window left at the end, and a prompt-driven change of punctuation (`is...`, `British?`). It stays opt-in. Not tried: refusing to cut when the leftover window would be short.

## 2026-10-15: `trimmed` never cuts the window below `partial_min_audio_ms`
- Reason: on natural speech `trimmed` ended with a hallucinated "Thank you." and lost 4-5 WER points, because the final decode ran on the short window left after the last cut. A cut now has to leave at least `partial_min_audio_ms` (1000 ms) of audio; if the last sentence end is too late, the one before it is used (`trim_point(..., latest_end_s)`). The limit is computed in whole samples, because a limit in seconds is wrong at the boundary (2.4 - 1.5 is 0.8999... in floats).
- Measured (turbo, clips A and B, minimum 0 / 1000 / 2000 / 3000 ms): WER A 8.3 / 5.0 / 5.0 / 5.0 %, B 5.8 / 1.4 / 0.0 / 0.0 %; `final_ms` A 536 / 1300 / 1563 / 1341, B 476 / 1481 / 2049 / 1569. 1000 ms gives most of the gain; longer costs `final_ms` for nothing on A. Final code: A 5.0 % / 1209 ms, B 0.0 % / 1557 ms. `redecode`: 3.3 % / 2229 ms and 1.4 % / 1803 ms.
- Rejected alternative: a separate config key for the minimum. The reason is the same as for `partial_min_audio_ms` (Whisper invents text on tiny clips), so the stream takes `min_tail_ms` (default `min_audio_ms`) and the service passes nothing.
- Cost: the cut comes later, so `final_ms` is about 0.7-1.0 s above the uncut-tail version, but still below `redecode` and not growing with the turn.

## 2026-10-15: `BatchStream` is deleted
- Reason: since S2.6 every strategy has its own stream, so the v0-like placeholder was only used by its own tests.

## 2026-10-15: `ASRStream.finish` takes the last audio, and Whisper's tokens are capped by the audio length
- Reason (finish): `StreamingTranscription.finish` used to push the pending audio through one more `feed` and then call `finish()`. For `redecode` and `trimmed` that feed is a full decode whose partial is thrown away, so the final waited for two decodes. `finish(audio)` hands the audio over; the stream adds it and decodes once. Turbo, 30 s clips, `final_ms` before -> after: `redecode` 2229 -> 1217 (clip A) and 1803 -> 1469 (B); `trimmed` 1209 -> 691 and 1557 -> 892. WER unchanged. The wait for a decode already in flight stays, so the saving is one decode, not half.
- Consequence: a failed last decode is no longer swallowed and retried by `finish`; it raises (the session turns it into an `asr_error`). `ChunkedStream.finish(audio)` feeds it like before, so chunks stay exactly `chunk_ms`.
- Reason (token cap): `chunked` at 500 ms hit a Whisper repetition loop ("I'm sorry" over a hundred times in one chunk) because generation was only limited by the 448-token context. `max_new_tokens` is now `ceil(12 x seconds) + 16`, at most `448 - 8 - prompt tokens`. The same chunk now stops after five repeats (WER on clip A 445 % -> 93 %; the rest is the garbling of tiny chunks). A full 224-token prompt on a 29.6 s window still decodes. It applies to every Whisper call, including the plain v0 batch decode.
- Rejected alternative (finish): skipping the flush feed when nothing is in flight cannot work, because `StreamingTranscription` does not know whether the stream decodes on every feed.
- Risk (cap): a very fast speaker (over 12 tokens per second sustained) would be cut off. English speech is 4-8, and the clip is capped at 30 s.


## 2026-10-17: NeMo installs in the same env on Python 3.13; the venv lives off `/mnt/d`
- Reason: `nemo-toolkit[asr]` 3.0.0 resolves on Python 3.13 without touching torch (2.14.0+cu130) or transformers (5.17.0); only `fsspec` and `packaging` step back a little. One process imports `nemo.collections.asr`, transformers and litellm, and CUDA is available. So `.python-version` stays 3.13 and the worker-subprocess fallback (SPEC_v0 6.7) is not needed.
- Reason (venv): the Makefile sets `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/astra` by default. A venv on `/mnt/d` (WSL2 9p mount) is dozens of times slower for thousands of small files. Plain `uv` commands need the same variable exported (README).
- Cost: 70 more packages (wandb, tensorboard, lhotse, lightning...), imported only when the Nemotron adapter loads. NeMo prints some OneLogger telemetry notices at import; telemetry is disabled.

## 2026-10-17: Nemotron runs through Transformers, not NeMo; the English checkpoint is the default
- Reason: `transformers` (already a dependency, 5.17) loads both Nemotron checkpoints (`AutoModelForRNNT`) and has cache-aware streaming `generate()`. The S4.3 spike streamed the fixture at 45-87 ms per chunk, 2.6 GB peak, with text equal to batch. No NeMo import, 70 packages fewer, same loading style as the Whisper adapter. `nemo-toolkit` is removed again (it was added and committed in S4.2, which stays as the record that it installs on 3.13).
- Reason (checkpoint): `nvidia/nemotron-speech-streaming-en-0.6b` read the fixture exactly where the multilingual 3.5 gave "graved" / "Ithaka", and its smallest lookahead is 1 (160 ms) against 3 (320 ms). v1 is English only. The registry key is `nemotron-streaming-en-0.6b`; the SPEC key `nemotron-3.5-asr-streaming-0.6b` would have named the wrong model. The 3.5 checkpoint works with the same adapter through a second config entry.
- Rejected alternative: NeMo `conformer_stream_step` with explicit cache state (what the SPEC assumed). Its audio buffer normalizes the whole audio first, so live feeding needs extra work, and it was not tried on these checkpoints.
- Constraint: the model object holds the streaming state, so the adapter allows one stream or batch decode at a time per model and raises `ASRError` on overlap instead of deadlocking the single ASR worker thread. Fine for a single-user v1.
- Detail: the last word of a native partial is tentative (it may still be growing, "Sir Fred"), the rest is committed; SPEC 6.4 said "no tentative text". `finish` commits everything.
- Licenses: English NVIDIA Open Model License, 3.5 OpenMDW-1.1; both fine for a personal project.

## 2026-10-17: Nemotron runs in float16; all three models stay resident
- Reason: on the fixture fp16 and bf16 gave the same text as fp32 (batch and stream WER 0.0 %) at half the weights (1.2 GB against 2.4 GB). With fp32, `whisper-large-v3`, turbo and Nemotron together used 11.9 GB of the 12.3 GB card (other programs take about 3.8 GB); with fp16 10.3 GB. All three fit, so v1 has no eviction logic (PLAN_v1 S4.6).
- Constraint: one clip only; S5 checks fp16 on longer clips and `dtype: float32` in `config.yaml` is the way back.
