# Voice Agent v1: Live Partial Transcripts

**Status:** Approved, in progress
**Author:** hhm
**Last updated:** 2026-10-04
**Builds on:** [SPEC_v0.md](SPEC_v0.md). Everything in v0 still holds unless this document changes it. v0 requirements keep their numbers (FR-1 to FR-14); new ones start at FR-15.

## 1. Overview

v1 makes the transcript **stream**. While the user holds the mic button, the words they are saying appear in their chat bubble and grow as they speak. On release, the final transcript replaces the live text and goes to the LLM exactly as in v0.

The voice agent is otherwise unchanged: push-to-talk, one turn at a time, a 30 s cap, review mode, a streamed LLM reply, LiteLLM.

v1 is also an **experiment bench**. The same model can stream in several ways (*strategies*), chosen per turn, so they can be compared on the same audio:

| Strategy | What each partial decodes | Final transcript on release | Works with |
|---|---|---|---|
| `chunked` | only the new chunk (e.g. 500 ms), once; texts are joined | the joined chunk texts (plus the last remainder) | any model |
| `redecode` | the whole turn so far; a word settles once two decodes in a row agree on it | one full decode (same as v0) | any model |
| `trimmed` | the audio since the last settled sentence, with the settled text as a prompt | settled text + a decode of the tail | models that return segment timestamps (Whisper) |
| `native` | each chunk once, carrying the model's own cache | flush of the last chunk | models built for streaming (Nemotron) |

v1 gets there in two steps:

1. **Whisper strategies.** `chunked`, `redecode` and `trimmed` on the Whisper models we already have. No new dependencies. This builds and proves the streaming seam (protocol, ASR interface, session, UI) and gives the first comparison.
2. **Nemotron native streaming.** NVIDIA's cache-aware streaming RNNT, through NeMo, processes each audio chunk once. The final transcript is then ready almost as soon as the user lets go.

**Relation to the project vision:** real-time ASR for live conversations, meetings and calls (see `README.md`). Streaming partials are the first real-time piece. VAD, long-form audio, WebRTC and translation come after v1.

## 2. Goals

- Live partial transcript in the UI while recording.
- A **streaming ASR interface** next to the batch one. Adding a streaming model is still one adapter file plus one config entry, and every batch model streams through the generic strategies.
- Selectable **streaming strategies** (`chunked`, `redecode`, `trimmed`, `native`), compared on accuracy and latency.
- A native streaming adapter (Nemotron) and a measured comparison against the Whisper strategies.
- Streaming latency metrics: how far the partials trail the audio, and how long the final transcript takes after release.

## 3. Non-Goals (v1)

- Voice activity detection, hands-free endpointing, barge-in (push-to-talk stays).
- Recordings longer than 30 s, long-form or sliding-window decoding.
- Sending partials to the LLM early (speculative LLM calls). The LLM still waits for the final transcript.
- Text-to-speech, WebRTC, translation, multilingual, diarization.
- Word timestamps or confidence scores in the UI.
- Everything already listed as a v0 non-goal that this document does not bring in.

## 4. Use Cases

| User | Need | Example scenario |
|------|------|------------------|
| Tester | See that they are being heard | Hold the button, speak, watch the words appear, release and get the reply |
| Developer (us) | Compare streaming strategies and models | Same sentence on `whisper-large-v3-turbo` with `chunked`, `redecode` and `trimmed`, then Nemotron `native`: compare how early the words appear, how often they change, `final_ms` and the final accuracy |
| Developer (us) | Test streaming without a browser | `compare_asr.py --stream` feeds the fixture clip in real time through every strategy and prints the partial timeline, latency and word error rate |

## 5. Requirements

### Functional

- [ ] FR-15: While a turn is recording, the server sends `partial` messages for it. Each one carries the full `committed` text so far and a `tentative` tail.
- [ ] FR-16: Committed text only grows: each `committed` starts with the previous one. Tentative text may change or vanish on the next `partial`.
- [ ] FR-17: The UI shows the partial text in the user's bubble while recording: committed text normally, tentative text dimmed. The final `transcript` replaces it, and it may differ from the partials.
- [ ] FR-18: Partials are best-effort. If a partial decode fails, the turn goes on without partials (the failure is logged, the user sees no error), and the final transcript is produced as in v0. Only a failed final transcript is an `asr_error` (FR-14).
- [ ] FR-19: Every registered ASR model streams. A model with no native streaming uses the generic strategies, with no per-model code in the session.
- [ ] FR-22: The streaming strategy is chosen per turn: `start_turn` may carry `stream_strategy`, and the `session` message lists the strategies each model supports and its default. A strategy a model does not support is rejected like an unknown model (FR-13 style `error`, turn not started).
- [ ] FR-20: The server never builds a backlog of partial decodes. When decoding falls behind, the next decode covers all audio received so far, and intermediate partials are skipped.
- [ ] FR-21: Review mode is unchanged. Partials show while recording, and the final transcript is what the user edits.

### Non-Functional

- **Latency (targets, not gates):** with `whisper-large-v3-turbo`, the first partial within ~1.5 s of the start of speech and `partial_lag_ms` (median) under ~1 s. With Nemotron, `final_ms` under ~300 ms for a 5 s utterance. The v0 target (ASR + LLM first token under ~3 s for 5 s of audio) still holds.
- **Accuracy:** with `redecode`, `trimmed` and `native`, the final transcript is no worse than v0 on the fixture clip for the same model (`trimmed` within a small WER margin). `chunked` is an experiment and is expected to be worse; its numbers are recorded, not gated.
- **Modularity:** a native streaming model = one adapter file + one config entry. Batch models stream with no extra code.
- **Observability:** the per-turn log line gains `stream_strategy`, `partial_count`, `partial_lag_ms` and `final_ms`. It still never holds text.
- **Privacy, compatibility:** unchanged from v0.

## 6. Technical Approach

### 6.1 Architecture

```
Browser ── PCM16 frames ──► Session ── frames ──► StreamingTranscription (one per turn)
   ▲                           │                     │ coalesces pending audio, one decode at a time
   │                           │                     ▼
   │                           │               ASR worker thread
   │                           │                     │ ASRStream.feed(audio) → PartialResult
   └──── partial ◄─────────────┘◄────────────────────┘
   └──── transcript ◄── end_turn → ASRStream.finish() → TranscriptResult → LLM (as v0)
```

### 6.2 Protocol changes (extends SPEC_v0 6.3)

Client → server:
- `start_turn` gains an optional field: `{"type": "start_turn", "asr_model": "...", "review": false, "stream_strategy": "redecode"}`. Missing means the model's default.

Server → client:
- `session` gains `"stream_strategies": {"<model>": {"supported": ["redecode", "chunked", "trimmed"], "default": "redecode"}, ...}`.
- New message:
  JSON `{"type": "partial", "turn_id": "...", "committed": "...", "tentative": "..."}`. Only sent between `turn_started` and `transcript`. Both fields hold full text, not deltas (a 30 s turn is short).

Turn lifecycle: `start_turn` → audio frames (with `partial`* coming back) → `end_turn` → `transcript` → (review) → `llm_delta`* → `llm_done`. A `partial` that arrives after `transcript` is a bug.

### 6.3 Timing definitions (extends SPEC_v0 6.3.2)

`llm_done.timings` gains three values, each null when it does not apply:
- `partial_lag_ms`: for each `partial`, the time from receiving the last audio frame that decode covered to sending the `partial`. The median over the turn. Null if the turn had no partials.
- `partial_count`: number of `partial` messages sent for the turn.
- `final_ms`: from receiving `end_turn` to sending `transcript`. This includes waiting for a partial decode already in progress. For `redecode` it is about `asr_ms` of a full decode; `trimmed` decodes only the tail, and `chunked` and `native` only the last remainder, so they should be much smaller.

`asr_ms` keeps its v0 meaning for the final transcript: time inside `finish()`. The `transcript` message and `llm_done.timings` also carry the `stream_strategy` used.

### 6.4 Streaming ASR interface (extends SPEC_v0 6.4)

```python
@dataclass(frozen=True)
class PartialResult:
    committed: str  # settled text, only ever grows
    tentative: str  # may change on the next partial


class ASRStream(Protocol):
    """One turn's stream. Every call is blocking and runs on the ASR worker thread."""

    def feed(self, audio: np.ndarray) -> PartialResult | None: ...  # new samples since last feed
    def finish(self) -> TranscriptResult: ...  # final transcript of the whole turn


class StreamingASRModel(ASRModel, Protocol):  # optional capability: the `native` strategy
    def open_stream(self, sample_rate: int = 16000, language: str | None = None) -> ASRStream: ...


@dataclass(frozen=True)
class Segment:
    text: str
    start_s: float
    end_s: float


class SegmentedASRModel(ASRModel, Protocol):  # optional capability: the `trimmed` strategy
    def transcribe_segments(
        self, audio: np.ndarray, sample_rate: int = 16000, prompt: str | None = None
    ) -> tuple[list[Segment], float]: ...  # segments with timestamps, latency_ms
```

- The `TranscriptionService` gives the session a stream for any model and strategy: `open_stream(model_name, strategy)`. It knows which strategies a model supports from the capabilities its adapter has: `chunked` and `redecode` for every model, `trimmed` with `SegmentedASRModel`, `native` with `StreamingASRModel`. The default is `native` when supported, else `asr.streaming.default_strategy`. The session never checks which kind it got.
- **`ChunkedStream`** (any `ASRModel`): every `chunk_ms` of new audio is transcribed on its own, once, and its text is appended to `committed` (there is no tentative text). `finish` transcribes the last remainder (if at least `partial_min_audio_ms`, else it is dropped) and returns the joined text. Words cut at chunk edges and the lack of context are the point of the experiment.
- **`RedecodeStream`** (any `ASRModel`): it keeps the whole turn's audio. A `feed` re-transcribes everything once at least `partial_min_audio_ms` of audio exists. Committed text uses **LocalAgreement-2**: the words two decodes in a row agree on, from the start of the text, beyond what is already committed. The rest of the latest decode is tentative. `finish` runs one full decode, so the final transcript equals v0's.
- **`TrimmedStream`** (`SegmentedASRModel`): like `RedecodeStream`, but it only decodes the audio after the last trim point, and passes the committed text as the prompt. When the committed text covers a whole segment that ends in sentence punctuation, the audio up to that segment's end is dropped and the trim point moves. Decodes stay short however long the turn gets. `finish` decodes only the remaining tail and returns committed text plus the tail (this approach comes from the `whisper_streaming` project).
- **Coalescing (FR-20):** each turn has one `StreamingTranscription` with a pending-audio buffer and at most one decode in flight. When a decode returns and at least `partial_interval_ms` of new audio is pending, all of it goes into the next `feed`. `end_turn` waits for the decode in flight, feeds the remainder, then calls `finish`.
- **Native streaming (Nemotron):** the adapter buffers fed audio into the model's chunk size and runs NeMo's cache-aware streaming step for each full chunk. It returns the hypothesis so far as committed (RNNT output for processed chunks does not change), with no tentative text. `finish` pads and flushes the last chunk. The chunk size (latency vs accuracy) is a config option.
- One ASR worker thread, as in v0. Partial decodes of one session can delay another session's decode. This is accepted for a single-user v1.

### 6.5 Config (extends SPEC_v0 6.2)

```yaml
asr:
  streaming:
    default_strategy: redecode # for models without native streaming
    partial_interval_ms: 500   # least new audio between two partial decodes (redecode, trimmed)
    partial_min_audio_ms: 1000 # no partial before this much audio (Whisper invents text on tiny clips)
    chunk_ms: 500              # chunk length for the `chunked` strategy
  models:
    nemotron-3.5-asr-streaming-0.6b:
      adapter: backend.asr.nemotron.NemotronASR
      model_id: nvidia/nemotron-3.5-asr-streaming-0.6b   # or the fallback, see SPEC_v0 6.4
      options:
        chunk_ms: 560          # exact allowed values fixed in PLAN_v1 S4.3
```

### 6.6 Repository layout (additions)

```
backend/asr/
  base.py          # + PartialResult, ASRStream, StreamingASRModel, Segment, SegmentedASRModel
  agreement.py     # LocalAgreement-2 (shared by redecode and trimmed)
  strategies.py    # ChunkedStream, RedecodeStream, TrimmedStream, strategy lookup
  streaming.py     # StreamingTranscription: coalescing, one decode in flight
  nemotron.py      # native streaming adapter (also batch transcribe())
  parakeet.py      # batch adapter (v0 T4.1), streams through RedecodeStream
scripts/compare_asr.py  # + --stream [--strategy ...]: real-time feed of a WAV, partial timeline and WER per model and strategy
```

## 7. Decisions

| Area | Choice | Why | Revisit when |
|------|--------|-----|--------------|
| First streaming model | Whisper strategies, then Nemotron | Proves the whole seam with models that already work and no NeMo risk. Nemotron then plugs into a tested interface | — |
| Streaming approach | Several strategies, chosen per turn | v1 is an experiment bench: compare independent chunks, full re-decode, trimmed re-decode and native streaming on the same audio | Once one strategy clearly wins, the others can go |
| Strategy choice | Per turn (`start_turn`), not per config entry | Comparing strategies needs no second copy of the weights in GPU memory | — |
| Partial stability | LocalAgreement-2 on words | Simple, well known (whisper_streaming), stops committed text from flickering | Partials still change too much |
| Partial format | Full committed + tentative text, not deltas | Turns are ≤ 30 s, so the text is short; easy to render; a lost or skipped partial costs nothing | Long-form audio (after v1) |
| Final transcript | Separate `finish()`; replaces partials | Each strategy decides: `redecode` keeps v0 accuracy, the others trade some of it for a much faster final | — |
| Turn control | Push-to-talk | Keeps VAD out of this step | v2 (VAD, hands-free) |
| NeMo placement | In-process, same env (SPEC_v0 6.7 fallback still applies) | Least plumbing | NeMo pins clash with the env |

## 8. Acceptance Criteria

- [ ] Holding the button with `whisper-large-v3-turbo`, I see my words appear while speaking. Committed text never changes. Release gives the final transcript and the reply as in v0.
- [ ] The same works on `whisper-large-v3` (partials may lag more) and on Nemotron, with no session code that names a model.
- [ ] I can pick the strategy in the UI. Each Whisper model runs `chunked`, `redecode` and `trimmed`, and Nemotron runs `native`, with no restart.
- [ ] A partial decode failure (forced in a test) leaves the turn working with a final transcript and no user-facing error.
- [ ] `compare_asr.py --stream` on the fixture prints, per model and strategy, the partial timeline, `partial_lag_ms`, `final_ms`, the final transcript and its WER against the reference. Every combination returns non-empty text.
- [ ] `llm_done.timings` and the turn log carry `stream_strategy`, `partial_count`, `partial_lag_ms` and `final_ms`. A comparison table (model × strategy: latency and WER) is recorded (PLAN_v1 S5).
- [ ] All v0 tests still pass. Review mode, the 30 s cap and failure handling behave as in v0.

## 9. Milestones

See [PLAN_v1.md](PLAN_v1.md):
- S0: docs and v0 baseline.
- S1: streaming seam with a fake model.
- S2: Whisper strategies (`chunked`, `redecode`, `trimmed`).
- S3: UI partials.
- S4: NeMo environment and Nemotron.
- S5: measurements and acceptance.

## 10. Open Questions

- [ ] Nemotron checkpoint and license (carried over from SPEC_v0 section 10): does `nvidia/nemotron-3.5-asr-streaming-0.6b` download, and does its license fit? Fallback: `nvidia/nemotron-speech-streaming-en-0.6b`. Settled in S4.
- [ ] Python version vs NeMo (carried over): does NeMo run on 3.13? Settled in S4.
- [ ] Which chunk sizes the Nemotron checkpoint supports, and which one to default to. Settled in S4.
- [ ] Whether `whisper-large-v3` `redecode` is fast enough to be useful on the RTX 3060, or whether `trimmed` or turbo should be its recommended setup. Settled in S2.
- [ ] Which `chunk_ms` makes `chunked` usable at all (500 ms is likely too short for Whisper; try 500, 1000 and 2000). Settled in S2.
- [ ] Default strategy for models without native streaming (`redecode` until the S2 numbers say otherwise). Settled in S2.
