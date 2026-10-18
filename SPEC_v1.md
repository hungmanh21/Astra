# Voice Agent v1: Live Partial Transcripts

**Status:** Accepted (2026-10-18)
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
- `session` gains `"stream_strategies": {"<model>": {"supported": ["redecode", "chunked", "trimmed"], "default": "redecode"}, ...}` and `"release_tail_ms": 400` (from `limits.release_tail_ms`, 0 to 3000, default 400): after the button is released the UI keeps recording this long before it sends `end_turn`. People let go as they finish the last word, and the end of it is still on its way from the microphone; without the tail the last word was often cut off (found in S5.2). The 30 s cap still applies. A server that does not send the field means 0 (stop at once).
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
    # final transcript of the whole turn; `audio` = the last samples, never fed
    def finish(self, audio: np.ndarray | None = None) -> TranscriptResult: ...


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
- **`ChunkedStream`** (any `ASRModel`): every `chunk_ms` of new audio is transcribed on its own, once, and its text is appended to `committed` (there is no tentative text). `finish` transcribes the last remainder, however short, and returns the joined text (a dropped remainder lost the last words of every turn). Words cut at chunk edges and the lack of context are the point of the experiment.
- **`RedecodeStream`** (any `ASRModel`): it keeps the whole turn's audio. A `feed` re-transcribes everything once at least `partial_min_audio_ms` of audio exists. Committed text uses **LocalAgreement-2**: the words two decodes in a row agree on, from the start of the text, beyond what is already committed. The rest of the latest decode is tentative. `finish` runs one full decode, so the final transcript equals v0's.
- **`TrimmedStream`** (`SegmentedASRModel`): like `RedecodeStream`, but it only decodes the audio after the last trim point, and passes the committed text as the prompt. When the committed text covers a whole segment that ends in sentence punctuation, the audio up to that segment's end is dropped and the trim point moves. A cut must leave at least `partial_min_audio_ms` of audio in the window (an earlier sentence end is used if the last one is too late), because Whisper invents text such as "Thank you." on a window of about a second or less; a trailing-off "..." is never a sentence end. Decodes stay short however long the turn gets. `finish` decodes only the remaining tail and returns committed text plus the tail (this approach comes from the `whisper_streaming` project).
- **Coalescing (FR-20):** each turn has one `StreamingTranscription` with a pending-audio buffer and at most one decode in flight. When a decode returns and at least `partial_interval_ms` of new audio is pending, all of it goes into the next `feed`. `end_turn` waits for the decode in flight, feeds the remainder, then calls `finish`.
- **Native streaming (Nemotron):** the adapter runs the checkpoint through 🤗 Transformers (`AutoModelForRNNT`, cache-aware streaming `generate()`), not NeMo (DECISIONS 2026-10-17). It buffers fed audio into the model's chunk size, feeds each full chunk once and reads the tokens decoded so far. Those tokens never change, so every complete word is committed; the last word may still be growing ("Sir Fred" before "Sir Frederick"), so it is the tentative text until a following word starts. `finish` appends 1.5 s of silence to the audio (the model emits the last word late: a turn cut right at the last sound ended in "Ith" for "Ithaca", and a quick 1 s "Is my English good?" lost "good" until 1.5 s had gone through; batch `transcribe` does the same), pads the last chunk, flushes it and commits everything. The chunk size (latency vs accuracy) is the `lookahead_tokens` config option, in 80 ms units. One stream (or batch decode) at a time per model: the model object holds the streaming state, so an overlapping call fails with an `ASRError`.
- One ASR worker thread, as in v0. Partial decodes of one session can delay another session's decode. This is accepted for a single-user v1.

### 6.5 Config (extends SPEC_v0 6.2)

```yaml
asr:
  streaming:
    default_strategy: redecode # for models without native streaming
    partial_interval_ms: 500   # least new audio between two partial decodes (redecode, trimmed)
    partial_min_audio_ms: 1000 # no partial before this much audio (Whisper invents text on tiny clips)
    chunk_ms: 2000             # chunk length for the `chunked` strategy (S2.7: 500 and 1000 garble Whisper)
  models:
    nemotron-streaming-en-0.6b:
      adapter: backend.asr.nemotron.NemotronASR
      model_id: nvidia/nemotron-speech-streaming-en-0.6b   # also runs: nvidia/nemotron-3.5-asr-streaming-0.6b
      options:
        lookahead_tokens: 6    # 0, 1, 6 or 13 (3.5: 0, 3, 6, 13); chunk = (n + 1) x 80 ms
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
| Nemotron backend | 🤗 Transformers, in-process, same env | Same loading as Whisper, no NeMo import, verified streaming in S4.3 | NeMo (installs on 3.13 but was not needed; SPEC_v0 6.7 fallback is moot) |

## 8. Acceptance Criteria

- [x] Holding the button with `whisper-large-v3-turbo`, I see my words appear while speaking. Committed text never changes. Release gives the final transcript and the reply as in v0. *(Checked in Chrome, S3.4: turbo and large-v3 with `chunked`, `redecode` and `trimmed`, review mode off and on, and the 30 s auto-stop.)*
- [x] The same works on `whisper-large-v3` (partials may lag more) and on Nemotron, with no session code that names a model. *(large-v3 in S3.4; Nemotron tried in Chrome after S4.4: fast, text decent. `session.py`, `protocol.py`, `streaming.py` and `service.py` name no model; only comments in `strategies.py` mention Whisper.)*
- [x] I can pick the strategy in the UI. Each Whisper model runs `chunked`, `redecode` and `trimmed`, and Nemotron runs `native`, with no restart. *(The Whisper strategies and Nemotron were each tried in Chrome. Switching between models, strategies and review mode in one server run is checked by `tests/test_acceptance.py`, S5.2. The `session` message lists the right strategies per model, checked on a clean clone.)*
- [x] A partial decode failure (forced in a test) leaves the turn working with a final transcript and no user-facing error. *(`test_a_failing_partial_decode_does_not_break_the_turn` and the streaming tests; with `native` a failed chunk ends the model's stream, so that turn gets an ASR error, see Known Issues.)*
- [x] `compare_asr.py --stream` on the fixture prints, per model and strategy, the partial timeline, `partial_lag_ms`, `final_ms`, the final transcript and its WER against the reference. Every combination returns non-empty text. *(S5.1: 3 models x 3-4 strategies on four clips, all non-empty.)*
- [x] `llm_done.timings` and the turn log carry `stream_strategy`, `partial_count`, `partial_lag_ms` and `final_ms`. A comparison table (model × strategy: latency and WER) is recorded (PLAN_v1 S5).
- [x] All v0 tests still pass. Review mode, the 30 s cap and failure handling behave as in v0. *(618 tests pass; review mode and the 30 s cap were checked in Chrome in S3.4. A forced LLM failure in the middle of a multi-model session is in `tests/test_acceptance.py`.)*

### Measured comparison (PLAN_v1 S5.1)

RTX 3060, fp16 (Nemotron fp16 too), `compare_asr.py --stream --strategy all`, real-time replay, partial interval 500 ms, `chunk_ms` 2000, `partial_min_audio_ms` 1000, Nemotron `lookahead_tokens: 6`. Measured 2026-10-17. "first" is when the first partial arrived from the start of the audio; "lag" is the median `partial_lag_ms`; "final" is `final_ms` (the batch rows are the v0 `asr_ms`). One run each; differences of a word of WER or ~10 % of a time are noise.

**5 s utterance** (`short5`, one LibriSpeech utterance, 12 words)

| Model | Strategy | Partials | first ms | lag ms | final ms | WER % |
|---|---|---|---|---|---|---|
| `whisper-large-v3-turbo` | batch | - | - | - | 275 | 0.0 |
| | `chunked` | 2 | 2250 | 242 | 185 | 16.7 |
| | `redecode` | 7 | 1204 | 259 | 550 | 0.0 |
| | `trimmed` | 7 | 1249 | 273 | 607 | 0.0 |
| `whisper-large-v3` | batch | - | - | - | 812 | 0.0 |
| | `chunked` | 2 | 2795 | 614 | 353 | 8.3 |
| | `redecode` | 6 | 1421 | 610 | 1165 | 0.0 |
| | `trimmed` | 5 | 1479 | 685 | 1025 | 0.0 |
| `nemotron-streaming-en-0.6b` | batch | - | - | - | 246 | 0.0 |
| | `chunked` | 2 | 2177 | 206 | 63 | 0.0 |
| | `redecode` | 8 | 1054 | 160 | 301 | 0.0 |
| | `native` | 6 | 1549 | 86 | 121 | 0.0 |

**28-30 s turns** (`longA`: three LibriSpeech utterances, 28.1 s; `longB`: one 29.7 s utterance with hard proper names; cells are `longA / longB`; the fixture, 9.9 s, lies between the two and gave the same ranking)

| Model | Strategy | Partials | first ms | lag ms | final ms | WER % |
|---|---|---|---|---|---|---|
| `whisper-large-v3-turbo` | batch | - | - | - | 782 / 859 | 3.0 / 10.3 |
| | `chunked` | 13 / 14 | 2272 / 2219 | 304 / 253 | 311 / 192 | 16.7 / 25.0 |
| | `redecode` | 42 / 46 | 1200 / 1186 | 468 / 491 | 846 / 1063 | 3.0 / 10.3 |
| | `trimmed` | 42 / 49 | 1237 / 1220 | 434 / 483 | 1213 / 1178 | 1.5 / 10.3 |
| `whisper-large-v3` | batch | - | - | - | 3594 / 3681 | 1.5 / 8.8 |
| | `chunked` | 13 / 14 | 2929 / 2733 | 555 / 566 | 998 / 417 | 6.1 / 17.6 |
| | `redecode` | 18 / 18 | 1383 / 1428 | 1247 / 1261 | 4896 / 4410 | 1.5 / 8.8 |
| | `trimmed` | 21 / 17 | 1437 / 1477 | 1203 / 1405 | 3134 / 3891 | 1.5 / 11.8 |
| `nemotron-streaming-en-0.6b` | batch | - | - | - | 754 / 787 | 1.5 / 13.2 |
| | `chunked` | 14 / 14 | 2123 / 2129 | 191 / 232 | 57 / 203 | 3.0 / 13.2 |
| | `redecode` | 48 / 53 | 1616 / 1061 | 316 / 375 | 670 / 831 | 1.5 / 13.2 |
| | `native` | 38 / 42 | 2086 / 2095 | 99 / 96 | 83 / 87 | 1.5 / 13.2 |

`longB`'s 9-13 % comes from the clip, not the strategy: proper names ("Linnell", "Birket Foster", "idylls") and "Mr." against "mister" in the reference. Every model makes them, and each model's streaming strategies mostly match its own batch row.

**Nemotron `native`, lookahead and precision** (same four clips; `la` is `lookahead_tokens`, chunk = (la + 1) x 80 ms; cells are `short5 / fixture / longA / longB`)

| Setup | Partials | first ms | lag ms | final ms | WER % |
|---|---|---|---|---|---|
| fp16, la 1 | 7 / 15 / 42 / 48 | 1139 / 1638 / 1662 / 1631 | 131 / 140 / 132 / 130 | 188 / 255 / 253 / 107 | 0 / 0 / 0 / 14.7 |
| fp16, la 6 (default) | 6 / 14 / 38 / 42 | 1609 / 1580 / 2113 / 2099 | 101 / 95 / 100 / 100 | 260 / 57 / 51 / 105 | 0 / 0 / 1.5 / 13.2 |
| fp16, la 13 | 4 / 8 / 22 / 23 | 1572 / 1627 / 2623 / 2581 | 132 / 117 / 96 / 174 | 119 / 72 / 194 / 115 | 0 / 0 / 0 / 13.2 |
| fp32, la 6 | 6 / 14 / 38 / 42 | 1578 / 1585 / 2160 / 2155 | 86 / 144 / 140 / 80 | 97 / 69 / 49 / 102 | 0 / 0 / 1.5 / 13.2 |

**Baseline (S0.2), v0 batch on the 9.9 s fixture:** `whisper-large-v3` 1499 ms, `whisper-large-v3-turbo` 385 ms. Today's batch rows on the same clip: 1505 ms and 400 ms (Nemotron 357 ms).

**Winners**
- **Fastest final:** Nemotron `native`, 83-121 ms at 5-30 s in the main run, 49-260 ms across the sweep runs (target: under 300 ms for 5 s). It does not grow with the turn. On Whisper, turbo `redecode` and `trimmed` need 0.5-1.2 s and large-v3 3-5 s on a long turn.
- **Steadiest partials (lowest lag):** Nemotron `native`, about 100 ms. Turbo `redecode` is 260-490 ms and rises with the turn; large-v3 is 0.6-1.4 s, so it is not a live model.
- **Most accurate:** large-v3 batch or `redecode` on Whisper (1.5 % and 8.8 %). Nemotron is as good on the clips with plain names (0 % on `short5` and the fixture, 1.5 % on `longA`). `redecode` always equals its model's batch result; `trimmed` is within a word of it on turbo and worse on large-v3 (11.8 %); `chunked` is clearly worst on Whisper (16.7-25 % on turbo) because it cuts words at the chunk edges.
- **Earliest first words:** the Whisper strategies show something at 1.2-1.5 s, which is `partial_min_audio_ms` plus a decode, but the first partial can be wrong ("it." on `longA` with turbo). Nemotron `native` shows real words from 1.5-2.1 s on a 0.5 s first window; lookahead 1 brings that to 1.1-1.7 s at about the same accuracy.

**Notes and hotspots for the next version**
- Lookahead: 1, 6 and 13 give the same text within a word. 1 is first and has the most partials but costs more GPU calls; 13 is coarse (22 partials in 29 s). 6 stays the default; 1 is the choice if words should appear sooner. fp16 and fp32 give the same text and similar times (checked on four clips), so fp16 stays.
- `redecode` and `trimmed` on Whisper pay a full decode per partial, so lag and `final_ms` rise with the turn (turbo `redecode` lag 259 -> 491 ms from 5 s to 30 s). `native` does not.
- The `native` partial keeps its last word tentative, so a 28 s turn shows the committed text one word behind what the model has heard.
- Not measured here: LLM first token and a ~5 s utterance in Chrome (the browser timings line has both; the earlier v0 number was ASR + first token about 2.0 s on large-v3).

## 9. Milestones

See [PLAN_v1.md](PLAN_v1.md):
- S0: docs and v0 baseline.
- S1: streaming seam with a fake model.
- S2: Whisper strategies (`chunked`, `redecode`, `trimmed`).
- S3: UI partials.
- S4: NeMo environment and Nemotron.
- S5: measurements and acceptance.

## 10. Open Questions

- [x] Nemotron checkpoint and license (carried over from SPEC_v0 section 10). Both download without a token. `nvidia/nemotron-3.5-asr-streaming-0.6b` is multilingual (OpenMDW-1.1); `nvidia/nemotron-speech-streaming-en-0.6b` is English only (NVIDIA Open Model License). **Default: the English one** (exact fixture text, smaller lookahead; v1 is English only). Both fit a personal project. The 3.5 checkpoint runs with the same adapter (DECISIONS 2026-10-17).
- [x] Python version vs NeMo (carried over): does NeMo run on 3.13? Yes: nemo-toolkit 3.0.0 resolves and imports on 3.13 next to transformers and litellm (S4.2, DECISIONS 2026-10-17).
- [x] Which chunk sizes the Nemotron checkpoint supports. English: lookahead 0, 1, 6, 13 (chunk 80 ms x (n + 1)); 3.5: 0, 3, 6, 13. 0 hangs in Transformers (its first chunk is 200 samples). Default `lookahead_tokens: 6` until S4.4 compares them on the clips.
- [x] Whether `whisper-large-v3` `redecode` is fast enough to be useful on the RTX 3060, or whether `trimmed` or turbo should be its recommended setup. **No: recommend `whisper-large-v3-turbo` for live partials.** On a 30 s turn large-v3 `redecode` has a `final_ms` of about 7.5 s and a median lag of 1.0-1.2 s, and `trimmed` on large-v3 drops or repeats sentences (DECISIONS 2026-10-15). large-v3 stays available for accuracy (its batch WER is 1.4-1.7 %).
- [x] Which `chunk_ms` makes `chunked` usable at all. **2000** (WER 13.3 % on a 29.6 s clip; 1000 gives 30.0 %, 500 gives a repetition loop and 445 %). Even 2000 is far behind `redecode` (3.3 %).
- [x] Default strategy for models without native streaming: **`redecode`**. Its final transcript equals the batch one (WER 3.3 % and 1.4 % against batch 3.3 % and 1.4 %); `trimmed` has a lower `final_ms` (0.7-0.9 s against 1.2-1.5 s on a 30 s turn, after the `finish(audio)` change; it does not grow with the turn) and, with the minimum-window rule, a WER of 5.0 % and 0.0 % on turbo (against 3.3 % and 1.4 % for `redecode`).
