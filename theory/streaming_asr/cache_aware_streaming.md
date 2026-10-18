# Cache-aware streaming ASR: a gentle introduction

The model family behind `native` streaming in v1 (PLAN_v1 S4): Nemotron Speech Streaming, a **cache-aware FastConformer-RNNT**. This note explains the idea from scratch. It assumes you have read the contrast table at the top of [reading_list.md](reading_list.md) and know roughly what a Conformer encoder and an RNN-T are.

Sources: [Stateful Conformer with Cache-based Inference for Streaming ASR](https://arxiv.org/abs/2312.17279) (Noroozi et al., NVIDIA; "the paper" below), the [Nemotron model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) and NVIDIA's [HF blog post](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents). Numbers are quoted from these; where I am inferring, I say so.

---

## 1. The one-paragraph version

An ordinary (offline) ASR encoder looks at the whole utterance, past and future, to encode each frame. That is great for accuracy and terrible for live use: every new chunk of audio would change every earlier encoder output. A cache-aware model fixes this in two steps. **(a)** It is *trained* with a restricted view: each frame may only see a bounded amount of the past and a small, fixed amount of the future. **(b)** Because of that restriction, the work done for old audio never changes, so at inference each layer **keeps its past activations in a cache** and every new chunk of audio goes through the encoder exactly once.

---

## 2. Motivation: why ordinary models are hard to cache

Suppose you want to stream an offline model. The obvious thing is to feed it audio as it arrives. Why can't you just keep the old encoder outputs and compute only the new ones? Because an offline Conformer depends on the future in **four** places:

| Where | Why it breaks caching |
|---|---|
| **Self-attention** | Bidirectional: frame *t* attends to every frame, including frames after it. New audio changes the output at *t*. |
| **Convolution** | Conformer blocks have a depthwise convolution with symmetric padding, so frame *t* mixes in frames on its right. |
| **Feature normalization** | Offline pipelines normalize the mel features with the mean and variance of the whole utterance, which are unknown until the audio ends. (The paper drops this step, and also swaps batch norm for layer norm, which works per frame.) |
| **Subsampling** | The convolutions that shrink the frame rate also look at neighbours on both sides. |

Change one frame at the input and, through 24 stacked layers, almost every encoder output moves. There is nothing safe to cache.

### What people do instead: buffered streaming

The workaround is **buffered streaming**: take the offline model and feed it overlapping windows (new chunk, plus some past audio, plus a bit of future audio), then stitch the outputs. This is what `redecode` and `trimmed` do with Whisper, and what NeMo's `FrameBatchASR` does with any model. It has two costs:

1. **Wasted compute.** The overlap is encoded again at every step. The paper's own words: these contexts "would result in re-computation and waste of compute".
2. **Train/inference mismatch.** The model was trained on full utterances, but at inference it sees windows with ragged edges. Frames at the window edge lack the context they were trained with, so accuracy drops. In the paper's LibriSpeech RNN-T result, the buffered model gets 11.3 % WER at 2 s latency, against 5.0 % for the same architecture run offline.

> Think of it as a book club that re-reads the whole book from page 1 each time a new page is written, because the meaning of every earlier page might change.

---

## 3. What the model is trying to fix

Three goals, all at once:

1. **Process each frame once.** Cost per step should not depend on how long the person has been talking.
2. **Bounded, predictable latency.** The delay should be a design knob (the chunk size), not a side effect of window size and agreement rules.
3. **No train/inference gap.** What the model sees at inference should be exactly what it saw in training.

The rest of the note shows how one idea, *train with the restriction you will use at inference*, delivers all three.

---

## 4. How the encoder is made streamable

Everything in the encoder that looked right is made unable to look further right than allowed.

### 4.1 Causal everything, except one controlled window

- **Convolutions are causal.** Left padding of `k-1`, no right padding, including the subsampling convolutions. A frame only mixes in frames at or before it.
- **Normalization that needs the whole utterance is removed.** Per-utterance feature normalization is dropped, and batch norm becomes layer norm (statistics per frame).
- **Self-attention gets a mask** that controls how far left and right a frame may look. This is where the latency lives.

### 4.2 The attention mask: three options

The paper compares three ways to limit attention:

| Option | A frame may attend to | Effective look-ahead |
|---|---|---|
| **Zero look-ahead** | the past only | 0 ms (lowest latency, worst accuracy) |
| **Regular look-ahead** | the past and *M* future frames, in every layer | *M* frames × *N* layers: it **stacks up** with depth |
| **Chunk-aware look-ahead** | its whole *chunk*, plus a limited number of past chunks | just the chunk size: it does **not** grow with depth |

Why regular look-ahead stacks: if layer 1 lets frame *t* see *t+M*, and layer 2 lets frame *t* see layer 1's output at *t+M*, which itself saw *t+2M*, then after *N* layers frame *t* is effectively informed by frames up to *t+N·M*. The paper's example for a regular setup comes to 1360 ms.

Chunk-aware avoids the stacking. Audio is cut into chunks, and every frame in a chunk sees every other frame in that chunk, plus the left context. Nothing outside the chunk to the right is visible in any layer, so the look-ahead is the chunk size, whatever the depth. The paper finds that chunk-aware is both lower-latency and more accurate than regular look-ahead (LibriSpeech test-other, RNN-T at 1360 ms: 6.3 % vs 7.1 % WER).

This is the first half of the Nemotron settings you saw in the reading list: `att_context_size = [left, right]`, with 70 frames of left context and 0, 1, 6 or 13 frames of right context. Since a chunk is its first frame plus the right context, the chunk is 1, 2, 7 or 14 frames. One frame is 80 ms (see 4.4), so chunks are 80, 160, 560 and 1120 ms.

```
time ─────────────────────────────────────────────────────────►

frames:  ... [ c-2 ][ c-1 ][ c0 ] [ c1 ][ c2 ] ...
                              ▲
                  the chunk being processed now

frames in c0 can see:  left context (bounded, e.g. 70 frames)  +  all of c0
frames in c0 cannot see:  c1, c2 ...   (not arrived yet)
```

### 4.3 Why this makes caching possible

Here is the central insight. With a chunk-aware mask, **a frame never depends on a chunk to its right**. So once chunk *c-1* has been processed, its activations at every layer are *final*: nothing that arrives later can change them. Final values are cacheable.

Compare with 2: in an offline model nothing is ever final until the audio ends.

### 4.4 What the cache holds

Each encoder layer keeps its own small cache. Per the paper:

| Cache | Contents | Size |
|---|---|---|
| **Attention cache** (one per layer) | The layer's activations for the last `left` frames, from which the keys and values are recomputed or reused | Grows from empty up to the left-context size (70 frames for Nemotron), then the oldest entries are dropped |
| **Convolution cache** (one per layer) | The last `K-1` activations for a depthwise convolution with kernel size `K` | Fixed; starts as zeros |
| **Subsampling** | No cache, instead the last few **mel frames** of the previous chunk are attached to the new chunk | A handful of frames |
| **Decoder state** | The RNN-T prediction network's hidden state and the last emitted token | Tiny |

Two things worth noticing:

- **Memory per stream is bounded.** Layers × (70 frames + a few) × hidden size. It does not grow with the length of the talk. That is why a server can hold hundreds of streams at once.
- **Frames are 80 ms** because FastConformer subsamples 8× (mel frames are 10 ms; 4× subsampling would give 40 ms). Fewer, longer frames mean fewer positions for attention to process and a smaller cache.

---

## 5. How it is trained

### 5.1 Train the way you infer

Training does **not** use caches. It runs on whole utterances in parallel, as any Transformer does, with the *same attention mask and causal convolutions* that inference will use. The cache is purely an inference-time trick that reproduces, step by step, what the mask does in one big pass. The paper states that streaming output matches processing the full audio in one pass.

That is how the train/inference gap disappears. A buffered system uses a model trained for one situation (full context) in another (windows). A cache-aware system is trained for the situation it runs in.

### 5.2 Multi-look-ahead training: one model, several latencies

If you trained a separate model for each chunk size you would have four checkpoints. Instead the paper picks the context size **at random for every batch** during training. One model learns to work under every size, and you choose at inference time. This is why the Nemotron card offers 80 / 160 / 560 / 1120 ms from one checkpoint ("without retraining").

It even helps: in the paper, the multi-look-ahead RNN-T model beat single-look-ahead models at each latency (LibriSpeech test-other: 5.5 % vs 5.9 % at 240 ms, 5.2 % vs 5.4 % at 520 ms).

### 5.3 Hybrid CTC + RNN-T on one encoder

The paper trains one encoder with *two* decoders, a CTC head and an RNN-T head, with a combined loss `0.3 × L_ctc + L_rnnt`. Either decoder can be used at inference. The CTC head speeds up convergence of the shared encoder. The paper also applies FastEmit (λ = 0.005) so that the RNN-T emits tokens early instead of drifting late (see reading list, section 3, question 3). Nemotron uses the RNN-T head.

### 5.4 What Nemotron was trained on

| | Paper's models | Nemotron Speech Streaming (en, 0.6B) |
|---|---|---|
| Size | ~114M parameters | ~600M parameters, 24 encoder layers |
| Data | LibriSpeech; NeMo ASRSET 3.0 (~26k hours public speech) | ~530k hours: NVIDIA Riva set (~250k h US English) plus Granary English (YouTube-Commons 109.5k h, YODAS2 102k h, LibriLight 49.5k h, Mosel 14k h), plus LibriSpeech, Fisher, Switchboard, WSJ, VCTK, VoxPopuli, MLS, Common Voice, People's Speech, AMI |
| Labels | human transcripts | a mix of human and ASR-generated transcripts |
| Output | lowercase words | **text with punctuation and capitalization** |

Caveats: the model card does not describe Nemotron's loss or how it samples context sizes. A comment reply by the blog's author says it was trained in a single stage, and that Granary's pipeline produced the pseudo punctuation and capitalization labels. That Nemotron follows the paper's multi-look-ahead recipe is my inference from the card's "runtime-selectable latency modes" claim, not something the card states.

---

## 6. How it runs at inference

For Nemotron with the 560 ms setting (`right = 6`, so a chunk is 7 frames):

```
  audio arrives (16 kHz PCM)
        │  wait until a full chunk (560 ms) is buffered
        ▼
  mel features for the chunk  (+ a few mel frames kept from the previous chunk)
        │
        ▼
  subsampling (8×)  ──►  7 encoder frames
        │
        ▼
  for each of the 24 layers:
        attention over [ cached 70 frames  |  these 7 frames ]
        convolution using the cached K-1 frames
        update this layer's caches (drop the oldest)
        │
        ▼
  7 final encoder frames
        │
        ▼
  RNN-T greedy decoding, continuing from the previous decoder state
        │
        ▼
  new tokens: committed forever, never revised
```

A few consequences:

- **Cost per step is constant.** Step 100 costs the same as step 1. For `redecode` the cost per step grows with the turn. Our S4.3 spike measured 45 to 87 ms per chunk on the RTX 3060 (DECISIONS.md, 2026-10-17).
- **Latency is chunk size plus compute.** A word that falls at the start of a chunk waits for the chunk to fill, so the worst-case wait is about the chunk length. The paper's "latency" is this algorithmic delay only; it ignores network compute time, which is why it is quoted as a property of the configuration.
- **Output is final.** The cache means old frames do not change, and greedy RNN-T decoding never goes back. This is why `native` only ever reports `committed` text (SPEC_v1 6.4). There is no agreement rule and no `tentative` text.
- **The end of the turn.** There is no future audio to fill the last chunk. The adapter pads or flushes it (`ASRStream.finish`), so `final_ms` is one short step, not a full re-decode.

---

## 7. What you get: the latency/accuracy dial

Nemotron on the HF OpenASR leaderboard (average WER %, no punctuation or case; from the model card):

| Chunk | Right context | Avg WER | LibriSpeech other | AMI (meeting) |
|---|---|---|---|---|
| 1.12 s | 13 frames | 6.93 | 4.84 | 11.73 |
| 0.56 s | 6 frames | 7.07 | 5.07 | 11.88 |
| 0.16 s | 1 frame | 7.67 | 5.57 | 14.71 |
| 0.08 s | 0 frames | 8.43 | 6.01 | 18.29 |

Reading the table:

- More look-ahead is better, with **diminishing returns**: going from 560 ms to 1120 ms buys 0.14 WER points, going from 80 ms to 160 ms buys 0.76.
- **Conversational audio suffers most** from small chunks (AMI goes from 11.73 to 18.29), because overlapping speech and short words need context from the right to disambiguate.
- A sensible default for a voice agent is 160 ms to 560 ms. Our adapter's smallest look-ahead is 1 frame (160 ms, DECISIONS 2026-10-17); varying it is an easy experiment for the bench.

Efficiency (NVIDIA's claims, treat as vendor numbers): 560 concurrent streams on one H100 at 320 ms chunks, against 180 for their baseline. A reader of the blog pointed out that the baseline was a larger model, so not all of the 3× is down to caching. The qualitative point stands anyway: a bounded cache and no recomputation is the property that makes many-streams-per-GPU possible.

---

## 8. Limits and gotchas

- **Fixed look-ahead is a hard floor on latency.** You cannot go below the chunk length however fast the GPU is.
- **Accuracy at small chunks is lower** than offline models (Whisper large-v3 sees the whole turn). Streaming costs accuracy; the question for the bench is how much.
- **Left context is bounded** (70 frames = 5.6 s). Something said 10 s ago is not directly visible to the encoder, though the RNN-T prediction network still carries a summary of earlier words.
- **Not a free lunch for training.** The model must be trained this way. You cannot take Whisper, or any offline checkpoint, and add caches to it.
- **English only** for the checkpoint we default to (`nvidia/nemotron-speech-streaming-en-0.6b`).

---

## 9. Where this shows up in Astra

| In the repo | Relation |
|---|---|
| `native` strategy (SPEC_v1, PLAN_v1 S4) | Runs this model through its streaming interface; every partial is `committed`. |
| `chunked`, `redecode`, `trimmed` | Buffered streaming for Whisper. The comparison in the bench is buffered vs cache-aware, made concrete. |
| Nemotron adapter (S4.4, DECISIONS 2026-10-17) | Loads the checkpoint with `transformers` and uses its cache-aware streaming `generate()`, so the caches described in 4.4 live inside the library rather than in our code. |

## 10. Check yourself

Try these before looking at the paper; answers are all above.

1. Name two reasons that an offline Conformer cannot reuse its old encoder outputs when new audio arrives, apart from attention.
2. Why does chunk-aware attention not accumulate look-ahead across layers, while regular look-ahead does?
3. What is cached per layer, and why is memory per stream bounded?
4. Training does not use caches at all. Why is the cache-aware model still free of a train/inference mismatch?
5. Why can one Nemotron checkpoint serve four chunk sizes?
6. Where does the accuracy cost of a small chunk come from, and which kind of audio suffers most?

## 11. Next steps for reading

- The paper itself, sections on chunk-aware attention and the cache diagrams: [arXiv 2312.17279](https://arxiv.org/abs/2312.17279).
- The code: NeMo's `setup_streaming_params` and `get_initial_cache_state` in `conformer_encoder.py`, and `conformer_stream_step` in `mixins.py` (reading_list.md, section 9.1).
- Contrast with another way to make a Conformer streamable: Emformer's memory bank (reading_list.md, section 6).
