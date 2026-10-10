# Streaming ASR: Reading List

Papers, blog posts and production write-ups on **why some ASR architectures stream and others do not**: attention encoder-decoder (AED) vs CTC vs RNN-T vs TDT, how encoders are made streamable, how offline models like Whisper are forced to stream, and what real production systems run. Background for v1 of the app (`SPEC_v1.md`), where we compare the `chunked`, `redecode`, `trimmed` and `native` strategies.

Builds on [../fundamentals/audio_representations.md](../fundamentals/audio_representations.md) (mel spectrograms are what all of these models consume).

## How this file works

Same idea as the fundamentals files, lighter. Each section has its links in a suggested reading order, then a few questions to answer under `> Your answer:` once you have read them. Ask me to review when you are done, and I will fill in `### Finalized notes`.

**Suggested path through the list:** the two surveys' sections on model types (0), then AED (1), CTC (2), transducers (3) and TDT (4) for the core contrast. Then streaming encoders (6), Whisper streaming (7, which is what our S2 strategies copy) and production (8). Section 5 (making attention stream) is optional, for depth. Section 9 is the source code: read 9.2 alongside section 7 before PLAN_v1 S2, and 9.1 before S4.

## The contrast in one table (orientation only; check it against the papers)

| | AED (LAS, Whisper) | CTC | RNN-T / Transducer | TDT |
|---|---|---|---|---|
| Output depends on earlier output tokens? | yes (autoregressive decoder) | no (each frame independent) | yes (prediction network) | yes (like RNN-T) |
| How the decoder sees the audio | cross-attention over the **whole** encoder output | one prediction per encoder frame | one joint step per (frame, token) | like RNN-T, but also predicts how many frames to skip |
| Alignment to time | implicit, learned by attention | monotonic, via blanks | monotonic, via blanks | monotonic, blanks + durations |
| Natural fit for streaming | poor: needs the full utterance to attend over and to know when to stop | good, if the encoder is streaming | good, if the encoder is streaming | good, and faster decoding |
| Example | `whisper-large-v3` (and turbo) in this repo | Wav2Vec2 / HuBERT fine-tunes (HF course, Unit 3) | Nemotron streaming (RNNT), PLAN_v1 S4 | Parakeet TDT, PLAN_v1 S4.5 |

The key point to check while reading: **streamability is mostly about the decoder's dependence on the full input (AED) versus a monotonic, frame-synchronous alignment (CTC, transducers)**. The encoder also has to be made causal or chunked (section 6), whatever the decoder.

---

## 0. Big picture (start here)

- [End-to-End Speech Recognition: A Survey](https://arxiv.org/abs/2303.03329) (Prabhavalkar, Hori, Sainath, Schlüter, Watanabe, 2023, IEEE/ACM TASLP). The most complete overview. Read the modelling sections (CTC, RNN-T, AED) and the part on streaming.
- [Recent Advances in End-to-End Automatic Speech Recognition](https://arxiv.org/abs/2111.01690) (Jinyu Li, Microsoft, 2021/2022). Shorter and very practical, written from production experience; strong on streaming and on why industry moved to transducers.
- [Hugging Face Audio Course, Unit 5: choosing an ASR model](https://huggingface.co/learn/audio-course/chapter5/asr_models) and [Unit 3: CTC architectures](https://huggingface.co/learn/audio-course/chapter3/ctc). An easy, hands-on comparison of CTC (encoder-only) and seq2seq (encoder-decoder) models.

**Questions:**

1. The surveys split end-to-end models into three families. Name them and say, for each, what the output at one step is conditioned on.
   > Your answer:

2. Why did production systems (Google, Microsoft, Amazon) move to transducers for on-device and streaming ASR, while Whisper-style AED models dominate offline benchmarks?
   > Your answer:

### Finalized notes

---

## 1. Attention encoder-decoder (AED): why it does not stream naturally

- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) (Chan, Jaitly, Le, Vinyals, 2015). The classic AED model for speech: a pyramidal encoder (listener) plus an attention decoder (speller).
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Radford et al., OpenAI, 2022). The Whisper paper. Note the fixed 30 s input window and the long-form "sequential decoding" heuristics; both explain behaviours we see in the app (padding to 30 s, invented text on silence).

**Questions:**

1. At each decoding step, an AED decoder attends over the encoder output. What does it need from the *future* of the audio, and what goes wrong if you give it only the first second?
   > Your answer:

2. How does an AED model decide that the transcript is finished (hint: the end-of-sequence token)? Why is that hard when the audio is still arriving?
   > Your answer:

3. Whisper pads every input to 30 s. What does that mean for the cost of transcribing 0.5 s vs 20 s, and for our `chunked` strategy?
   > Your answer:

### Finalized notes

---

## 2. CTC

- [Sequence Modeling with CTC](https://distill.pub/2017/ctc/) (Awni Hannun, Distill, 2017). The best visual explanation of CTC: blanks, collapsing repeats, the alignment lattice and the conditional independence assumption. Read this before the paper.
- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (Graves, Fernández, Gomez, Schmidhuber, ICML 2006). The original paper.
- [Hybrid CTC/Attention Architecture for End-to-End Speech Recognition](https://www.merl.com/publications/docs/TR2017-190.pdf) (Watanabe, Hori, Kim, Hershey, Hayashi, IEEE JSTSP 2017). Combines CTC's monotonic alignment with an attention decoder, the basis of ESPnet. It shows what each side contributes.

**Questions:**

1. What is the blank token for, and how does the collapsing rule turn frame-level outputs into a transcript?
   > Your answer:

2. CTC assumes each frame's output is independent of the others given the audio. Why does that make it easy to stream, and why does it hurt accuracy (spelling, no internal language model)?
   > Your answer:

### Finalized notes

---

## 3. RNN-T / Transducers

- [Sequence-to-sequence learning with Transducers](https://lorenlugosch.github.io/posts/2020/11/transducer/) (Loren Lugosch, blog, 2020). A clear tutorial with code: encoder, predictor and joiner, the alignment lattice, and training memory costs. Read before the paper.
- [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711) (Alex Graves, 2012). The original RNN-T paper.
- [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/abs/2002.02562) (Zhang et al., Google, 2020). Limiting self-attention's left and right context makes a Transformer encoder streamable, with only a small accuracy loss.
- [FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization](https://arxiv.org/abs/2010.11148) (Yu et al., Google, ICASSP 2021). Transducers learn to *delay* their output to see more context. FastEmit pushes them to emit earlier (90th-percentile latency from 210 ms to 30 ms on LibriSpeech). This is a latency problem that only streaming models have.

**Questions:**

1. What do the three parts of a transducer (encoder, prediction network, joint network) each do? How does it differ from CTC, given that it also emits blanks?
   > Your answer:

2. Why is an RNN-T's output for already-processed audio final (no re-decoding needed)? Connect this to the `committed`-only partials of our `native` strategy (SPEC_v1 6.4).
   > Your answer:

3. What is "emission delay" in a streaming transducer, and why would a model trained only for accuracy learn to emit late?
   > Your answer:

### Finalized notes

---

## 4. TDT (Token-and-Duration Transducer)

- [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795) (Xu et al., NVIDIA, ICML 2023). The joint network predicts a token **and** how many frames it covers, so inference can skip frames: up to 2.82x faster than a plain transducer with better accuracy. This is the architecture behind Parakeet TDT.

**Questions:**

1. What does a TDT predict that an RNN-T does not, and why does that let decoding skip frames?
   > Your answer:

2. Most of an RNN-T's steps emit a blank. Why does that make it slow, and how does TDT avoid it?
   > Your answer:

### Finalized notes

---

## 5. Making attention stream (optional, for depth)

- [Monotonic Chunkwise Attention (MoChA)](https://arxiv.org/abs/1712.05382) (Chiu, Raffel, Google Brain, ICLR 2018). Attention restricted to small chunks that move forward monotonically, which allows online, linear-time decoding of an AED.
- [Two-Pass End-to-End Speech Recognition](https://arxiv.org/abs/1908.10992) (Sainath et al., Google, Interspeech 2019). The production compromise: a streaming RNN-T gives the partials, then a LAS (AED) second pass rescores the result at the end, for 17-22% lower WER and under 200 ms extra latency. Compare it with our `redecode` strategy (fast partials, one full decode at release).

**Questions:**

1. MoChA and the two-pass system make AED usable for streaming in two different ways. What does each give up?
   > Your answer:

### Finalized notes

---

## 6. Streaming encoders: the other half of the problem

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., Google, 2020). The encoder most modern ASR models build on.
- [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084) (Rekesh et al., NVIDIA, 2023). 8x downsampling and limited-context attention, 2.8x faster than Conformer. The encoder in Parakeet and Nemotron.
- [Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition](https://arxiv.org/abs/2312.17279) (Noroozi et al., NVIDIA). **Cache-aware streaming**: limited look-ahead and past context, plus cached activations so each frame is processed once, trained the same way it runs so there is no train/inference gap. The method behind Nemotron streaming and our S4.
- [Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition](https://arxiv.org/abs/2010.10759) (Shi et al., Meta, 2020). Meta's streaming encoder: a memory bank for long history and a key/value cache for the left context. Shows another production design.
- [Zipformer: A faster and better encoder for automatic speech recognition](https://arxiv.org/abs/2310.11230) (Yao et al., ICLR 2024). The k2/icefall encoder (next-gen Kaldi), with streaming variants used widely on-device through sherpa-onnx (section 8).

**Questions:**

1. Even a CTC or transducer model cannot stream if the encoder uses full self-attention over the utterance. Why not? What do "look-ahead" (right context) and "left context" mean for latency and accuracy?
   > Your answer:

2. What is the difference between **buffered** streaming (re-run overlapping windows) and **cache-aware** streaming (process each frame once)? Which of our strategies is closest to buffered streaming?
   > Your answer:

3. Nemotron's chunk sizes (80, 160, 560, 1120 ms) trade latency for accuracy. Where in the encoder does that trade-off come from?
   > Your answer:

### Finalized notes

---

## 7. Forcing an offline AED (Whisper) to stream: our S2 strategies

- [Turning Whisper into Real-Time Transcription System](https://arxiv.org/abs/2307.14743) (Macháček, Dabre, Bojar, IJCNLP-AACL 2023 demo) and its code [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming). **LocalAgreement** plus buffer trimming at confirmed sentences, with the committed text as prompt. This is exactly our `redecode` (LocalAgreement-2) and `trimmed` strategies. About 3.3 s latency on long-form speech.
- [Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection](https://arxiv.org/abs/2406.10052) (2024). **AlignAtt**: uses Whisper's cross-attention to see which audio a token is attending to, and stops emitting when it looks at the unfinished edge of a chunk. Plus a small detector for words cut at chunk boundaries. About 1.46% absolute WER loss at 1 s chunks, without fine-tuning.
- [ufal/SimulStreaming](https://github.com/ufal/SimulStreaming) (2025). The successor to whisper_streaming: AlignAtt plus the whisper_streaming interface, about 5x faster than it, and the best system at the IWSLT 2025 simultaneous speech translation task. Paper: [Simultaneous Translation with Offline Speech and LLM Models in CUNI Submission to IWSLT 2025](https://arxiv.org/abs/2506.17077). Relevant later for the translation part of the project vision.

**Questions:**

1. LocalAgreement re-decodes the buffer from the start each time. AlignAtt decides per token whether to emit. Which one wastes more compute, and why?
   > Your answer:

2. Why does whisper_streaming trim the buffer at *sentence* boundaries and pass the confirmed text as a prompt? What would go wrong if it trimmed at arbitrary points?
   > Your answer:

3. Simul-Whisper detects words cut at chunk boundaries. Which of our strategies suffers most from that problem?
   > Your answer:

### Finalized notes

---

## 8. Production systems and real pipelines

- [An All-Neural On-Device Speech Recognizer](https://research.google/blog/an-all-neural-on-device-speech-recognizer/) (Google AI blog, March 2019), with its paper [Streaming End-to-end Speech Recognition for Mobile Devices](https://arxiv.org/abs/1811.06621) (He et al., 2018). RNN-T in Gboard: on-device, character by character as you speak, an 80 MB model after 4x quantization. A landmark production streaming deployment.
- [Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents) (NVIDIA on the Hugging Face blog, January 2026). Buffered vs cache-aware streaming in production terms: redundant computation, GPU memory under concurrency, latency drift. Claims up to 3x efficiency.
- Model cards: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) (cache-aware FastConformer-RNNT, chunk sizes 80 / 160 / 560 / 1120 ms, punctuation and capitalization) and [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b). The two checkpoints PLAN_v1 S4.3 chooses between. Check the license on each card.
- [NeMo ASR inference docs (streaming pipelines)](https://docs.nvidia.com/nemo/speech/latest/asr/inference.html). NeMo's streaming pipeline API: buffered CTC/RNNT/TDT pipelines (any offline model) and cache-aware CTC/RNNT pipelines, with a tutorial. The reference for building S4.4.
- [Introducing Universal-Streaming](https://assemblyai.com/blog/introducing-universal-streaming) (AssemblyAI). A commercial streaming API for voice agents that emits **immutable** transcripts (no partials-now, finals-later) in about 300 ms, with built-in turn detection. The post does not say which architecture it uses; read it for the product requirements, which match our `committed` text.
- [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753) (Kyutai, 2025) and its code [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling). A different route: a decoder-only model over time-aligned audio and text streams, with ASR as the text stream delayed behind the audio. Worth reading for where the field is heading.
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (with [icefall](https://github.com/k2-fsa/icefall) for training). An open-source production toolkit: streaming Zipformer transducers on ONNX Runtime, from servers down to phones and a Raspberry Pi, with a WebSocket streaming server. A good real pipeline to read the code of.

**Questions:**

1. Gboard (2019), NVIDIA Nemotron (2026) and sherpa-onnx all use transducers for streaming. List the reasons the readings give for that choice.
   > Your answer:

2. AssemblyAI sells "immutable" transcripts, and our protocol has `committed` and `tentative` text. Why does a voice agent care whether text can change after it is shown?
   > Your answer:

3. After this list, predict our S2.7 and S5.1 results. Which strategy should have the lowest `final_ms`, the steadiest partials, and the best WER?
   > Your answer:

### Finalized notes

---

## 9. Source code and framework pipelines

The papers above, as running code. Every repo, file and symbol below was checked against GitHub on 2026-10-04. Links point at the default branch, so line numbers drift; search for the symbol named. Where a file is long, the **Read** column says what to look for.

**Note:** NeMo's speech code moved from `NVIDIA/NeMo` (which now redirects) to [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech). Older blog posts and model cards still use the old paths.

### 9.1 Native cache-aware streaming: NVIDIA NeMo (our S4, `native`)

| Code | Read |
|---|---|
| [examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py) | **Start here.** The whole streaming loop in one script: `CacheAwareStreamingAudioBuffer` cuts the audio into chunks, `asr_model.conformer_stream_step(...)` runs one step and returns the updated caches, and `att_context_size` picks the look-ahead (chunk size). This is almost exactly what our Nemotron `ASRStream` will wrap. |
| [nemo/collections/asr/parts/mixins/mixins.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/mixins/mixins.py) | `conformer_stream_step`: encoder step with caches in and out, then a greedy CTC or RNNT decode that keeps the previous hypotheses. |
| [nemo/collections/asr/modules/conformer_encoder.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/modules/conformer_encoder.py) | `setup_streaming_params` (chunk size, shift, cache sizes derived from `att_context_size`) and `get_initial_cache_state` (the per-stream attention and convolution caches). The cache-aware paper (section 6) in code. |
| [nemo/collections/asr/parts/utils/streaming_utils.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/utils/streaming_utils.py) | `CacheAwareStreamingAudioBuffer` (cache-aware), and `FrameBatchASR` / `BatchedFrameASRRNNT` (**buffered** streaming: overlapping windows with any offline model, the NeMo equivalent of our `redecode`). Compare the two in one file. |
| [examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py) | The buffered approach as a runnable script, for side-by-side comparison with the cache-aware one. |
| [nemo/collections/asr/inference/pipelines/](https://github.com/NVIDIA-NeMo/Speech/tree/main/nemo/collections/asr/inference/pipelines) | NeMo's newer streaming pipeline API: `cache_aware_rnnt_pipeline.py`, `cache_aware_ctc_pipeline.py`, `buffered_rnnt_pipeline.py`, `buffered_ctc_pipeline.py`, with per-stream state under [`inference/streaming/`](https://github.com/NVIDIA-NeMo/Speech/tree/main/nemo/collections/asr/inference/streaming) and its entry script [examples/asr/asr_streaming_inference/asr_streaming_infer.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_streaming_inference/asr_streaming_infer.py). The production-shaped version of the script above. |
| [examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming.yaml](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming.yaml) | How a streaming model is *trained*: `att_context_size` (several look-aheads in one model), causal convolutions. Shows the "train as you infer" point from the paper. |
| [nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py) | TDT greedy decoding: where the predicted duration makes the decoder skip frames (section 4). |
| Tutorials: [Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb), [Online_ASR_Microphone_Demo_Buffered_Streaming.ipynb](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Buffered_Streaming.ipynb), [Buffered_Transducer_Inference.ipynb](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Buffered_Transducer_Inference.ipynb) | Microphone demos of both approaches, a good first hands-on run for S4.3. |
| [nemo/agents/voice_agent/pipecat/services/nemo/streaming_asr.py](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/agents/voice_agent/pipecat/services/nemo/streaming_asr.py) | NeMo's own streaming ASR service inside a Pipecat voice agent (see 9.4): how NVIDIA wires cache-aware ASR into an STT → LLM → TTS agent. |
| [NVIDIA-AI-Blueprints/nemotron-voice-agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) | NVIDIA's reference voice agent around Nemotron ASR, with scaling and latency benchmarks in [`benchmarking_tools/`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/tree/main/benchmarking_tools). |

### 9.2 Making Whisper stream (our S2: `chunked`, `redecode`, `trimmed`)

| Code | Read |
|---|---|
| [ufal/whisper_streaming: whisper_online.py](https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py) | **The reference for our S2.** `HypothesisBuffer.insert` / `flush` is LocalAgreement-2 (our `redecode`). `OnlineASRProcessor.process_iter` is one streaming step, `chunk_completed_segment` / `chunk_at` trim the buffer at a confirmed segment (our `trimmed`), and `finish` flushes at the end. About 300 lines that map one-to-one onto PLAN_v1 S2.3-S2.6. |
| [ufal/SimulStreaming: simul_whisper.py](https://github.com/ufal/SimulStreaming/blob/main/simulstreaming/whisper/simul_whisper/simul_whisper.py) | AlignAtt (Simul-Whisper). `PaddedAlignAttWhisper.infer` stops emitting when the most-attended audio frame is within `frame_threshold` of the end of the buffer. Compare with LocalAgreement: decide per token instead of re-decoding everything. |
| [ufal/SimulStreaming: eow_detection.py](https://github.com/ufal/SimulStreaming/blob/main/simulstreaming/whisper/simul_whisper/eow_detection.py) | The truncation (end-of-word) detector from the Simul-Whisper paper, for words cut at chunk edges. That is exactly the weakness of our `chunked` strategy. |
| [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) ([local_agreement/online_asr.py](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/local_agreement/online_asr.py), [audio_processor.py](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/audio_processor.py)) | A full app (server + browser UI) built on both LocalAgreement and SimulStreaming. Closest in shape to Astra: audio over a WebSocket, partial text back. |
| [collabora/WhisperLive](https://github.com/collabora/WhisperLive) ([backend/base.py](https://github.com/collabora/WhisperLive/blob/main/whisper_live/backend/base.py), [backend/faster_whisper_backend.py](https://github.com/collabora/WhisperLive/blob/main/whisper_live/backend/faster_whisper_backend.py)) | Another real-time Whisper server with several backends. Look at how it decides when a segment is final. |
| [ggml-org/whisper.cpp: examples/stream](https://github.com/ggml-org/whisper.cpp/tree/master/examples/stream) ([stream.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp), [README](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/README.md)) | The simplest possible approach: re-transcribe the last `--length` ms every `--step` ms (a fixed sliding window), or a VAD-triggered mode with `--step 0`. A useful baseline next to LocalAgreement. |
| [openai/whisper: transcribe.py](https://github.com/openai/whisper/blob/main/whisper/transcribe.py), [decoding.py](https://github.com/openai/whisper/blob/main/whisper/decoding.py) | How Whisper itself handles long audio: sequential 30 s windows, `condition_on_previous_text` (the prompt idea `trimmed` reuses), and the fallback temperatures and thresholds that fight invented text. |
| [huggingface/transformers: pipelines/automatic_speech_recognition.py](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py), [models/whisper/generation_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py) | The code our v0 `WhisperASR` calls. `chunk_length_s` / `stride_length_s` do offline chunking with overlap; `generate` handles timestamps and `prompt_ids` (needed for S2.5). |
| [SYSTRAN/faster-whisper: transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), [vad.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py) | Whisper on CTranslate2, the default backend of whisper_streaming. Word timestamps plus Silero VAD; read it if S2 speed becomes the bottleneck. |

### 9.3 Production transducer and CTC toolkits

| Code | Read |
|---|---|
| [k2-fsa/icefall: zipformer/](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer) ([zipformer.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py), [streaming_decode.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/streaming_decode.py), [decode_stream.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/decode_stream.py), [export-onnx-streaming.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/export-onnx-streaming.py)) | Training to deployment for a streaming Zipformer transducer: the model, the per-stream state (`decode_stream.py`), batched streaming decoding of many streams at once, and export to ONNX for sherpa-onnx. |
| [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ([csrc/online-recognizer.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.h), [csrc/online-transducer-model.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-model.h), [streaming_server.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/streaming_server.py), [speech-recognition-from-microphone.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone.py), [online-decode-files.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-decode-files.py)) | The deployment side: a C++ online recognizer API (create a stream, accept waveform, decode, get result, endpointing) and a WebSocket streaming server. Compare its stream API with our `ASRStream`. |
| [k2-fsa/k2: rnnt_loss.py](https://github.com/k2-fsa/k2/blob/master/k2/python/k2/rnnt_loss.py) | Pruned RNN-T loss: why a naive RNN-T loss needs a (time × tokens × vocabulary) tensor, and how pruning avoids it. Goes with the memory discussion in Lugosch's post. |
| [HawkAaron/warp-transducer: cpu_rnnt.h](https://github.com/HawkAaron/warp-transducer/blob/master/include/detail/cpu_rnnt.h) | A small, readable RNN-T loss (the forward-backward recursion over the alignment lattice) in C++. Older and no longer maintained, but the clearest code version of Graves 2012. |
| [wenet-e2e/wenet](https://github.com/wenet-e2e/wenet) ([wenet/utils/mask.py](https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/mask.py), [runtime/core/decoder/asr_decoder.h](https://github.com/wenet-e2e/wenet/blob/main/runtime/core/decoder/asr_decoder.h)) | U2, unified streaming and non-streaming in one model. `subsequent_chunk_mask` and `add_optional_chunk_mask` are **dynamic chunk training** (random chunk sizes so one model serves any latency). `AsrDecoder` streams CTC prefix beam search, then `AttentionRescoring` runs an attention decoder at the end: the two-pass idea (section 5) in production C++. Papers: [WeNet](https://arxiv.org/abs/2102.01547), [WeNet 2.0](https://arxiv.org/abs/2203.15455). |
| [espnet/espnet](https://github.com/espnet/espnet) ([espnet2/bin/asr_inference_streaming.py](https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_inference_streaming.py), [contextual_block_conformer_encoder.py](https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/contextual_block_conformer_encoder.py)) | The research toolkit's streaming path: a blockwise (contextual block) Conformer encoder with hybrid CTC/attention decoding (section 2's Watanabe paper). |
| [pytorch/audio](https://github.com/pytorch/audio) ([src/torchaudio/models/emformer.py](https://github.com/pytorch/audio/blob/main/src/torchaudio/models/emformer.py), [rnnt.py](https://github.com/pytorch/audio/blob/main/src/torchaudio/models/rnnt.py), [rnnt_decoder.py](https://github.com/pytorch/audio/blob/main/src/torchaudio/models/rnnt_decoder.py), [examples/asr/emformer_rnnt](https://github.com/pytorch/audio/tree/main/examples/asr/emformer_rnnt)) | Meta's Emformer (section 6) plus a compact RNN-T model and beam search in plain PyTorch. The smallest complete streaming transducer to read end to end. torchaudio has been cutting back its scope, so check that these are still supported in the release you install. |
| [alphacep/vosk-api](https://github.com/alphacep/vosk-api) ([src/recognizer.cc](https://github.com/alphacep/vosk-api/blob/master/src/recognizer.cc), [python/example/test_microphone.py](https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py)) | The pre-end-to-end way: Kaldi hybrid (HMM-DNN) streaming, still widely deployed offline. `PartialResult` vs `Result` is the same partial/final split as our protocol. |
| [kensho-technologies/pyctcdecode: decoder.py](https://github.com/kensho-technologies/pyctcdecode/blob/main/pyctcdecode/decoder.py) | CTC prefix beam search with an n-gram language model in pure Python: the "CTC has no internal language model" fix from section 2. |
| [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) ([scripts/stt_from_file_pytorch.py](https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/stt_from_file_pytorch.py)), [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | Kyutai's decoder-only streaming STT (section 8) as runnable code. |

### 9.4 Voice agent frameworks: where streaming ASR plugs in

| Code | Read |
|---|---|
| [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) ([services/nvidia/stt.py](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/stt.py), [services/deepgram/stt.py](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/deepgram/stt.py), [services/whisper/stt.py](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/whisper/stt.py), [processors/aggregators/llm_response.py](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/processors/aggregators/llm_response.py)) | A real STT → LLM → TTS pipeline. Compare a streaming STT service (NVIDIA, Deepgram: interim and final transcripts) with a segmented one (Whisper: transcribe after the user stops). Then read how the aggregator decides when the user's turn is over and the LLM may start, which is the next step after our push-to-talk. |
| [livekit/agents](https://github.com/livekit/agents) ([stt/stt.py](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/stt/stt.py), [stt/stream_adapter.py](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/stt/stream_adapter.py), [voice/agent_activity.py](https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/voice/agent_activity.py)) | `stt.py` defines the event types (`INTERIM_TRANSCRIPT`, `FINAL_TRANSCRIPT`, ...), the same partial/final split as ours. `StreamAdapter` makes a non-streaming STT "stream" by cutting audio with a VAD. It is LiveKit's version of the problem our strategies solve, and a different answer to it. |

**Questions:**

1. In NeMo, `streaming_utils.py` holds both `CacheAwareStreamingAudioBuffer` and `FrameBatchASR`. Explain the difference in what each one recomputes per chunk, and say which of our four strategies each one resembles.
   > Your answer:

2. Map `whisper_online.py` onto PLAN_v1: which method corresponds to S2.3 (LocalAgreement), which to S2.4 (`redecode`), and which to S2.6 (`trimmed`)? What does it do that our plan leaves out?
   > Your answer:

3. sherpa-onnx's online recognizer API, NeMo's pipeline API and our `ASRStream` all model "a stream per utterance". What does each keep in the per-stream state?
   > Your answer:

4. LiveKit's `StreamAdapter` uses a VAD to turn a batch STT into a streaming one. How is that different from our `chunked` strategy, and why might it be more accurate?
   > Your answer:

### Finalized notes

---

## All sources

Every paper and article link this list was built from, in one place, with alternative copies (PDF mirrors, HTML versions, publisher pages) where the search found them. Source code is in section 9.

**Surveys and courses**
- End-to-End Speech Recognition: A Survey: [arXiv](https://arxiv.org/abs/2303.03329), [PDF](https://arxiv.org/pdf/2303.03329), [IEEE Xplore](https://ieeexplore.ieee.org/document/10301513)
- Recent Advances in End-to-End Automatic Speech Recognition: [arXiv](https://arxiv.org/abs/2111.01690), [PDF](https://arxiv.org/pdf/2111.01690), [Microsoft Research talk](https://www.microsoft.com/en-us/research/video/recent-advances-in-end-to-end-automatic-speech-recognition/)
- Hugging Face Audio Course: [Unit 5, ASR models](https://huggingface.co/learn/audio-course/chapter5/asr_models), [Unit 3, CTC](https://huggingface.co/learn/audio-course/chapter3/ctc), [Unit 1, audio data](https://huggingface.co/learn/audio-course/en/chapter1/audio_data)

**Attention encoder-decoder**
- Listen, Attend and Spell: [arXiv](https://arxiv.org/abs/1508.01211), [Google Research](https://research.google/pubs/listen-attend-and-spell/), [Hugging Face papers](https://huggingface.co/papers/1508.01211)
- Robust Speech Recognition via Large-Scale Weak Supervision (Whisper): [arXiv](https://arxiv.org/abs/2212.04356), [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2212.04356), [OpenAI PDF](https://cdn.openai.com/papers/whisper.pdf)

**CTC**
- Sequence Modeling with CTC (Distill): [article](https://distill.pub/2017/ctc/), [ML Anthology entry](https://mlanthology.org/distill/2017/hannun2017distill-sequence)
- Connectionist Temporal Classification (Graves et al., 2006): [PDF (Toronto)](https://www.cs.toronto.edu/~graves/icml_2006.pdf), [PDF (TUM mirror)](https://mediatum.ub.tum.de/doc/1292048/341531.pdf), [ML Anthology entry](https://mlanthology.org/icml/2006/graves2006icml-connectionist)
- Hybrid CTC/Attention Architecture: [MERL PDF](https://www.merl.com/publications/docs/TR2017-190.pdf), [IEEE Xplore](https://ieeexplore.ieee.org/document/8068205), [Joint CTC/attention decoding (ACL 2017)](https://aclanthology.org/P17-1048/)

**Transducers and TDT**
- Sequence-to-sequence learning with Transducers (Lugosch): [blog](https://lorenlugosch.github.io/posts/2020/11/transducer/)
- Sequence Transduction with Recurrent Neural Networks (RNN-T): [arXiv](https://arxiv.org/abs/1211.3711), [ar5iv HTML](https://ar5iv.arxiv.org/html/1211.3711)
- Transformer Transducer: [arXiv](https://arxiv.org/abs/2002.02562), [PDF](https://arxiv.org/pdf/2002.02562)
- FastEmit: [arXiv](https://arxiv.org/abs/2010.11148), [Google Research](https://research.google/pubs/fastemit-low-latency-streaming-asr-with-sequence-level-emission-regularization/)
- Token-and-Duration Transducer (TDT): [arXiv PDF](https://arxiv.org/pdf/2304.06795), [ICML proceedings](https://proceedings.mlr.press/v202/xu23g.html), [NVIDIA Research](https://research.nvidia.com/labs/conv-ai/publications/2023/2023-token-duration-transducer)
- A Neural Transducer (Jaitly et al., an early online seq2seq model): [arXiv](https://arxiv.org/abs/1511.04868), [Google Research](https://research.google/pubs/a-neural-transducer/)

**Making attention stream**
- Monotonic Chunkwise Attention (MoChA): [arXiv PDF](https://arxiv.org/pdf/1712.05382), [Google Research](https://research.google/pubs/monotonic-chunkwise-attention/)
- Two-Pass End-to-End Speech Recognition: [arXiv PDF](https://arxiv.org/pdf/1908.10992), [Interspeech 2019](https://www.isca-archive.org/interspeech_2019/sainath19_interspeech.html)

**Streaming encoders**
- Conformer: [arXiv](https://arxiv.org/abs/2005.08100), [Interspeech 2020](https://www.isca-archive.org/interspeech_2020/gulati20_interspeech.html)
- Fast Conformer: [arXiv](https://arxiv.org/abs/2305.05084), [HTML](https://arxiv.org/html/2305.05084v6), [NVIDIA Research](https://research.nvidia.com/publication/2023-05_fast-conformer-linearly-scalable-attention-efficient-speech-recognition)
- Stateful Conformer with Cache-based Inference (cache-aware streaming): [arXiv](https://arxiv.org/abs/2312.17279), [HTML](https://arxiv.org/html/2312.17279v3), [NVIDIA Research](https://research.nvidia.com/labs/conv-ai/publications/2024/2024-cacheaware-streaming)
- Emformer: [arXiv](https://arxiv.org/abs/2010.10759), [Meta AI](https://ai.meta.com/research/publications/emformer-efficient-memory-transformer-based-acoustic-model-for-low-latency-streaming-speech-recognition/)
- Zipformer: [arXiv PDF](https://arxiv.org/pdf/2310.11230), [HTML](https://arxiv.org/html/2310.11230v3), [ICLR 2024 oral](https://iclr.cc/virtual/2024/oral/19784)

**Streaming Whisper**
- Turning Whisper into Real-Time Transcription System: [arXiv PDF](https://arxiv.org/pdf/2307.14743), [HTML](https://arxiv.org/html/2307.14743), [code: ufal/whisper_streaming](https://github.com/ufal/whisper_streaming)
- Simul-Whisper (AlignAtt): [arXiv PDF](https://arxiv.org/pdf/2406.10052), [HTML](https://arxiv.org/html/2406.10052v1)
- SimulStreaming: [code: ufal/SimulStreaming](https://github.com/ufal/SimulStreaming), [IWSLT 2025 paper](https://arxiv.org/abs/2506.17077)
- WhisperLiveKit (a whisper_streaming-based app): [README on Hugging Face Spaces](https://huggingface.co/spaces/langtech-innovation/WhisperLiveKit/blob/2ba48bcbf436dae9ebf126cb7df6c960a8bd8cc2/README.md)

**Production systems, models and toolkits**
- Google, An All-Neural On-Device Speech Recognizer: [Google Research blog](https://research.google/blog/an-all-neural-on-device-speech-recognizer/), [original AI blog URL](https://ai.googleblog.com/2019/03/an-all-neural-on-device-speech.html)
- Streaming End-to-end Speech Recognition for Mobile Devices: [arXiv](https://arxiv.org/abs/1811.06621), [PDF](https://arxiv.org/pdf/1811.06621)
- NVIDIA, Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR: [Hugging Face blog](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
- NVIDIA Nemotron Speech: [FAQ: streaming transcription for agents](https://perspectives.nvidia.com/nemotron-speech/task/faq/asr-models-streaming-transcription-real-time-agent-response/), [FAQ: open ASR throughput benchmarks](https://perspectives.nvidia.com/nemotron-speech/task/faq/open-asr-models-throughput-benchmarks/), [build.nvidia.com model card](https://build.nvidia.com/nvidia/nemotron-asr-streaming/modelcard)
- Model cards: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b), [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- NeMo streaming inference: [docs (latest)](https://docs.nvidia.com/nemo/speech/latest/asr/inference.html), [NeMo voice agent: speech recognition](https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/speech-recognition)
- Daily, Building Voice Agents with NVIDIA Open Models: [blog](https://www.daily.co/blog/building-voice-agents-with-nvidia-open-models/)
- AssemblyAI: [Introducing Universal-Streaming](https://assemblyai.com/blog/introducing-universal-streaming), [Voice agent architecture](https://assemblyai.com/blog/voice-agent-architecture)
- Kyutai, Delayed Streams Modeling: [arXiv HTML](https://arxiv.org/html/2509.08753v2), [code](https://github.com/kyutai-labs/delayed-streams-modeling), [Kyutai STT](https://kyutai.org/stt)
- WeNet: [WeNet paper](https://arxiv.org/abs/2102.01547), [WeNet 2.0 paper](https://arxiv.org/abs/2203.15455), [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2102.01547)
- k2-fsa: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [icefall](https://github.com/k2-fsa/icefall), [streaming LSTM transducer recipe (icefall docs)](https://icefall.readthedocs.io/en/latest/recipes/Streaming-ASR/librispeech/lstm_pruned_stateless_transducer.html)
