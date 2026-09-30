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
