# Signal Processing Basics

Fundamentals needed before working with audio for the real-time ASR translation project (see [README.md](../../README.md)).

## How this file works

This file has two phases per topic:

1. **Phase 1 — Explore.** Each topic below has a set of questions with a light hint. Research them yourself (docs, articles, experimentation) and write your answers directly under the `> Your answer:` line for each question. Leave nothing else in the file edited.
2. **Phase 2 — Finalize.** Once you've filled in your answers, ask me to review this file. I'll check each answer, correct/fill gaps, and write the finalized explanation into the `### Finalized notes` block under that topic. The status badge for the topic will flip from `Not started` to `Answered` to `Finalized`.

Work through topics in order — later ones build on earlier ones (e.g. aliasing depends on understanding sampling rate).

**General resource:** [Hugging Face Audio Course — Chapter 1: Introduction to audio data](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) is a short, practical overview covering sampling rate, bit depth, and waveforms/spectrograms, written for people building with HF audio/ASR models — a good primer across several topics below before you dig into per-topic questions.

---

## 1. Continuous vs discrete signals

**Status:** Not started

**Questions:**

1. What makes a signal "continuous" vs "discrete"? Is this about the time axis, the amplitude axis, or both?
   > Your answer:

2. Sound in the real world (a voice, a room) — is it continuous or discrete? What has to happen for it to become data a computer can store?
   > Your answer:

3. What's the difference between *discretizing time* and *discretizing amplitude*? (These are two separate steps — name them.)
   > Your answer:

**Resources:**
- [Continuous signal — Wikipedia](https://en.wikipedia.org/wiki/Continuous_signal)
- [Discrete-time signal — Wikipedia](https://en.wikipedia.org/wiki/Discrete-time_signal)

### Finalized notes
*(filled in during Phase 2)*

---

## 2. Sampling and sampling rate

**Status:** Not started

**Questions:**

1. What does it mean to "sample" a signal? What is a sampling rate (units: Hz) measuring exactly?
   > Your answer:

2. Common sampling rates you'll see in speech/audio work include 8000 Hz, 16000 Hz, 44100 Hz, 48000 Hz. Why do these particular numbers show up — what are they each typically used for? (Hint: think telephone audio, speech models like Whisper, music/CD audio.)
   > Your answer:

3. If you record at a higher sampling rate, what do you gain, and what does it cost you (storage, compute)?
   > Your answer:

**Resources:**
- [Sampling (signal processing) — Wikipedia](https://en.wikipedia.org/wiki/Sampling_(signal_processing))

### Finalized notes
*(filled in during Phase 2)*

---

## 3. Nyquist–Shannon sampling theorem

**Status:** Not started

**Questions:**

1. State the Nyquist–Shannon theorem in your own words. What relationship does it define between sampling rate and the highest frequency you can represent?
   > Your answer:

2. What is the "Nyquist frequency" for a signal sampled at 16000 Hz?
   > Your answer:

3. Human speech's meaningful energy roughly spans up to ~8000 Hz (with most intelligibility content below ~4000 Hz). Given that, why is 16000 Hz a common choice for speech/ASR models rather than 44100 Hz?
   > Your answer:

**Resources:**
- [Nyquist–Shannon sampling theorem — Wikipedia](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem)

### Finalized notes
*(filled in during Phase 2)*

---

## 4. Aliasing

**Status:** Not started

**Questions:**

1. What is aliasing, and why does it happen when a signal is sampled below its Nyquist rate?
   > Your answer:

2. What does aliasing sound/look like in practice (e.g. in a spectrogram, or a classic visual example like a wagon wheel appearing to spin backwards in film)?
   > Your answer:

3. What's an "anti-aliasing filter" and where in an audio pipeline would you expect one to be applied — before or after sampling? Why?
   > Your answer:

**Resources:**
- [Aliasing — Wikipedia](https://en.wikipedia.org/wiki/Aliasing)

### Finalized notes
*(filled in during Phase 2)*

---

## 5. Bit depth

**Status:** Not started

**Questions:**

1. What does "bit depth" describe — is it about time or amplitude? How does it relate to the discretization concept from Topic 1?
   > Your answer:

2. What's the difference between 8-bit, 16-bit, and 24-bit audio in terms of the range of amplitude values they can represent?
   > Your answer:

3. What is "quantization noise/error" and how does bit depth affect it?
   > Your answer:

4. What bit depth is most commonly used for speech/ASR datasets and why is that usually "enough"?
   > Your answer:

**Resources:**
- [Audio bit depth — Wikipedia](https://en.wikipedia.org/wiki/Audio_bit_depth)

### Finalized notes
*(filled in during Phase 2)*

---

## 6. PCM encoding

**Status:** Not started

**Questions:**

1. What does PCM (Pulse-Code Modulation) mean, and how do sampling rate + bit depth combine to define a PCM stream?
   > Your answer:

2. What's the difference between integer PCM (e.g. `int16`) and float PCM (e.g. `float32`) representations? Which do audio ML libraries (e.g. librosa, torchaudio, Hugging Face `datasets`) typically expect?
   > Your answer:

**Resources:**
- [Pulse-code modulation — Wikipedia](https://en.wikipedia.org/wiki/Pulse-code_modulation)
- [Load audio data — Hugging Face `datasets` docs](https://huggingface.co/docs/datasets/audio_load)

### Finalized notes
*(filled in during Phase 2)*

---

## 7. Mono vs stereo & channels

**Status:** Not started

**Questions:**

1. What is a "channel" in audio? What's the difference between mono and stereo in terms of raw data layout?
   > Your answer:

2. For a real-time speech/ASR pipeline built on a single speaker's voice, why would you typically downmix to mono before feeding audio into a model?
   > Your answer:

### Finalized notes
*(filled in during Phase 2)*

---

## 8. WAV file structure

**Status:** Not started

**Questions:**

1. WAV is a container format (specifically RIFF-based). What are the major "chunks" a basic PCM WAV file has, and what does each one hold? (Hint: at minimum, look up `RIFF` header, `fmt ` chunk, `data` chunk.)
   > Your answer:

2. In the `fmt ` chunk, what key fields are stored, and how do they map back to the concepts above (sampling rate, bit depth, channels)?
   > Your answer:

3. Is WAV audio typically compressed or uncompressed? How does that affect file size vs. formats like MP3 or FLAC?
   > Your answer:

**Resources:**
- [WAVE PCM soundfile format (soundfile.sapp.org)](http://soundfile.sapp.org/doc/WaveFormat/): the canonical byte-by-byte breakdown of the RIFF/WAV chunk layout.

### Finalized notes
*(filled in during Phase 2)*

---

## 9. Common audio formats overview (WAV / FLAC / MP3)

**Status:** Not started

**Questions:**

1. Group WAV, FLAC, and MP3 into "uncompressed", "lossless compressed", and "lossy compressed". Which is which?
   > Your answer:

2. For a real-time ASR/translation pipeline capturing live mic/call audio, which of these formats (or none — raw PCM buffers) would you actually want to work with internally, and why?
   > Your answer:

**Resources:**
- [Comparison of audio coding formats — Wikipedia](https://en.wikipedia.org/wiki/Comparison_of_audio_coding_formats)

### Finalized notes
*(filled in during Phase 2)*

---

## Summary (filled in after all topics are finalized)

*(A short consolidated cheat-sheet will go here once every topic above is finalized — the key numbers and rules you'll actually reach for while building the project.)*
