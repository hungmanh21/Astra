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

**Status:** Finalized

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
"Continuous vs discrete" is about **both** axes independently, and it's easy to conflate them:

- **Time axis:** continuous-time (defined at every instant) vs discrete-time (defined only at fixed intervals).
- **Amplitude axis:** continuous-valued (any real value) vs discrete-valued / quantized (snapped to one of a finite set of levels).

Real-world sound (air pressure fluctuating over time) is continuous on *both* axes — a true analog signal. Turning it into computer-storable data requires analog-to-digital conversion (ADC), which is exactly the two separate steps named in Q3:

1. **Sampling** — discretizes the *time* axis: measure the signal's value only at fixed, regular instants.
2. **Quantization** — discretizes the *amplitude* axis: round each measured value to the nearest representable level.

Sampling is covered in Topic 2, quantization (via bit depth) in Topic 5. Together, sampling + quantization = ADC, and the result is a PCM stream (Topic 6).

---

## 2. Sampling and sampling rate

**Status:** Finalized

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
**Sampling** = measuring/reading the value of a continuous signal only at regular, discrete instants in time. **Sampling rate** (Hz) = how many of those measurements are taken per second (`fs = 1 / sampling_period`).

Why those specific common rates show up:

| Rate | Typical use | Why |
|---|---|---|
| 8000 Hz | Telephone / narrowband speech codecs (G.711) | Matches the ~300–3400 Hz bandwidth telephone networks were engineered for; "just enough" for intelligible speech |
| 16000 Hz | Speech/ASR models (Whisper, wav2vec2, most HF speech models) | Nyquist = 8000 Hz, which covers essentially all speech intelligibility content (see Topic 3) without wasting compute on frequencies speech doesn't use |
| 44100 Hz | CD audio, general music | Nyquist = 22050 Hz, covers the full ~20 Hz–20 kHz human hearing range plus margin for anti-aliasing filter roll-off; historically inherited from early digital audio tape formats |
| 48000 Hz | Professional audio/video production | Similar rationale to 44.1 kHz but divides evenly into common video frame rates, which is why it's the standard for film/broadcast audio |

Trade-off of a higher sampling rate: you raise the Nyquist frequency (Topic 3), so you can faithfully capture higher frequency content — but storage and file size grow linearly with rate, and so does the compute/data volume a model or pipeline has to process per second. For a real-time, low-latency ASR pipeline this matters directly: more samples per second to move and process means more work per unit of audio, so you want the *lowest* rate that still captures what the model needs (hence 16 kHz for speech, not 44.1 kHz).

---

## 3. Nyquist–Shannon sampling theorem

**Status:** Finalized

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
**Theorem:** a continuous signal can be perfectly reconstructed from its samples only if it was sampled at a rate at least twice the highest frequency component present in it: `fs ≥ 2 × f_max`. Equivalently, a sampling rate `fs` can faithfully represent frequencies only up to `fs / 2` — go above that and information is lost/corrupted (see Topic 4).

**Nyquist frequency at fs = 16000 Hz:** `16000 / 2 = 8000 Hz` — the highest frequency that sampling rate can represent without aliasing.

**Why 16 kHz for speech/ASR, not 44.1 kHz:** speech's meaningful energy lives almost entirely below ~8000 Hz (most intelligibility content below ~4000 Hz). A 16 kHz sampling rate gives a Nyquist frequency of exactly 8000 Hz, so it captures essentially everything relevant to recognizing speech. Sampling at 44.1 kHz (Nyquist 22050 Hz) would capture frequencies far beyond anything speech produces — extra data and extra compute per second with no benefit to ASR accuracy. 44.1 kHz earns its keep for music (which uses the full audible spectrum), not for speech.

---

## 4. Aliasing

**Status:** Finalized

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
**What it is:** when a signal is sampled *below* its Nyquist rate (`fs < 2 × f_max`), frequency components above `fs / 2` don't get dropped — they get *folded back* and masquerade as a lower, false frequency in the resulting samples. Once sampled, those fake-low-frequency samples are mathematically indistinguishable from samples of a genuinely low-frequency signal — the information about which one actually happened is gone for good. For a pure tone, the apparent (aliased) frequency is `|f_true − n·fs|`, picking the integer `n` that lands the result in `[0, fs/2]`.

**Worked example:** sample a `900 Hz` tone at `fs = 1000 Hz`. Nyquist is `500 Hz`, and `900 Hz` blows straight past it. Aliased frequency = `|900 − 1000| = 100 Hz`. The samples taken every 1 ms are *exactly* the values a genuine 100 Hz cosine would produce at those same instants — the ADC (or anything downstream) has no way to tell it was ever a 900 Hz signal.

```
+1.0 |O....    '        '        '        '       '        '        '        '    ....O
     |     ...         '        '        '         '        '        '         ...
     | '      O.                                                             .O      '
     |          *                                                           *
     |           ..      '        '                       '        '      ..
     |             ..                   '  '     '  '                   ..
     |               .         '                             '         .
     |                O                                               O
     |  '    '         ..                                           ..         '    '
     |           '       .                                         .       '
 0.0 |                    *                                       *
     |                     .       '                     '       .
     |                      ..         '    '   '    '         ..
     |                        O                               O
     |               '         .                             .         '
     |   '  '                   ..                         ..                   '  '
     |            '        '      ..                     ..      '        '
     |                              *                   *
     |                               .O      ' '      O.
     |     '        '        '         ...         ...         '        '        '
-1.0 |    '        '        '        '    ....O....    '        '        '        '
     +---------------------------------------------------------------------------------
      0ms                              5ms                                      10ms
```

- `O` = the actual samples taken (11 of them, every 1 ms — that's `fs = 1000 Hz`).
- `.` = the true continuous 900 Hz signal — oscillating fast, threading through the sample points 9 times over the 10 ms window (visually thicker/denser because it's drawn at higher resolution than just the sample instants).
- `'`/`*` = the false 100 Hz signal — the one slow hump that any reconstruction or spectrum analysis built from *only the `O` samples* would report.

Both curves pass through the exact same 11 dots. That's the whole problem: given only the samples, "it was a fast 900 Hz tone" and "it was a slow 100 Hz tone" are equally valid explanations, and there's no way to recover which one actually happened.

**Classic real-world analogy:** the wagon-wheel effect in old film — a camera's frame rate is a temporal sampling rate, and if the wheel's spokes rotate faster than half that frame rate, the spokes appear to spin slowly, freeze, or even reverse, for exactly the same reason as the diagram above.

**Anti-aliasing filter:** a low-pass filter applied to the *analog* signal **before** it reaches the sampler/ADC, removing any frequency content above the Nyquist frequency so it never gets the chance to fold back. It must happen before sampling — once a signal has been sampled below its Nyquist rate, the aliased (wrong) frequencies are already indistinguishably mixed into the digital data, and no amount of filtering afterward can separate genuine content from alias corruption. This is also why *downsampling* a digital signal (e.g. 48 kHz → 16 kHz) needs a digital low-pass filter applied before discarding samples — same principle, digital domain.

---

## 5. Bit depth

**Status:** Finalized

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
**Bit depth is about amplitude, not time** — it's the number of bits used to represent each individual sample's amplitude value. It's exactly the "quantization" half of the ADC process named in Topic 1 (sampling discretizes time; bit depth is how finely amplitude gets discretized).

**Range of levels:**

| Bit depth | Distinct amplitude levels |
|---|---|
| 8-bit | 2⁸ = 256 |
| 16-bit | 2¹⁶ = 65,536 |
| 24-bit | 2²⁴ = 16,777,216 |

More bits = finer amplitude resolution = ability to represent quiet detail and a wider dynamic range without audible "stair-stepping."

**Quantization noise/error:** the difference between a sample's true continuous amplitude and the nearest representable discrete level it gets rounded to. Since it's essentially rounding error, it shows up as a noise floor added to the signal. More bits → smaller maximum rounding error per sample → lower quantization noise floor → more usable dynamic range (roughly +6 dB of dynamic range per added bit: 16-bit ≈ 96 dB, 24-bit ≈ 144 dB).

**Speech/ASR datasets** almost universally use **16-bit** PCM. That's already far beyond the practical dynamic range of speech and typical microphone noise floors, so 24-bit adds no information useful for recognition — just more storage and bandwidth for no accuracy gain.

---

## 6. PCM encoding

**Status:** Finalized

**Questions:**

1. What does PCM (Pulse-Code Modulation) mean, and how do sampling rate + bit depth combine to define a PCM stream?
   > Your answer:

2. What's the difference between integer PCM (e.g. `int16`) and float PCM (e.g. `float32`) representations? Which do audio ML libraries (e.g. librosa, torchaudio, Hugging Face `datasets`) typically expect?
   > Your answer:

**Resources:**
- [Pulse-code modulation — Wikipedia](https://en.wikipedia.org/wiki/Pulse-code_modulation)
- [Load audio data — Hugging Face `datasets` docs](https://huggingface.co/docs/datasets/audio_load)

### Finalized notes
**PCM (Pulse-Code Modulation)** is the standard method of representing a sampled + quantized analog signal digitally: a plain stream of numbers, each one a quantized amplitude sample taken at fixed intervals. Sampling rate (Topic 2, time discretization) + bit depth (Topic 5, amplitude discretization) + channel count (Topic 7) together fully specify a PCM stream.

**Integer vs float PCM:**
- **Integer PCM** (e.g. `int16`): samples stored as fixed-point integers, e.g. `-32768..32767` for 16-bit — this is what's actually stored on disk in most WAV files.
- **Float PCM** (e.g. `float32`): samples stored as floating-point values, normalized to `-1.0..1.0`.

Audio ML libraries (librosa, torchaudio, Hugging Face `datasets`) load/convert audio into **float32 in `[-1, 1]`**, because neural network models expect normalized floating-point tensors as input — so there's typically a conversion step from the file's native `int16` storage format to `float32` at load time.

---

## 7. Mono vs stereo & channels

**Status:** Finalized

**Questions:**

1. What is a "channel" in audio? What's the difference between mono and stereo in terms of raw data layout?
   > Your answer:

2. For a real-time speech/ASR pipeline built on a single speaker's voice, why would you typically downmix to mono before feeding audio into a model?
   > Your answer:

### Finalized notes
A **channel** is one independent stream of audio samples — one "recording" of a signal, e.g. from one microphone or virtual position. **Mono** = 1 channel, stored as a single flat stream of samples. **Stereo** = 2 channels (left, right), typically interleaved sample-by-sample in the raw data (`L0, R0, L1, R1, ...`), so a stereo file has roughly double the raw data of the equivalent mono file.

For a real-time single-speaker ASR pipeline, the spatial/positional information stereo carries adds nothing about *what was said* — and ASR models are trained on mono input. Downmixing to mono keeps only the waveform content the model actually uses, avoiding doubled data volume and compute for no accuracy benefit.

---

## 8. WAV file structure

**Status:** Finalized

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
WAV is a **RIFF**-based container. A basic PCM WAV file has three chunks:

- **`RIFF` header chunk:** starts with the `"RIFF"` magic bytes, the total file size, then the `"WAVE"` format identifier.
- **`fmt ` subchunk:** describes the audio format — PCM type, channel count, sample rate, byte rate, block align, bits per sample.
- **`data` subchunk:** the actual raw PCM sample bytes, plus a field giving their size.

The `fmt ` chunk's fields map directly onto topics already covered:

| `fmt ` field | Maps to |
|---|---|
| `AudioFormat` | PCM type (Topic 6) |
| `NumChannels` | mono/stereo (Topic 7) |
| `SampleRate` | sampling rate (Topic 2) |
| `BitsPerSample` | bit depth (Topic 5) |
| `ByteRate` (derived) | `SampleRate × NumChannels × BitsPerSample / 8` |
| `BlockAlign` (derived) | `NumChannels × BitsPerSample / 8` |

Standard PCM WAV is **uncompressed** — the raw samples are stored directly with no encode/decode step, so file size is fully determined by `sample rate × bit depth × channels × duration`. That's why WAV files are large compared to FLAC (lossless compression — smaller, but bit-exact and reversible) or MP3 (lossy compression — much smaller, but permanently discards detail).

---

## 9. Common audio formats overview (WAV / FLAC / MP3)

**Status:** Finalized

**Questions:**

1. Group WAV, FLAC, and MP3 into "uncompressed", "lossless compressed", and "lossy compressed". Which is which?
   > Your answer:

2. For a real-time ASR/translation pipeline capturing live mic/call audio, which of these formats (or none — raw PCM buffers) would you actually want to work with internally, and why?
   > Your answer:

**Resources:**
- [Comparison of audio coding formats — Wikipedia](https://en.wikipedia.org/wiki/Comparison_of_audio_coding_formats)

### Finalized notes
- **Uncompressed:** WAV (typically holds raw PCM, Topic 8).
- **Lossless compressed:** FLAC (smaller than WAV, but bit-exactly recoverable back to the original).
- **Lossy compressed:** MP3 (smallest files, but permanently discards detail deemed perceptually unimportant — not recoverable).

For a **real-time capture pipeline** (live mic/call audio), you generally want to work with **raw PCM buffers directly in memory** — no container format at all. Live audio arrives as a continuous sample stream, so there's no file to decode in the first place; feeding int16/float32 PCM frames straight into the ASR model avoids the CPU overhead and added latency of any encode/decode step, which matters directly for a low-latency pipeline. A file format like WAV or FLAC only becomes relevant once you need to *save or transmit* audio persistently (e.g. logging a call, exporting a clip) — not for the live processing path.

---

## Summary

Cheat-sheet of the numbers and rules that will actually come up while building this project:

- **ADC = sampling (discretize time) + quantization (discretize amplitude, via bit depth).** Together they turn a continuous analog signal into a PCM stream.
- **Sampling rate:** use **16000 Hz** for speech/ASR (Nyquist 8000 Hz covers speech content); 44.1/48 kHz is for music/video, not needed here.
- **Nyquist rule:** `fs ≥ 2 × f_max`, or you get aliasing — sample below it and high frequencies fold into false low frequencies that can't be recovered afterward. Always low-pass filter *before* sampling/downsampling, never after.
- **Bit depth:** **16-bit** PCM is the standard and sufficient choice for speech; 24-bit adds no ASR-relevant information.
- **PCM format:** files store `int16`; ML libraries (librosa/torchaudio/HF `datasets`) load as `float32` in `[-1, 1]` — expect a conversion step.
- **Channels:** downmix to **mono** for single-speaker ASR; stereo just doubles data with no benefit here.
- **WAV structure:** `RIFF` header → `fmt ` chunk (sample rate, bit depth, channels) → `data` chunk (raw PCM), uncompressed.
- **Formats:** WAV = uncompressed, FLAC = lossless, MP3 = lossy. For the live real-time pipeline itself, skip files entirely and work with raw PCM buffers in memory — only reach for WAV/FLAC when persisting audio to disk.
