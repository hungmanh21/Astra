# Audio Representations

How the same audio can be viewed as a waveform, a frequency spectrum, a spectrogram, or a mel spectrogram — what each one shows, why more than one exists, and which one you actually feed to which kind of model. Builds directly on [signal_basics.md](signal_basics.md) (sampling, Nyquist, PCM) — do that one first if you haven't.

## How this file works

Same two phases as `signal_basics.md`:

1. **Phase 1 — Explore.** Each topic has questions with a light hint, plus a **Code example** pointing at a runnable script so you can actually see the representation, not just read about it. Fill in your answers under `> Your answer:`.
2. **Phase 2 — Finalize.** Ask me to review once you're done; I'll fill in `### Finalized notes` per topic and flip `Status` to `Finalized`.

**Companion code:** [`theory/fundamentals/examples/audio_representations.py`](examples/audio_representations.py). It builds a short synthetic signal (no external audio file needed — a steady two-tone segment followed by a rising chirp, so the frequency content visibly changes partway through) and renders all four representations from it. Run it with:

```bash
uv run theory/fundamentals/examples/audio_representations.py
```

This writes `theory/fundamentals/examples/output/audio_representations.png` (a 2×2 figure — waveform, spectrum, spectrogram, mel spectrogram) and `.../output/synthetic_signal.wav` (so you can listen to what generated those plots). That `output/` folder is gitignored — regenerate it locally rather than expecting it to already exist after a fresh clone.

**General resource:** [Hugging Face Audio Course — Chapter 1](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) also covers waveforms and spectrograms and is worth skimming before the per-topic questions below.

---

## 1. Waveform (time domain)

**Status:** Finalized

**Questions:**

1. What do the two axes of a waveform plot represent? What can you directly read off it (hint: loudness/amplitude over time), and what can't you tell just by looking at it?
   > Your answer:

2. A digital waveform plot is literally the PCM samples from `signal_basics.md` Topic 6, plotted as points/lines. Given that, what determines how "smooth" vs "blocky" the plotted curve looks?
   > Your answer:

**Code example:** `plot_waveform()` in the companion script — look at the top-left panel of the generated PNG. The signal has two very different segments (a steady two-tone sound, then a rising chirp) stitched together at t = 0.5s.
   > After looking at the plot: could you tell, from the waveform alone, where the frequency content changes? What's missing from this view that would make that obvious?
   > Your answer:

### Finalized notes

- **Axes:** x = time, y = amplitude — the raw PCM sample value at each instant (a proxy for instantaneous air pressure deviation, per `signal_basics.md`).
- **What you can read directly:** loudness/envelope over time (quiet vs loud passages), onsets and silences, rough duration, and for a very simple, slowly-varying tone at high zoom, a rough sense of periodicity. What you *can't* tell: which frequencies are present, or timbre — a loud low hum and a loud hiss can produce very similar-looking envelopes.
- **Smooth vs blocky:** the plotted curve is literally the discrete PCM samples connected by straight lines. Zoomed out (many samples per pixel), it renders as a dense filled band and looks "smooth." Zoomed in near individual samples, you see the actual discrete points and straight-line segments between them — "blocky." Higher sampling rate = more points per unit time = smoother-looking curve at a given zoom level, but this is a rendering effect, not new information.
- **Code example:** No — at the zoom level of the full 1-second plot, both the steady two-tone segment and the 1–3 kHz chirp compress into a dense oscillating band that looks visually similar throughout; you cannot pick out where the frequency content changes just from the envelope shape. What's missing is a *frequency-domain* view — the waveform tells you "how loud," not "what pitch." That gap is exactly what Topics 2–4 address.

---

## 2. Frequency spectrum (Fourier Transform / FFT)

**Status:** Finalized

**Questions:**

1. Conceptually, what does a Fourier Transform do to a signal — what is it decomposing the signal into? What do the x-axis and y-axis of a magnitude spectrum plot represent?
   > Your answer:

2. A single FFT computed over an entire signal loses all information about *when* something happened — why? (What assumption is it implicitly making about the signal?)
   > Your answer:

3. Given the Nyquist theorem from `signal_basics.md` Topic 3, what's the highest frequency a spectrum computed from 16000 Hz audio could possibly show on its x-axis?
   > Your answer:

**Resources:**
- [But what is the Fourier Transform? A visual introduction — 3Blue1Brown](https://www.3blue1brown.com/lessons/fourier-transforms/)

**Code example:** `plot_spectrum()` — top-right panel. You should see two sharp peaks (the steady tones) plus a low, broad hump (the chirp's energy, smeared across the range it swept through).
   > Can you tell from this plot whether the tones happened before or after the chirp? Why not?
   > Your answer:

### Finalized notes

- **What it decomposes into:** a Fourier Transform decomposes a signal into a sum of sinusoids (single-frequency waves), each with a magnitude and a phase. A magnitude spectrum plot puts frequency on the x-axis and "how much of that frequency is present" (magnitude) on the y-axis.
- **Why a single whole-signal FFT loses timing (validated):** it implicitly treats the entire analyzed window as one stationary block — the DFT math assumes the window repeats periodically forever, i.e. the same frequency content the whole way through. Each output bin is one number per frequency covering the *entire* window, with no per-bin timestamp. Two tones that never overlapped and two tones playing simultaneously the whole time can produce a nearly identical spectrum — the "when" is thrown away, only "what's present overall" survives.
- **Highest frequency visible at 16 kHz:** 8000 Hz (`sr / 2`), the Nyquist frequency — a spectrum can never show energy above that, by construction of the FFT bins for real-valued audio.
- **Code example:** No, you can't tell ordering from the spectrum plot — it shows two sharp peaks (440/880 Hz) and a broad hump (the chirp's energy spread across 1–3 kHz) with no indication of which came first, exactly because the FFT collapsed the time axis away.

---

## 3. Spectrogram (Short-Time Fourier Transform)

**Status:** Finalized

**Questions:**

1. What problem does the STFT solve relative to a single whole-signal FFT? (Hint: "short-time" — what does it do differently?)
   > Your answer:

2. STFT works by splitting the signal into short, overlapping windows and running an FFT on each one. This creates a trade-off controlled by the window length: shorter windows give better ___ resolution but worse ___ resolution, and vice versa. Fill in the blanks and explain why the trade-off exists.
   > Your answer:

3. A spectrogram plot has three dimensions of information on a 2D image. What are they, and which one is usually shown as color/intensity rather than as an axis?
   > Your answer:

**Resources:**
- [Short-time Fourier transform — Wikipedia](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)

**Code example:** `plot_spectrogram()` — bottom-left panel (uses `librosa.stft`). You should see two horizontal lines for the first half (the steady tones) and a diagonal line for the second half (the chirp sweeping upward).
   > Your answer: does this match the segments described in `generate_signal()`? What does the diagonal line's slope represent physically?
   > Your answer:

### Finalized notes

- **Problem solved vs a whole-signal FFT:** instead of one FFT over the entire signal, the STFT splits it into many short, overlapping windows and runs an FFT on each one separately. That recovers a time axis: you get a frequency spectrum *per window*, so you can see frequency content change over time, not just its overall presence.
- **The window-length trade-off:** shorter windows give better **time** resolution but worse **frequency** resolution, and vice versa. This is unavoidable, not a tuning bug — see the ELI5 + detailed explanation above: FFT frequency resolution is `sample_rate / N` Hz per bin, so fewer samples (short window) means coarser frequency bins but a narrower, more precise time slice; more samples (long window) means finer frequency bins but content within that longer window gets smeared together in time. It's a time–frequency uncertainty trade-off — you choose the window length based on whether you care more about *when* or about *exact pitch*.
- **Three dimensions on a 2D plot:** time (x-axis), frequency (y-axis), and magnitude/energy at that time-frequency point — shown as color/intensity (usually in dB), since a 2D image only has two spatial axes to spend on time and frequency.
- **Code example:** Yes, it matches `generate_signal()` exactly — two flat horizontal lines (440 Hz and its 880 Hz harmonic) for the first 0.5s, then a rising diagonal line from roughly 1000→3000 Hz for the second 0.5s. The diagonal's slope is physically the chirp's **sweep rate** — how many Hz the instantaneous frequency increases per second (here, `(3000-1000) Hz / 0.5 s = 4000 Hz/s`).

---

## 4. Mel spectrogram

**Status:** Finalized

**Questions:**

1. What is the mel scale trying to approximate about human hearing? (Hint: is our sensitivity to a 100 Hz difference the same at 100 Hz vs at 8000 Hz?)
   > Your answer:

2. Converting a linear-frequency spectrogram to a mel-scaled one changes the spacing of the frequency axis. Does it give *more* or *fewer* bins of resolution to low frequencies vs high frequencies, relative to a linear scale?
   > Your answer:

3. Why might a speech/ASR model prefer a mel spectrogram as its input feature over a raw linear spectrogram or the raw waveform?
   > Your answer:

**Resources:**
- [Mel scale — Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)

**Code example:** `plot_mel_spectrogram()` — bottom-right panel (uses `librosa.feature.melspectrogram`). Compare it directly to the bottom-left (linear spectrogram) panel.
   > Your answer: on the mel axis, do the two steady tones (440/880 Hz) take up more or less vertical space, proportionally, than they did on the linear axis? Does that match what you'd expect from Q2?
   > Your answer:

### Finalized notes

- **What the mel scale approximates:** human pitch perception is non-linear — we're much more sensitive to a given Hz difference at low frequencies than at high ones. A 100 Hz → 200 Hz jump sounds like a huge pitch change; an 8000 Hz → 8100 Hz jump is barely perceptible even though it's the same 100 Hz gap. The mel scale is a frequency remapping designed so that equal *mel* distances correspond to roughly equal *perceived* pitch distances.
- **Effect on resolution:** converting to mel gives **more** bins/resolution to low frequencies and **fewer** to high frequencies, relative to a linear scale — the mel scale compresses the high end and expands (relatively) the low end, matching where human hearing (and speech) cares most.
- **Why ASR models prefer it:** most phonetically important speech energy sits in the low/mid frequency range, which mel spectrograms represent with disproportionately more resolution — for the perceptually irrelevant task of speech recognition. It also compresses the input (far fewer mel bins than raw linear FFT bins), reducing compute and giving the model a smaller, denser, more perceptually-relevant feature space than either a raw linear spectrogram or the raw waveform, which empirically improves training efficiency and accuracy for speech models.
- **Code example:** Yes — on the mel axis, the 440 Hz and 880 Hz tones take up noticeably *more* vertical space, proportionally, than they did on the linear axis, exactly matching Q2: mel expands low-frequency spacing, so two low tones that were close together on a linear axis get pushed further apart on the mel axis.

---

## 5. Choosing a representation for this project

**Status:** Finalized

**Questions:**

1. Whisper's encoder takes an 80-channel (128 for `large-v3`) log-mel spectrogram as input, computed with 25 ms windows and a 10 ms stride on 16 kHz audio. Given that, what preprocessing would your pipeline need to do to raw microphone/call audio before it could be fed to a Whisper model?
   > Your answer:

2. `wav2vec2`-family models instead take the **raw waveform** directly as input (just resampled to 16 kHz) — no spectrogram computed at all. What does that imply is happening *inside* the model that, for Whisper, instead happens in a separate preprocessing step?
   > Your answer:

3. This project is real-time (live meetings/calls). What practical latency/compute implication does "compute a mel spectrogram per audio chunk before the model even runs" have, compared to a model that consumes raw waveform chunks directly? Is that a reason to prefer one model family over the other, or is it a non-issue in practice?
   > Your answer:

**Resources:**
- [Whisper — Hugging Face Transformers docs](https://huggingface.co/docs/transformers/en/model_doc/whisper)
- [facebook/wav2vec2-base-960h — Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h)

### Finalized notes

- **Whisper preprocessing:** resample/downmix raw mic or call audio to 16 kHz mono, then compute a log-mel spectrogram (80 channels for base/small/medium, 128 for `large-v3`) using a 25 ms window with a 10 ms stride (400/160 samples at 16 kHz), normalized the way Whisper's feature extractor expects. The model never sees raw audio — only the resulting log-mel frames, typically chunked/padded to the 30s window Whisper was trained on.
- **wav2vec2 implication:** since it takes the raw waveform directly (just resampled to 16 kHz), whatever frequency/feature extraction Whisper does explicitly as a *preprocessing* step (the mel spectrogram) must instead happen **inside** the model — concretely, a stack of 1D convolutional layers at the front of wav2vec2 (its CNN feature encoder) learns a data-driven transformation that plays the same role a hand-crafted mel spectrogram plays for Whisper, just learned from data instead of fixed by a formula.
- **Real-time latency implication:** computing a mel spectrogram per chunk is a fast, deterministic, FFT-based operation (`O(n log n)`) — in practice it's a negligible fraction of total pipeline latency compared to the model's own forward pass, especially for encoder-decoder models the size of Whisper. It's generally **not** a meaningful reason to prefer wav2vec2 over Whisper on latency grounds alone; for this project, model accuracy, streaming/chunking support, and available fine-tunes matter far more than whether feature extraction happens before or inside the model.

---

## Summary

- **Waveform:** time vs amplitude. Shows loudness/timing, not frequency content.
- **Spectrum (FFT):** frequency vs magnitude, over an entire window. Shows *what* frequencies are present, not *when* — a single FFT assumes the window is stationary throughout.
- **Spectrogram (STFT):** many short FFTs over sliding windows → time *and* frequency, at the cost of a hard trade-off: short windows = good time / poor frequency resolution, long windows = good frequency / poor time resolution (`resolution ≈ sample_rate / window_length`).
- **Mel spectrogram:** a spectrogram with frequency remapped to the mel scale, which matches human (and speech-relevant) pitch perception — more resolution at low frequencies, less at high, and a smaller, denser feature space than a linear spectrogram.
- **For this project:** Whisper expects a precomputed log-mel spectrogram (80/128 mel bins, 25 ms/10 ms), while wav2vec2-family models expect the raw 16 kHz waveform and learn the equivalent transformation internally via convolutional layers. The mel-computation step itself is cheap relative to model inference, so it isn't a real-time blocker either way.
