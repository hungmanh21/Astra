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

**Status:** Not started

**Questions:**

1. What do the two axes of a waveform plot represent? What can you directly read off it (hint: loudness/amplitude over time), and what can't you tell just by looking at it?
   > Your answer:

2. A digital waveform plot is literally the PCM samples from `signal_basics.md` Topic 6, plotted as points/lines. Given that, what determines how "smooth" vs "blocky" the plotted curve looks?
   > Your answer:

**Code example:** `plot_waveform()` in the companion script — look at the top-left panel of the generated PNG. The signal has two very different segments (a steady two-tone sound, then a rising chirp) stitched together at t = 0.5s.
   > After looking at the plot: could you tell, from the waveform alone, where the frequency content changes? What's missing from this view that would make that obvious?
   > Your answer:

### Finalized notes
*(filled in during Phase 2)*

---

## 2. Frequency spectrum (Fourier Transform / FFT)

**Status:** Not started

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
*(filled in during Phase 2)*

---

## 3. Spectrogram (Short-Time Fourier Transform)

**Status:** Not started

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
*(filled in during Phase 2)*

---

## 4. Mel spectrogram

**Status:** Not started

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
*(filled in during Phase 2)*

---

## 5. Choosing a representation for this project

**Status:** Not started

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
*(filled in during Phase 2)*

---

## Summary (filled in after all topics are finalized)

*(A short consolidated cheat-sheet will go here once every topic above is finalized.)*
