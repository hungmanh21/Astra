"""
Companion code for theory/fundamentals/audio_representations.md

Builds a short synthetic signal (no external audio file needed, fully
reproducible) and renders it as the four representations discussed in
that file: waveform, frequency spectrum, spectrogram, mel spectrogram.

Run with:
    uv run theory/fundamentals/examples/audio_representations.py

Output (figures + the synthetic .wav itself) is written to ./output/
next to this script.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

SR = 16_000  # same 16 kHz used throughout signal_basics.md for speech/ASR
OUT_DIR = Path(__file__).parent / "output"


def generate_signal(sr: int = SR) -> np.ndarray:
    """A signal whose frequency content changes over time, on purpose.

    A single FFT over the whole thing (Topic 2 in the markdown) can only
    report "these frequencies are present somewhere" — it can't say when.
    That's the exact limitation a spectrogram (Topic 3) exists to fix, so
    the signal here is built to make that limitation obvious.

    Segment A (0.0-0.5s): a steady 440 Hz tone (A4) with a quiet 880 Hz
    harmonic on top - mimics a sustained vowel-like sound.
    Segment B (0.5-1.0s): a chirp sweeping 1000 Hz -> 3000 Hz - mimics a
    fast-changing consonant/transient.
    Light white noise is layered in throughout for realism.
    """
    t_a = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    seg_a = 0.6 * np.sin(2 * np.pi * 440 * t_a) + 0.2 * np.sin(2 * np.pi * 880 * t_a)

    t_b = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    f0, f1 = 1000, 3000
    instantaneous_freq = f0 + (f1 - f0) * (t_b / 0.5)
    seg_b = 0.6 * np.sin(2 * np.pi * instantaneous_freq * t_b)

    y = np.concatenate([seg_a, seg_b])
    rng = np.random.default_rng(seed=0)
    y = y + 0.01 * rng.standard_normal(y.shape)
    return y.astype(np.float32)


def plot_waveform(y: np.ndarray, sr: int, ax: plt.Axes) -> None:
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform (time domain)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")


def plot_spectrum(y: np.ndarray, sr: int, ax: plt.Axes) -> None:
    """Single FFT over the whole signal — frequency content, no timing.

    Windowed the same way the HF Audio Course does it: a Hann window
    before the FFT, magnitude converted to dB, frequency axis on a log
    scale. The Hann taper just reduces spectral-leakage artifacts at the
    edges of the analyzed window — it doesn't change the point Topic 2 is
    making, since we still run one FFT over the *whole* signal.
    """
    window = np.hanning(len(y))
    dft = np.fft.rfft(y * window)
    amplitude_db = librosa.amplitude_to_db(np.abs(dft), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=len(y))

    ax.plot(freqs, amplitude_db, linewidth=0.7)
    ax.set_title("Frequency spectrum (FFT of the whole signal)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xscale("log")
    ax.set_xlim(20, sr / 2)  # 20 Hz: log scale can't start at 0; sr/2 = Nyquist


def plot_spectrogram(y: np.ndarray, sr: int, ax: plt.Axes) -> None:
    """STFT: many short FFTs over sliding windows -> time AND frequency."""
    stft = librosa.stft(y, n_fft=512, hop_length=128)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = librosa.display.specshow(
        magnitude_db, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=ax
    )
    ax.set_title("Spectrogram (STFT, linear frequency axis)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_mel_spectrogram(y: np.ndarray, sr: int, ax: plt.Axes) -> None:
    """Same idea as the spectrogram, but frequency axis is mel-scaled."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=128, x_axis="time", y_axis="mel", ax=ax
    )
    ax.set_title("Mel spectrogram")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    y = generate_signal(SR)
    sf.write(OUT_DIR / "synthetic_signal.wav", y, SR)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_waveform(y, SR, axes[0, 0])
    plot_spectrum(y, SR, axes[0, 1])
    plot_spectrogram(y, SR, axes[1, 0])
    plot_mel_spectrogram(y, SR, axes[1, 1])
    fig.tight_layout()

    out_path = OUT_DIR / "audio_representations.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    print(f"Wrote {OUT_DIR / 'synthetic_signal.wav'}")


if __name__ == "__main__":
    main()
