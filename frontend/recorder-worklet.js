// Runs on the audio rendering thread. Turns the mic stream (at whatever rate the
// AudioContext runs at) into 16 kHz mono PCM16 frames and posts them to the main thread.

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 800; // 50 ms at 16 kHz, inside the 20-100 ms range in SPEC 6.3.1

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Input samples that make up one output sample. 1 when the context already runs at 16 kHz.
    this.ratio = sampleRate / TARGET_RATE;
    this.active = false;
    this.runId = 0;
    this.maxSamples = Infinity;
    this.resetState();

    this.port.onmessage = ({ data }) => {
      if (data.cmd === "start") {
        this.resetState();
        this.runId = data.runId;
        this.maxSamples = data.maxSamples;
        this.active = true;
      } else if (data.cmd === "stop" && this.active) {
        this.active = false;
        this.flushFrame();
        this.port.postMessage({ type: "stopped", runId: this.runId });
      }
    };
  }

  resetState() {
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.frameLen = 0;
    this.emitted = 0;
    this.sum = 0;
    this.remaining = this.ratio;
  }

  // Area-averaging resampler: each output sample is the mean of the input span it covers.
  // The averaging doubles as a crude anti-aliasing low-pass filter.
  pushInput(x) {
    let left = 1;
    while (left > 1e-9) {
      const take = Math.min(left, this.remaining);
      this.sum += x * take;
      this.remaining -= take;
      left -= take;
      if (this.remaining < 1e-9) {
        this.pushOutput(this.sum / this.ratio);
        this.sum = 0;
        this.remaining = this.ratio;
      }
    }
  }

  pushOutput(v) {
    // The hard cap lives here so the server never receives more than the maximum turn length.
    if (this.emitted >= this.maxSamples) return;
    this.emitted++;
    const c = Math.max(-1, Math.min(1, v));
    this.frame[this.frameLen++] = Math.round(c < 0 ? c * 0x8000 : c * 0x7fff);
    if (this.frameLen === FRAME_SAMPLES) this.flushFrame();
  }

  flushFrame() {
    if (this.frameLen === 0) return;
    const pcm = this.frame.slice(0, this.frameLen).buffer;
    this.frameLen = 0;
    this.port.postMessage({ type: "frame", runId: this.runId, pcm }, [pcm]);
  }

  process(inputs) {
    if (!this.active) return true;
    const channels = inputs[0];
    if (!channels || channels.length === 0) return true;

    const n = channels[0].length;
    for (let i = 0; i < n; i++) {
      let x = 0;
      for (let c = 0; c < channels.length; c++) x += channels[c][i];
      this.pushInput(x / channels.length);
    }
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
