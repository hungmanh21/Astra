// Main-thread side of the recorder: owns the mic stream and the AudioWorklet,
// and hands 16 kHz mono PCM16 frames to the app.

export const SAMPLE_RATE = 16000;

export class Recorder {
  // onFrame(arrayBuffer, totalSamples), onStopped(totalSamples), onLimit()
  constructor({ maxSeconds, onFrame, onStopped, onLimit }) {
    this.maxSamples = Math.round(maxSeconds * SAMPLE_RATE);
    this.onFrame = onFrame;
    this.onStopped = onStopped;
    this.onLimit = onLimit;
    this.ctx = null;
    this.node = null;
    this.active = false;
    this.runId = 0;
    this.total = 0;
  }

  get sampleRate() {
    return this.ctx ? this.ctx.sampleRate : null;
  }

  // Must be called from a user gesture (permission prompt + AudioContext resume).
  async init() {
    if (this.ctx) {
      await this.ctx.resume();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("microphone access needs HTTPS or localhost");
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });

    // Ask for 16 kHz. Where the browser ignores or rejects it, the worklet resamples itself.
    let ctx;
    try {
      ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    } catch {
      ctx = new AudioContext();
    }

    try {
      await ctx.audioWorklet.addModule(new URL("recorder-worklet.js", import.meta.url));
      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "recorder-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      node.port.onmessage = ({ data }) => this.handleMessage(data);

      // Some browsers only run a worklet that reaches the destination; keep it silent.
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(node).connect(mute).connect(ctx.destination);

      await ctx.resume();
      this.ctx = ctx;
      this.node = node;
    } catch (err) {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      throw err;
    }
  }

  start() {
    this.runId++;
    this.total = 0;
    this.active = true;
    this.node.port.postMessage({ cmd: "start", runId: this.runId, maxSamples: this.maxSamples });
  }

  // Stop and deliver any remaining audio; onStopped fires once it has all been posted.
  stop() {
    if (!this.active) return;
    this.active = false;
    this.node.port.postMessage({ cmd: "stop" });
  }

  // Stop and drop everything still in flight from this recording.
  cancel() {
    if (!this.node) return;
    this.active = false;
    this.node.port.postMessage({ cmd: "stop" });
    this.runId++;
  }

  handleMessage(data) {
    if (data.runId !== this.runId) return; // stale message from a cancelled recording
    if (data.type === "frame") {
      this.total += data.pcm.byteLength / 2;
      this.onFrame(data.pcm, this.total);
      if (this.active && this.total >= this.maxSamples) this.onLimit();
    } else if (data.type === "stopped") {
      this.onStopped(this.total);
    }
  }
}
