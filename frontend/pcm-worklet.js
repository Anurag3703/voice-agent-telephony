/**
 * R6 – AudioWorklet capture processor
 * Runs on the audio render thread; posts Float32 mono frames to the main thread.
 */
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._active = true;
    this.port.onmessage = (ev) => {
      if (ev.data && ev.data.type === "stop") this._active = false;
    };
  }

  process(inputs) {
    if (!this._active) return false;
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    // Mono: use channel 0 (getUserMedia is requested mono)
    const channel = input[0];
    // Copy – underlying buffer is reused by the engine
    const copy = new Float32Array(channel.length);
    copy.set(channel);
    this.port.postMessage({ type: "samples", samples: copy }, [copy.buffer]);
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
