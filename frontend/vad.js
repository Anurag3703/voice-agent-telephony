/**
 * R2 – Endpointing VAD with hysteresis
 * R3 – Echo-aware barge-in residual gate
 */

export class EndpointVad {
  constructor(opts = {}) {
    this.startThreshold = opts.startThreshold ?? 0.018;
    this.endThreshold = opts.endThreshold ?? 0.010;
    this.minSpeechMs = opts.minSpeechMs ?? 100;
    this.hangoverMs = opts.hangoverMs ?? 450;
    this.minUtteranceMs = opts.minUtteranceMs ?? 200;

    this.inSpeech = false;
    this.speechStartAt = null;
    this.silenceFrom = null;
    this.utteranceStartAt = null;
  }

  reset() {
    this.inSpeech = false;
    this.speechStartAt = null;
    this.silenceFrom = null;
    this.utteranceStartAt = null;
  }

  forceSpeechStart(now = performance.now()) {
    this.inSpeech = true;
    this.utteranceStartAt = now;
    this.speechStartAt = null;
    this.silenceFrom = null;
  }

  process(rms, now) {
    if (!this.inSpeech) {
      if (rms >= this.startThreshold) {
        if (this.speechStartAt == null) this.speechStartAt = now;
        if (now - this.speechStartAt >= this.minSpeechMs) {
          this.inSpeech = true;
          this.utteranceStartAt = this.speechStartAt;
          this.silenceFrom = null;
          this.speechStartAt = null;
          return "speech_start";
        }
      } else {
        this.speechStartAt = null;
      }
      return null;
    }

    if (rms < this.endThreshold) {
      if (this.silenceFrom == null) this.silenceFrom = now;
      if (now - this.silenceFrom >= this.hangoverMs) {
        const dur = this.utteranceStartAt != null ? now - this.utteranceStartAt : 0;
        this.inSpeech = false;
        this.silenceFrom = null;
        this.utteranceStartAt = null;
        return dur < this.minUtteranceMs ? "speech_end_short" : "speech_end";
      }
    } else {
      this.silenceFrom = null;
    }
    return null;
  }
}

/**
 * R3 – Residual barge-in detector
 *
 * While the agent is playing, the mic picks up residual loudspeaker echo even with AEC.
 * We maintain a fast-adaptive baseline of mic energy during agent playback
 * and fire barge-in when energy exceeds threshold consistently.
 */
export class EchoAwareBargeIn {
  constructor(opts = {}) {
    this.margin = opts.margin ?? 2.0;           // how far above baseline
    this.absoluteFloor = opts.absoluteFloor ?? 0.018;
    this.confirmMs = opts.confirmMs ?? 45;      // fast response (45 ms confirmation)
    this.baselineAlpha = opts.baselineAlpha ?? 0.08; // track echo floor
    this.baseline = 0.01;
    this.aboveSince = null;
    this.enabled = false;
  }

  start() {
    this.enabled = true;
    this.aboveSince = null;
    this.baseline = Math.max(0.006, this.baseline * 0.9);
  }

  stop() {
    this.enabled = false;
    this.aboveSince = null;
  }

  /**
   * @param {number} level  current RMS
   * @param {number} now    performance.now()
   * @returns {{ barge: boolean, residual: number, threshold: number, baseline: number }}
   */
  process(level, now) {
    if (!this.enabled) {
      return { barge: false, residual: 0, threshold: 0, baseline: this.baseline };
    }

    const threshold = Math.max(this.absoluteFloor, this.baseline * this.margin);
    const residual = Math.max(0, level - this.baseline);

    // Adapt baseline only when below threshold to avoid tracking user speech
    if (level < threshold) {
      this.baseline = this.baseline * (1 - this.baselineAlpha) + level * this.baselineAlpha;
    }

    if (level >= threshold) {
      if (this.aboveSince == null) this.aboveSince = now;
      if (now - this.aboveSince >= this.confirmMs) {
        return { barge: true, residual, threshold, baseline: this.baseline };
      }
    } else if (level < threshold * 0.75) {
      // Allow slight dip without immediately dropping aboveSince
      this.aboveSince = null;
    }

    return { barge: false, residual, threshold, baseline: this.baseline };
  }
}

export function rms(buffer) {
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / buffer.length);
}
