/**
 * R2 – Better VAD + endpointing (builds on R1 turn epochs)
 */

import { EndpointVad, EchoAwareBargeIn, rms } from "/static/vad.js";
import { ResilientWS } from "/static/ws.js";

const TARGET_SAMPLE_RATE = 16000;
const BUFFER_SIZE = 4096;
const TTFA_TARGET_MS = 300;

const el = {
  connStatus: document.getElementById("conn-status"),
  btnConnect: document.getElementById("btn-connect"),
  btnMic: document.getElementById("btn-mic"),
  btnStop: document.getElementById("btn-stop"),
  btnCancel: document.getElementById("btn-cancel"),
  btnSend: document.getElementById("btn-send"),
  typedInput: document.getElementById("typed-input"),
  micHint: document.getElementById("mic-hint"),
  vadStart: document.getElementById("vad-start"),
  vadEnd: document.getElementById("vad-end"),
  vadMinSpeech: document.getElementById("vad-min-speech"),
  vadHangover: document.getElementById("vad-hangover"),
  levelBar: document.getElementById("level-bar"),
  vadState: document.getElementById("vad-state"),
  partial: document.getElementById("partial"),
  final: document.getElementById("final"),
  llmResponse: document.getElementById("llm-response"),
  audioState: document.getElementById("audio-state"),
  latencyLog: document.getElementById("latency-log"),
  ttfaStatus: document.getElementById("ttfa-status"),
  worstStage: document.getElementById("worst-stage"),
  sessionStats: document.getElementById("session-stats"),
  turnHistory: document.getElementById("turn-history"),
  eventLog: document.getElementById("event-log"),
};

let ws = null;
let captureCtx = null;
let mediaStream = null;
let processor = null;
let workletNode = null;
let sourceNode = null;
let captureMode = "none"; // worklet | script | none
let isListening = false;
let isAgentSpeaking = false;

let playCtx = null;
let nextPlayTime = 0;
let activeSources = [];

let t0 = null, t1 = null, t3 = null, t5 = null, t6 = null;
let tInterrupt = null, tAudioStopped = null;
let toolLatencyMs = null;
let turnCounter = 0;
let activeTurnId = 0;
let clientState = "idle";

const history = [];
let vad = createVadFromUI();
const barge = new EchoAwareBargeIn({
  margin: 2.0,
  absoluteFloor: 0.018,
  confirmMs: 45,
  baselineAlpha: 0.08,
});

function createVadFromUI() {
  return new EndpointVad({
    startThreshold: parseFloat(el.vadStart?.value) || 0.018,
    endThreshold: parseFloat(el.vadEnd?.value) || 0.010,
    minSpeechMs: parseInt(el.vadMinSpeech?.value, 10) || 100,
    hangoverMs: parseInt(el.vadHangover?.value, 10) || 450,
    minUtteranceMs: 200,
  });
}

function syncVadParams() {
  vad.startThreshold = parseFloat(el.vadStart?.value) || 0.018;
  vad.endThreshold = parseFloat(el.vadEnd?.value) || 0.010;
  vad.minSpeechMs = parseInt(el.vadMinSpeech?.value, 10) || 100;
  vad.hangoverMs = parseInt(el.vadHangover?.value, 10) || 450;
}

["vad-start", "vad-end", "vad-min-speech", "vad-hangover"].forEach((id) => {
  const node = document.getElementById(id);
  if (node) node.addEventListener("change", syncVadParams);
});

function logEvent(msg) {
  const ts = new Date().toISOString().slice(11, 23);
  el.eventLog.textContent = `[${ts}] ${msg}\n` + el.eventLog.textContent.slice(0, 7000);
}

function setConnStatus(text, ok = null) {
  el.connStatus.textContent = text;
  el.connStatus.style.color = ok === true ? "var(--success)" : ok === false ? "var(--danger)" : "var(--muted)";
}

function setVadState(text, speaking = false) {
  el.vadState.textContent = text;
  el.vadState.style.color = speaking ? "var(--success)" : "var(--muted)";
}

function setClientState(s) {
  clientState = s;
  if (s === "listening") setVadState("Listening", false);
  else if (s === "user_speaking") setVadState("User speaking…", true);
  else if (s === "processing") setVadState("Processing…", false);
  else if (s === "agent_speaking") setVadState("Agent speaking (interrupt OK)", false);
  else if (s === "idle") setVadState("Idle", false);
}

function stageDeltas() {
  if (t0 == null) return null;
  const d = {};
  if (t1 != null) d.stt = t1 - t0;
  if (t3 != null) d.llm = t3 - (t1 ?? t0);
  if (t3 != null) d.t0_to_llm = t3 - t0;
  if (t5 != null) d.tts = t5 - (t3 ?? t0);
  if (t5 != null) d.t0_to_tts = t5 - t0;
  if (t6 != null) d.browser = t6 - (t5 ?? t0);
  if (t6 != null) d.ttfa = t6 - t0;
  if (toolLatencyMs != null) d.tool = toolLatencyMs;
  return d;
}

function largestContributor(d) {
  if (!d || d.ttfa == null) return null;
  const candidates = [
    { name: "STT finalization", ms: d.stt ?? 0 },
    { name: "LLM (to first token)", ms: d.llm ?? 0 },
    { name: "TTS (to first audio)", ms: d.tts ?? 0 },
    { name: "Browser playback", ms: d.browser ?? 0 },
  ];
  candidates.sort((a, b) => b.ms - a.ms);
  return candidates[0];
}

function renderCurrentTurn() {
  const d = stageDeltas();
  if (!d) {
    el.latencyLog.textContent = "Waiting for speech end…";
    el.ttfaStatus.textContent = "—";
    el.ttfaStatus.style.color = "var(--muted)";
    if (el.worstStage) el.worstStage.textContent = "";
    return;
  }
  const lines = [];
  lines.push(`TURN #${turnCounter || "—"}  (id=${activeTurnId || "—"})`);
  lines.push(``);
  if (t0 != null) lines.push(`Speech ended (T0)`);
  if (d.stt != null) lines.push(`  STT final              +${d.stt.toFixed(0)} ms`);
  if (d.t0_to_llm != null) lines.push(`  LLM first token        +${d.t0_to_llm.toFixed(0)} ms`);
  if (d.t0_to_tts != null) lines.push(`  TTS first audio        +${d.t0_to_tts.toFixed(0)} ms`);
  if (d.ttfa != null) lines.push(`  Browser playback       +${d.ttfa.toFixed(0)} ms`);
  if (d.tool != null) lines.push(``, `  Tool latency           ${d.tool.toFixed(0)} ms  (separate)`);
  if (tInterrupt != null && tAudioStopped != null) {
    lines.push(``, `  Barge-in stop          ${(tAudioStopped - tInterrupt).toFixed(1)} ms`);
  }
  el.latencyLog.textContent = lines.join("\n");

  if (d.ttfa != null) {
    const ok = d.ttfa < TTFA_TARGET_MS;
    el.ttfaStatus.textContent = `TTFA: ${d.ttfa.toFixed(0)} ms    STATUS: ${ok ? "✓ < 300 ms" : "✗ ≥ 300 ms"}`;
    el.ttfaStatus.style.color = ok ? "var(--success)" : "var(--danger)";
    if (el.worstStage) {
      if (!ok) {
        const worst = largestContributor(d);
        el.worstStage.textContent = worst ? `Largest contributor: ${worst.name} (${worst.ms.toFixed(0)} ms)` : "";
      } else el.worstStage.textContent = "";
    }
  }
}

function finalizeTurn() {
  const d = stageDeltas();
  if (!d || d.ttfa == null) return;
  const worst = d.ttfa >= TTFA_TARGET_MS ? largestContributor(d) : null;
  history.unshift({
    id: turnCounter,
    ttfa: d.ttfa,
    stages: d,
    worst: worst ? `${worst.name} ${worst.ms.toFixed(0)}ms` : null,
    ok: d.ttfa < TTFA_TARGET_MS,
    toolMs: d.tool ?? null,
  });
  if (history.length > 50) history.pop();
  renderHistory();
  renderSessionStats();
}

function renderHistory() {
  if (!el.turnHistory) return;
  if (history.length === 0) {
    el.turnHistory.innerHTML = `<div class="empty-hist">Turns will appear here after each response.</div>`;
    return;
  }
  el.turnHistory.innerHTML = history.map((h) => {
    const br = [
      h.stages.stt != null ? `STT ${h.stages.stt.toFixed(0)}` : null,
      h.stages.t0_to_llm != null ? `LLM ${h.stages.t0_to_llm.toFixed(0)}` : null,
      h.stages.t0_to_tts != null ? `TTS ${h.stages.t0_to_tts.toFixed(0)}` : null,
      h.toolMs != null ? `tool ${h.toolMs.toFixed(0)}` : null,
    ].filter(Boolean).join(" · ");
    return `<div class="turn-row">
      <span class="turn-id">#${h.id}</span>
      <span class="turn-breakdown">${br}</span>
      <span class="turn-ttfa ${h.ok ? "ok" : "bad"}">${h.ttfa.toFixed(0)} ms</span>
      <span class="turn-worst">${h.worst ? "⚠ " + h.worst : ""}</span>
    </div>`;
  }).join("");
}

function renderSessionStats() {
  if (!el.sessionStats) return;
  if (history.length === 0) {
    el.sessionStats.textContent = "No turns yet";
    return;
  }
  const ttfas = history.map((h) => h.ttfa);
  const avg = ttfas.reduce((a, b) => a + b, 0) / ttfas.length;
  const sorted = [...ttfas].sort((a, b) => a - b);
  const p50 = sorted[Math.floor(sorted.length * 0.5)];
  const okCount = history.filter((h) => h.ok).length;
  el.sessionStats.textContent =
    `${history.length} turns · P50 ${p50.toFixed(0)} ms · avg ${avg.toFixed(0)} ms · ${okCount}/${history.length} under 300 ms`;
}

let masterGain = null;

async function ensurePlayCtx() {
  if (!playCtx) {
    playCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 24000,
      latencyHint: "interactive",
    });
    masterGain = playCtx.createGain();
    masterGain.gain.setValueAtTime(1.0, playCtx.currentTime);
    masterGain.connect(playCtx.destination);
  }
  if (playCtx.state === "suspended") await playCtx.resume();
  return playCtx;
}

function base64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const buf = new ArrayBuffer(binary.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
  return buf;
}

function base64ToInt16(b64) {
  const buf = base64ToArrayBuffer(b64);
  return new Int16Array(buf);
}

async function playAudioChunk(data) {
  const isFirst = !!data.is_first;
  const fmt = data.format || "pcm16";
  
  if (fmt === "mp3" || fmt === "audio/mpeg") {
    const arrayBuf = base64ToArrayBuffer(data.audio_b64);
    const ctx = await ensurePlayCtx();
    try {
      const audioBuf = await ctx.decodeAudioData(arrayBuf);
      const source = ctx.createBufferSource();
      source.buffer = audioBuf;
      source.connect(masterGain || ctx.destination);
      const now = ctx.currentTime;
      if (nextPlayTime < now + 0.005) nextPlayTime = now + 0.008;
      source.start(nextPlayTime);
      nextPlayTime += audioBuf.duration;
      activeSources.push(source);
      source.onended = () => { activeSources = activeSources.filter(s => s !== source); };

      if (isFirst && t6 == null) {
        t6 = performance.now();
        isAgentSpeaking = true;
        barge.start();
        el.audioState.textContent = "Playing… (speak to interrupt)";
        el.audioState.style.color = "var(--success)";
        logEvent(`T6 first audio scheduled (+${(t6 - t0).toFixed(0)} ms)`);
        renderCurrentTurn();
        finalizeTurn();
      }
    } catch (err) {
      console.error("Decode error:", err);
    }
  } else {
    playPcm16(base64ToInt16(data.audio_b64), data.sample_rate || 24000, isFirst);
  }
}

async function playPcm16(int16, sampleRate, isFirst) {
  if (!isAgentSpeaking && !isFirst) return;
  const ctx = await ensurePlayCtx();
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  const buffer = ctx.createBuffer(1, float32.length, sampleRate);
  buffer.getChannelData(0).set(float32);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(masterGain || ctx.destination);
  const now = ctx.currentTime;
  if (nextPlayTime < now + 0.005) nextPlayTime = now + 0.008;
  source.start(nextPlayTime);
  nextPlayTime += buffer.duration;
  activeSources.push(source);
  source.onended = () => { activeSources = activeSources.filter(s => s !== source); };

  if (isFirst && t6 == null) {
    t6 = performance.now();
    isAgentSpeaking = true;
    barge.start();
    el.audioState.textContent = "Playing… (speak to interrupt)";
    el.audioState.style.color = "var(--success)";
    logEvent(`T6 first audio scheduled  (+${(t6 - t0).toFixed(0)} ms)`);
    renderCurrentTurn();
    finalizeTurn();
  }
}

function stopPlaybackImmediate(opts = {}) {
  tAudioStopped = performance.now();
  if (playCtx && masterGain) {
    try {
      const now = playCtx.currentTime;
      masterGain.gain.setValueAtTime(masterGain.gain.value, now);
      masterGain.gain.linearRampToValueAtTime(0.0001, now + 0.005);
    } catch (_) {}
  }
  for (const src of activeSources) {
    try { src.stop(0); } catch (_) {}
    try { src.disconnect(); } catch (_) {}
  }
  activeSources = [];
  nextPlayTime = 0;
  if (playCtx && masterGain) {
    try {
      const resetTime = playCtx.currentTime + 0.008;
      masterGain.gain.setValueAtTime(1.0, resetTime);
    } catch (_) {}
  }
  isAgentSpeaking = false;
  barge.stop();
  if (opts.resumeListening) {
    el.audioState.textContent = "Your turn — keep talking";
    el.audioState.style.color = "var(--accent)";
  } else {
    el.audioState.textContent = "Interrupted / Stopped";
    el.audioState.style.color = "var(--warn)";
  }
}

function connect() {
  if (ws && ws.ready) return;
  if (ws) ws.close();

  const persona = document.getElementById("persona-select")?.value || "friend";

  ws = new ResilientWS({
    urlFn: () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const key = localStorage.getItem("VOICE_API_KEY") || new URLSearchParams(location.search).get("api_key") || "";
      const q = key ? `&api_key=${encodeURIComponent(key)}` : "";
      return `${proto}://${location.host}/ws/stt?persona=${encodeURIComponent(persona)}${q}`;
    },
    onMessage: (data) => handleServerEvent(data),
    onStatus: (status, detail) => {
      if (status === "connected") {
        const resumed = detail === "resumed";
        setConnStatus(resumed ? "Reconnected (persistent)" : "Connected (persistent)", true);
        el.btnMic.disabled = false;
        if (el.btnSend) el.btnSend.disabled = false;
        logEvent(resumed ? "WebSocket reconnected" : "WebSocket connected");
        // If mic was active across a drop, user must restart capture (browser MediaStream is local)
        if (resumed && isListening) {
          logEvent("Connection resumed – mic still local; continue speaking");
        }
      } else if (status === "connecting") {
        setConnStatus("Connecting…", null);
      } else if (status === "reconnecting") {
        setConnStatus(`Reconnecting… ${detail || ""}`, false);
        el.btnMic.disabled = true;
        logEvent(`WS ${status}: ${detail || ""}`);
      } else if (status === "dead") {
        setConnStatus("Connection dead – recovering", false);
        logEvent(`WS dead: ${detail || ""}`);
      } else if (status === "disconnected") {
        setConnStatus("Disconnected", false);
        el.btnMic.disabled = true;
        if (el.btnSend) el.btnSend.disabled = true;
        el.btnStop.disabled = true;
        el.btnCancel.disabled = true;
        logEvent("WebSocket disconnected");
      } else if (status === "error") {
        setConnStatus("Error", false);
        logEvent(`WS error: ${detail || ""}`);
      }
    },
    heartbeatMs: 5000,
    deadMs: 15000,
    maxBackoffMs: 8000,
  });
  ws.connect();
}

function handleServerEvent(data) {
  const scoped = new Set([
    "partial","final","llm_start","llm_first_token","llm_token","llm_done",
    "first_phrase","tts_start","tts_first_audio","tts_audio","tts_done",
    "tool_call","tool_start","tool_result","llm_cancelled","tts_cancelled",
  ]);
  if (scoped.has(data.type) && data.turn_id != null && activeTurnId !== 0 && data.turn_id !== activeTurnId) {
    logEvent(`DROP late ${data.type} turn=${data.turn_id} (active=${activeTurnId})`);
    return;
  }

  switch (data.type) {
    case "turn_start":
      activeTurnId = data.turn_id;
      setClientState("processing");
      logEvent(`Turn #${data.turn_id} started`);
      break;
    case "partial":
      el.partial.textContent = data.text || "…";
      break;
    case "final":
      t1 = performance.now();
      el.final.textContent = data.text;
      el.partial.textContent = "";
      logEvent(`FINAL "${data.text}"`);
      renderCurrentTurn();
      break;
    case "llm_start":
      el.llmResponse.textContent = "";
      el.btnCancel.disabled = false;
      logEvent("LLM start");
      break;
    case "llm_first_token":
      t3 = performance.now();
      el.llmResponse.textContent = data.text;
      logEvent(`LLM first token (+${(t3 - t0).toFixed(0)} ms)`);
      renderCurrentTurn();
      break;
    case "llm_token":
      el.llmResponse.textContent = data.accumulated || el.llmResponse.textContent + data.text;
      break;
    case "llm_done":
      logEvent("LLM done");
      break;
    case "first_phrase":
      logEvent(`First phrase → TTS: "${(data.text || "").slice(0, 48)}"`);
      break;
    case "tts_start":
      logEvent("TTS start");
      nextPlayTime = 0;
      break;
    case "tts_first_audio":
      t5 = performance.now();
      setClientState("agent_speaking");
      logEvent(`TTS first audio recv (+${(t5 - t0).toFixed(0)} ms)`);
      renderCurrentTurn();
      break;
    case "tts_audio": {
      if (!isAgentSpeaking && t6 != null) break;
      playAudioChunk(data);
      break;
    }
    case "tts_done":
      el.btnCancel.disabled = true;
      isAgentSpeaking = false;
      barge.stop();
      setClientState("listening");
      el.audioState.textContent = "Done";
      logEvent("TTS done");
      break;
    case "tool_call":
      logEvent(`TOOL CALL  ${data.name}`);
      el.llmResponse.textContent = `(calling ${data.name}…)`;
      break;
    case "tool_start":
      logEvent(`TOOL START ${data.name}`);
      break;
    case "tool_result":
      logEvent(`TOOL RESULT ${data.name}  ${data.ok ? "ok" : "ERR"}  ${data.latency_ms?.toFixed(0) ?? "?"} ms`);
      if (data.latency_ms != null) toolLatencyMs = data.latency_ms;
      renderCurrentTurn();
      break;
    case "cancelled":
    case "llm_cancelled":
    case "tts_cancelled":
      el.btnCancel.disabled = true;
      // If the user is already in a new utterance, don't clobber that state
      if (clientState !== "user_speaking" && clientState !== "processing") {
        activeTurnId = 0;
        setClientState("listening");
        el.audioState.textContent = "Your turn — keep talking";
        el.audioState.style.color = "var(--accent)";
      }
      logEvent("Server confirmed cancel – ready for next utterance");
      renderCurrentTurn();
      break;

    case "server_hello":
      logEvent("Server hello (session ok)");
      if (data.deadlines) {
        logEvent(`Deadlines: STT≤${data.deadlines.stt_final_ms} LLM≤${data.deadlines.llm_first_token_ms} tool≤${data.deadlines.tool_ms} turn≤${data.deadlines.turn_total_ms} ms`);
      }
      break;
    case "timeout":
      activeTurnId = 0;
      stopPlaybackImmediate();
      el.btnCancel.disabled = true;
      setClientState("listening");
      el.llmResponse.textContent = `(timeout: ${data.stage || "unknown"})`;
      el.audioState.textContent = "Timed out";
      el.audioState.style.color = "var(--danger)";
      logEvent(`TIMEOUT stage=${data.stage} reason=${data.reason} +${data.ms_since_speech_end?.toFixed?.(0) ?? "?"}ms`);
      if (el.ttfaStatus) {
        el.ttfaStatus.textContent = `TIMEOUT at ${data.stage}`;
        el.ttfaStatus.style.color = "var(--danger)";
      }
      break;
    case "error":
      logEvent("ERROR " + (data.message || ""));
      break;
  }
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

function downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) result[i] = buffer[Math.round(i * ratio)] || 0;
  return result;
}

function doBargeIn(now = performance.now()) {
  if (!isAgentSpeaking) return;
  tInterrupt = now;
  const interruptedTurn = activeTurnId;
  logEvent("BARGE-IN detected — agent yielded; this speech is the next turn");
  stopPlaybackImmediate({ resumeListening: true });
  if (ws && ws.ready) ws.sendJson({ type: "cancel", turn_id: interruptedTurn });
  activeTurnId = 0;
  el.llmResponse.textContent = "";
  el.btnCancel.disabled = true;
  vad.forceSpeechStart(now);
  t0 = t1 = t3 = t5 = t6 = null;
  toolLatencyMs = null;
  setClientState("user_speaking");
  renderCurrentTurn();
}

function onSpeechEnd() {
  turnCounter += 1;
  t0 = performance.now();
  t1 = t3 = t5 = t6 = null;
  tInterrupt = null;
  tAudioStopped = null;
  toolLatencyMs = null;
  setClientState("processing");
  logEvent(`VAD speech end (T0)  local turn #${turnCounter}`);
  if (ws && ws.ready) ws.sendJson({ type: "finalize" });
  renderCurrentTurn();
}


async function startMic() {
  if (isListening) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    logEvent("Mic error: " + msg);
    if (el.micHint) {
      el.micHint.textContent =
        "Microphone blocked in this preview. Type a message below — you will still hear the reply.";
      el.micHint.classList.add("warn");
    }
    if (el.typedInput) el.typedInput.focus();
    return;
  }

  syncVadParams();
  vad.reset();

  captureCtx = new (window.AudioContext || window.webkitAudioContext)();
  await captureCtx.resume();
  sourceNode = captureCtx.createMediaStreamSource(mediaStream);

  const onFrame = (input) => {
    if (!isListening || !ws || !ws.ready) return;
    const down = downsample(input, captureCtx.sampleRate, TARGET_SAMPLE_RATE);
    ws.send(floatTo16BitPCM(down));

    const level = rms(down);
    const now = performance.now();
    el.levelBar.style.width = Math.min(100, level * 800) + "%";

    if (isAgentSpeaking) {
      const r = barge.process(level, now);
      const hot = level >= r.threshold;
      el.levelBar.style.background = hot ? "var(--danger)" : "var(--accent)";
      if (r.barge) {
        logEvent(
          `Barge residual gate  level=${level.toFixed(3)} baseline=${r.baseline.toFixed(3)} thresh=${r.threshold.toFixed(3)}`
        );
        doBargeIn(now);
      }
      return;
    }

    el.levelBar.style.background = level > vad.startThreshold ? "var(--success)" : "var(--accent)";
    const ev = vad.process(level, now);
    if (ev === "speech_start") {
      setClientState("user_speaking");
      logEvent("VAD speech_start (hysteresis + minSpeech)");
    } else if (ev === "speech_end") {
      onSpeechEnd();
    } else if (ev === "speech_end_short") {
      logEvent("VAD ignored short utterance");
      setClientState("listening");
    }
  };

  // Prefer AudioWorklet (R6); fall back to ScriptProcessor
  let usedWorklet = false;
  if (captureCtx.audioWorklet) {
    try {
      await captureCtx.audioWorklet.addModule("/static/pcm-worklet.js");
      workletNode = new AudioWorkletNode(captureCtx, "pcm-capture-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
      });
      workletNode.port.onmessage = (ev) => {
        if (ev.data && ev.data.type === "samples") {
          onFrame(ev.data.samples);
        }
      };
      sourceNode.connect(workletNode);
      // Keep graph alive on some browsers
      const mute = captureCtx.createGain();
      mute.gain.value = 0;
      sourceNode.connect(mute);
      mute.connect(captureCtx.destination);
      usedWorklet = true;
      captureMode = "worklet";
      logEvent("Capture: AudioWorklet (low jitter)");
    } catch (err) {
      logEvent("AudioWorklet failed, falling back: " + err.message);
      workletNode = null;
    }
  }

  if (!usedWorklet) {
    processor = captureCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);
    sourceNode.connect(processor);
    processor.connect(captureCtx.destination);
    processor.onaudioprocess = (e) => {
      onFrame(e.inputBuffer.getChannelData(0));
    };
    captureMode = "script";
    logEvent("Capture: ScriptProcessor (fallback)");
  }

  isListening = true;
  el.btnMic.disabled = true;
  el.btnStop.disabled = false;
  setClientState("listening");
  el.final.textContent = "Listening…";
  el.partial.textContent = "…";
  el.llmResponse.textContent = "—";
  el.audioState.textContent = "No audio yet";
  logEvent("Mic started – mode=" + captureMode);
}

function stopMic() {
  isListening = false;
  if (workletNode) {
    try { workletNode.port.postMessage({ type: "stop" }); } catch (_) {}
    try { workletNode.disconnect(); } catch (_) {}
    workletNode = null;
  }
  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
    processor = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (captureCtx) {
    captureCtx.close().catch(() => {});
    captureCtx = null;
  }
  stopPlaybackImmediate();
  vad.reset();
  captureMode = "none";
  el.btnMic.disabled = !(ws && ws.ready);
  el.btnStop.disabled = true;
  setClientState("idle");
  el.levelBar.style.width = "0%";
  logEvent("Mic stopped");
}

el.btnConnect.addEventListener("click", connect);
el.btnMic.addEventListener("click", startMic);
el.btnStop.addEventListener("click", stopMic);
el.btnCancel.addEventListener("click", () => {
  tInterrupt = performance.now();
  const interruptedTurn = activeTurnId;
  stopPlaybackImmediate({ resumeListening: true });
  activeTurnId = 0;
  if (ws && ws.ready) ws.sendJson({ type: "cancel", turn_id: interruptedTurn });
  vad.reset();
  setClientState("listening");
  logEvent("Manual cancel — listening");
  renderCurrentTurn();
});

function sendTyped() {
  const text = (el.typedInput?.value || "").trim();
  if (!text || !ws || !ws.ready) return;
  turnCounter += 1;
  t0 = performance.now();
  t1 = t3 = t5 = t6 = null;
  setClientState("processing");
  el.final.textContent = text;
  el.llmResponse.textContent = "";
  logEvent(`Typed (T0) "${text}"`);
  ws.sendJson({ type: "user_text", text });
  el.typedInput.value = "";
  renderCurrentTurn();
}
if (el.btnSend) el.btnSend.addEventListener("click", sendTyped);
if (el.typedInput) {
  el.typedInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendTyped();
    }
  });
}

// Quick HTTP Pipeline Test (Direct Hugging Face LLM + Neural Voice Playback)
const btnQuickTest = document.getElementById("btn-quick-test");
if (btnQuickTest) {
  btnQuickTest.addEventListener("click", async () => {
    const inputVal = (el.typedInput?.value || "").trim() || "Hey, what are you doing today?";
    btnQuickTest.disabled = true;
    btnQuickTest.textContent = "⏳ Generating...";
    logEvent(`Direct Test: Sending "${inputVal}" to Hugging Face Llama-3.3`);
    
    try {
      const resp = await fetch("/api/talk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputVal })
      });
      const data = await resp.json();
      logEvent(`Direct Test Reply (${data.llm_provider}): "${data.reply_text}"`);
      el.llmResponse.textContent = data.reply_text;
      
      if (data.audio_b64) {
        logEvent(`Direct Test Audio: Playing ${data.audio_bytes} bytes`);
        playAudioChunk({
          audio_b64: data.audio_b64,
          format: "pcm16",
          sample_rate: data.sample_rate || 24000,
          is_first: true
        });
      }
    } catch (err) {
      logEvent(`Direct Test Error: ${err.message}`);
    } finally {
      btnQuickTest.disabled = false;
      btnQuickTest.textContent = "⚡ Test Hugging Face Voice";
    }
  });
}

if (window.self !== window.top && el.micHint) {
  el.micHint.textContent =
    "This preview may block the microphone. If Start Mic fails, type below — you still hear the reply.";
}

// ===== THREAT INTELLIGENCE & SOC DASHBOARD POLLER =====
async function fetchThreatStats() {
  try {
    const res = await fetch("/api/threats/stats");
    if (!res.ok) return;
    const data = await res.json();
    
    const elWasted = document.getElementById("stat-time-wasted");
    const elCost = document.getElementById("stat-cost-inflicted");
    const elMules = document.getElementById("stat-mule-count");
    const elCrypto = document.getElementById("stat-crypto-count");

    if (elWasted) elWasted.textContent = data.total_time_wasted_formatted || "0m 0s";
    if (elCost) elCost.textContent = data.scammer_financial_loss_usd || "$0.00";
    if (elMules) elMules.textContent = String(data.unique_mule_accounts_captured || 0);
    if (elCrypto) elCrypto.textContent = String(data.unique_crypto_wallets_captured || 0);
  } catch (_) {}
}

async function fetchThreatFeed() {
  try {
    const res = await fetch("/api/threats/feed?limit=10");
    if (!res.ok) return;
    const records = await res.json();
    const listEl = document.getElementById("threat-feed-list");
    if (!listEl) return;

    if (!records || records.length === 0) {
      listEl.innerHTML = `<div class="empty-hist">No threat records captured yet.</div>`;
      return;
    }

    listEl.innerHTML = records.map((r) => {
      const mules = r.mule_accounts?.length ? `· Mule: <b>${r.mule_accounts.join(", ")}</b>` : "";
      const rats = r.remote_access_codes?.length ? `· AnyDesk: <b>${r.remote_access_codes.join(", ")}</b>` : "";
      const wallets = r.crypto_wallets?.length ? `· Crypto: <b>${r.crypto_wallets[0].slice(0, 10)}...</b>` : "";
      const dur = `${Math.round(r.duration_sec)}s`;
      
      return `<div class="turn-row" style="border-left: 3px solid #ef4444; padding: 6px 10px; margin-bottom: 6px; background: rgba(239,68,68,0.05);">
        <span class="turn-id" style="color: #ef4444; font-weight: bold;">${r.id}</span>
        <span class="turn-breakdown">${r.caller_id} · ${r.scam_category} ${mules} ${rats} ${wallets}</span>
        <span class="turn-ttfa" style="color: #10b981;">Wasted ${dur}</span>
      </div>`;
    }).join("");
  } catch (_) {}
}

const btnRefreshThreats = document.getElementById("btn-refresh-threats");
if (btnRefreshThreats) {
  btnRefreshThreats.addEventListener("click", () => {
    fetchThreatStats();
    fetchThreatFeed();
  });
}

// Initial load & 10-second polling
fetchThreatStats();
fetchThreatFeed();
setInterval(fetchThreatStats, 8000);
setInterval(fetchThreatFeed, 12000);

logEvent("Conversational barge-in & Threat Intelligence Engine active.");
setConnStatus("Disconnected");
