# R6 – AudioWorklet Capture

## Why
`ScriptProcessorNode` runs on the main thread and is deprecated. Under UI load
it increases capture jitter and can delay VAD / barge-in.

`AudioWorklet` runs on the audio rendering thread and posts frames to main
via `MessagePort` — lower jitter, better real-time behavior.

## Files
- `frontend/pcm-worklet.js` – `pcm-capture-processor`
- `frontend/app.js` – prefers worklet; falls back to ScriptProcessor

## Behaviour
1. `audioWorklet.addModule("/static/pcm-worklet.js")`
2. `AudioWorkletNode` receives mono frames, posts `{type:"samples", samples}`
3. Main thread: downsample → PCM16 → WebSocket + VAD/barge-in
4. On failure → ScriptProcessor path (logged)

## Note
Worklet module URL must be same-origin (served from `/static/`).
