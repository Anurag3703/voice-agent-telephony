# Stage 6 – Fast Barge-in / Interruption

## Requirement
When the user starts speaking while the agent is speaking:

1. STOP AUDIO PLAYBACK
2. CANCEL TTS
3. CANCEL CURRENT LLM STREAM
4. DISCARD UNSPOKEN RESPONSE
5. SWITCH TO LISTENING

**Target:** interruption detected → agent audio stops **< 100 ms**

## Implementation

### Client
- Mic stays open (continuous listening)
- While `isAgentSpeaking`, energy VAD with a slightly higher threshold detects user speech
- On detect:
  - Immediately `source.stop(0)` on every active BufferSource + close AudioContext
  - Send `{"type":"cancel"}` to server
  - Clear partial agent text
  - Reset turn timers and continue streaming mic audio for the new utterance

### Server
- `cancel` message:
  - Sets `cancelled = True` first
  - Calls `llm.cancel()` + `tts.cancel()` (flags checked in generators)
  - Cancels asyncio tasks and awaits them with `return_exceptions=True`
  - Replies `{"type":"cancelled"}`

### Measurement
Latency panel shows:
```
BARGE-IN
  detect → audio stop   X.X ms
```

## Notes
- Echo cancellation is enabled on the mic constraint; threshold is raised slightly during agent speech to reduce false barge-ins.
- Real acoustic echo cancellation / residual echo suppression will be needed for production hardware.
