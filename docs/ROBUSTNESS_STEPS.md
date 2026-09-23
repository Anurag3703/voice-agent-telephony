# Robustness Roadmap

Build in this order. Each step should stay measurable (dashboard + evals).

| Step | Name | Goal |
|------|------|------|
| **R1** | Turn epochs + state machine | ✅ Done |
| **R2** | Better VAD + endpointing | ✅ Hysteresis VAD, min speech, hangover, min utterance |
| **R3** | Echo-aware barge-in | ✅ Adaptive residual gate + confirm window |
| **R4** | WebSocket heartbeats + reconnect | ✅ Ping/pong, dead detect, backoff reconnect |
| **R5** | Provider timeouts + hard deadlines | ✅ Stage ceilings + turn watchdog |
| **R6** | AudioWorklet capture | ✅ Worklet capture + ScriptProcessor fallback |
| **R7** | Context cap + idempotent tools | ✅ Cap history + tool_call_id dedupe |

## R1 acceptance
- [ ] Server and client share `turn_id` (monotonic per session)
- [ ] `cancel` increments epoch / invalidates current turn
- [ ] Late `tts_audio` / `llm_token` for old turns are ignored
- [ ] Explicit client states: idle → listening → user_speaking → processing → agent_speaking
- [ ] Dashboard still reports TTFA; barge-in still works


## Robustness track complete
R1–R7 address turn safety, VAD, barge-in, reconnect, deadlines, capture jitter, and multi-turn stability while protecting TTFA measurement.
