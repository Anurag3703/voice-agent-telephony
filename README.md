# Low-Latency Voice Agent

**Hard requirement:** P50 TTFA (informational) < 300 ms, P95 < 600 ms.  
**Barge-in:** user speech → audio stop < 100 ms.  
**Tools:** latency measured separately.

## Stages
| Stage | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Browser audio playback latency |
| 2 | ✅ | Streaming STT |
| 3 | ✅ | Streaming LLM |
| 4 | ✅ | Streaming TTS + overlap |
| 5 | ✅ | Tightened chunker / first phrase |
| 6 | ✅ | Fast barge-in / continuous listening |
| 7 | ✅ | Local/fake tools |
| 8 | ✅ | Latency dashboard |
| 9 | ✅ | Automated evaluations |
| 10 | ✅ | 100-turn benchmark + regression gate |
| R1 | ✅ | Turn epochs + state machine (robustness) |
| R2 | ✅ | Hysteresis VAD + endpointing |
| R3 | ✅ | Echo-aware residual barge-in |
| R4 | ✅ | WS heartbeats + auto-reconnect |
| R5 | ✅ | Provider timeouts + turn deadlines |
| R6 | ✅ | AudioWorklet mic capture |
| R7 | ✅ | Context cap + idempotent tools |
| P0 | ✅ | Real STT/LLM/TTS provider adapters (env) |

## Quick start – server
```bash
export PATH="/root/.local/bin:$PATH"
export PYTHONPATH="/root/.local/lib/python3.12/site-packages"
cd backend && python3 server.py
# open http://localhost:8080
```

## Quick start – benchmark
```bash
export PYTHONPATH="backend:.:/root/.local/lib/python3.12/site-packages"
python -m evals.performance --turns 100 --baseline evals/baseline.json
python evals/benchmark.py --update-baseline
```

## 100-turn mock results
- Informational TTFA P50: **~248 ms** ✓
- Overall P95: **~354 ms** ✓
- Tool-path P50: ~335 ms (tool time included; reported separately)

## Architecture
```
Mic → VAD → Streaming STT → (optional Tool) → Streaming LLM
     → Phrase chunker → Streaming TTS → Web Audio playback
```
Provider-agnostic protocols in `backend/*_protocol.py` with deterministic mocks for CI.


## Real providers (P0)
```bash
# Free-tier friendly
export LLM_PROVIDER=groq
export TTS_PROVIDER=elevenlabs
export GROQ_API_KEY=gsk_...
export ELEVENLABS_API_KEY=...

# Optional STT
export STT_PROVIDER=deepgram   # needs DEEPGRAM_API_KEY
# default STT/LLM/TTS = mock without keys

cd backend && python3 server.py
python -m evals.provider_smoke
```
Without keys, providers automatically fall back to mocks.


## Steps A–D (post-P0)
| Step | Status |
|------|--------|
| A Faster path (chunker, max_tokens, prompt) | ✅ |
| B Deepgram STT adapter | ✅ (needs `DEEPGRAM_API_KEY`) |
| C Barge-in eval `python -m evals.bargein_eval` | ✅ |
| D Optional `VOICE_API_KEY` + session_id | ✅ |

Live TTFA (server): `python -m evals.live_ttfa --turns 3`
