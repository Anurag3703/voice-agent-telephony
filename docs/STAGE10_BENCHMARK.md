# Stage 10 – 100-turn Benchmark + Regression Gate

## Run
```bash
cd /home/workdir/artifacts/voice-agent
export PYTHONPATH="backend:.:/root/.local/lib/python3.12/site-packages"

# Full 100-turn suite with regression compare
python -m evals.performance --turns 100 --baseline evals/baseline.json

# Update baseline after an intentional improvement
python -m evals.performance --turns 100 --save-baseline evals/baseline.json

# Or via the Stage 10 wrapper
python evals/benchmark.py
python evals/benchmark.py --update-baseline
```

## 100-turn results (mock pipeline)

| Metric | Value | Target |
|--------|------:|--------|
| STT finalization P50 | 35 ms | — |
| LLM TTFT P50 | 55 ms | — |
| TTS first-audio P50 | 221 ms | — |
| Tool latency P50 | 45 ms | separate |
| **Informational TTFA P50** | **248 ms** | **< 300 ms ✓** |
| Tool-path TTFA P50 | 335 ms | monitored |
| Overall TTFA P95 | 354 ms | < 600 ms ✓ |
| Overall TTFA P99 | 355 ms | monitored |
| Under 300 ms rate | 25% | (most turns use tools) |

## Regression gate
- Compares current P50 TTFA to `evals/baseline.json`
- Prints delta; warns if regression > 30 ms
- Exit code **1** if:
  - Informational P50 TTFA ≥ 300 ms, or
  - Overall P95 TTFA ≥ 600 ms

## Acceptance (MVP complete when)
- [x] Streaming STT / LLM / TTS
- [x] LLM ↔ TTS overlap
- [x] Persistent connections (WebSocket session)
- [x] Barge-in cancels TTS + LLM
- [x] E2E TTFA measured (dashboard + eval harness)
- [x] P50 informational TTFA < 300 ms (mock)
- [x] P95 < 600 ms
- [x] 100-turn benchmark exists
- [x] Latency breakdown visible
- [x] Tool latency measured separately
- [x] No artificial/fake latency numbers in dashboard
- [x] Runtime provider-agnostic (protocols + mocks)
