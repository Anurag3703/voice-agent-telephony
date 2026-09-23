# Stage 9 – Automated Evaluations

## Run
```bash
cd /home/workdir/artifacts/voice-agent
export PYTHONPATH="backend:.:/root/.local/lib/python3.12/site-packages"

python -m evals.performance                  # 30 turns default
python -m evals.performance --turns 100
python -m evals.performance --turns 50 --save-baseline evals/baseline.json
python -m evals.performance --turns 50 --baseline evals/baseline.json
```

## What it measures
Each turn drives the mock pipeline (no browser):

| Metric | Meaning |
|--------|---------|
| STT finalization | T0 → STT FINAL |
| LLM TTFT | LLM start → first token / tool decision |
| Tool latency | Tool execution only (separate) |
| TTS first-audio | TTS start → first audio chunk |
| TTFA proxy | T0 → first TTS audio (server-side) |

## Reporting
- **Informational TTFA P50** – turns with no tool (primary <300 ms target)
- **Tool-path TTFA P50** – includes tool time (reported separately)
- P90 / P95 / P99 overall
- Regression delta vs `--baseline` JSON

## Design notes
- TTS uses `realtime_pace=False` so evals measure pipeline latency, not spoken duration
- Exit code 1 if informational P50 ≥ 300 ms or overall P95 ≥ 600 ms
- Deterministic mocks → stable CI numbers
