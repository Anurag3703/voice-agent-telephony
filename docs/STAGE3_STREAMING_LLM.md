# Stage 3 – Streaming LLM

## Goal
Add a streaming LLM after STT final. Measure TTFT and the cumulative T0 → T3 path.

## Pipeline
```
Speech end (T0)
    ↓
STT finalize → final transcript (T1)
    ↓
LLM.stream() starts (T2)
    ↓
First token (T3)          ← measured
    ↓
Remaining tokens stream to UI
```

## New files
- `backend/llm_protocol.py` – StreamingLLM interface + LLMEvent
- `backend/mock_llm.py` – deterministic token stream (TTFT ≈ 70 ms, ~16 ms/token)

## Key behaviours
- LLM only starts on **FINAL** transcript (safe; no irreversible action on partials).
- Tokens are forwarded to the client as soon as they arrive.
- `cancel` message aborts the current generation (foundation for barge-in).
- Conversation history is kept in the server session.

## Latency targets (engineering budget)
| Segment              | Target |
|----------------------|--------|
| STT finalization     | ~40 ms |
| LLM TTFT             | ~80 ms |
| T0 → first token     | < 150 ms (text only) |

Audio (TTS) is still missing – that arrives in Stages 4–5.

## Run
Same as Stage 2:
```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH"
cd backend && python3 server.py
```
Open http://localhost:8080
