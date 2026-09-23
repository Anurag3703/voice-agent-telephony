# Stage 7 – Tools

## Goal
Add tool calling without destroying the latency path.

- Informational questions → direct speech (no tool)
- Tool-required questions → LLM tool_call → async tool → LLM summary → TTS
- Tool latency measured **separately** from TTFA
- Tools have `timeout_ms` and run via `asyncio.wait_for`

## Fake tools (deterministic)
| Tool | Typical latency | Purpose |
|------|-----------------|---------|
| `get_vehicle_status` | ~45 ms | location, lock, odometer, alerts |
| `get_battery` | ~30 ms | percent, range, charging |
| `get_climate` | ~25 ms | cabin / target temp |
| `honk_flash` | ~80 ms | locate vehicle |

## Example utterances
- “What’s my battery level?”
- “Where’s my car?”
- “What’s the cabin temperature?”
- “Honk the horn”
- “Hello” → no tool, direct greeting

## Flow
```
STT final
  → LLM (may emit tool_call)
      → run_tool (timed)
      → inject tool result
  → LLM summary tokens
  → chunker → TTS → audio
```

TTFA still measures speech-end → first playable audio.
Tool time is reported as its own line in the latency panel.
