# R7 – Context Cap + Idempotent Tools

## Context cap (`backend/context.py`)
- Keeps system prompts
- Retains the most recent **12** dialogue messages once total exceeds **24**
- Inserts a short system note when history was truncated
- Avoids starting the window on a lone `tool` message

Applied after each user utterance and each assistant reply.

## Idempotent tools (`backend/tools.py`)
| Tool | Idempotent |
|------|------------|
| get_vehicle_status | yes |
| get_battery | yes |
| get_climate | yes |
| honk_flash | **no** (deduped by `tool_call_id`) |

Non-idempotent tools cache results by `tool_call_id` for the session so retries
do not double-execute (e.g. double honk).

## Why
Unbounded context grows LLM latency and cost. Non-idempotent tools without
dedupe cause user-visible double actions on retry/timeout paths.
