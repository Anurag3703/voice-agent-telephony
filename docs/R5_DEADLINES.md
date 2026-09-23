# R5 – Provider Timeouts + Hard Deadlines

## Config (`backend/deadlines.py`)

| Deadline | Default | Meaning |
|----------|--------:|---------|
| stt_final_ms | 2000 | T0 → STT final |
| llm_first_token_ms | 2500 | LLM decision / first token |
| tool_ms | 800 | Single tool execution |
| llm_speech_first_token_ms | 2000 | Speech LLM pass |
| tts_first_audio_ms | 2000 | TTS start → first audio |
| turn_first_audio_ms | 8000 | Soft ceiling T0 → first audio |
| turn_total_ms | 20000 | Hard watchdog for whole turn |

These are **safety ceilings**, not the 300 ms product target.

## Behaviour
- Exceeding a stage limit → `cancel_turn` → client gets `{"type":"timeout","stage":...}`
- Turn epoch invalidated (same as user cancel)
- Tool path uses `asyncio.wait_for` in addition to tools.py internal timeout
- Watchdog task sleeps `turn_total_ms` then force-cancels if still active

## Client
- Shows timeout stage in agent text + audio state
- Logs deadline values from `server_hello`
