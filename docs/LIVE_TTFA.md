# Live TTFA (server-side)

```bash
export LLM_PROVIDER=groq TTS_PROVIDER=elevenlabs STT_PROVIDER=mock
export GROQ_API_KEY=... GROQ_LLM_MODEL=openai/gpt-oss-120b
export ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=sOTW4BKyQlWYiVcj0TpO
PYTHONPATH=backend:. python -m evals.live_ttfa --turns 3
```

## Sample results (3 turns, mock STT)
| Metric | P50 | P95 |
|--------|----:|----:|
| LLM TTFT | ~434 ms | ~569 ms |
| TTS first audio | ~605 ms | ~752 ms |
| T0 → first audio | ~640 ms | ~788 ms |

Browser T6 adds output latency on top. Target &lt;300 ms needs faster model/TTS or more aggressive first-phrase + regional edge.
