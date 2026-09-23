# Steps A–D

| Step | Status | What |
|------|--------|------|
| **A** Faster path | ✅ | Tighter chunker (8/16/90), short system prompt, max_tokens=120, Groq default model |
| **B** Real STT | ✅ | Deepgram adapter ready (`STT_PROVIDER=deepgram` + `DEEPGRAM_API_KEY`) |
| **C** Barge-in eval | ✅ | `python -m evals.bargein_eval` |
| **D** Auth + sessions | ✅ | Optional `VOICE_API_KEY`; `session_id` on `server_hello` |

## Run
```bash
# A – measure
export LLM_PROVIDER=groq TTS_PROVIDER=elevenlabs STT_PROVIDER=mock
export GROQ_API_KEY=... GROQ_LLM_MODEL=openai/gpt-oss-120b
export ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=sOTW4BKyQlWYiVcj0TpO
PYTHONPATH=backend:. python -m evals.live_ttfa --turns 3

# C – barge-in (mock is fine)
PYTHONPATH=backend:. python -m evals.bargein_eval --turns 5

# D – enforce API key
export VOICE_API_KEY=dev-secret
# browser: http://localhost:8080/?api_key=dev-secret
```
