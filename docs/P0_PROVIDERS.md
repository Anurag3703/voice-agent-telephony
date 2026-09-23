# P0 – Real Provider Adapters

## Status
| Step | Work | Status |
|------|------|--------|
| P0.1 | Provider registry + env selection | ✅ |
| P0.2 | OpenAI streaming LLM | ✅ |
| P0.3 | Deepgram streaming STT | ✅ |
| P0.4 | OpenAI streaming TTS | ✅ |
| P0.5 | Wire into server | ✅ |
| P0.5b | **Groq LLM + ElevenLabs TTS** (free-tier path) | ✅ |
| P0.6 | Smoke tests | ✅ |

## Free-tier recommended stack
```bash
export STT_PROVIDER=mock          # or deepgram + DEEPGRAM_API_KEY (trial)
export LLM_PROVIDER=groq
export TTS_PROVIDER=elevenlabs
export GROQ_API_KEY=gsk_...
export ELEVENLABS_API_KEY=...
# optional:
export GROQ_LLM_MODEL=llama-3.3-70b-versatile
export ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
export ELEVENLABS_MODEL_ID=eleven_flash_v2_5
```

## All providers
| Layer | Values | Key |
|-------|--------|-----|
| STT | `mock`, `deepgram` | `DEEPGRAM_API_KEY` |
| LLM | `mock`, `openai`, **`groq`** | `OPENAI_API_KEY` / **`GROQ_API_KEY`** |
| TTS | `mock`, `openai`, **`elevenlabs`** | `OPENAI_API_KEY` / **`ELEVENLABS_API_KEY`** |

Missing keys → automatic mock fallback (safe default).

## Smoke tests
```bash
export PYTHONPATH=backend:.
python -m evals.provider_smoke
```
