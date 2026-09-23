# Stage 4 – Streaming TTS

## Goal
Stream TTS audio as soon as the first safe phrase is available from the LLM.
LLM token generation and TTS audio generation **overlap**.

## Pipeline
```
T0  speech end
T1  STT final
T2  LLM start
T3  LLM first token
    ↓ phrase chunker (min ~18 chars / punctuation)
T4  TTS start
T5  first audio chunk leaves server / arrives client
T6  first audio scheduled in Web Audio (playable)
```

## New pieces
- `tts_protocol.py` – StreamingTTS interface
- `mock_tts.py` – PCM16 @ 24 kHz, ~90 ms first-audio, chunked
- `chunker.py` – ResponseChunker (min_chars / max_chars / split on punctuation)
- Server now runs LLM and TTS concurrently; text phrases are queued to TTS as soon as the chunker flushes
- Frontend plays PCM chunks with minimal buffering (`latencyHint: "interactive"`)

## Expected mock TTFA (T6 − T0)
Roughly:
- STT finalization     ~40 ms
- LLM TTFT             ~70 ms
- First phrase form    ~30–60 ms (depends on tokens)
- TTS first audio      ~90 ms
- Browser schedule     ~10–30 ms
→ **typically 220–280 ms** under the 300 ms P50 target
