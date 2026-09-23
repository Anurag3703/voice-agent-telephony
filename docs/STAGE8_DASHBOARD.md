# Stage 8 – Latency Dashboard

## What it shows

### Current turn panel
```
TURN #12

Speech ended (T0)
  STT final              +38 ms
  LLM first token        +92 ms
  TTS first audio        +181 ms
  Browser playback       +244 ms

  Tool latency           45 ms  (separate)

TTFA: 244 ms    STATUS: ✓ < 300 ms
```

When over budget:
```
TTFA: 421 ms    STATUS: ✗ ≥ 300 ms
Largest contributor: LLM (to first token) (180 ms)
```

### Session stats
`12 turns · P50 241 ms · avg 256 ms · 11/12 under 300 ms`

### Turn history
Scrollable list of recent turns with compact breakdown and TTFA status.

## Notes
- TTFA = T6 − T0 (first *playable* audio, not first network byte)
- Tool latency is displayed but does not redefine TTFA
- Barge-in stop time is shown when an interruption occurs
- History keeps the last 50 turns in memory
