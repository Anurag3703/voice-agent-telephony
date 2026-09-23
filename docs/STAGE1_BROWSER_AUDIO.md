# Stage 1 – Browser Audio Round-Trip / Playback Latency

## Goal
Quantify the **browser contribution** to the overall Time-To-First-Audio (TTFA) budget.

We measure the path:

```
TTS audio chunk arrives in JS
        ↓
create AudioBuffer + BufferSource
        ↓
source.start()
        ↓
AudioContext processing (baseLatency)
        ↓
output device (outputLatency)
```

This number is one of the fixed costs that must fit inside the 300 ms P50 budget.

## Files
- `frontend/index.html` – measurement UI
- `frontend/app.js` – measurement logic using Web Audio API
- `frontend/styles.css`

## How to run
```bash
cd frontend
python3 -m http.server 8080
# open http://localhost:8080
```

1. Click **Initialize AudioContext** (required user gesture).
2. If the context is suspended, click **Resume Context**.
3. Run the measurement suite (default 20 runs).

## Key metrics reported
| Metric | Meaning |
|--------|---------|
| `jsToStartMs` | Pure JS overhead (chunk received → `source.start()` called) |
| `baseLatencyMs` | Browser processing pipeline (Chrome/Edge) |
| `outputLatencyMs` | Time to hardware output (Chrome/Edge) |
| **`browserContributionMs`** | Sum that matters for the budget |

## Expected order of magnitude
On a modern desktop Chrome:
- JS scheduling: 0.5 – 3 ms
- baseLatency: 5 – 15 ms (interactive hint)
- outputLatency: 10 – 30 ms
- **Total browser contribution: typically 20 – 50 ms**

Mobile Safari / Firefox usually report higher or incomplete latency numbers.

## Budget allocation reminder
```
Speech-end detection       ~20 ms
STT finalization           ~40 ms
LLM first token            ~80 ms
TTS first audio            ~100 ms
Browser / network / play   ~50 ms   ← this stage
--------------------------------
Target                     <300 ms
```

If the browser alone is consuming > 60–70 ms on the target devices, we must:
- Use smaller initial chunks
- Prefer `latencyHint: "interactive"`
- Consider AudioWorklet for even lower overhead later
- Avoid large pre-buffering

## Next
Stage 2 will add streaming STT on top of a warmed connection and start feeding partial transcripts into the conversation engine.
