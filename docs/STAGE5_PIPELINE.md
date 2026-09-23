# Stage 5 – Tightened LLM → TTS Pipeline

## Focus
Reduce time-to-first-audio by making the first TTS phrase available earlier and keeping LLM / TTS fully overlapped.

## Changes vs Stage 4
1. **Aggressive first chunk** – `first_min_chars=12` (subsequent chunks use 22).
2. **Smarter boundaries** – prefer sentence end → clause/comma → word boundary.
3. **Avoid tiny trailers** – `min_final_chars` prevents emitting noise fragments.
4. **Lower mock component latencies** (for clearer headroom measurement):
   - LLM TTFT ≈ 55 ms
   - TTS first-audio ≈ 75 ms
5. **First-phrase event** – client is notified the moment the first phrase is handed to TTS.
6. **Phrase count** reported on `llm_done`.

## Expected mock TTFA (T6 − T0)
| Segment                | Approx |
|------------------------|--------|
| STT finalization       | 35–45 ms |
| LLM TTFT               | ~55 ms |
| First phrase formation | 15–40 ms |
| TTS first audio        | ~75 ms |
| Browser schedule       | 10–25 ms |
| **Total TTFA**         | **~190–240 ms** |

Still under the 300 ms P50 hard target with comfortable margin for real providers.

## What is still sequential (by design)
- We only start the LLM on **FINAL** transcript (safe).
- Speculative LLM on partials remains future work (Stage 6+ / advanced).
