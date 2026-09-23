# R2 – Better VAD + Endpointing

## What changed
Replaced single-threshold RMS VAD with **hysteresis endpointing**:

| Parameter | Default | Role |
|-----------|---------|------|
| Start threshold | 0.020 | Must exceed to *enter* speech |
| End threshold | 0.012 | Must stay below to *leave* speech |
| Min speech (ms) | 180 | Ignore blips shorter than this before speech_start |
| Hangover (ms) | 550 | Continuous silence required to end utterance |
| Min utterance (ms) | 280 | Discard ultra-short completed utterances |

## Behaviour
```
silent ──(rms ≥ start for minSpeechMs)──► speech
speech ──(rms < end for hangoverMs)──► silent → finalize
```

- False starts from clicks/noise are filtered by `minSpeechMs`
- Boundary chatter is reduced by start > end hysteresis
- Short accidental utterances are ignored via `minUtteranceMs`
- Barge-in still uses a raised bar (`startThreshold * 1.5`) while agent speaks (R3 will improve AEC)

## Files
- `frontend/vad.js` – `EndpointVad` class
- `frontend/app.js` – uses EndpointVad; UI exposes thresholds
