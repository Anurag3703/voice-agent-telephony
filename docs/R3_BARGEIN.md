# R3 – Echo-aware Barge-in

## Problem
Browser `echoCancellation: true` helps but is incomplete. Loudspeaker audio
still leaks into the mic and can look like user speech, causing false barge-ins
or requiring thresholds so high that real interrupts feel sluggish.

## Approach
`EchoAwareBargeIn` maintains a **slow adaptive baseline** of mic RMS while the
agent is speaking (echo floor). Barge-in requires:

```
level >= max(absoluteFloor, baseline × margin)   for confirmMs (~120 ms)
```

Defaults:
- `margin = 2.8`
- `absoluteFloor = 0.025`
- `confirmMs = 120`
- `baselineAlpha = 0.08` (only adapts when below threshold)

## Lifecycle
- `barge.start()` on first agent audio (T6)
- `barge.process(level)` each capture frame while `isAgentSpeaking`
- `barge.stop()` on interrupt, TTS done, or mic stop

## Still measured
Dashboard shows `Barge-in stop` = detect → local audio silenced (target < 100 ms).

## Limits
This is residual-energy gating, not full AEC. True hardware/software AEC plus
optional dual-end analysis remains better on hard speakerphone cases. R3 is the
practical browser-side improvement without native modules.
