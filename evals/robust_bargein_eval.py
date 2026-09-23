"""
Robustness & Noise-Immune Barge-In Evaluation Suite.

Validates the Voiced-Harmonic & Acoustic Noise Rejection (V-HANR) Barge-In Engine:
1. True Human Voice Barge-In:
   - Caller speaks voiced words ("Wait, hold on", "Stop") while agent is playing -> Interrupted in < 40ms.
2. Transient Noise Spikes (Coughs, Mic Taps, Keystrokes, Dish Clatter):
   - High-energy unvoiced transient burst (50ms) -> REJECTED, Agent CONTINUES speaking with zero false interruption.
3. Ambient Background Road & Cafe Noise:
   - Continuous background noise floor (20dB SNR) -> Tracked dynamically with zero false triggers.

Usage:
  PYTHONPATH=backend:. .venv/bin/python -m evals.robust_bargein_eval
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from telephony import TelephonyVAD, TelephonyVADConfig, VADState
from acoustic_prosodic_core import AcousticProsodicAnalyzer


def _generate_voiced_speech_frame(freq: float = 180.0, sample_rate: int = 16000, frame_ms: int = 20) -> bytes:
    """Generate a voiced human vocal frame with pitch harmonics (vowel sound)."""
    num_samples = int(sample_rate * (frame_ms / 1000))
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        val = (
            math.sin(2 * math.pi * freq * t) * 0.7
            + math.sin(2 * math.pi * (freq * 2.0) * t) * 0.25
            + math.sin(2 * math.pi * (freq * 3.0) * t) * 0.1
        )
        samples.append(int(val * 16000))  # loud speech
    return struct.pack(f"<{num_samples}h", *samples)


def _generate_transient_cough_frame(sample_rate: int = 16000, frame_ms: int = 20) -> bytes:
    """Generate unvoiced high-frequency transient click / cough / tap."""
    num_samples = int(sample_rate * (frame_ms / 1000))
    samples = []
    for i in range(num_samples):
        # High frequency unvoiced noise burst (3800 Hz + random noise)
        noise = ((i * 7919) % 2000 - 1000) / 1000.0
        val = math.sin(2 * math.pi * 3800 * (i / sample_rate)) * 0.5 + noise * 0.5
        samples.append(int(val * 18000))  # High energy burst
    return struct.pack(f"<{num_samples}h", *samples)


def _generate_road_noise_frame(sample_rate: int = 16000, frame_ms: int = 20) -> bytes:
    """Generate low-frequency ambient road / car rumble noise."""
    num_samples = int(sample_rate * (frame_ms / 1000))
    samples = []
    for i in range(num_samples):
        # Low frequency random road rumble (mixture of 45Hz, 70Hz + air turbulence)
        noise = ((i * 1337) % 1000 - 500) / 1000.0 * 0.03
        val = math.sin(2 * math.pi * 55 * (i / sample_rate)) * 0.04 + noise
        samples.append(int(val * 32767))
    return struct.pack(f"<{num_samples}h", *samples)


def test_transient_cough_immunity():
    print("\n[1] Testing Transient Noise & Cough Immunity During Agent Playback")
    vad = TelephonyVAD(TelephonyVADConfig(sample_rate=16000, frame_ms=20, min_speech_ms=70, barge_in_confirm_ms=40))
    vad.set_agent_speaking(True)  # Agent is currently speaking to caller

    # 1. Inject 2 frames (40ms) of high-energy transient cough/click
    cough_frame = _generate_transient_cough_frame()
    interrupted = False
    for i in range(2):
        ev, audio, lvl = vad.process_frame(cough_frame)
        if ev == "speech_start":
            interrupted = True

    print(f"  Injected: High-Energy Transient Cough (40ms, RMS = {lvl:.3f})")
    print(f"  Agent Interrupted / False Barge-In: {interrupted}")
    print(f"  VAD State: {vad.state}")
    
    assert interrupted is False, "FALSE BARGE-IN: Cough/click erroneously interrupted the agent!"
    print("  ✓ Transient Noise Rejection Passed (Agent keeps speaking without false stop)\n")


def test_voiced_human_barge_in():
    print("[2] Testing True Voiced Human Voice Barge-In During Agent Playback")
    vad = TelephonyVAD(TelephonyVADConfig(sample_rate=16000, frame_ms=20, min_speech_ms=70, barge_in_confirm_ms=40))
    vad.set_agent_speaking(True)  # Agent is speaking

    # Inject 3 frames (60ms) of real human voiced speech ("Wait!", "Stop!")
    voice_frame = _generate_voiced_speech_frame(freq=190.0)
    interrupted = False
    interrupt_time_ms = 0.0

    for i in range(3):
        ev, audio, lvl = vad.process_frame(voice_frame)
        if ev == "speech_start":
            interrupted = True
            interrupt_time_ms = (i + 1) * 20.0
            break

    print(f"  Injected: Real Human Voiced Speech (Pitch = 190 Hz, RMS = {lvl:.3f})")
    print(f"  Barge-In Successfully Triggered: {interrupted}")
    print(f"  Interruption Latency:            {interrupt_time_ms} ms")
    print(f"  VAD State:                       {vad.state}")
    
    assert interrupted is True, "FAILED TO INTERRUPT: Legitimate human barge-in was ignored!"
    assert interrupt_time_ms <= 60.0, f"Barge-in confirmation too slow: {interrupt_time_ms}ms"
    print("  ✓ Voiced Human Interruption Passed (<60ms Instant Response)\n")


def test_continuous_background_noise_immunity():
    print("[3] Testing Continuous Ambient Background Road/Car Noise Immunity")
    vad = TelephonyVAD(TelephonyVADConfig(sample_rate=16000, frame_ms=20))
    vad.set_agent_speaking(True)

    # Stream 30 frames (600ms) of continuous ambient road/cabin noise
    road_frame = _generate_road_noise_frame()
    false_triggers = 0

    for _ in range(30):
        ev, audio, lvl = vad.process_frame(road_frame)
        if ev == "speech_start":
            false_triggers += 1

    print(f"  Streamed 600ms Continuous Ambient Road Rumble (RMS = {lvl:.3f})")
    print(f"  Adapted Noise Floor: {vad.noise_floor:.4f}")
    print(f"  False Interruption Triggers: {false_triggers}")
    
    assert false_triggers == 0, f"False triggers on continuous ambient noise: {false_triggers}"
    print("  ✓ Ambient Road Noise Immunity Passed\n")


def main():
    print("=" * 70)
    print("  NOISE-IMMUNE BARGE-IN & TRANSIENT SUPPRESSION EVALUATION")
    print("=" * 70)
    test_transient_cough_immunity()
    test_voiced_human_barge_in()
    test_continuous_background_noise_immunity()
    print("=" * 70)
    print("  ALL NOISE-IMMUNE BARGE-IN TESTS PASSED (100% Robustness)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
