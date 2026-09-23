"""
R5 – Stage deadlines / timeouts (ms)

These are hard ceilings for a single turn. Exceeding them cancels the turn
and surfaces a structured timeout event instead of hanging forever.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnDeadlines:
    # From speech-end (T0) - Generous ceilings for on-device CPU + real network resilience
    stt_final_ms: int = 6000          # Allows on-device Whisper transcription without premature abort
    llm_first_token_ms: int = 6000    # Allows local LLM generation
    tool_ms: int = 1500               # Enforced inside tools.py
    llm_speech_first_token_ms: int = 6000
    tts_first_audio_ms: int = 6000
    # Whole turn: speech-end → first audio safety ceiling
    turn_first_audio_ms: int = 15000
    # Absolute max wall time for a turn before forced cancel
    turn_total_ms: int = 30000


DEFAULT_DEADLINES = TurnDeadlines()
