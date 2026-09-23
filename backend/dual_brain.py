"""
Dual-Brain Reflex & Instant Backchannel Engine.

Architecture:
1. Brain-1 (Reflex / Sub-conscious Brain):
   - Ultra-low latency (< 50ms decision).
   - Fires on intermediate partial STT or immediate VAD speech-end.
   - Selects context-aware acoustic micro-fillers ("Yeah...", "Gotcha, checking that...",
     "Mhm, let's see...", "One sec...") when a turn requires external tools or deep reasoning.
   - Bridges the dead-air gap so perceived TTFA drops to < 100ms.

2. Brain-2 (Reasoning Brain):
   - Full LLM reasoning + external database / tool execution.
   - Computes the factual, verified response concurrently while Brain-1 is already vocalizing.
   - Seamlessly stitches the final answer as Brain-1 completes its filler.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ReflexIntent(str, Enum):
    IMMEDIATE_DIRECT = "immediate_direct"   # Simple greeting/acknowledgment (no tool needed)
    TOOL_LOOKUP = "tool_lookup"             # Query requiring DB / telematics / tools
    COMPLEX_REASONING = "complex_reasoning" # Multi-step or detailed query
    UNCERTAIN = "uncertain"


@dataclass
class ReflexFiller:
    text: str
    estimated_duration_ms: int
    intent: ReflexIntent


class DualBrainReflexEngine:
    """
    Sub-conscious Reflex Engine that eliminates conversational dead air.
    """
    # Categorized conversational fillers by persona and intent
    FILLERS: Dict[str, Dict[ReflexIntent, List[str]]] = {
        "friend": {
            ReflexIntent.TOOL_LOOKUP: [
                "Gotcha, checking that now.",
                "Yeah, let me pull that up.",
                "On it, one sec.",
                "Let's see here.",
                "Checking for you real quick.",
            ],
            ReflexIntent.COMPLEX_REASONING: [
                "Hmm, let me think about that.",
                "Yeah, good question, let's see.",
                "Right, give me just a second.",
            ],
            ReflexIntent.IMMEDIATE_DIRECT: [
                "Yeah,",
                "Totally,",
                "Oh nice,",
                "Got it,",
            ],
        },
        "assistant": {
            ReflexIntent.TOOL_LOOKUP: [
                "Certainly, checking that for you now.",
                "One moment, retrieving the latest data.",
                "Accessing your vehicle status now.",
                "Right away, looking that up.",
            ],
            ReflexIntent.COMPLEX_REASONING: [
                "Allow me a moment to review that.",
                "Checking the details for you.",
            ],
            ReflexIntent.IMMEDIATE_DIRECT: [
                "Understood.",
                "Certainly.",
                "Right away.",
            ],
        },
        "support": {
            ReflexIntent.TOOL_LOOKUP: [
                "Sure thing, pulling up your diagnostics.",
                "Let me check the system for you.",
                "Looking into that right now.",
            ],
            ReflexIntent.COMPLEX_REASONING: [
                "Let's check the parameters on that.",
                "One moment while I analyze that.",
            ],
            ReflexIntent.IMMEDIATE_DIRECT: [
                "Okay, got it.",
                "Right, understood.",
            ],
        },
    }

    TOOL_KEYWORDS = (
        "battery", "range", "charge", "charging",
        "climate", "temp", "temperature", "cabin", "ac", "heat",
        "honk", "horn", "flash", "lights",
        "status", "where", "location", "parked", "locked", "alerts",
        "schedule", "book", "appointment", "check"
    )

    @classmethod
    def analyze_intent(cls, text: str) -> ReflexIntent:
        """
        Fast intent classification (< 1ms) based on lexical analysis.
        """
        if not text:
            return ReflexIntent.UNCERTAIN

        t = text.lower()
        
        # Check if tool execution is required
        if any(k in t for k in cls.TOOL_KEYWORDS):
            return ReflexIntent.TOOL_LOOKUP
            
        # Check if complex question
        if any(t.startswith(q) for q in ("how", "why", "what if", "can you explain", "compare")):
            return ReflexIntent.COMPLEX_REASONING
            
        return ReflexIntent.IMMEDIATE_DIRECT

    @classmethod
    def get_reflex_filler(
        cls,
        user_text: str,
        persona: str = "friend",
        force_filler: bool = False
    ) -> Optional[str]:
        """
        Determine if an instant acoustic reflex filler should be spoken.
        Returns the filler text if advantageous, or None if direct answer is preferred.
        """
        intent = cls.analyze_intent(user_text)
        
        # Only fire reflex fillers when the request requires tool latency or deep reasoning
        if intent in (ReflexIntent.TOOL_LOOKUP, ReflexIntent.COMPLEX_REASONING) or force_filler:
            persona_fillers = cls.FILLERS.get(persona, cls.FILLERS["friend"])
            pool = persona_fillers.get(intent, persona_fillers[ReflexIntent.TOOL_LOOKUP])
            return random.choice(pool)
            
        return None
