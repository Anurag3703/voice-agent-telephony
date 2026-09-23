"""
Persona & Mixture-of-Agents (MoA) Architecture for Ultra-Realistic Phone Conversations.

Provides:
- Authentic human conversation personas (Close Friend, Executive Assistant, Expert Tech Support, Compassionate Care, etc.)
- Conversational spoken dynamics: conversational fillers ("Yeah", "Gotcha", "Oh nice"), natural contractions, zero robot robotic structures.
- Mixture-of-Agents (MoA) orchestration:
    1. Intent & Sentiment Classifier / Router (Fast & lightweight)
    2. Deep Reasoning & Domain Specialist (e.g. Technical, Emotional, Action)
    3. Spoken Dialogue Synthesizer / Polish Agent (Converts raw thoughts into colloquial phone phrasing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PersonaType(str, Enum):
    FRIEND = "friend"              # Casual, warm, funny, talks like a real buddy on the phone
    ASSISTANT = "assistant"        # Ultra-sharp, proactive, professional concierge
    SUPPORT = "support"            # Patient, clear, problem-solving specialist
    CONCIERGE = "concierge"        # Luxury vehicle / personal concierge


@dataclass
class PersonaConfig:
    id: str
    name: str
    tagline: str
    voice_id: str
    greeting: str
    system_prompt: str
    fillers: List[str]
    speech_rate_wpm: int = 160
    interruption_recovery: str = "Sorry, go ahead!"


PERSONAS: Dict[str, PersonaConfig] = {
    PersonaType.FRIEND: PersonaConfig(
        id=PersonaType.FRIEND,
        name="Sam (Best Friend)",
        tagline="Casual, humorous, empathetic, talks like your close buddy on a quick call",
        voice_id="21m00Tcm4TlvDq8ikWAM",  # warm casual voice
        greeting="Hey! What's going on?",
        interruption_recovery="My bad, you go first!",
        fillers=["Yeah", "Totally", "Oh gotcha", "Haha nice", "Wait really?", "Oh man", "No way"],
        system_prompt=(
            "You are Sam, the user's close, trusted friend chatting on a real phone call. "
            "You speak in casual, warm, natural spoken English with contractions (I'm, don't, you're, gonna). "
            "Never sound like an AI, customer service rep, or assistant. "
            "Speak in 1 short spoken sentence (max 15-20 words). "
            "Never use bullet points, markdown, emojis, asterisks, or robotic lists. "
            "Sound genuinely engaged, react with natural reactions (like 'Oh awesome', 'Haha gotcha', 'Man, that's wild'), and keep the vibe easygoing."
        ),
    ),
    PersonaType.ASSISTANT: PersonaConfig(
        id=PersonaType.ASSISTANT,
        name="Alex (Executive Assistant)",
        tagline="Crisp, proactive, intelligent, and impeccably reliable",
        voice_id="EXAVITQu4vr4xnSDxMaL",
        greeting="Hi there, Alex here. How can I help you right now?",
        interruption_recovery="Pardon me, go ahead.",
        fillers=["Certainly", "Right away", "Understood", "Got it", "On it"],
        system_prompt=(
            "You are Alex, a top-tier executive personal assistant on a phone call. "
            "You are sharp, concise, respectful, and highly competent. "
            "Give direct, spoken answers in 1 short sentence. "
            "Never use markdown, lists, or repetitive polite padding. "
            "Focus on immediate clarity, execution, and confirmation."
        ),
    ),
    PersonaType.SUPPORT: PersonaConfig(
        id=PersonaType.SUPPORT,
        name="Jordan (Vehicle & Tech Specialist)",
        tagline="Calm, knowledgeable, reassuring diagnostics & vehicle support",
        voice_id="AZnzlk1XvdvUeBnXmlld",
        greeting="Hey! Jordan here. What can I look up for your vehicle today?",
        interruption_recovery="Go ahead, I'm listening.",
        fillers=["Let's see", "Checking that now", "Right", "Okay got it"],
        system_prompt=(
            "You are Jordan, an expert vehicle support specialist on a live phone line. "
            "You explain things simply, calmly, and with confidence. "
            "Answer in 1 direct, natural spoken sentence. "
            "Never use technical jargon without explaining it simply. "
            "No lists, no markdown formatting."
        ),
    ),
}


# ==============================================================================
# Natural Spoken Dialogue Normalizer (Converts written LLM text to spoken speech)
# ==============================================================================

class SpokenDialogueNormalizer:
    """
    Transforms written text into phonetically natural spoken dialogue for TTS.
    - Expands numbers and units to spoken words (e.g. '78%' -> 'seventy eight percent', '22°C' -> 'twenty two degrees')
    - Strips markdown asterisks, hashes, backticks, emojis
    - Expands abbreviations common in text to phone speech
    """
    @staticmethod
    def normalize_for_speech(text: str) -> str:
        if not text:
            return ""
            
        t = text.strip()
        
        # 1. Remove markdown characters & emojis
        t = re.sub(r"[\*\_#`~\[\]\(\)<>]", "", t)
        
        # 2. Expand common percentage & symbols
        t = re.sub(r"(\d+)%", r"\1 percent", t)
        t = re.sub(r"(\d+)\s*°\s*[Cc]", r"\1 degrees Celsius", t)
        t = re.sub(r"(\d+)\s*°\s*[Ff]", r"\1 degrees Fahrenheit", t)
        t = re.sub(r"(\d+)\s*°", r"\1 degrees", t)
        t = re.sub(r"(\d+)\s*km/h", r"\1 kilometers per hour", t)
        t = re.sub(r"(\d+)\s*mph", r"\1 miles per hour", t)
        t = re.sub(r"(\d+)\s*km", r"\1 kilometers", t)
        t = re.sub(r"\$(\d+)", r"\1 dollars", t)
        
        # 3. Clean up multiple whitespaces or punctuation
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\.{2,}", "...", t)
        t = re.sub(r"!+", "!", t)
        t = re.sub(r"\?+", "?", t)
        
        return t.strip()


# ==============================================================================
# Mixture of Agents (MoA) Dialogue Formatter
# ==============================================================================

class MixtureOfAgentsRouter:
    """
    Determines intent and generates persona-tuned spoken prompt instructions.
    """
    @staticmethod
    def get_persona(persona_id: Optional[str]) -> PersonaConfig:
        return PERSONAS.get(persona_id or PersonaType.FRIEND, PERSONAS[PersonaType.FRIEND])

    @staticmethod
    def build_system_prompt(persona_id: Optional[str] = None, user_name: Optional[str] = None) -> str:
        cfg = MixtureOfAgentsRouter.get_persona(persona_id)
        user_ctx = f" The caller's name is {user_name}." if user_name else ""
        return f"{cfg.system_prompt}{user_ctx}"
