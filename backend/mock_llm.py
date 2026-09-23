"""
Mock Streaming LLM – Stage 7 (with tool calls)

- Streams tokens with realistic TTFT
- Can emit a tool_call event when the user asks for vehicle data
- After tool results are injected, produces a spoken summary
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Optional

from llm_protocol import (
    StreamingLLM,
    LLMEvent,
    LLMEventType,
    ChatMessage,
)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in " \n.,!?;:":
            if current.strip():
                tokens.append(current)
            current = ""
    if current.strip():
        tokens.append(current)
    return tokens


# Responses used when no tool is needed
SIMPLE_RESPONSES = {
    "hello": "Hi there! How can I help you with your vehicle today?",
    "hi": "Hi there! How can I help you with your vehicle today?",
    "default": "I can check vehicle status, battery, climate, or honk to help you find it. What do you need?",
}


class MockStreamingLLM(StreamingLLM):
    def __init__(
        self,
        ttft_ms: float = 55.0,
        inter_token_ms: float = 16.0,
    ):
        self.ttft_ms = ttft_ms
        self.inter_token_ms = inter_token_ms
        self._connected = False
        self._cancelled = False
        self._closed = False
        self._t_connect = 0.0
        self._t_stream_start = 0.0
        self._t_first_token = 0.0

    @property
    def supports_streaming(self) -> bool:
        return True

    async def connect(self) -> None:
        self._connected = True
        self._t_connect = time.perf_counter()
        await asyncio.sleep(0.010)

    def _decide(self, messages: list[ChatMessage]) -> tuple[str, Optional[str]]:
        """
        Returns (mode, tool_name_or_none).
        mode: "tool" | "speak"
        """
        # If the last message is a tool result, we should speak a summary
        for m in reversed(messages):
            if m.role == "tool":
                return "speak", None
            if m.role == "user":
                break

        user_text = ""
        for m in reversed(messages):
            if m.role == "user":
                user_text = m.content.lower()
                break

        if any(k in user_text for k in ("battery", "range", "charge")):
            return "tool", "get_battery"
        if any(k in user_text for k in ("climate", "temperature", "cabin", "ac")):
            return "tool", "get_climate"
        if any(k in user_text for k in ("honk", "flash", "find my car", "locate")):
            return "tool", "honk_flash"
        if any(k in user_text for k in ("status", "where", "location", "vehicle", "car")):
            return "tool", "get_vehicle_status"
        if any(k in user_text for k in ("hello", "hi", "hey")):
            return "speak", None
        return "speak", None

    def _summary_from_tools(self, messages: list[ChatMessage]) -> str:
        """Build a short spoken answer from the most recent tool result(s)."""
        tool_data = {}
        for m in messages:
            if m.role == "tool" and m.content:
                try:
                    tool_data[m.name or "tool"] = json.loads(m.content)
                except Exception:
                    tool_data[m.name or "tool"] = m.content

        # Determine persona style from system prompt if present
        is_friend = any("Sam" in (m.content or "") for m in messages if m.role == "system")

        if "get_battery" in tool_data:
            d = tool_data["get_battery"]
            if is_friend:
                return f"You're at {d.get('percent', '?')} percent, with about {d.get('range_km', '?')} kilometers left. {'Charging right now.' if d.get('charging') else 'Not plugged in.'}"
            return (
                f"Your battery is at {d.get('percent', '?')} percent, "
                f"with about {d.get('range_km', '?')} kilometers of range. "
                f"{'Currently charging.' if d.get('charging') else 'Not charging right now.'}"
            )
        if "get_climate" in tool_data:
            d = tool_data["get_climate"]
            if is_friend:
                return f"Cabin is at {d.get('cabin_c', '?')} degrees, set to {d.get('target_c', '?')}."
            return (
                f"Cabin temperature is {d.get('cabin_c', '?')} degrees. "
                f"Target is {d.get('target_c', '?')} degrees."
            )
        if "honk_flash" in tool_data:
            return "Honked the horn and flashed the lights for you!" if is_friend else "Done — I honked the horn and flashed the lights."
        if "get_vehicle_status" in tool_data:
            d = tool_data["get_vehicle_status"]
            loc = d.get("location", "unknown location")
            locked = "locked" if d.get("locked") else "unlocked"
            alerts = d.get("alerts") or []
            alert_txt = "No alerts." if not alerts else f"Alerts: {', '.join(alerts)}."
            if is_friend:
                return f"Car is at {loc}, and it's {locked}. {alert_txt}"
            return (
                f"Your vehicle is at {loc}, and it is {locked}. {alert_txt}"
            )
        return "I have the information, but something went wrong summarizing it."

    def _plain_response(self, messages: list[ChatMessage]) -> str:
        is_friend = any("Sam" in (m.content or "") for m in messages if m.role == "system")
        is_margaret = any("Margaret" in (m.content or "") for m in messages if m.role == "system")
        is_arthur = any("Arthur" in (m.content or "") for m in messages if m.role == "system")
        
        user_text = ""
        for m in reversed(messages):
            if m.role == "user":
                user_text = m.content.lower().strip()
                break

        # Honeypot Personas
        if is_margaret:
            if any(k in user_text for k in ("anydesk", "teamviewer", "download", "code", "app")):
                return "Oh dear, let me put on my reading glasses, now which button did you say I should click?"
            if any(k in user_text for k in ("police", "arrest", "warrant", "tax", "irs", "ato", "fraud")):
                return "Oh goodness me, an arrest? My heart can't take this, please tell me what I did wrong officer!"
            if any(k in user_text for k in ("money", "transfer", "bank", "account", "card")):
                return "Let me find my chequebook, it's somewhere here in the drawer next to my knitting."
            return "Can you speak a little louder dear? My hearing aid is acting up today."

        if is_arthur:
            if any(k in user_text for k in ("anydesk", "teamviewer", "code", "number")):
                return "Hold on, my pen ran out of ink, was that code starting with a five or an eight?"
            return "Fascinating, let me review my notes here. Could you repeat that last point?"

        # Conversational Friend Persona (Sam)
        if any(k in user_text for k in ("how you doing", "how are you", "how are you doing", "how's it going")):
            return "Doing great, man! How are you holding up today?"
        if any(k in user_text for k in ("what the fuck", "what's happening", "whats happening", "what is happening", "what happened")):
            return "Man, things are a bit wild right now, but I've got your back. What's going on?"
        if any(k in user_text for k in ("not a good idea", "bad idea", "not really a good idea", "terrible idea")):
            return "Yeah, you're totally right, let's definitely rethink that."
        if any(k in user_text for k in ("good idea", "great idea", "awesome")):
            return "Totally, I'm with you on that!"
        if any(k in user_text for k in ("hello", "hi", "hey", "sup", "yo")):
            return "Hey! What's up?" if is_friend else "Hello! How can I help you today?"
        if any(k in user_text for k in ("who are you", "what are you", "your name")):
            return "I'm Sam, your close friend and assistant." if is_friend else "I am your voice assistant."
        if any(k in user_text for k in ("there", "listen", "hear me", "can you hear")):
            return "Yeah, I can hear you loud and clear!"
        if any(k in user_text for k in ("what are you doing", "what r u doing")):
            return "Just chilling and keeping an eye on things for you. What are you up to?"
        if any(k in user_text for k in ("joke", "funny")):
            return "Why don't scientists trust atoms? Because they make up everything!"
        if any(k in user_text for k in ("scammer", "scam", "bot")):
            return "Haha no way man, I'm your real AI buddy right here on the line."
        if any(k in user_text for k in ("help", "accident", "emergency")):
            return "I'm right here with you! Are you alright? Tell me what happened."
        if any(k in user_text for k in ("thanks", "thank you", "appreciate")):
            return "Anytime man! Always got your back." if is_friend else "You are very welcome!"
        if any(k in user_text for k in ("bye", "goodbye", "see ya", "later")):
            return "Catch you later! Take care!"
            
        # Contextual natural fallback without rigid echo
        if is_friend:
            return f"I hear you on that. Let's figure it out together."
        return f"I understand. How else can I assist you with that?"

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[LLMEvent]:
        if not self._connected or self._closed:
            yield LLMEvent(type=LLMEventType.ERROR, text="LLM not connected")
            return

        self._cancelled = False
        self._t_stream_start = time.perf_counter()
        mode, tool_name = self._decide(messages)

        # --- Tool call path ---
        if mode == "tool" and tool_name:
            await asyncio.sleep(self.ttft_ms / 1000.0 * 0.6)  # slightly faster decision
            if self._cancelled:
                return
            yield LLMEvent(
                type=LLMEventType.TOOL_CALL,
                tool_name=tool_name,
                tool_args={},
                tool_call_id=f"call_{tool_name}_{int(time.time()*1000)}",
                meta={"decision_ms": self.ttft_ms * 0.6},
            )
            yield LLMEvent(type=LLMEventType.DONE, meta={"tool_call": tool_name})
            return

        # --- Speak path ---
        if any(m.role == "tool" for m in messages):
            text = self._summary_from_tools(messages)
        else:
            text = self._plain_response(messages)

        tokens = _tokenize(text)
        await asyncio.sleep(self.ttft_ms / 1000.0)
        if self._cancelled:
            return

        self._t_first_token = time.perf_counter()

        for i, tok in enumerate(tokens):
            if self._cancelled:
                return
            yield LLMEvent(
                type=LLMEventType.TOKEN,
                text=tok,
                meta={"index": i, "ttft_ms": (self._t_first_token - self._t_stream_start) * 1000 if i == 0 else None},
            )
            if i < len(tokens) - 1:
                await asyncio.sleep(self.inter_token_ms / 1000.0)

        if not self._cancelled:
            yield LLMEvent(
                type=LLMEventType.DONE,
                meta={
                    "total_tokens": len(tokens),
                    "ttft_ms": (self._t_first_token - self._t_stream_start) * 1000,
                    "total_ms": (time.perf_counter() - self._t_stream_start) * 1000,
                },
            )

    async def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        self._closed = True
        self._cancelled = True
