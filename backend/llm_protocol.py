"""
Streaming LLM Protocol

Provider-agnostic. Real adapters (OpenAI, Anthropic, Groq, local vLLM, etc.)
must implement token streaming. Batch-only LLMs are not acceptable for the
realtime path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Any, Optional
import time


class LLMEventType(str, Enum):
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DONE = "done"
    ERROR = "error"


@dataclass
class LLMEvent:
    type: LLMEventType
    text: str = ""
    # For tool calls
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_call_id: Optional[str] = None
    # Timestamps
    t_event: float = field(default_factory=time.perf_counter)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class StreamingLLM(ABC):
    """
    Streaming LLM interface.

    Usage:
        await llm.connect()          # optional warm
        async for event in llm.stream(messages, tools=None):
            if event.type == LLMEventType.TOKEN:
                ...
        await llm.close()
    """

    @abstractmethod
    async def connect(self) -> None:
        """Warm / establish persistent connection if the provider supports it."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[LLMEvent]:
        """
        Yield tokens (and optional tool calls) as they are generated.
        Must not buffer the entire response before yielding.
        """
        ...

    @abstractmethod
    async def cancel(self) -> None:
        """Abort the current generation as fast as possible (barge-in)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        ...
