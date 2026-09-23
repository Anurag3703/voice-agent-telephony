"""
Streaming STT Protocol / Interfaces

Provider-agnostic. Real adapters (Deepgram, AssemblyAI, etc.) must implement this.
A batch-only STT must NOT be silently treated as realtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional, Any
import time


class TranscriptType(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    ERROR = "error"


@dataclass
class TranscriptEvent:
    type: TranscriptType
    text: str = ""
    is_final: bool = False
    confidence: Optional[float] = None
    # High-resolution monotonic timestamps (seconds)
    t_event: float = field(default_factory=time.perf_counter)
    # Optional provider-specific metadata
    meta: dict[str, Any] = field(default_factory=dict)

    def age_ms(self, since: float) -> float:
        return (self.t_event - since) * 1000


class StreamingSTT(ABC):
    """
    Streaming Speech-to-Text interface.

    Lifecycle:
        await connect()
        async for event in receive_events():
            ...
        await send_audio(chunk)
        ...
        await finalize()   # optional, forces final
        await close()
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish persistent connection to the provider."""
        ...

    @abstractmethod
    async def send_audio(self, chunk: bytes, sample_rate: int = 16000) -> None:
        """Send a raw audio chunk (PCM16 LE mono recommended)."""
        ...

    @abstractmethod
    async def receive_events(self) -> AsyncIterator[TranscriptEvent]:
        """Yield transcript events as they arrive (partials + finals)."""
        ...

    @abstractmethod
    async def finalize(self) -> None:
        """Signal end-of-utterance; provider should emit a final transcript soon."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Tear down the connection cleanly."""
        ...

    @property
    @abstractmethod
    def supports_partials(self) -> bool:
        ...
