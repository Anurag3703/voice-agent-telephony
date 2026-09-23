"""
Streaming TTS Protocol

Provider-agnostic. Real adapters (ElevenLabs, Cartesia, OpenAI, Deepgram Aura, etc.)
must support text streaming in and audio chunk streaming out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Any, Optional
import time


class TTSEventType(str, Enum):
    AUDIO = "audio"          # PCM or encoded audio chunk
    MARK = "mark"            # optional alignment / word boundary
    DONE = "done"
    ERROR = "error"


@dataclass
class TTSEvent:
    type: TTSEventType
    # Raw audio bytes (PCM16 LE mono recommended for lowest latency)
    audio: bytes = b""
    sample_rate: int = 24000
    # Optional text that this chunk corresponds to
    text: str = ""
    t_event: float = field(default_factory=time.perf_counter)
    meta: dict[str, Any] = field(default_factory=dict)


class StreamingTTS(ABC):
    """
    Streaming Text-to-Speech interface.

    Usage:
        await tts.connect()
        async for event in tts.stream(text_chunk_iterator):
            if event.type == TTSEventType.AUDIO:
                play(event.audio)
        await tts.close()
    """

    @abstractmethod
    async def connect(self) -> None:
        """Warm persistent connection."""
        ...

    @abstractmethod
    async def stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[TTSEvent]:
        """
        Consume text chunks as they arrive and yield audio chunks
        as soon as they are ready. Must overlap generation with upstream LLM.
        """
        ...

    @abstractmethod
    async def cancel(self) -> None:
        """Stop current synthesis immediately (barge-in)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def audio_format(self) -> str:
        """e.g. 'pcm16', 'mulaw', 'opus'"""
        ...
