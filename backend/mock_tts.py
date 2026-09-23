"""
Mock Streaming TTS for deterministic latency testing.

- Consumes text chunks as they arrive
- Emits PCM16 LE mono @ 24 kHz audio chunks
- First-audio latency is configurable (default ~90 ms)
- Duration is roughly proportional to text length (≈ 14 chars/sec spoken)
"""

from __future__ import annotations

import asyncio
import math
import struct
import time
from typing import AsyncIterator, Optional

from tts_protocol import StreamingTTS, TTSEvent, TTSEventType


SAMPLE_RATE = 24000
CHARS_PER_SECOND = 14.0          # rough speaking rate
FIRST_AUDIO_MS = 75.0            # time until first audio chunk
CHUNK_DURATION_MS = 40.0         # size of each emitted audio frame


def _generate_tone_chunk(
    num_samples: int,
    sample_rate: int = SAMPLE_RATE,
    freq: float = 180.0,
    volume: float = 0.18,
    phase: float = 0.0,
) -> tuple[bytes, float]:
    """
    Generate a soft buzz / formant-ish tone so something is audible.
    Returns (pcm16_bytes, next_phase).
    """
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Simple two-partial tone with quick envelope to avoid clicks
        env = 1.0
        if i < 32:
            env = i / 32
        elif i > num_samples - 32:
            env = (num_samples - i) / 32
        val = (
            math.sin(2 * math.pi * freq * t + phase) * 0.7
            + math.sin(2 * math.pi * freq * 2.1 * t + phase) * 0.3
        )
        samples.append(max(-1.0, min(1.0, val * volume * env)))

    pcm = struct.pack(f"<{num_samples}h", *[int(s * 32767) for s in samples])
    next_phase = phase + 2 * math.pi * freq * (num_samples / sample_rate)
    return pcm, next_phase


class MockStreamingTTS(StreamingTTS):
    def __init__(
        self,
        first_audio_ms: float = 75.0,
        chunk_duration_ms: float = CHUNK_DURATION_MS,
        sample_rate: int = SAMPLE_RATE,
        realtime_pace: bool = True,
    ):
        self._first_audio_ms = first_audio_ms
        self._chunk_duration_ms = chunk_duration_ms
        self._sample_rate = sample_rate
        self._realtime_pace = realtime_pace

        self._connected = False
        self._cancelled = False
        self._closed = False
        self._phase = 0.0
        self._t_connect = 0.0
        self._t_first_audio = 0.0

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def audio_format(self) -> str:
        return "pcm16"

    async def connect(self) -> None:
        self._connected = True
        self._t_connect = time.perf_counter()
        await asyncio.sleep(0.010)  # warm

    async def stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[TTSEvent]:
        if not self._connected or self._closed:
            yield TTSEvent(type=TTSEventType.ERROR, text="TTS not connected")
            return

        self._cancelled = False
        first_chunk = True
        samples_per_chunk = int(self._sample_rate * self._chunk_duration_ms / 1000)

        async for text in text_stream:
            if self._cancelled:
                break

            text = text.strip()
            if not text:
                continue

            # Estimated spoken duration
            duration_s = max(0.15, len(text) / CHARS_PER_SECOND)
            total_samples = int(duration_s * self._sample_rate)
            remaining = total_samples

            if first_chunk:
                await asyncio.sleep(self._first_audio_ms / 1000.0)
                if self._cancelled:
                    break
                self._t_first_audio = time.perf_counter()
                first_chunk = False

            while remaining > 0 and not self._cancelled:
                n = min(samples_per_chunk, remaining)
                pcm, self._phase = _generate_tone_chunk(
                    n, self._sample_rate, phase=self._phase
                )
                remaining -= n

                yield TTSEvent(
                    type=TTSEventType.AUDIO,
                    audio=pcm,
                    sample_rate=self._sample_rate,
                    text=text if remaining <= 0 else "",
                    meta={
                        "samples": n,
                        "is_first": self._t_first_audio == time.perf_counter(),  # approx
                    },
                )

                # Real-time-ish pacing so the client buffer doesn't explode.
                # Disabled in eval mode so we measure pipeline latency, not speech duration.
                if self._realtime_pace:
                    await asyncio.sleep(self._chunk_duration_ms / 1000.0 * 0.85)

        if not self._cancelled:
            yield TTSEvent(type=TTSEventType.DONE)

    async def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        self._closed = True
        self._cancelled = True

    def metrics(self) -> dict:
        return {
            "t_connect": self._t_connect,
            "t_first_audio": self._t_first_audio,
            "first_audio_ms": self._first_audio_ms,
        }
