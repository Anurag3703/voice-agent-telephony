"""
Mock Streaming STT for deterministic latency testing.

Simulates realistic partial → final behavior without external API keys.
Useful for measuring the rest of the pipeline and for CI.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional

from stt_protocol import StreamingSTT, TranscriptEvent, TranscriptType


# Scripted progressive partials for multi-turn conversational demo phrases
DEMO_CONVERSATION_TURNS = [
    [
        "Can you",
        "Can you tell",
        "Can you tell me",
        "Can you tell me the status",
        "Can you tell me the status of my vehicle?",
    ],
    [
        "What is",
        "What is my",
        "What is my battery",
        "What is my battery level?",
    ],
    [
        "Is the",
        "Is the climate",
        "Is the climate on",
        "Is the climate on right now?",
    ],
    [
        "Can you",
        "Can you honk",
        "Can you honk the horn",
        "Can you honk the horn for me?",
    ],
    [
        "Where is",
        "Where is my",
        "Where is my car",
        "Where is my car parked?",
    ],
]


class MockStreamingSTT(StreamingSTT):
    def __init__(
        self,
        partial_interval_ms: float = 120.0,
        final_delay_ms: float = 40.0,
        sample_rate: int = 16000,
    ):
        self.partial_interval_ms = partial_interval_ms
        self.final_delay_ms = final_delay_ms
        self.sample_rate = sample_rate

        self._connected = False
        self._audio_bytes = 0
        self._speech_started = False
        self._finalized = False
        self._closed = False
        self._event_queue: asyncio.Queue[Optional[TranscriptEvent]] = asyncio.Queue()
        self._partial_task: Optional[asyncio.Task] = None
        self._part_index = 0
        self._turn_index = 0
        self._t_connect = 0.0
        self._t_first_audio = 0.0
        self._t_speech_end = 0.0

    @property
    def supports_partials(self) -> bool:
        return True

    async def connect(self) -> None:
        self._connected = True
        self._t_connect = time.perf_counter()
        # Simulate connection warm-up cost (should be paid once per session)
        await asyncio.sleep(0.015)

    def _reset_utterance(self) -> None:
        """Allow the next user turn after finalize / barge-in."""
        self._finalized = False
        self._speech_started = False
        self._audio_bytes = 0
        self._part_index = 0
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        self._partial_task = None

    async def send_audio(self, chunk: bytes, sample_rate: int = 16000) -> None:
        if not self._connected or self._closed:
            return

        if self._audio_bytes == 0:
            self._t_first_audio = time.perf_counter()

        self._audio_bytes += len(chunk)

        # Simple energy-less heuristic: after ~300 ms of audio, start "speech"
        duration_s = self._audio_bytes / (2 * sample_rate)  # PCM16
        if not self._speech_started and duration_s > 0.25:
            self._speech_started = True
            await self._event_queue.put(
                TranscriptEvent(type=TranscriptType.SPEECH_START, text="")
            )
            self._partial_task = asyncio.create_task(self._emit_partials())

    async def _emit_partials(self) -> None:
        """Emit progressive partials on a timer while audio is flowing."""
        phrase_parts = DEMO_CONVERSATION_TURNS[self._turn_index % len(DEMO_CONVERSATION_TURNS)]
        try:
            while not self._finalized and not self._closed:
                if self._part_index < len(phrase_parts) - 1:
                    text = phrase_parts[self._part_index]
                    await self._event_queue.put(
                        TranscriptEvent(
                            type=TranscriptType.PARTIAL,
                            text=text,
                            is_final=False,
                            confidence=0.6 + self._part_index * 0.04,
                        )
                    )
                    self._part_index += 1
                await asyncio.sleep(self.partial_interval_ms / 1000.0)
        except asyncio.CancelledError:
            pass

    async def finalize(self) -> None:
        """Called when VAD detects end of speech (or user stops)."""
        if self._closed:
            return
        # Always produce a FINAL so barge-in / turn N+1 is never stuck.
        self._finalized = True
        self._t_speech_end = time.perf_counter()

        if self._partial_task:
            self._partial_task.cancel()
            try:
                await self._partial_task
            except asyncio.CancelledError:
                pass
            self._partial_task = None

        await asyncio.sleep(self.final_delay_ms / 1000.0)

        phrase_parts = DEMO_CONVERSATION_TURNS[self._turn_index % len(DEMO_CONVERSATION_TURNS)]
        idx = min(self._part_index, len(phrase_parts) - 1)
        final_text = phrase_parts[idx] if phrase_parts else ""
        self._turn_index += 1
        await self._event_queue.put(
            TranscriptEvent(
                type=TranscriptType.FINAL,
                text=final_text,
                is_final=True,
                confidence=0.94,
                meta={
                    "stt_finalization_ms": self.final_delay_ms,
                    "audio_duration_s": self._audio_bytes / (2 * self.sample_rate),
                },
            )
        )
        # Ready for the next conversational turn
        self._reset_utterance()

    async def receive_events(self) -> AsyncIterator[TranscriptEvent]:
        while not self._closed:
            event = await self._event_queue.get()
            if event is None:
                # End of current utterance; keep the generator alive for next one
                # In a real multi-turn system we would reset state here.
                continue
            yield event

    async def close(self) -> None:
        self._closed = True
        if self._partial_task:
            self._partial_task.cancel()
        await self._event_queue.put(None)

    def metrics(self) -> dict:
        return {
            "t_connect": self._t_connect,
            "t_first_audio": self._t_first_audio,
            "t_speech_end": self._t_speech_end,
            "audio_bytes": self._audio_bytes,
            "parts_emitted": self._part_index,
        }
