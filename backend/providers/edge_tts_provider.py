"""
Real-World Neural Voice TTS Adapter using Edge TTS (Microsoft Neural Voices).
Zero API keys required; produces crystal clear, natural human speech decoded to 24kHz PCM16.
"""

from __future__ import annotations

import asyncio
import io
import os
from typing import AsyncIterator, Optional

from tts_protocol import StreamingTTS, TTSEvent, TTSEventType

try:
    import edge_tts
    import av
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False


PERSONA_VOICE_MAP = {
    "friend": "en-US-GuyNeural",         # Sam (Casual warm male)
    "assistant": "en-US-AriaNeural",      # Alex (Crisp professional female)
    "support": "en-US-ChristopherNeural", # Jordan (Calm technical male)
    "margaret": "en-US-AnaNeural",        # Margaret (Elderly grandmother)
    "arthur": "en-US-RogerNeural",        # Arthur (Older distinguished male)
}


class EdgeStreamingTTS(StreamingTTS):
    def __init__(
        self,
        voice: Optional[str] = None,
        rate: str = "+8%",
        sample_rate: int = 24000,
    ):
        self.voice = voice or "en-US-GuyNeural"
        self.rate = rate
        self._sample_rate = sample_rate
        self._connected = False
        self._cancelled = False
        self._closed = False

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
        self._closed = False

    async def stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[TTSEvent]:
        if not HAS_EDGE_TTS or self._closed:
            yield TTSEvent(type=TTSEventType.ERROR, text="edge-tts or PyAV not installed")
            return

        self._cancelled = False
        first_audio = True

        try:
            async for phrase in text_stream:
                if self._cancelled:
                    break
                phrase = (phrase or "").strip()
                if not phrase:
                    continue

                communicate = edge_tts.Communicate(phrase, self.voice, rate=self.rate)
                mp3_data = bytearray()
                
                async for chunk in communicate.stream():
                    if self._cancelled:
                        break
                    if chunk["type"] == "audio" and chunk["data"]:
                        mp3_data.extend(chunk["data"])

                if self._cancelled or not mp3_data:
                    continue

                # Decode complete phrase MP3 into pure PCM16 24kHz using PyAV
                def _decode_to_pcm(raw_mp3: bytes) -> bytes:
                    try:
                        container = av.open(io.BytesIO(raw_mp3))
                        resampler = av.AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
                        pcm_parts = []
                        for frame in container.decode(audio=0):
                            frame.pts = None
                            for resampled in resampler.resample(frame):
                                pcm_parts.append(bytes(resampled.planes[0]))
                        return b"".join(pcm_parts)
                    except Exception as e:
                        print(f"[EdgeTTS] Decode error: {e}")
                        return b""

                pcm_bytes = await asyncio.to_thread(_decode_to_pcm, bytes(mp3_data))

                if pcm_bytes and not self._cancelled:
                    # Stream in ~40ms chunks (1920 bytes @ 24kHz 16-bit mono)
                    CHUNK_SIZE = int(self._sample_rate * 0.04 * 2)
                    for offset in range(0, len(pcm_bytes), CHUNK_SIZE):
                        if self._cancelled:
                            break
                        chunk_pcm = pcm_bytes[offset:offset + CHUNK_SIZE]
                        yield TTSEvent(
                            type=TTSEventType.AUDIO,
                            audio=chunk_pcm,
                            sample_rate=self._sample_rate,
                            text=phrase if first_audio else "",
                            meta={"format": "pcm16", "voice": self.voice},
                        )
                        first_audio = False
                        await asyncio.sleep(0.005)

            if not self._cancelled:
                yield TTSEvent(type=TTSEventType.DONE)
        except Exception as e:
            if not self._cancelled:
                yield TTSEvent(type=TTSEventType.ERROR, text=str(e))

    async def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        self._closed = True
        self._cancelled = True
