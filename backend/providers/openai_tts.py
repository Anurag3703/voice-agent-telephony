"""
P0 – OpenAI TTS adapter (streaming audio out).

OpenAI TTS takes text per request and can stream PCM audio bytes.
We synthesize each incoming phrase as its own request so LLM→TTS overlap
still works at phrase granularity (matches ResponseChunker).
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

import aiohttp

from tts_protocol import StreamingTTS, TTSEvent, TTSEventType


class OpenAIStreamingTTS(StreamingTTS):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        sample_rate: int = 24000,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self._sample_rate = sample_rate
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._current_resp: Optional[aiohttp.ClientResponse] = None
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
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )

    async def stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[TTSEvent]:
        if self._closed:
            yield TTSEvent(type=TTSEventType.ERROR, text="TTS closed")
            return
        await self.connect()
        self._cancelled = False
        first = True

        try:
            async for phrase in text_stream:
                if self._cancelled:
                    break
                phrase = (phrase or "").strip()
                if not phrase:
                    continue
                async for ev in self._synthesize_phrase(phrase, is_turn_first=first):
                    if self._cancelled:
                        break
                    if ev.type == TTSEventType.AUDIO and first:
                        first = False
                    yield ev
                    if ev.type == TTSEventType.ERROR:
                        return
            if not self._cancelled:
                yield TTSEvent(type=TTSEventType.DONE)
        except Exception as e:
            if not self._cancelled:
                yield TTSEvent(type=TTSEventType.ERROR, text=str(e))

    async def _synthesize_phrase(self, text: str, is_turn_first: bool) -> AsyncIterator[TTSEvent]:
        assert self._session is not None
        url = f"{self.base_url}/audio/speech"
        body = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",  # raw PCM16 LE
        }
        # Some models accept stream; request streaming body when available
        try:
            async with self._session.post(url, json=body) as resp:
                self._current_resp = resp
                if resp.status >= 400:
                    err = await resp.text()
                    yield TTSEvent(type=TTSEventType.ERROR, text=f"OpenAI TTS {resp.status}: {err[:300]}")
                    return
                # Stream response body in chunks
                buf = b""
                chunk_bytes = int(self._sample_rate * 0.04 * 2)  # ~40ms PCM16 mono
                async for data in resp.content.iter_chunked(chunk_bytes):
                    if self._cancelled:
                        return
                    buf += data
                    while len(buf) >= chunk_bytes:
                        piece = buf[:chunk_bytes]
                        buf = buf[chunk_bytes:]
                        yield TTSEvent(
                            type=TTSEventType.AUDIO,
                            audio=piece,
                            sample_rate=self._sample_rate,
                            text=text if is_turn_first else "",
                        )
                if buf and not self._cancelled:
                    yield TTSEvent(
                        type=TTSEventType.AUDIO,
                        audio=buf,
                        sample_rate=self._sample_rate,
                    )
        except Exception as e:
            if not self._cancelled:
                yield TTSEvent(type=TTSEventType.ERROR, text=str(e))
        finally:
            self._current_resp = None

    async def cancel(self) -> None:
        self._cancelled = True
        if self._current_resp and not self._current_resp.closed:
            try:
                self._current_resp.close()
            except Exception:
                pass
            self._current_resp = None

    async def close(self) -> None:
        self._closed = True
        self._cancelled = True
        if self._current_resp and not self._current_resp.closed:
            try:
                self._current_resp.close()
            except Exception:
                pass
            self._current_resp = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
