"""
Low-latency ElevenLabs TTS via WebSocket stream-input.

Feeds text incrementally (LLM tokens/phrases) and yields PCM audio as soon as
ElevenLabs generates it. Uses aggressive chunk_length_schedule + flush for TTFA.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import aiohttp

from tts_protocol import StreamingTTS, TTSEvent, TTSEventType


class ElevenLabsStreamingTTS(StreamingTTS):
    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_flash_v2_5",
        sample_rate: int = 24000,
        base_url: str = "https://api.elevenlabs.io/v1",
        # Aggressive schedule: start audio ASAP (quality tradeoff acceptable for voice agents)
        chunk_schedule: Optional[list[int]] = None,
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self._sample_rate = sample_rate
        self.base_url = base_url.rstrip("/")
        self.chunk_schedule = chunk_schedule or [50, 80, 120, 200]
        self._session: Optional[aiohttp.ClientSession] = None
        self._current_ws: Optional[aiohttp.ClientWebSocketResponse] = None
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
            timeout = aiohttp.ClientTimeout(total=30, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
            # Warm DNS/TLS to ElevenLabs so the first turn is not cold
            try:
                async with self._session.get(
                    "https://api.elevenlabs.io/v1/user",
                    headers={"xi-api-key": self.api_key},
                ) as resp:
                    await resp.read()
            except Exception:
                pass

    def _ws_url(self) -> str:
        qs = urlencode({
            "model_id": self.model_id,
            "output_format": f"pcm_{self._sample_rate}",
            "optimize_streaming_latency": "4",
            "inactivity_timeout": "30",
        })
        return f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?{qs}"

    async def stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[TTSEvent]:
        if self._closed:
            yield TTSEvent(type=TTSEventType.ERROR, text="TTS closed")
            return
        await self.connect()
        self._cancelled = False

        # Prefer WebSocket path; fall back to HTTP phrase streaming on failure
        try:
            async for ev in self._stream_ws(text_stream):
                yield ev
        except Exception as e:
            yield TTSEvent(type=TTSEventType.ERROR, text=f"ElevenLabs WS failed: {e}")

    async def _stream_ws(self, text_stream: AsyncIterator[str]) -> AsyncIterator[TTSEvent]:
        assert self._session is not None
        url = self._ws_url()
        audio_q: asyncio.Queue = asyncio.Queue()
        first_audio = True

        async with self._session.ws_connect(
            url,
            headers={"xi-api-key": self.api_key},
            heartbeat=20.0,
            max_msg_size=4 * 1024 * 1024,
        ) as ws:
            self._current_ws = ws
            # Initialize connection
            await ws.send_json({
                "text": " ",
                "voice_settings": {
                    "stability": 0.35,
                    "similarity_boost": 0.75,
                    "use_speaker_boost": False,
                },
                "generation_config": {
                    "chunk_length_schedule": self.chunk_schedule,
                },
                # auto_mode helps when text arrives as a stream from an LLM
                "try_trigger_generation": True,
                "xi_api_key": self.api_key,
            })

            async def reader():
                try:
                    async for msg in ws:
                        if self._cancelled:
                            break
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            continue
                        data = json.loads(msg.data)
                        if data.get("audio"):
                            try:
                                raw = base64.b64decode(data["audio"])
                            except Exception:
                                continue
                            if raw:
                                await audio_q.put(("audio", raw))
                        if data.get("isFinal"):
                            break
                        if data.get("error"):
                            await audio_q.put(("error", str(data["error"])))
                            break
                except Exception as e:
                    await audio_q.put(("error", str(e)))
                finally:
                    await audio_q.put(("done", None))

            reader_task = asyncio.create_task(reader())

            async def writer():
                try:
                    total = 0
                    flushed_first = False
                    async for piece in text_stream:
                        if self._cancelled:
                            break
                        piece = piece or ""
                        if not piece:
                            continue
                        total += len(piece)
                        # Each phrase from our chunker is a natural unit — send + flush
                        # so first audio does not wait for the 50-char schedule alone.
                        do_flush = (not flushed_first) or any(c in piece for c in ".!?;")
                        await ws.send_json({"text": piece, "flush": bool(do_flush)})
                        if do_flush:
                            flushed_first = True
                    if not self._cancelled:
                        await ws.send_json({"text": " ", "flush": True})
                        await ws.send_json({"text": ""})
                except Exception as e:
                    await audio_q.put(("error", str(e)))

            writer_task = asyncio.create_task(writer())

            try:
                while True:
                    if self._cancelled:
                        break
                    kind, payload = await audio_q.get()
                    if kind == "audio" and payload:
                        yield TTSEvent(
                            type=TTSEventType.AUDIO,
                            audio=payload,
                            sample_rate=self._sample_rate,
                            text="" if not first_audio else "",
                        )
                        first_audio = False
                    elif kind == "error":
                        yield TTSEvent(type=TTSEventType.ERROR, text=str(payload))
                        break
                    elif kind == "done":
                        break
            finally:
                self._cancelled = True
                self._current_ws = None
                for t in (writer_task, reader_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(writer_task, reader_task, return_exceptions=True)

        if not self._cancelled:
            yield TTSEvent(type=TTSEventType.DONE)

    async def cancel(self) -> None:
        self._cancelled = True
        if self._current_ws and not self._current_ws.closed:
            try:
                await self._current_ws.close()
            except Exception:
                pass
            self._current_ws = None

    async def close(self) -> None:
        self._closed = True
        self._cancelled = True
        if self._current_ws and not self._current_ws.closed:
            try:
                await self._current_ws.close()
            except Exception:
                pass
            self._current_ws = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
