"""
P0 – Deepgram live streaming STT adapter.

WebSocket: send binary PCM16 LE mono; receive JSON partial/final transcripts.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import aiohttp

from stt_protocol import StreamingSTT, TranscriptEvent, TranscriptType


class DeepgramStreamingSTT(StreamingSTT):
    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        sample_rate: int = 16000,
        language: str = "en-US",
    ):
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.language = language
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False

    @property
    def supports_partials(self) -> bool:
        return True

    async def connect(self) -> None:
        if self._ws and not self._ws.closed:
            return
        params = urlencode({
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "channels": "1",
            "punctuate": "true",
            "interim_results": "true",
            "endpointing": "300",
            "language": self.language,
        })
        url = f"wss://api.deepgram.com/v1/listen?{params}"
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(
            url,
            headers={"Authorization": f"Token {self.api_key}"},
            heartbeat=20.0,
        )
        self._closed = False
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "Results":
                        channel = (data.get("channel") or {})
                        alts = (channel.get("alternatives") or [{}])
                        text = (alts[0] or {}).get("transcript") or ""
                        is_final = bool(data.get("is_final"))
                        speech_final = bool(data.get("speech_final"))
                        conf = (alts[0] or {}).get("confidence")
                        if not text and not is_final and not speech_final:
                            continue
                        await self._queue.put(TranscriptEvent(
                            type=TranscriptType.FINAL if (is_final or speech_final) else TranscriptType.PARTIAL,
                            text=text,
                            is_final=is_final or speech_final,
                            confidence=conf,
                            meta={"speech_final": speech_final},
                        ))
                    elif data.get("type") == "Error":
                        await self._queue.put(TranscriptEvent(
                            type=TranscriptType.ERROR,
                            text=str(data),
                        ))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception as e:
            await self._queue.put(TranscriptEvent(type=TranscriptType.ERROR, text=str(e)))
        finally:
            await self._queue.put(None)  # sentinel

    async def send_audio(self, chunk: bytes, sample_rate: int = 16000) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send_bytes(chunk)

    async def receive_events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                break
            yield ev

    async def finalize(self) -> None:
        # Deepgram: send empty JSON close-stream message for Finalize
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_json({"type": "Finalize"})
            except Exception:
                pass

    async def close(self) -> None:
        self._closed = True
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_json({"type": "CloseStream"})
            except Exception:
                pass
            await self._ws.close()
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except Exception:
                pass
        if self._session and not self._session.closed:
            await self._session.close()
        self._ws = None
        self._session = None
