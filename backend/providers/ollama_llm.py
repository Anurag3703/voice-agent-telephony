"""
Local Ollama Streaming LLM Adapter (Uses local qwen3:4b / qwen3:8b via Ollama).
Enables true, intelligent, dynamic conversational responses with zero cloud API keys!
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional, List, Any

import aiohttp

from llm_protocol import (
    StreamingLLM,
    LLMEvent,
    LLMEventType,
    ChatMessage,
)


class OllamaStreamingLLM(StreamingLLM):
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._current_resp: Optional[aiohttp.ClientResponse] = None
        self._cancelled = False
        self._closed = False

    @property
    def supports_streaming(self) -> bool:
        return True

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60, connect=5)
            )

    def _to_ollama_messages(self, messages: List[ChatMessage]) -> list[dict]:
        out = []
        for m in messages:
            role = m.role if m.role in ("system", "user", "assistant") else "user"
            content = m.content or ""
            out.append({"role": role, "content": content})
        return out[-8:]  # Keep last 8 turns for tight conversational context

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[LLMEvent]:
        if self._closed:
            yield LLMEvent(type=LLMEventType.ERROR, text="LLM closed")
            return
        await self.connect()
        self._cancelled = False

        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages),
            "stream": True,
            "options": {
                "temperature": 0.6,
                "top_p": 0.9,
                "num_predict": 45,
                "stop": ["\n\n", "User:", "<|eot_id|>"],
            }
        }

        url = f"{self.base_url}/api/chat"
        try:
            assert self._session is not None
            resp = await self._session.post(url, json=body)
            self._current_resp = resp
            if resp.status >= 400:
                err = await resp.text()
                yield LLMEvent(type=LLMEventType.ERROR, text=f"Ollama {resp.status}: {err[:200]}")
                return

            buf = ""
            async for chunk in resp.content.iter_any():
                if self._cancelled:
                    break
                buf += chunk.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = obj.get("message") or {}
                    token = msg.get("content", "")
                    if token:
                        yield LLMEvent(type=LLMEventType.TOKEN, text=token)

                    if obj.get("done"):
                        yield LLMEvent(type=LLMEventType.DONE)
                        return
        except Exception as e:
            if not self._cancelled:
                yield LLMEvent(type=LLMEventType.ERROR, text=str(e))
        finally:
            if self._current_resp and not self._current_resp.closed:
                try:
                    self._current_resp.close()
                except Exception:
                    pass
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
