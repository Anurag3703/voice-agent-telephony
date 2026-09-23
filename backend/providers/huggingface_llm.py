"""
Hugging Face Inference API Adapter (OpenAI-compatible / router endpoint).
Free tier friendly via Hugging Face Access Token.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional, List, Any
import aiohttp

from llm_protocol import StreamingLLM, LLMEvent, LLMEventType, ChatMessage


class HuggingFaceStreamingLLM(StreamingLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        base_url: str = "https://router.huggingface.co/v1",
    ):
        self.api_key = api_key
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
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=45, connect=5),
            )

    async def close(self) -> None:
        self._closed = True
        if self._session and not self._session.closed:
            await self._session.close()

    async def cancel(self) -> None:
        self._cancelled = True
        if self._current_resp and not self._current_resp.closed:
            self._current_resp.close()

    def _to_messages(self, messages: List[ChatMessage]) -> list[dict]:
        out = []
        for m in messages:
            role = m.role if m.role in ("system", "user", "assistant") else "user"
            out.append({"role": role, "content": m.content or ""})
        return out[-8:]

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

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_messages(messages),
            "stream": True,
            "max_tokens": 40,
            "temperature": 0.7,
        }

        url = f"{self.base_url}/chat/completions"
        try:
            assert self._session is not None
            resp = await self._session.post(url, json=payload)
            self._current_resp = resp
            if resp.status >= 400:
                err_text = await resp.text()
                yield LLMEvent(type=LLMEventType.ERROR, text=f"HF API {resp.status}: {err_text}")
                return

            async for line_bytes in resp.content:
                if self._cancelled:
                    break
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield LLMEvent(type=LLMEventType.TOKEN, text=token)
                except Exception:
                    continue

            yield LLMEvent(type=LLMEventType.DONE)
        except Exception as e:
            yield LLMEvent(type=LLMEventType.ERROR, text=str(e))
