"""
P0 – OpenAI Chat Completions streaming LLM adapter.

Uses SSE token stream. cancel() sets a flag checked between chunks.
Tool calls: best-effort parsing of streamed tool_call deltas (simplified).
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


class OpenAIStreamingLLM(StreamingLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
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
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

    def _to_openai_messages(self, messages: List[ChatMessage]) -> list[dict]:
        out = []
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "tool",
                    "content": m.content or "",
                    "tool_call_id": m.tool_call_id or "tool_call",
                    **({"name": m.name} if m.name else {}),
                })
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

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
            "messages": self._to_openai_messages(messages),
            "stream": True,
            "temperature": 0.3,
            "max_tokens": 120,  # keep spoken answers short → faster first phrase + TTS
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        try:
            assert self._session is not None
            async with self._session.post(url, json=body) as resp:
                self._current_resp = resp
                if resp.status >= 400:
                    err = await resp.text()
                    yield LLMEvent(type=LLMEventType.ERROR, text=f"OpenAI {resp.status}: {err[:300]}")
                    return

                # Accumulate tool call fragments if any
                tool_acc: dict[int, dict] = {}

                async for raw in resp.content:
                    if self._cancelled:
                        return
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    for part in line.split("\n"):
                        part = part.strip()
                        if not part.startswith("data:"):
                            continue
                        data = part[5:].strip()
                        if data == "[DONE]":
                            # Flush tool calls
                            for tc in tool_acc.values():
                                if tc.get("name"):
                                    args = {}
                                    try:
                                        args = json.loads(tc.get("arguments") or "{}")
                                    except Exception:
                                        args = {}
                                    yield LLMEvent(
                                        type=LLMEventType.TOOL_CALL,
                                        tool_name=tc["name"],
                                        tool_args=args,
                                        tool_call_id=tc.get("id"),
                                    )
                            yield LLMEvent(type=LLMEventType.DONE)
                            return
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield LLMEvent(type=LLMEventType.TOKEN, text=content)
                        # tool_calls streaming
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            slot = tool_acc.setdefault(idx, {"name": "", "arguments": "", "id": None})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = (slot["name"] or "") + fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] = (slot["arguments"] or "") + fn["arguments"]
        except Exception as e:
            if not self._cancelled:
                yield LLMEvent(type=LLMEventType.ERROR, text=str(e))
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
