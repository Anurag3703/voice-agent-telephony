"""
P0 – Groq streaming LLM adapter (OpenAI-compatible API, free tier friendly).
"""

from __future__ import annotations

from typing import Optional

from providers.openai_llm import OpenAIStreamingLLM


class GroqStreamingLLM(OpenAIStreamingLLM):
    """Groq exposes an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url)
