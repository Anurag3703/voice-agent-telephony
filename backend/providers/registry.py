"""
P0 – Provider registry

Creates STT / LLM / TTS implementations from environment variables.
Falls back to mocks when keys are missing or provider=mock.
"""

from __future__ import annotations

import os

from stt_protocol import StreamingSTT
from llm_protocol import StreamingLLM
from tts_protocol import StreamingTTS


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def create_stt() -> StreamingSTT:
    provider = _env("STT_PROVIDER", "whisper").lower()
    if provider == "deepgram":
        key = _env("DEEPGRAM_API_KEY")
        if not key:
            print("[providers] STT_PROVIDER=deepgram but DEEPGRAM_API_KEY missing → whisper")
        else:
            from providers.deepgram_stt import DeepgramStreamingSTT
            return DeepgramStreamingSTT(
                api_key=key,
                model=_env("DEEPGRAM_MODEL", "nova-2"),
            )
    if provider in ("whisper", "local", "faster-whisper", "auto"):
        try:
            from providers.whisper_stt import LocalWhisperStreamingSTT
            return LocalWhisperStreamingSTT(model_size=_env("WHISPER_MODEL", "tiny.en"))
        except Exception as e:
            print(f"[providers] Local Whisper failed: {e}")

    from mock_stt import MockStreamingSTT
    return MockStreamingSTT(partial_interval_ms=110, final_delay_ms=35)


def create_llm() -> StreamingLLM:
    provider = _env("LLM_PROVIDER", "huggingface").lower()
    if provider == "openai":
        key = _env("OPENAI_API_KEY")
        if not key:
            print("[providers] LLM_PROVIDER=openai but OPENAI_API_KEY missing → llama")
        else:
            from providers.openai_llm import OpenAIStreamingLLM
            return OpenAIStreamingLLM(
                api_key=key,
                model=_env("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            )
    if provider == "groq":
        key = _env("GROQ_API_KEY")
        if not key:
            print("[providers] LLM_PROVIDER=groq but GROQ_API_KEY missing → llama")
        else:
            from providers.groq_llm import GroqStreamingLLM
            return GroqStreamingLLM(
                api_key=key,
                model=_env("GROQ_LLM_MODEL", "openai/gpt-oss-120b"),
            )
    if provider in ("huggingface", "hf"):
        key = _env("HF_TOKEN") or _env("HUGGINGFACE_API_KEY")
        if not key:
            print("[providers] LLM_PROVIDER=huggingface but HF_TOKEN missing → fallback")
        else:
            from providers.huggingface_llm import HuggingFaceStreamingLLM
            return HuggingFaceStreamingLLM(
                api_key=key,
                model=_env("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            )
    if provider in ("llama", "llama3", "llama3.2", "ollama", "local", "auto"):
        try:
            from providers.ollama_llm import OllamaStreamingLLM
            model_name = _env("OLLAMA_MODEL", "llama3.2:3b")
            return OllamaStreamingLLM(model=model_name)
        except Exception as e:
            print(f"[providers] Ollama LLM failed: {e}")

    from mock_llm import MockStreamingLLM
    return MockStreamingLLM(ttft_ms=45, inter_token_ms=12)


def create_tts(persona: str = "friend") -> StreamingTTS:
    provider = _env("TTS_PROVIDER", "edge").lower()
    if provider == "openai":
        key = _env("OPENAI_API_KEY")
        if not key:
            print("[providers] TTS_PROVIDER=openai but OPENAI_API_KEY missing → edge/mock")
        else:
            from providers.openai_tts import OpenAIStreamingTTS
            return OpenAIStreamingTTS(
                api_key=key,
                model=_env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
                voice=_env("OPENAI_TTS_VOICE", "alloy"),
            )
    if provider == "elevenlabs":
        key = _env("ELEVENLABS_API_KEY")
        if not key:
            print("[providers] TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY missing → edge/mock")
        else:
            from providers.elevenlabs_tts import ElevenLabsStreamingTTS
            return ElevenLabsStreamingTTS(
                api_key=key,
                voice_id=_env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
                model_id=_env("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
            )
    if provider in ("edge", "edge-tts", "auto", "mock"):
        try:
            from providers.edge_tts_provider import EdgeStreamingTTS, PERSONA_VOICE_MAP
            voice = PERSONA_VOICE_MAP.get(persona, "en-US-GuyNeural")
            return EdgeStreamingTTS(voice=voice)
        except Exception:
            pass

    from mock_tts import MockStreamingTTS
    return MockStreamingTTS(first_audio_ms=75, chunk_duration_ms=40)
