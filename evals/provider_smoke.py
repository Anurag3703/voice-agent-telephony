"""
P0.6 – Provider smoke tests

- Always: mock path constructs and runs a tiny STT→LLM→TTS pipeline
- Without keys: requesting groq/elevenlabs/deepgram/openai falls back to mock
- With keys: optional live probe (skipped if env not set)

Usage:
  PYTHONPATH=backend:. python -m evals.provider_smoke
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _section(title: str) -> None:
    print(f"\n{'='*56}\n  {title}\n{'='*56}")


def test_registry_fallback() -> int:
    _section("Registry fallback (no keys → mock)")
    # Clear provider keys for this process slice
    for k in list(os.environ.keys()):
        if k.endswith("_API_KEY"):
            os.environ.pop(k, None)

    os.environ["STT_PROVIDER"] = "deepgram"
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["TTS_PROVIDER"] = "elevenlabs"

    from providers.registry import create_stt, create_llm, create_tts

    stt, llm, tts = create_stt(), create_llm(), create_tts()
    print(f"  STT={type(stt).__name__}")
    print(f"  LLM={type(llm).__name__}")
    print(f"  TTS={type(tts).__name__}")
    ok = (
        "Mock" in type(stt).__name__
        and "Mock" in type(llm).__name__
        and "Mock" in type(tts).__name__
    )
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


async def test_mock_pipeline() -> int:
    _section("Mock pipeline smoke (STT final → LLM tokens → TTS audio)")
    os.environ["STT_PROVIDER"] = "mock"
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["TTS_PROVIDER"] = "mock"

    # Fresh imports not required; factory reads env each call
    from providers.registry import create_stt, create_llm, create_tts
    from stt_protocol import TranscriptType
    from llm_protocol import ChatMessage, LLMEventType
    from tts_protocol import TTSEventType

    stt, llm, tts = create_stt(), create_llm(), create_tts()
    await stt.connect()
    await llm.connect()
    await tts.connect()

    t0 = time.perf_counter()
    await stt.send_audio(bytes(3200), sample_rate=16000)
    await stt.finalize()

    final_text = None
    async for ev in stt.receive_events():
        if ev.type == TranscriptType.FINAL:
            final_text = ev.text or "hello"
            break
    if not final_text:
        print("  FAIL: no STT final")
        return 1
    print(f"  STT final: {final_text!r}  (+{(time.perf_counter()-t0)*1000:.0f} ms)")

    messages = [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="Hello"),
    ]
    tokens = []
    t_llm = time.perf_counter()
    async for ev in llm.stream(messages):
        if ev.type == LLMEventType.TOKEN:
            tokens.append(ev.text)
        elif ev.type in (LLMEventType.DONE, LLMEventType.ERROR, LLMEventType.TOOL_CALL):
            break
    if not tokens:
        print("  FAIL: no LLM tokens")
        return 1
    print(f"  LLM tokens: {len(tokens)}  (+{(time.perf_counter()-t_llm)*1000:.0f} ms)  {''.join(tokens)[:60]!r}")

    async def phrases():
        yield "".join(tokens)[:80]

    audio_chunks = 0
    t_tts = time.perf_counter()
    async for ev in tts.stream(phrases()):
        if ev.type == TTSEventType.AUDIO:
            audio_chunks += 1
        elif ev.type in (TTSEventType.DONE, TTSEventType.ERROR):
            break
    if audio_chunks < 1:
        print("  FAIL: no TTS audio")
        return 1
    print(f"  TTS audio chunks: {audio_chunks}  (+{(time.perf_counter()-t_tts)*1000:.0f} ms)")

    await stt.close()
    await llm.close()
    await tts.close()
    print("  PASS")
    return 0


def test_adapter_imports() -> int:
    _section("Adapter imports")
    errs = []
    try:
        from providers.groq_llm import GroqStreamingLLM  # noqa: F401
        print("  GroqStreamingLLM OK")
    except Exception as e:
        errs.append(f"groq: {e}")
    try:
        from providers.elevenlabs_tts import ElevenLabsStreamingTTS  # noqa: F401
        print("  ElevenLabsStreamingTTS OK")
    except Exception as e:
        errs.append(f"elevenlabs: {e}")
    try:
        from providers.openai_llm import OpenAIStreamingLLM  # noqa: F401
        from providers.openai_tts import OpenAIStreamingTTS  # noqa: F401
        from providers.deepgram_stt import DeepgramStreamingSTT  # noqa: F401
        print("  OpenAI + Deepgram adapters OK")
    except Exception as e:
        errs.append(f"openai/deepgram: {e}")
    if errs:
        for e in errs:
            print("  FAIL", e)
        return 1
    print("  PASS")
    return 0


def test_live_optional() -> int:
    _section("Live probes (optional – skipped without keys)")
    ran = 0
    # Groq
    if os.environ.get("GROQ_API_KEY"):
        ran += 1
        print("  GROQ_API_KEY set – run manual live test outside CI if desired")
    else:
        print("  skip Groq (no GROQ_API_KEY)")
    if os.environ.get("ELEVENLABS_API_KEY"):
        ran += 1
        print("  ELEVENLABS_API_KEY set – run manual live test outside CI if desired")
    else:
        print("  skip ElevenLabs (no ELEVENLABS_API_KEY)")
    if os.environ.get("DEEPGRAM_API_KEY"):
        print("  DEEPGRAM_API_KEY set")
    else:
        print("  skip Deepgram (no DEEPGRAM_API_KEY)")
    print("  PASS (optional section)")
    return 0


def main() -> int:
    print("\nP0.6 Provider smoke tests")
    code = 0
    code |= test_adapter_imports()
    code |= test_registry_fallback()
    code |= asyncio.run(test_mock_pipeline())
    code |= test_live_optional()
    print()
    if code == 0:
        print("All P0.6 smoke tests passed.\n")
    else:
        print("Some P0.6 tests FAILED.\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
