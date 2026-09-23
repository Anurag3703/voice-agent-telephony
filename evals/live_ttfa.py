"""
Live TTFA component timing (Groq LLM + ElevenLabs TTS + mock/real STT)

Measures server-side pipeline (not browser T6):
  STT final → LLM first token → first phrase → TTS first audio

Usage:
  PYTHONPATH=backend:. python -m evals.live_ttfa --turns 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from providers.registry import create_stt, create_llm, create_tts
from stt_protocol import TranscriptType
from llm_protocol import ChatMessage, LLMEventType
from tts_protocol import TTSEventType
from chunker import ResponseChunker


UTTERANCES = [
    "Hello, how are you?",
    "Thanks, that helps.",
    "Can you hear me clearly?",
]


async def one_turn(utterance: str, turn: int) -> dict:
    stt = create_stt()
    llm = create_llm()
    tts = create_tts()
    await stt.connect()
    await llm.connect()
    await tts.connect()

    t0 = time.perf_counter()

    await stt.send_audio(bytes(3200), sample_rate=16000)
    await stt.finalize()
    stt_text = utterance
    t_stt = None
    async for ev in stt.receive_events():
        if ev.type == TranscriptType.FINAL:
            t_stt = time.perf_counter()
            if os.environ.get("STT_PROVIDER", "mock").lower() == "mock":
                stt_text = utterance
            else:
                stt_text = ev.text or utterance
            break
    stt_ms = ((t_stt or time.perf_counter()) - t0) * 1000

    messages = [
        ChatMessage(
            role="system",
            content="You are a vehicle voice assistant. Answer in 1 short spoken sentence. No lists.",
        ),
        ChatMessage(role="user", content=stt_text),
    ]

    chunker = ResponseChunker(first_min_chars=6, min_chars=14, max_chars=80)
    t_llm_start = time.perf_counter()
    ttft_ms = None
    first_phrase = None
    full = ""
    text_q: asyncio.Queue = asyncio.Queue()

    async def speech():
        nonlocal ttft_ms, first_phrase, full
        async for ev in llm.stream(messages):
            if ev.type == LLMEventType.TOKEN:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_llm_start) * 1000
                full += ev.text
                for ph in chunker.add(ev.text):
                    if first_phrase is None:
                        first_phrase = ph
                    await text_q.put(ph)
            elif ev.type == LLMEventType.DONE:
                for ph in chunker.flush():
                    if first_phrase is None:
                        first_phrase = ph
                    await text_q.put(ph)
                await text_q.put(None)
                break
            elif ev.type == LLMEventType.ERROR:
                await text_q.put(None)
                raise RuntimeError(ev.text or "LLM error")
            elif ev.type == LLMEventType.TOOL_CALL:
                # Latency probe: skip tools; synthesize a short spoken line
                if not full:
                    full = "I can help with that."
                    await text_q.put(full)
                await text_q.put(None)
                break

    async def tts_consume():
        async def gen():
            while True:
                item = await text_q.get()
                if item is None:
                    break
                yield item

        t_tts_start = time.perf_counter()
        first_audio = None
        chunks = 0
        async for ev in tts.stream(gen()):
            if ev.type == TTSEventType.AUDIO:
                if first_audio is None:
                    first_audio = time.perf_counter()
                chunks += 1
            elif ev.type == TTSEventType.ERROR:
                raise RuntimeError(ev.text or "TTS error")
            elif ev.type == TTSEventType.DONE:
                break
        return t_tts_start, first_audio, chunks

    tts_task = asyncio.create_task(tts_consume())
    await speech()
    t_tts_start, first_audio, chunks = await tts_task

    tts_ttfa = ((first_audio or time.perf_counter()) - t_tts_start) * 1000 if first_audio else float("nan")
    t0_to_audio = ((first_audio or time.perf_counter()) - t0) * 1000 if first_audio else float("nan")

    await stt.close()
    await llm.close()
    await tts.close()

    return {
        "turn": turn,
        "utterance": utterance,
        "stt_ms": stt_ms,
        "llm_ttft_ms": ttft_ms or float("nan"),
        "tts_ttfa_ms": tts_ttfa,
        "t0_to_first_audio_ms": t0_to_audio,
        "tts_chunks": chunks,
        "reply": full[:80],
        "first_phrase": (first_phrase or "")[:60],
    }


async def run(n: int) -> int:
    print(f"\nLive TTFA probe – {n} turns")
    print(f"  STT={os.environ.get('STT_PROVIDER','mock')}  "
          f"LLM={os.environ.get('LLM_PROVIDER','mock')}  "
          f"TTS={os.environ.get('TTS_PROVIDER','mock')}")
    print(f"  LLM model={os.environ.get('GROQ_LLM_MODEL') or os.environ.get('OPENAI_LLM_MODEL') or 'default'}")
    print(f"  TTS voice={os.environ.get('ELEVENLABS_VOICE_ID', 'default')}\n")

    rows = []
    for i in range(n):
        u = UTTERANCES[i % len(UTTERANCES)]
        try:
            r = await one_turn(u, i + 1)
            rows.append(r)
            print(
                f"  #{r['turn']}  T0→audio={r['t0_to_first_audio_ms']:7.0f}ms  "
                f"STT={r['stt_ms']:5.0f}  LLM={r['llm_ttft_ms']:6.0f}  "
                f"TTS={r['tts_ttfa_ms']:6.0f}  chunks={r['tts_chunks']}  "
                f"{r['reply']!r}"
            )
        except Exception as e:
            print(f"  #{i+1}  FAIL: {e}")
            return 1

    if not rows:
        return 1

    def pct(xs, p):
        s = sorted(xs)
        k = int(round((p / 100) * (len(s) - 1)))
        return s[max(0, min(k, len(s) - 1))]

    ttfas = [r["t0_to_first_audio_ms"] for r in rows]
    llms = [r["llm_ttft_ms"] for r in rows]
    ttss = [r["tts_ttfa_ms"] for r in rows]
    print("\n" + "=" * 56)
    print(f"  n={len(rows)}")
    print(f"  LLM TTFT   P50={pct(llms,50):.0f}  P95={pct(llms,95):.0f}  ms")
    print(f"  TTS TTFA   P50={pct(ttss,50):.0f}  P95={pct(ttss,95):.0f}  ms")
    print(f"  T0→audio   P50={pct(ttfas,50):.0f}  P95={pct(ttfas,95):.0f}  avg={statistics.mean(ttfas):.0f}  ms")
    print("=" * 56)
    print("  Note: server-side only (no browser T6). STT mock is ~35ms.\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args.turns)))


if __name__ == "__main__":
    main()
