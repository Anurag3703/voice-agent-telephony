"""
Step C – Barge-in / cancel latency eval

Usage:
  PYTHONPATH=backend:. python -m evals.bargein_eval --turns 5
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

from providers.registry import create_llm, create_tts
from llm_protocol import ChatMessage, LLMEventType
from tts_protocol import TTSEventType


async def one_barge_turn(turn: int) -> dict:
    llm = create_llm()
    tts = create_tts()
    await llm.connect()
    await tts.connect()

    messages = [
        ChatMessage(role="system", content="Speak a long helpful answer in several sentences."),
        ChatMessage(role="user", content="Tell me a long story about a road trip."),
    ]

    text_q: asyncio.Queue = asyncio.Queue()
    cancel_at_tokens = 8
    tokens_seen = 0
    t_cancel = None
    t_llm_cancel_done = None
    t_tts_cancel_done = None
    cancelled = False

    async def gen_speech():
        nonlocal tokens_seen, t_cancel, t_llm_cancel_done, cancelled
        try:
            async for ev in llm.stream(messages):
                if cancelled:
                    break
                if ev.type == LLMEventType.TOKEN:
                    tokens_seen += 1
                    await text_q.put(ev.text)
                    if tokens_seen >= cancel_at_tokens and t_cancel is None:
                        t_cancel = time.perf_counter()
                        cancelled = True
                        await llm.cancel()
                        t_llm_cancel_done = time.perf_counter()
                        await text_q.put(None)
                        break
                elif ev.type in (LLMEventType.DONE, LLMEventType.ERROR, LLMEventType.TOOL_CALL):
                    await text_q.put(None)
                    break
        finally:
            if t_cancel is None:
                await text_q.put(None)

    async def gen_tts():
        nonlocal t_tts_cancel_done

        async def phrases():
            while True:
                item = await text_q.get()
                if item is None:
                    break
                yield item

        try:
            async for ev in tts.stream(phrases()):
                if cancelled and t_cancel is not None:
                    await tts.cancel()
                    t_tts_cancel_done = time.perf_counter()
                    break
                if ev.type in (TTSEventType.DONE, TTSEventType.ERROR):
                    break
        except Exception:
            t_tts_cancel_done = time.perf_counter()

    await asyncio.gather(gen_speech(), gen_tts())
    await llm.close()
    await tts.close()

    llm_ms = ((t_llm_cancel_done - t_cancel) * 1000) if t_cancel and t_llm_cancel_done else None
    tts_ms = ((t_tts_cancel_done - t_cancel) * 1000) if t_cancel and t_tts_cancel_done else None

    return {
        "turn": turn,
        "tokens_before_cancel": tokens_seen,
        "llm_cancel_ms": llm_ms,
        "tts_cancel_ms": tts_ms,
        "ok": t_cancel is not None and llm_ms is not None and llm_ms < 500,
    }


async def run(n: int) -> int:
    print(f"\nBarge-in eval – {n} turns")
    print(f"  LLM={os.environ.get('LLM_PROVIDER','mock')}  TTS={os.environ.get('TTS_PROVIDER','mock')}\n")
    rows = []
    for i in range(n):
        try:
            r = await one_barge_turn(i + 1)
            rows.append(r)
            llm_s = f"{r['llm_cancel_ms']:.1f}ms" if r["llm_cancel_ms"] is not None else "n/a"
            tts_s = f"{r['tts_cancel_ms']:.1f}ms" if r["tts_cancel_ms"] is not None else "n/a"
            mark = "OK" if r["ok"] else "FAIL"
            print(
                f"  #{r['turn']}  tokens={r['tokens_before_cancel']}  "
                f"llm_cancel={llm_s}  tts_cancel={tts_s}  {mark}"
            )
        except Exception as e:
            print(f"  #{i+1} FAIL: {e}")
            return 1

    llm_vals = [r["llm_cancel_ms"] for r in rows if r["llm_cancel_ms"] is not None]
    tts_vals = [r["tts_cancel_ms"] for r in rows if r["tts_cancel_ms"] is not None]
    print("\n" + "=" * 50)
    if llm_vals:
        print(f"  LLM cancel P50={statistics.median(llm_vals):.1f} ms")
    if tts_vals:
        print(f"  TTS cancel P50={statistics.median(tts_vals):.1f} ms")
    print("  Target: cancel path << 100 ms server-side")
    print("=" * 50 + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=5)
    args = ap.parse_args()
    if not os.environ.get("LLM_PROVIDER"):
        os.environ["LLM_PROVIDER"] = "mock"
    if not os.environ.get("TTS_PROVIDER"):
        os.environ["TTS_PROVIDER"] = "mock"
    raise SystemExit(asyncio.run(run(args.turns)))


if __name__ == "__main__":
    main()
