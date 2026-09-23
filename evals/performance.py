"""
Stage 9 – Automated performance evaluation

Runs N scripted turns through the mock STT → LLM → (tools) → chunker → TTS
pipeline and reports component + end-to-end latencies.

Usage:
  cd /home/workdir/artifacts/voice-agent
  PYTHONPATH=backend:. python -m evals.performance
  PYTHONPATH=backend:. python -m evals.performance --turns 100
  PYTHONPATH=backend:. python -m evals.performance --turns 50 --baseline evals/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# Ensure backend package is importable
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mock_stt import MockStreamingSTT
from mock_llm import MockStreamingLLM
from mock_tts import MockStreamingTTS
from chunker import ResponseChunker
from tools import run_tool, list_tools_for_llm
from llm_protocol import ChatMessage, LLMEventType
from tts_protocol import TTSEventType
from stt_protocol import TranscriptType


# Scripted user utterances covering tool + non-tool paths
UTTERANCES = [
    "Can you tell me the status of my vehicle?",
    "What's my battery level?",
    "Where is my car right now?",
    "What's the cabin temperature?",
    "Honk the horn please",
    "Hello",
    "How much range do I have left?",
    "Is my vehicle locked?",
]


@dataclass
class TurnResult:
    turn: int
    utterance: str
    stt_final_ms: float
    llm_ttft_ms: float          # first token (or tool decision)
    tool_ms: Optional[float]
    tts_ttfa_ms: float          # TTS first audio after TTS start
    t0_to_t5_ms: float          # speech-end → first TTS audio (server-side proxy for TTFA)
    total_pipeline_ms: float
    used_tool: bool
    ok_under_300: bool


@dataclass
class SuiteResult:
    n_turns: int
    turns: list[TurnResult] = field(default_factory=list)

    def percentile(self, values: list[float], p: float) -> float:
        if not values:
            return float("nan")
        s = sorted(values)
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] * (c - k) + s[c] * (k - f)

    def summary(self) -> dict:
        ttfas = [t.t0_to_t5_ms for t in self.turns]
        stts = [t.stt_final_ms for t in self.turns]
        llms = [t.llm_ttft_ms for t in self.turns]
        ttss = [t.tts_ttfa_ms for t in self.turns]
        tools = [t.tool_ms for t in self.turns if t.tool_ms is not None]
        info = [t.t0_to_t5_ms for t in self.turns if not t.used_tool]
        tool_path = [t.t0_to_t5_ms for t in self.turns if t.used_tool]

        return {
            "n_turns": self.n_turns,
            "stt_p50": self.percentile(stts, 50),
            "llm_ttft_p50": self.percentile(llms, 50),
            "tts_ttfa_p50": self.percentile(ttss, 50),
            "tool_p50": self.percentile(tools, 50) if tools else None,
            "ttfa_p50": self.percentile(ttfas, 50),
            "ttfa_p90": self.percentile(ttfas, 90),
            "ttfa_p95": self.percentile(ttfas, 95),
            "ttfa_p99": self.percentile(ttfas, 99),
            "ttfa_avg": statistics.mean(ttfas) if ttfas else float("nan"),
            "info_ttfa_p50": self.percentile(info, 50) if info else None,
            "tool_path_ttfa_p50": self.percentile(tool_path, 50) if tool_path else None,
            "under_300_rate": sum(1 for t in self.turns if t.ok_under_300) / max(1, len(self.turns)),
            "tool_turns": sum(1 for t in self.turns if t.used_tool),
            "info_turns": sum(1 for t in self.turns if not t.used_tool),
        }


async def run_one_turn(turn_id: int, utterance: str) -> TurnResult:
    """
    Drive one full turn through mocks, measuring component times.
    Mirrors server.py: optional tool call, then speech tokens → chunker → TTS.
    """
    stt = MockStreamingSTT(partial_interval_ms=100, final_delay_ms=35)
    llm = MockStreamingLLM(ttft_ms=55, inter_token_ms=8)
    tts = MockStreamingTTS(first_audio_ms=75, chunk_duration_ms=35, realtime_pace=False)

    await stt.connect()
    await llm.connect()
    await tts.connect()

    conversation = [
        ChatMessage(role="system", content="You are a helpful vehicle assistant. Be concise."),
    ]

    t_speech_end = time.perf_counter()

    fake_pcm = bytes(3200)
    await stt.send_audio(fake_pcm, sample_rate=16000)
    await stt.finalize()

    t_stt_final = time.perf_counter()
    async for ev in stt.receive_events():
        if ev.type == TranscriptType.FINAL:
            t_stt_final = ev.t_event
            break

    stt_final_ms = (t_stt_final - t_speech_end) * 1000
    conversation.append(ChatMessage(role="user", content=utterance))

    tool_ms = None
    used_tool = False
    first_token_t = None
    t_llm_start = time.perf_counter()

    chunker = ResponseChunker(first_min_chars=6, min_chars=16, max_chars=80)
    text_q: asyncio.Queue = asyncio.Queue()

    async def feed_tts():
        async def text_iter():
            while True:
                item = await text_q.get()
                if item is None:
                    break
                yield item

        t_tts_start = time.perf_counter()
        first_audio_t = None
        async for tev in tts.stream(text_iter()):
            if tev.type == TTSEventType.AUDIO and first_audio_t is None:
                first_audio_t = tev.t_event
            if tev.type == TTSEventType.DONE:
                break
        return t_tts_start, first_audio_t

    tts_task = asyncio.create_task(feed_tts())

    async def push_text(txt: str):
        for phrase in chunker.add(txt):
            await text_q.put(phrase)

    tool_call = None
    async for event in llm.stream(conversation, tools=list_tools_for_llm()):
        if event.type == LLMEventType.TOOL_CALL:
            tool_call = event
            if first_token_t is None:
                first_token_t = event.t_event
            break
        elif event.type == LLMEventType.TOKEN:
            if first_token_t is None:
                first_token_t = event.t_event
            await push_text(event.text)
        elif event.type == LLMEventType.DONE:
            for phrase in chunker.flush():
                await text_q.put(phrase)
            await text_q.put(None)
            break

    if tool_call and tool_call.tool_name:
        used_tool = True
        await text_q.put(None)
        if not tts_task.done():
            tts_task.cancel()

        result = await run_tool(tool_call.tool_name, tool_call.tool_args or {})
        tool_ms = result.latency_ms
        conversation.append(ChatMessage(
            role="tool",
            content=json.dumps(result.data if result.ok else {"error": result.error}),
            name=result.name,
            tool_call_id=tool_call.tool_call_id,
        ))

        # Restart TTS and chunker for tool summary
        chunker.reset()
        text_q = asyncio.Queue()
        tts_task = asyncio.create_task(feed_tts())
        first_token_t = None
        t_llm_start = time.perf_counter()

        async for event in llm.stream(conversation, tools=list_tools_for_llm()):
            if event.type == LLMEventType.TOKEN:
                if first_token_t is None:
                    first_token_t = event.t_event
                await push_text(event.text)
            elif event.type == LLMEventType.DONE:
                for phrase in chunker.flush():
                    await text_q.put(phrase)
                await text_q.put(None)
                break
            elif event.type == LLMEventType.TOOL_CALL:
                break

    t_tts_start, first_audio_t = await tts_task
    t_end = time.perf_counter()

    llm_ttft_ms = ((first_token_t or t_llm_start) - t_llm_start) * 1000
    tts_ttfa_ms = ((first_audio_t or t_tts_start) - t_tts_start) * 1000 if first_audio_t else float("nan")
    t0_to_t5_ms = ((first_audio_t or t_end) - t_speech_end) * 1000
    total_ms = (t_end - t_speech_end) * 1000

    await stt.close()
    await llm.close()
    await tts.close()

    return TurnResult(
        turn=turn_id,
        utterance=utterance,
        stt_final_ms=stt_final_ms,
        llm_ttft_ms=llm_ttft_ms,
        tool_ms=tool_ms,
        tts_ttfa_ms=tts_ttfa_ms,
        t0_to_t5_ms=t0_to_t5_ms,
        total_pipeline_ms=total_ms,
        used_tool=used_tool,
        ok_under_300=t0_to_t5_ms < 300,
    )


async def run_suite(n_turns: int) -> SuiteResult:
    suite = SuiteResult(n_turns=n_turns)
    for i in range(n_turns):
        utt = UTTERANCES[i % len(UTTERANCES)]
        result = await run_one_turn(i + 1, utt)
        suite.turns.append(result)
        status = "✓" if result.ok_under_300 else "✗"
        tool = f" tool={result.tool_ms:.0f}ms" if result.tool_ms is not None else ""
        print(
            f"  #{result.turn:3d}  TTFA={result.t0_to_t5_ms:6.1f}ms  "
            f"STT={result.stt_final_ms:5.1f}  LLM={result.llm_ttft_ms:5.1f}  "
            f"TTS={result.tts_ttfa_ms:5.1f}{tool}  {status}  {utt[:40]}"
        )
    return suite


def print_report(suite: SuiteResult, baseline: Optional[dict] = None) -> None:
    s = suite.summary()
    print()
    print("=" * 56)
    print(f"  Performance suite – {s['n_turns']} turns")
    print("=" * 56)
    print(f"  STT finalization P50     {s['stt_p50']:7.1f} ms")
    print(f"  LLM TTFT P50             {s['llm_ttft_p50']:7.1f} ms")
    print(f"  TTS first-audio P50      {s['tts_ttfa_p50']:7.1f} ms")
    if s["tool_p50"] is not None:
        print(f"  Tool latency P50         {s['tool_p50']:7.1f} ms")
    print("-" * 56)
    print(f"  TTFA (T0→first audio) P50  {s['ttfa_p50']:6.1f} ms", end="")
    print("  ✓" if s["ttfa_p50"] < 300 else "  ✗")
    print(f"  TTFA P90                   {s['ttfa_p90']:6.1f} ms")
    print(f"  TTFA P95                   {s['ttfa_p95']:6.1f} ms", end="")
    print("  ✓" if s["ttfa_p95"] < 600 else "  ✗")
    print(f"  TTFA P99                   {s['ttfa_p99']:6.1f} ms")
    print(f"  TTFA avg                   {s['ttfa_avg']:6.1f} ms")
    print(f"  Under 300 ms               {s['under_300_rate']*100:5.1f}%  ({s['tool_turns']} tool turns)")
    print("=" * 56)

    if s.get("info_ttfa_p50") is not None:
        print(f"  Informational TTFA P50   {s['info_ttfa_p50']:6.1f} ms  (no tool)")
    if s.get("tool_path_ttfa_p50") is not None:
        print(f"  Tool-path TTFA P50       {s['tool_path_ttfa_p50']:6.1f} ms  (includes tool)")
    print()
    print("                    Target")
    # Primary conversational target is informational path; tool path reported separately
    info_ok = s.get("info_ttfa_p50") is not None and s["info_ttfa_p50"] < 300
    overall_p95_ok = s["ttfa_p95"] < 600
    print(f"  P50 TTFA (info)    <300ms  {'✓' if info_ok else '✗'}")
    print(f"  P95 TTFA (all)     <600ms  {'✓' if overall_p95_ok else '✗'}")
    print(f"  P99 TTFA           monitored")
    print(f"  Tool latency       measured separately (P50 {s['tool_p50']:.0f} ms)" if s.get("tool_p50") else "")

    if baseline:
        prev = baseline.get("ttfa_p50")
        if prev is not None:
            delta = s["ttfa_p50"] - prev
            sign = "+" if delta >= 0 else ""
            print()
            print(f"  Regression vs baseline P50: {sign}{delta:.1f} ms "
                  f"(was {prev:.1f} ms)")
            if delta > 30:
                print("  ⚠  Significant regression detected (>30 ms)")
            elif delta < -10:
                print("  ↓  Improvement vs baseline")


def save_baseline(suite: SuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = suite.summary()
    data["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(data, indent=2))
    print(f"\n  Baseline written to {path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Voice agent performance evaluation")
    parser.add_argument("--turns", type=int, default=30, help="Number of turns (default 30)")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline JSON for regression compare")
    parser.add_argument("--save-baseline", type=str, default=None, help="Write current results as baseline JSON")
    args = parser.parse_args(argv)

    print(f"\nRunning {args.turns} realtime turns (mock pipeline)…\n")
    t0 = time.perf_counter()
    suite = asyncio.run(run_suite(args.turns))
    elapsed = time.perf_counter() - t0
    print(f"\n  Wall time: {elapsed:.1f}s ({elapsed/max(1,args.turns)*1000:.0f} ms/turn)")

    baseline = None
    if args.baseline:
        bpath = Path(args.baseline)
        if bpath.exists():
            baseline = json.loads(bpath.read_text())
        else:
            print(f"  Warning: baseline file not found: {bpath}")

    print_report(suite, baseline)

    if args.save_baseline:
        save_baseline(suite, Path(args.save_baseline))

    s = suite.summary()
    # Exit non-zero if hard targets missed.
    # Informational P50 is the primary conversational budget; tool path is separate.
    info_p50 = s.get("info_ttfa_p50")
    info_fail = info_p50 is not None and info_p50 >= 300
    if info_fail or s["ttfa_p95"] >= 600:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
