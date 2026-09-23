"""
P0 – Real provider adapters

Builds on R1 turn epochs. Each stage has a ceiling; exceeding it cancels the
turn and emits a structured timeout instead of hanging.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse, StreamingResponse
import os
import base64
import uuid

from providers.registry import create_stt, create_llm, create_tts
from mock_stt import MockStreamingSTT
from mock_llm import MockStreamingLLM
from mock_tts import MockStreamingTTS
from chunker import ResponseChunker
from tools import run_tool, list_tools_for_llm
from deadlines import DEFAULT_DEADLINES, TurnDeadlines
from context import cap_messages
from stt_protocol import TranscriptType
from llm_protocol import ChatMessage, LLMEventType
from tts_protocol import TTSEventType
from telephony_session import TelephonySession
from personas import PERSONAS, PersonaType, MixtureOfAgentsRouter, SpokenDialogueNormalizer
from threat_store import ThreatIntelligenceStore
from cybersecurity import AntiFraudDetector, ThreatLevel, HONEYPOT_PERSONAS, MultiFactorThreatScorer
from syndicate_graph import SyndicateAttributionGraph, DynamicCognitiveTrapEngine
from acoustic_prosodic_core import AcousticProsodicAnalyzer, AcousticSignalProcessor
from dual_brain import DualBrainReflexEngine, ReflexIntent

app = FastAPI(title="Voice Agent – Telephony, Cyber Honeypot & Realtime Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _preview_headers(request, call_next):
    response = await call_next(request)
    # Hint the browser we want mic; parent iframe policy still wins.
    response.headers["Permissions-Policy"] = "microphone=*, camera=()"
    response.headers["Feature-Policy"] = "microphone *"
    return response


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "ok", "stage": "P0", "service": "Voice Agent & Threat Intelligence Engine"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "stt_provider": os.environ.get("STT_PROVIDER", "mock"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "mock"),
        "tts_provider": os.environ.get("TTS_PROVIDER", "mock"),
    }


@app.get("/api/threats/stats")
async def get_threat_stats():
    store = ThreatIntelligenceStore.get_instance()
    return store.get_stats()


@app.get("/api/threats/feed")
async def get_threat_feed(limit: int = 50):
    store = ThreatIntelligenceStore.get_instance()
    return store.get_feed(limit=limit)


@app.get("/api/threats/syndicates")
async def get_syndicates():
    graph = SyndicateAttributionGraph.get_instance()
    clusters = []
    for syn_id, cluster in graph.syndicates.items():
        clusters.append({
            "syndicate_id": cluster.syndicate_id,
            "name": cluster.name,
            "primary_category": cluster.primary_category,
            "estimated_members": cluster.estimated_members,
            "confidence_score": cluster.confidence_score,
            "total_calls_attributed": cluster.total_calls_attributed,
            "total_time_wasted_seconds": cluster.total_time_wasted_seconds,
            "associated_phones": list(cluster.associated_phone_numbers),
            "associated_mules": list(cluster.associated_mule_accounts),
            "associated_crypto": list(cluster.associated_crypto_wallets),
            "associated_remote_ids": list(cluster.associated_remote_ids),
        })
    return {"syndicates": clusters}


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice_webhook(request: Request):
    """
    Returns TwiML instruction to connect an incoming Twilio voice call
    directly to the /ws/twilio streaming WebSocket.
    """
    host = request.headers.get("host") or "localhost:8080"
    scheme = "wss" if request.url.scheme == "https" or "ngrok" in host else "ws"
    ws_url = f"{scheme}://{host}/ws/twilio"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.api_route("/twilio/honeypot", methods=["GET", "POST"])
async def twilio_honeypot_webhook(request: Request):
    """
    TwiML webhook to route scam / robocalls directly to Margaret Honeypot.
    """
    host = request.headers.get("host") or "localhost:8080"
    scheme = "wss" if request.url.scheme == "https" or "ngrok" in host else "ws"
    ws_url = f"{scheme}://{host}/ws/honeypot"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="persona" value="margaret" />
        </Stream>
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.post("/chat/completions")
@app.post("/vapi/chat/completions")
@app.post("/vapi/chat/completions/chat/completions")
@app.post("/vapi")
async def custom_llm_chat_completions(request: Request):
    """
    OpenAI-compatible streaming Chat Completions endpoint for Vapi and custom voice bridges.
    Routes every phone turn through our natural phone conversation persona and LLM engine.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    messages_raw = body.get("messages", [])
    stream = body.get("stream", True)
    
    # Extract messages into ChatMessage format
    chat_msgs = []
    has_system = False
    for m in messages_raw:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            has_system = True
        chat_msgs.append(ChatMessage(role=role, content=content))

    if not has_system:
        # Prepend our high-fidelity human phone call persona
        system_content = (
            "You are having a casual phone conversation with a friend. "
            "Rules:\n"
            "1. Respond in EXACTLY ONE short, natural, human spoken sentence (under 10-12 words).\n"
            "2. Sound like a real person on a call (use casual language, natural fillers like 'Oh', 'Yeah', 'Gotcha', 'Haha').\n"
            "3. NEVER act like an AI, customer support bot, or say 'How can I assist/help you today?'.\n"
            "4. If insulted or teased, brush it off casually or banter with light humor like a real friend.\n"
            "5. No lists, emojis, quotes, or markdown. Just one crisp spoken line."
        )
        chat_msgs.insert(0, ChatMessage(role="system", content=system_content))

    llm = create_llm()
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    async def sse_stream():
        try:
            async for event in llm.stream(chat_msgs):
                if event.type == LLMEventType.TOKEN and event.text:
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": "voice-agent-custom-llm",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": event.text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif event.type == LLMEventType.ERROR:
                    err_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": "voice-agent-custom-llm",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": "Hey, sorry, couldn't hear that properly."},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
        except Exception as e:
            err_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "voice-agent-custom-llm",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Hey, what did you say?"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"

        # Final termination packet
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "voice-agent-custom-llm",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(sse_stream(), media_type="text/event-stream")
    else:
        # Non-streaming fallback
        full_text = []
        async for event in llm.stream(chat_msgs):
            if event.type == LLMEventType.TOKEN and event.text:
                full_text.append(event.text)
        content_str = "".join(full_text)
        return JSONResponse({
            "id": chunk_id,
            "object": "chat.completion",
            "created": created_time,
            "model": "voice-agent-custom-llm",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content_str},
                "finish_reason": "stop"
            }]
        })


SYSTEM_PROMPT = (
    "You are a vehicle voice assistant. "
    "Answer in 1 short spoken sentence when possible. "
    "Never use lists or markdown."
)

DEADLINES: TurnDeadlines = DEFAULT_DEADLINES

# Step D – optional API key (set VOICE_API_KEY to enforce)
# Clients send ?api_key=... on the WebSocket URL or header Authorization: Bearer ...
import uuid

def _expected_api_key() -> str:
    return os.environ.get("VOICE_API_KEY", "").strip()


def _extract_api_key(ws: WebSocket) -> str:
    q = ws.query_params.get("api_key") or ws.query_params.get("key") or ""
    if q:
        return q.strip()
    auth = ws.headers.get("authorization") or ws.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""



def _ms_since(t0: Optional[float]) -> Optional[float]:
    if t0 is None:
        return None
    return (time.perf_counter() - t0) * 1000


@app.websocket("/ws/stt")
async def voice_websocket(ws: WebSocket):
    expected = _expected_api_key()
    if expected:
        provided = _extract_api_key(ws)
        if provided != expected:
            await ws.close(code=4401, reason="unauthorized")
            return

    persona_param = ws.query_params.get("persona") or PersonaType.FRIEND
    persona_cfg = MixtureOfAgentsRouter.get_persona(persona_param)
    current_system_prompt = MixtureOfAgentsRouter.build_system_prompt(persona_param)

    await ws.accept()
    session_id = str(uuid.uuid4())

    stt = create_stt()
    llm = create_llm()
    tts = create_tts(persona_param)
    print(
        f"[session {session_id[:8]}] "
        f"Persona={persona_cfg.name} STT={type(stt).__name__} LLM={type(llm).__name__} TTS={type(tts).__name__}"
    )

    await stt.connect()
    await llm.connect()
    await tts.connect()

    turn_id = 0
    active_turn = 0
    cancelled = False

    t0: Optional[float] = None
    t1: Optional[float] = None
    t2: Optional[float] = None
    t3: Optional[float] = None
    t4: Optional[float] = None
    t5: Optional[float] = None
    tool_latency_ms: Optional[float] = None

    conversation: list[ChatMessage] = [
        ChatMessage(role="system", content=current_system_prompt)
    ]

    llm_task: Optional[asyncio.Task] = None
    tts_task: Optional[asyncio.Task] = None
    watchdog_task: Optional[asyncio.Task] = None

    async def send(obj: dict, for_turn: Optional[int] = None):
        if for_turn is not None and for_turn != active_turn:
            return
        if for_turn is not None:
            obj = {**obj, "turn_id": for_turn}
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    async def cancel_turn(reason: str, stage: str, my_turn: Optional[int] = None):
        nonlocal cancelled, active_turn, llm_task, tts_task, watchdog_task
        if active_turn == 0:
            return
        if my_turn is not None and my_turn != 0 and my_turn != active_turn:
            return
        cancelled = True
        dead = active_turn
        active_turn = 0
        if watchdog_task and not watchdog_task.done():
            watchdog_task.cancel()

        cur_llm = llm_task
        cur_tts = tts_task
        llm_task = None
        tts_task = None

        if cur_llm and not cur_llm.done():
            cur_llm.cancel()
        if cur_tts and not cur_tts.done():
            cur_tts.cancel()

        try:
            await asyncio.gather(llm.cancel(), tts.cancel(), return_exceptions=True)
            if hasattr(stt, "_reset_utterance"):
                stt._reset_utterance()
        except Exception:
            pass

        try:
            await ws.send_json({
                "type": "timeout" if reason == "timeout" else "cancelled",
                "stage": stage,
                "reason": reason,
                "turn_id": dead,
                "ms_since_speech_end": _ms_since(t0),
            })
        except Exception:
            pass

    async def turn_watchdog(my_turn: int, started: float):
        """Hard ceiling for the whole turn."""
        try:
            await asyncio.sleep(DEADLINES.turn_total_ms / 1000.0)
            if active_turn == my_turn:
                await cancel_turn("timeout", "turn_total", my_turn)
        except asyncio.CancelledError:
            return

    async def send_audio(pcm: bytes, sample_rate: int, is_first: bool, for_turn: int, fmt: str = "pcm16"):
        await send({
            "type": "tts_audio",
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": sample_rate,
            "format": fmt,
            "is_first": is_first,
            "ms_since_speech_end": _ms_since(t0),
        }, for_turn=for_turn)

    async def run_tts(text_queue: asyncio.Queue, my_turn: int):
        nonlocal t4, t5, tts_task

        async def text_iterator() -> AsyncIterator[str]:
            while True:
                item = await text_queue.get()
                if item is None:
                    break
                yield item

        if my_turn != active_turn:
            return

        t4 = time.perf_counter()
        await send({
            "type": "tts_start",
            "ms_since_speech_end": _ms_since(t0),
        }, for_turn=my_turn)

        first_audio_sent = False
        try:
            async def _consume():
                nonlocal t5, first_audio_sent
                async for event in tts.stream(text_iterator()):
                    if cancelled or my_turn != active_turn:
                        break
                    if event.type == TTSEventType.AUDIO:
                        is_first = not first_audio_sent
                        if is_first:
                            t5 = event.t_event
                            first_audio_sent = True
                            await send({
                                "type": "tts_first_audio",
                                "ms_since_speech_end": _ms_since(t0),
                                "tts_ttfa_ms": (t5 - t4) * 1000 if t4 else None,
                            }, for_turn=my_turn)
                        fmt = event.meta.get("format", getattr(tts, "audio_format", "pcm16"))
                        await send_audio(event.audio, event.sample_rate, is_first, my_turn, fmt=fmt)
                    elif event.type == TTSEventType.DONE:
                        if watchdog_task and not watchdog_task.done():
                            watchdog_task.cancel()
                        await send({
                            "type": "tts_done",
                            "timings": {
                                "t0_to_t1": (t1 - t0) * 1000 if t0 and t1 else None,
                                "t0_to_t3": (t3 - t0) * 1000 if t0 and t3 else None,
                                "t0_to_t5": (t5 - t0) * 1000 if t0 and t5 else None,
                                "tts_ttfa_ms": (t5 - t4) * 1000 if t4 and t5 else None,
                                "tool_latency_ms": tool_latency_ms,
                            },
                        }, for_turn=my_turn)
                    elif event.type == TTSEventType.ERROR:
                        await send({"type": "error", "message": event.text}, for_turn=my_turn)

            await asyncio.wait_for(_consume(), timeout=DEADLINES.tts_first_audio_ms / 1000.0 + 15.0)
            # Note: wait_for wraps whole TTS; first-audio soft check below
            if not first_audio_sent and my_turn == active_turn:
                elapsed = _ms_since(t0) or 0
                if elapsed > DEADLINES.tts_first_audio_ms:
                    await cancel_turn("timeout", "tts_first_audio", my_turn)
        except asyncio.TimeoutError:
            if my_turn == active_turn:
                await cancel_turn("timeout", "tts", my_turn)
        except asyncio.CancelledError:
            await send({"type": "tts_cancelled"}, for_turn=my_turn)
        finally:
            tts_task = None

    async def stream_speech(text_queue: asyncio.Queue, messages: list[ChatMessage], my_turn: int):
        nonlocal t2, t3

        chunker = ResponseChunker(first_min_chars=6, min_chars=14, max_chars=80)
        phrases_sent = 0
        full_response = ""

        t2 = time.perf_counter()
        await send({
            "type": "llm_start",
            "ms_since_speech_end": _ms_since(t0),
        }, for_turn=my_turn)

        try:
            async def _gen():
                nonlocal t3, phrases_sent, full_response
                async for event in llm.stream(messages, tools=list_tools_for_llm()):
                    if cancelled or my_turn != active_turn:
                        break
                    if event.type == LLMEventType.TOKEN:
                        if t3 is None:
                            t3 = event.t_event
                            await send({
                                "type": "llm_first_token",
                                "text": event.text,
                                "ttft_ms": (t3 - t2) * 1000,
                                "ms_since_speech_end": _ms_since(t0),
                            }, for_turn=my_turn)
                        full_response += event.text
                        await send({
                            "type": "llm_token",
                            "text": event.text,
                            "accumulated": full_response,
                        }, for_turn=my_turn)
                        for phrase in chunker.add(event.text):
                            phrases_sent += 1
                            await text_queue.put(phrase)
                            if phrases_sent == 1:
                                await send({
                                    "type": "first_phrase",
                                    "text": phrase,
                                    "ms_since_speech_end": _ms_since(t0),
                                }, for_turn=my_turn)
                    elif event.type == LLMEventType.DONE:
                        for phrase in chunker.flush():
                            phrases_sent += 1
                            await text_queue.put(phrase)
                        await text_queue.put(None)
                        if my_turn == active_turn and full_response:
                            conversation.append(ChatMessage(role="assistant", content=full_response))
                            conversation[:] = cap_messages(conversation)
                        await send({
                            "type": "llm_done",
                            "text": full_response,
                            "meta": {**(event.meta or {}), "phrases_sent": phrases_sent},
                        }, for_turn=my_turn)
                    elif event.type == LLMEventType.ERROR:
                        await text_queue.put(None)
                        await send({"type": "error", "message": event.text}, for_turn=my_turn)

            await asyncio.wait_for(
                _gen(),
                timeout=DEADLINES.llm_speech_first_token_ms / 1000.0 + 12.0,
            )
        except asyncio.TimeoutError:
            await text_queue.put(None)
            if my_turn == active_turn:
                await cancel_turn("timeout", "llm_speech", my_turn)
        except asyncio.CancelledError:
            await text_queue.put(None)
            await send({"type": "llm_cancelled"}, for_turn=my_turn)

    async def run_turn(user_text: str, my_turn: int):
        nonlocal t2, t3, t4, t5, tool_latency_ms, llm_task, tts_task, cancelled

        if my_turn != active_turn:
            return

        conversation.append(ChatMessage(role="user", content=user_text))
        conversation[:] = cap_messages(conversation)
        tool_latency_ms = None
        t2 = t3 = t4 = t5 = None
        cancelled = False

        # STT deadline check (T0 already set)
        if t0 is not None and _ms_since(t0) is not None and _ms_since(t0) > DEADLINES.stt_final_ms:
            await cancel_turn("timeout", "stt_final", my_turn)
            return

        t2 = time.perf_counter()
        await send({
            "type": "llm_start",
            "ms_since_speech_end": _ms_since(t0),
        }, for_turn=my_turn)

        # --- BRAIN-1: Sub-Conscious Instant Reflex Trigger ---
        reflex_filler = DualBrainReflexEngine.get_reflex_filler(user_text, persona=persona_param)

        # Pipeline: start TTS consumer immediately and stream tokens directly from LLM
        text_queue: asyncio.Queue = asyncio.Queue()
        tts_task = asyncio.create_task(run_tts(text_queue, my_turn))
        chunker = ResponseChunker(first_min_chars=6, min_chars=16, max_chars=80)
        phrases_sent = 0
        full_response = ""
        tool_call = None

        if reflex_filler:
            phrases_sent += 1
            await text_queue.put(reflex_filler)
            await send({
                "type": "reflex_filler",
                "text": reflex_filler,
                "ms_since_speech_end": _ms_since(t0),
            }, for_turn=my_turn)

        try:
            async def _stream_llm():
                nonlocal t3, phrases_sent, full_response, tool_call, tts_task
                async for event in llm.stream(conversation, tools=list_tools_for_llm()):
                    if cancelled or my_turn != active_turn:
                        break
                    if event.type == LLMEventType.TOKEN:
                        if t3 is None:
                            t3 = event.t_event
                            await send({
                                "type": "llm_first_token",
                                "text": event.text,
                                "ttft_ms": (t3 - t2) * 1000,
                                "ms_since_speech_end": _ms_since(t0),
                            }, for_turn=my_turn)
                        full_response += event.text
                        await send({
                            "type": "llm_token",
                            "text": event.text,
                            "accumulated": full_response,
                        }, for_turn=my_turn)
                        for phrase in chunker.add(event.text):
                            clean_phrase = SpokenDialogueNormalizer.normalize_for_speech(phrase)
                            if clean_phrase:
                                phrases_sent += 1
                                await text_queue.put(clean_phrase)
                                if phrases_sent == 1:
                                    await send({
                                        "type": "first_phrase",
                                        "text": clean_phrase,
                                        "ms_since_speech_end": _ms_since(t0),
                                    }, for_turn=my_turn)
                    elif event.type == LLMEventType.TOOL_CALL:
                        tool_call = event
                        break
                    elif event.type == LLMEventType.DONE:
                        for phrase in chunker.flush():
                            clean_phrase = SpokenDialogueNormalizer.normalize_for_speech(phrase)
                            if clean_phrase:
                                phrases_sent += 1
                                await text_queue.put(clean_phrase)
                        await text_queue.put(None)
                        if my_turn == active_turn and full_response:
                            conversation.append(ChatMessage(role="assistant", content=full_response))
                            conversation[:] = cap_messages(conversation)
                        await send({
                            "type": "llm_done",
                            "text": full_response,
                            "meta": {**(event.meta or {}), "phrases_sent": phrases_sent},
                        }, for_turn=my_turn)
                    elif event.type == LLMEventType.ERROR:
                        await text_queue.put(None)
                        await send({"type": "error", "message": event.text}, for_turn=my_turn)

            await asyncio.wait_for(
                _stream_llm(),
                timeout=DEADLINES.llm_speech_first_token_ms / 1000.0 + 12.0,
            )
        except asyncio.TimeoutError:
            await text_queue.put(None)
            if my_turn == active_turn:
                await cancel_turn("timeout", "llm", my_turn)
            return
        except asyncio.CancelledError:
            await text_queue.put(None)
            await send({"type": "llm_cancelled"}, for_turn=my_turn)
            return

        if cancelled or my_turn != active_turn:
            return

        # If a tool call was requested:
        if tool_call and tool_call.tool_name:
            await send({
                "type": "tool_call",
                "name": tool_call.tool_name,
                "args": tool_call.tool_args or {},
                "id": tool_call.tool_call_id,
            }, for_turn=my_turn)
            await send({"type": "tool_start", "name": tool_call.tool_name}, for_turn=my_turn)

            try:
                result = await asyncio.wait_for(
                    run_tool(
                        tool_call.tool_name,
                        tool_call.tool_args or {},
                        tool_call_id=tool_call.tool_call_id,
                    ),
                    timeout=DEADLINES.tool_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                if my_turn == active_turn:
                    await cancel_turn("timeout", "tool", my_turn)
                return

            if cancelled or my_turn != active_turn:
                return

            tool_latency_ms = result.latency_ms
            await send({
                "type": "tool_result",
                "name": result.name,
                "ok": result.ok,
                "data": result.data,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }, for_turn=my_turn)

            conversation.append(ChatMessage(
                role="tool",
                content=json.dumps(result.data if result.ok else {"error": result.error}),
                name=result.name,
                tool_call_id=tool_call.tool_call_id,
            ))

            if cancelled or my_turn != active_turn:
                return

            # Soft product budget check before tool summary synthesis
            if t0 is not None and (_ms_since(t0) or 0) > DEADLINES.turn_first_audio_ms:
                await cancel_turn("timeout", "turn_first_audio", my_turn)
                return

            # Now stream spoken summary after tool execution into active text_queue
            await stream_speech(text_queue, conversation, my_turn)

    async def stt_event_loop():
        nonlocal t1, llm_task, cancelled, active_turn, watchdog_task

        async for event in stt.receive_events():
            for_turn = active_turn if active_turn else turn_id
            payload = {
                "type": event.type.value,
                "text": event.text,
                "is_final": event.is_final,
                "confidence": event.confidence,
            }
            if t0 is not None:
                payload["ms_since_speech_end"] = _ms_since(t0)

            if event.type == TranscriptType.FINAL:
                t1 = event.t_event
                payload["stt_finalization_ms"] = (t1 - t0) * 1000 if t0 else None
                my_turn = active_turn
                await send(payload, for_turn=my_turn)

                # STT took too long?
                if t0 is not None and (t1 - t0) * 1000 > DEADLINES.stt_final_ms:
                    await cancel_turn("timeout", "stt_final", my_turn)
                    continue

                cancelled = False
                if llm_task and not llm_task.done():
                    llm_task.cancel()
                    await llm.cancel()
                if tts_task and not tts_task.done():
                    tts_task.cancel()
                    await tts.cancel()

                if my_turn and my_turn == active_turn:
                    llm_task = asyncio.create_task(run_turn(event.text, my_turn))
                continue

            await send(payload, for_turn=for_turn if for_turn else None)

    stt_task = asyncio.create_task(stt_event_loop())

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                await stt.send_audio(message["bytes"], sample_rate=16000)

            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "finalize":
                    turn_id += 1
                    active_turn = turn_id
                    cancelled = False
                    t0 = time.perf_counter()
                    t1 = t2 = t3 = t4 = t5 = None
                    tool_latency_ms = None
                    if watchdog_task and not watchdog_task.done():
                        watchdog_task.cancel()
                    watchdog_task = asyncio.create_task(turn_watchdog(active_turn, t0))
                    await send({"type": "turn_start", "turn_id": active_turn}, for_turn=active_turn)
                    await stt.finalize()

                elif msg_type == "user_text":
                    # Typed utterance — skip STT (preview iframes often deny the mic)
                    user_text = (data.get("text") or "").strip()
                    if not user_text:
                        continue
                    turn_id += 1
                    active_turn = turn_id
                    cancelled = False
                    t0 = time.perf_counter()
                    t1 = t0
                    t2 = t3 = t4 = t5 = None
                    tool_latency_ms = None
                    if watchdog_task and not watchdog_task.done():
                        watchdog_task.cancel()
                    if llm_task and not llm_task.done():
                        llm_task.cancel()
                        await llm.cancel()
                    if tts_task and not tts_task.done():
                        tts_task.cancel()
                        await tts.cancel()
                    watchdog_task = asyncio.create_task(turn_watchdog(active_turn, t0))
                    await send({"type": "turn_start", "turn_id": active_turn}, for_turn=active_turn)
                    await send({
                        "type": "final",
                        "text": user_text,
                        "is_final": True,
                        "stt_finalization_ms": 0,
                        "ms_since_speech_end": 0,
                    }, for_turn=active_turn)
                    llm_task = asyncio.create_task(run_turn(user_text, active_turn))

                elif msg_type == "cancel":
                    # Kill active turn or specific turn requested by client
                    target = data.get("turn_id")
                    if watchdog_task and not watchdog_task.done():
                        watchdog_task.cancel()
                    await cancel_turn("user_cancel", "cancel", target)

                elif msg_type == "reset":
                    cancelled = True
                    active_turn = 0
                    turn_id = 0
                    if watchdog_task and not watchdog_task.done():
                        watchdog_task.cancel()
                    if llm_task and not llm_task.done():
                        llm_task.cancel()
                    if tts_task and not tts_task.done():
                        tts_task.cancel()
                    await stt.close()
                    await llm.close()
                    await tts.close()
                    stt = create_stt()
                    llm = create_llm()
                    tts = create_tts()
                    await stt.connect()
                    await llm.connect()
                    await tts.connect()
                    t0 = t1 = t2 = t3 = t4 = t5 = None
                    tool_latency_ms = None
                    conversation = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
                    stt_task.cancel()
                    stt_task = asyncio.create_task(stt_event_loop())
                    await send({"type": "reset_ok"})

                elif msg_type == "ping":
                    await send({"type": "pong", "t": time.perf_counter()})

                elif msg_type == "client_hello":
                    await send({
                        "type": "server_hello",
                        "session": "ok",
                        "session_id": session_id,
                        "deadlines": {
                            "stt_final_ms": DEADLINES.stt_final_ms,
                            "llm_first_token_ms": DEADLINES.llm_first_token_ms,
                            "tool_ms": DEADLINES.tool_ms,
                            "tts_first_audio_ms": DEADLINES.tts_first_audio_ms,
                            "turn_total_ms": DEADLINES.turn_total_ms,
                        },
                        "t": time.perf_counter(),
                    })

    except WebSocketDisconnect:
        pass
    finally:
        cancelled = True
        active_turn = 0
        for task in (llm_task, tts_task, stt_task, watchdog_task):
            if task and not task.done():
                task.cancel()
        await stt.close()
        await llm.close()
        await tts.close()


@app.websocket("/ws/twilio")
async def twilio_websocket(ws: WebSocket):
    """
    Twilio Media Streams bi-directional audio bridge.
    G.711 u-law 8kHz audio with instant carrier-side buffer clearing on barge-in.
    """
    expected = _expected_api_key()
    if expected:
        provided = _extract_api_key(ws)
        if provided != expected:
            await ws.close(code=4401, reason="unauthorized")
            return

    await ws.accept()
    session: Optional[TelephonySession] = None

    async def send_to_twilio(msg: dict | bytes):
        if isinstance(msg, dict):
            await ws.send_json(msg)
        else:
            b64_payload = base64.b64encode(msg).decode("ascii")
            if session and session.stream_sid:
                await ws.send_json({
                    "event": "media",
                    "streamSid": session.stream_sid,
                    "media": {"payload": b64_payload},
                })

    try:
        while True:
            msg_raw = await ws.receive()
            if msg_raw["type"] == "websocket.disconnect":
                break

            if "text" in msg_raw and msg_raw["text"]:
                try:
                    data = json.loads(msg_raw["text"])
                except Exception:
                    continue

                event_type = data.get("event")

                if event_type == "start":
                    start_data = data.get("start") or {}
                    custom_params = start_data.get("customParameters") or {}
                    persona_id = custom_params.get("persona") or PersonaType.FRIEND
                    stream_sid = start_data.get("streamSid") or data.get("streamSid") or str(uuid.uuid4())
                    
                    persona_obj = MixtureOfAgentsRouter.get_persona(persona_id)
                    session = TelephonySession(
                        send_fn=send_to_twilio,
                        encoding="ulaw_8k",
                        protocol="twilio",
                        stream_sid=stream_sid,
                        deadlines=DEADLINES,
                        greeting_text=persona_obj.greeting,
                    )
                    # Override system prompt with persona
                    session.conversation = [ChatMessage(
                        role="system",
                        content=MixtureOfAgentsRouter.build_system_prompt(persona_id)
                    )]
                    await session.start()

                elif event_type == "media":
                    if session:
                        media_data = data.get("media") or {}
                        payload = media_data.get("payload") or ""
                        if payload:
                            try:
                                ulaw_bytes = base64.b64decode(payload)
                                await session.process_inbound_audio(ulaw_bytes)
                            except Exception:
                                pass

                elif event_type == "stop":
                    if session:
                        await session.close()
                        session = None
                    break

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            await session.close()


@app.websocket("/ws/telephony")
async def generic_telephony_websocket(ws: WebSocket):
    """
    Generic full-duplex Telephony / SIP / WebRTC streaming bridge.
    Accepts raw binary audio frames (8kHz u-law or 16kHz PCM16) or JSON.
    """
    expected = _expected_api_key()
    if expected:
        provided = _extract_api_key(ws)
        if provided != expected:
            await ws.close(code=4401, reason="unauthorized")
            return

    encoding = ws.query_params.get("encoding", "ulaw_8k")
    persona_id = ws.query_params.get("persona", PersonaType.FRIEND)
    persona_obj = MixtureOfAgentsRouter.get_persona(persona_id)
    
    await ws.accept()

    async def send_to_client(data: dict | bytes):
        if isinstance(data, dict):
            await ws.send_json(data)
        else:
            await ws.send_bytes(data)

    session = TelephonySession(
        send_fn=send_to_client,
        encoding=encoding,
        protocol="raw",
        deadlines=DEADLINES,
        greeting_text=persona_obj.greeting,
    )
    session.conversation = [ChatMessage(
        role="system",
        content=MixtureOfAgentsRouter.build_system_prompt(persona_id)
    )]
    await session.start()

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                await session.process_inbound_audio(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "cancel":
                        await session.cancel_turn("client_cancel", "cancel")
                    elif data.get("type") == "audio_b64" and data.get("data"):
                        raw = base64.b64decode(data["data"])
                        await session.process_inbound_audio(raw)
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


@app.websocket("/ws/honeypot")
async def honeypot_telephony_websocket(ws: WebSocket):
    """
    Dedicated Cybersecurity Honeypot & Threat Intelligence Streaming Endpoint.
    Connects scam calls to AI baiter personas (Margaret / Arthur), stalls scammers,
    and extracts forensic threat intelligence (Mule accounts, crypto wallets, AnyDesk IDs).
    """
    expected = _expected_api_key()
    if expected:
        provided = _extract_api_key(ws)
        if provided != expected:
            await ws.close(code=4401, reason="unauthorized")
            return

    await ws.accept()
    session: Optional[TelephonySession] = None
    store = ThreatIntelligenceStore.get_instance()

    async def send_to_carrier(msg: dict | bytes):
        if isinstance(msg, dict):
            await ws.send_json(msg)
        else:
            b64_payload = base64.b64encode(msg).decode("ascii")
            if session and session.stream_sid:
                await ws.send_json({
                    "event": "media",
                    "streamSid": session.stream_sid,
                    "media": {"payload": b64_payload},
                })

    try:
        while True:
            msg_raw = await ws.receive()
            if msg_raw["type"] == "websocket.disconnect":
                break

            if "text" in msg_raw and msg_raw["text"]:
                try:
                    data = json.loads(msg_raw["text"])
                except Exception:
                    continue

                event_type = data.get("event")

                if event_type == "start":
                    start_data = data.get("start") or {}
                    custom_params = start_data.get("customParameters") or {}
                    persona_id = custom_params.get("persona") or "margaret"
                    stream_sid = start_data.get("streamSid") or data.get("streamSid") or str(uuid.uuid4())
                    
                    hp_config = HONEYPOT_PERSONAS.get(persona_id, HONEYPOT_PERSONAS["margaret"])
                    session = TelephonySession(
                        send_fn=send_to_carrier,
                        encoding="ulaw_8k",
                        protocol="twilio",
                        stream_sid=stream_sid,
                        deadlines=DEADLINES,
                        greeting_text="Hello? Who is this calling?",
                        persona_id=persona_id,
                    )
                    session.conversation = [ChatMessage(
                        role="system",
                        content=hp_config["system_prompt"]
                    )]
                    store.start_call(stream_sid=stream_sid, caller_id=start_data.get("from") or "Scam Syndicate Caller", persona=persona_id)
                    await session.start()

                elif event_type == "media":
                    if session:
                        media_data = data.get("media") or {}
                        payload = media_data.get("payload") or ""
                        if payload:
                            try:
                                ulaw_bytes = base64.b64decode(payload)
                                await session.process_inbound_audio(ulaw_bytes)
                            except Exception:
                                pass

                elif event_type == "stop":
                    if session:
                        store.end_call(session.stream_sid)
                        await session.close()
                        session = None
                    break

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            store.end_call(session.stream_sid)
            await session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False, log_level="info")
