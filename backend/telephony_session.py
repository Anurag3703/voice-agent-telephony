"""
Telephony Session Orchestrator for Real Phone Calls (Twilio, SIP, Telnyx, WebRTC).

Features:
- Full-duplex continuous audio streaming (G.711 u-law 8kHz / PCM16 16kHz)
- Server-side VAD with pre-speech audio ring-buffer (zero syllable loss)
- Sub-15ms carrier buffer clearing (Twilio 'clear' event / telephony purge)
- Overlapped LLM -> Chunker -> TTS streaming with 20ms frame pacing
- Resilient turn epoch state machine and tool handling
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
import uuid
from typing import Optional, Callable, Awaitable, List

from telephony import (
    TelephonyVAD,
    TelephonyVADConfig,
    ulaw_to_pcm16,
    pcm16_to_ulaw,
    resample_pcm16,
)
from providers.registry import create_stt, create_llm, create_tts
from chunker import ResponseChunker
from tools import run_tool, list_tools_for_llm
from deadlines import DEFAULT_DEADLINES, TurnDeadlines
from context import cap_messages
from stt_protocol import TranscriptType, StreamingSTT
from llm_protocol import ChatMessage, LLMEventType, StreamingLLM
from tts_protocol import TTSEventType, StreamingTTS
from personas import SpokenDialogueNormalizer
from dual_brain import DualBrainReflexEngine, ReflexIntent
from enterprise import EnterprisePIISanitizer, EnterpriseGuardrails, WarmTransferEngine, HandoverDossier
from cybersecurity import AntiFraudDetector, ThreatLevel, HONEYPOT_PERSONAS
from threat_store import ThreatIntelligenceStore
from resilience import AdaptiveJitterBuffer, AcousticEchoSuppressor, ProviderCircuitBreaker


SYSTEM_PROMPT = (
    "You are a helpful vehicle voice assistant on a live phone call. "
    "Answer in 1 short, natural spoken sentence when possible. "
    "Never use markdown, bullet points, or lists."
)


class TelephonySession:
    """
    Manages a live bidirectional telephony audio call session.
    """
    def __init__(
        self,
        send_fn: Callable[[dict | bytes], Awaitable[None]],
        encoding: str = "ulaw_8k",  # "ulaw_8k" (Twilio/G.711) or "pcm16_16k"
        protocol: str = "twilio",    # "twilio" | "raw"
        stream_sid: Optional[str] = None,
        deadlines: TurnDeadlines = DEFAULT_DEADLINES,
        greeting_text: Optional[str] = None,
        persona_id: str = "friend",
    ):
        self.send_fn = send_fn
        self.encoding = encoding
        self.protocol = protocol
        self.stream_sid = stream_sid or str(uuid.uuid4())
        self.deadlines = deadlines
        self.greeting_text = greeting_text
        self.persona_id = persona_id
        
        self.session_id = str(uuid.uuid4())
        self.turn_id = 0
        self.active_turn = 0
        self.cancelled = False
        
        # Server-side VAD
        self.vad = TelephonyVAD(TelephonyVADConfig(
            sample_rate=16000,
            frame_ms=20,
            start_threshold=0.018,
            end_threshold=0.010,
            min_speech_ms=70,
            hangover_ms=380,
            pre_speech_buffer_ms=240,
            barge_in_confirm_ms=40,
        ))
        
        # Providers
        self.stt: StreamingSTT = create_stt()
        self.llm: StreamingLLM = create_llm()
        self.tts: StreamingTTS = create_tts(self.persona_id)
        
        self.conversation: List[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT)
        ]
        
        # State tracking
        self.t0: Optional[float] = None
        self.t1: Optional[float] = None
        self.t3: Optional[float] = None
        self.t5: Optional[float] = None
        self.tool_latency_ms: Optional[float] = None
        
        # Async tasks
        self.llm_task: Optional[asyncio.Task] = None
        self.tts_task: Optional[asyncio.Task] = None
        self.audio_out_task: Optional[asyncio.Task] = None
        self.stt_task: Optional[asyncio.Task] = None
        self.watchdog_task: Optional[asyncio.Task] = None
        
        # Outbound audio queue for 20ms paced framing
        self.audio_out_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.is_agent_speaking = False
        self._running = False
        self._in_speech = False
        self._accumulated_transcript: list[str] = []
        
        # Production Resilience Layer (Jitter buffer + PLC + Acoustic Echo Suppressor)
        self.jitter_buffer = AdaptiveJitterBuffer(sample_rate=16000, frame_ms=20)
        self.echo_suppressor = AcousticEchoSuppressor(sample_rate=16000, history_ms=800)
        self._inbound_seq = 0

    async def start(self):
        """Initialize connections and start background consumers."""
        self._running = True
        await self.stt.connect()
        await self.llm.connect()
        await self.tts.connect()
        
        self.stt_task = asyncio.create_task(self._stt_event_loop())
        self.audio_out_task = asyncio.create_task(self._audio_out_loop())
        
        if self.greeting_text:
            # Trigger initial greeting turn
            self.turn_id += 1
            self.active_turn = self.turn_id
            self.t0 = time.perf_counter()
            text_q: asyncio.Queue = asyncio.Queue()
            self.tts_task = asyncio.create_task(self._run_tts(text_q, self.active_turn))
            await text_q.put(self.greeting_text)
            await text_q.put(None)
            self.conversation.append(ChatMessage(role="assistant", content=self.greeting_text))

    async def close(self):
        """Tear down all resources cleanly."""
        self._running = False
        await self.cancel_turn("session_closed", "close")
        
        for t in (self.llm_task, self.tts_task, self.stt_task, self.audio_out_task, self.watchdog_task):
            if t and not t.done():
                t.cancel()
                
        await asyncio.gather(self.stt.close(), self.llm.close(), self.tts.close(), return_exceptions=True)

    async def _send_carrier_clear(self):
        """Immediately clear telephone carrier output buffer on barge-in."""
        if self.protocol == "twilio":
            clear_msg = {
                "event": "clear",
                "streamSid": self.stream_sid,
            }
            try:
                await self.send_fn(clear_msg)
            except Exception:
                pass
        else:
            try:
                await self.send_fn({"type": "clear", "stream_sid": self.stream_sid})
            except Exception:
                pass

    async def cancel_turn(self, reason: str, stage: str, target_turn: Optional[int] = None):
        """Instant cancellation of active turn and purge of output audio buffers."""
        if self.active_turn == 0:
            return
        if target_turn is not None and target_turn != 0 and target_turn != self.active_turn:
            return
            
        self.cancelled = True
        dead_turn = self.active_turn
        self.active_turn = 0
        self.is_agent_speaking = False
        self.vad.set_agent_speaking(False)
        
        # 1. Drain and clear outbound audio queue immediately
        while not self.audio_out_queue.empty():
            try:
                self.audio_out_queue.get_nowait()
            except Exception:
                break
                
        # 2. Clear carrier-side buffer
        await self._send_carrier_clear()
        
        # 3. Cancel watchdog
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
            
        # 4. Abort LLM & TTS tasks and provider streams concurrently
        tasks = []
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
            tasks.append(self.llm_task)
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            tasks.append(self.tts_task)
            
        tasks.append(asyncio.create_task(self.llm.cancel()))
        tasks.append(asyncio.create_task(self.tts.cancel()))
        
        try:
            if hasattr(self.stt, "_reset_utterance"):
                self.stt._reset_utterance()
        except Exception:
            pass
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self.llm_task = None
        self.tts_task = None

    async def _audio_out_loop(self):
        """Paces outbound 20ms audio frames over the telephony connection."""
        FRAME_DURATION = 0.020  # 20ms
        
        while self._running:
            try:
                chunk = await self.audio_out_queue.get()
                if chunk is None or not self._running:
                    continue
                    
                if self.cancelled or self.active_turn == 0:
                    continue
                    
                t_start = time.perf_counter()
                
                # Send frame over protocol
                if self.protocol == "twilio":
                    # Twilio expects base64 encoded G.711 u-law payload
                    b64_payload = base64.b64encode(chunk).decode("ascii")
                    msg = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": b64_payload,
                        }
                    }
                    await self.send_fn(msg)
                else:
                    # Generic / raw binary frame
                    await self.send_fn(chunk)
                    
                # Pace sending to avoid flooding carrier buffer
                elapsed = time.perf_counter() - t_start
                sleep_dur = FRAME_DURATION - elapsed
                if sleep_dur > 0.001:
                    await asyncio.sleep(sleep_dur * 0.92)  # slight lead for jitter tolerance
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def process_inbound_audio(self, raw_bytes: bytes):
        """
        Ingest continuous inbound audio from telephony stream.
        Applies Acoustic Echo Suppression (AES), Jitter Buffer smoothing & PLC,
        decodes to 16kHz PCM16 and runs server-side VAD.
        """
        if not self._running:
            return
            
        # 1. Convert inbound audio to 16kHz Linear PCM
        if self.encoding == "ulaw_8k":
            pcm_8k = ulaw_to_pcm16(raw_bytes)
            pcm_16k = resample_pcm16(pcm_8k, 8000, 16000)
        elif self.encoding == "pcm16_8k":
            pcm_16k = resample_pcm16(raw_bytes, 8000, 16000)
        else:
            pcm_16k = raw_bytes  # Already 16kHz PCM16
            
        # 2. Feed 20ms frames through AES -> Jitter Buffer / PLC -> Server-Side VAD
        FRAME_BYTES_16K = 640  # 20ms @ 16kHz 16-bit mono
        now = time.perf_counter()
        
        for offset in range(0, len(pcm_16k), FRAME_BYTES_16K):
            raw_frame = pcm_16k[offset:offset + FRAME_BYTES_16K]
            if len(raw_frame) < FRAME_BYTES_16K:
                break
                
            self._inbound_seq += 1
            
            # Step A: Acoustic Echo Suppression (Suppresses speakerphone bleed)
            aes_frame, was_echo = self.echo_suppressor.suppress_echo(
                raw_frame, agent_speaking=self.is_agent_speaking
            )
            
            # Step B: Push to Adaptive Jitter Buffer & pull smooth/PLC concealed frame
            self.jitter_buffer.push(self._inbound_seq, aes_frame)
            frame, is_plc = self.jitter_buffer.pop_frame()
                
            event, audio_payload, _ = self.vad.process_frame(frame, now=now)
            
            if event == "speech_start":
                self._in_speech = True
                self._accumulated_transcript.clear()
                self.t0 = time.perf_counter()
                
                # --- ZERO-LATENCY BARGE-IN TRIGGER ---
                if self.is_agent_speaking or self.active_turn != 0:
                    await self.cancel_turn("barge_in", "barge_in")
                    
                self.turn_id += 1
                self.active_turn = self.turn_id
                self.cancelled = False
                
                # Forward buffered pre-speech audio to STT immediately
                if audio_payload:
                    await self.stt.send_audio(audio_payload, sample_rate=16000)
                    
            elif self._in_speech:
                await self.stt.send_audio(frame, sample_rate=16000)
                
            if event == "speech_end":
                self._in_speech = False
                await self.stt.finalize()

    async def _stt_event_loop(self):
        """Consume STT transcripts and trigger response turns."""
        while self._running:
            try:
                async for event in self.stt.receive_events():
                    if event.type == TranscriptType.FINAL:
                        self.t1 = event.t_event
                        chunk_text = (event.text or "").strip()
                        if chunk_text:
                            self._accumulated_transcript.append(chunk_text)
                            
                        # If Deepgram speech_final is true or VAD finalized
                        if event.meta.get("speech_final") or not self._in_speech or len(self._accumulated_transcript) >= 1:
                            full_text = " ".join(self._accumulated_transcript).strip()
                            self._accumulated_transcript.clear()
                            if full_text and self.active_turn != 0 and not self.cancelled:
                                my_turn = self.active_turn
                                if self.llm_task and not self.llm_task.done():
                                    self.llm_task.cancel()
                                self.llm_task = asyncio.create_task(self._run_turn(full_text, my_turn))
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.05)

    async def _run_tts(self, text_queue: asyncio.Queue, my_turn: int):
        """Synthesize TTS audio chunks, convert to telephony format, and queue frames."""
        async def text_iter():
            while True:
                item = await text_queue.get()
                if item is None:
                    break
                yield item

        self.is_agent_speaking = True
        self.vad.set_agent_speaking(True)
        first_audio = False
        
        try:
            async for event in self.tts.stream(text_iter()):
                if self.cancelled or my_turn != self.active_turn:
                    break
                if event.type == TTSEventType.AUDIO and event.audio:
                    if not first_audio:
                        first_audio = True
                        self.t5 = event.t_event
                        
                    # Resample TTS audio (typically 24kHz or 16kHz) to Telephony 8kHz or 16kHz
                    src_rate = event.sample_rate
                    pcm_16k_ref = resample_pcm16(event.audio, src_rate, 16000)
                    self.echo_suppressor.record_outbound_playback(pcm_16k_ref)

                    if self.encoding == "ulaw_8k":
                        pcm_8k = resample_pcm16(event.audio, src_rate, 8000)
                        ulaw_data = pcm16_to_ulaw(pcm_8k)
                        # Slice into 20ms frames (160 bytes of ulaw @ 8kHz)
                        FRAME_SIZE = 160
                        for i in range(0, len(ulaw_data), FRAME_SIZE):
                            frame = ulaw_data[i:i + FRAME_SIZE]
                            if frame:
                                await self.audio_out_queue.put(frame)
                    else:
                        FRAME_SIZE = 640
                        for i in range(0, len(pcm_16k_ref), FRAME_SIZE):
                            frame = pcm_16k_ref[i:i + FRAME_SIZE]
                            if frame:
                                await self.audio_out_queue.put(frame)
                elif event.type == TTSEventType.DONE:
                    break
        except Exception:
            pass
        finally:
            if my_turn == self.active_turn:
                # Wait for outbound queue to finish sending before lowering speaking flag
                while not self.audio_out_queue.empty() and not self.cancelled:
                    await asyncio.sleep(0.02)
                self.is_agent_speaking = False
                self.vad.set_agent_speaking(False)

    async def _run_turn(self, user_text: str, my_turn: int):
        """Dual-Brain Overlapped LLM -> Chunker -> TTS streaming for telephony calls with Enterprise Intelligence."""
        if my_turn != self.active_turn or self.cancelled:
            return
            
        # 1. Enterprise PII Redaction & Cyber Threat Intelligence Scan
        clean_user_text, redacted_entities = EnterprisePIISanitizer.sanitize(user_text)
        self.conversation.append(ChatMessage(role="user", content=clean_user_text))
        self.conversation[:] = cap_messages(self.conversation)
        
        # Threat Intelligence extraction
        threat_level, cat, matches = AntiFraudDetector.scan_utterance(user_text)
        extracted_intel = AntiFraudDetector.extract_threat_intelligence(user_text)
        ThreatIntelligenceStore.get_instance().update_call(
            stream_sid=self.stream_sid,
            turn_transcript=user_text,
            threat_level=threat_level.value,
            category=cat,
            extracted_intel=extracted_intel,
        )
        
        # 2. Warm Escalation Detection
        hist_dicts = [{"role": m.role, "content": m.content} for m in self.conversation]
        dossier = WarmTransferEngine.analyze_escalation(hist_dicts, caller_id=self.stream_sid)
        if dossier and dossier.should_escalate:
            # Transfer to human agent
            escalation_phrase = f"I'm connecting you with our {dossier.suggested_department} right now. One moment."
            text_queue: asyncio.Queue = asyncio.Queue()
            self.tts_task = asyncio.create_task(self._run_tts(text_queue, my_turn))
            await text_queue.put(escalation_phrase)
            await text_queue.put(None)
            
            # Send structured handover event over protocol
            if self.protocol == "twilio":
                await self.send_fn({
                    "event": "handover",
                    "streamSid": self.stream_sid,
                    "dossier": {
                        "intent": dossier.customer_intent,
                        "sentiment": dossier.sentiment,
                        "bullets": dossier.briefing_bullets,
                        "target": dossier.target_sip_or_phone,
                    }
                })
            else:
                await self.send_fn({
                    "type": "handover",
                    "stream_sid": self.stream_sid,
                    "dossier": {
                        "intent": dossier.customer_intent,
                        "sentiment": dossier.sentiment,
                        "bullets": dossier.briefing_bullets,
                        "target": dossier.target_sip_or_phone,
                    }
                })
            return
        
        # --- BRAIN-1: Sub-Conscious Instant Reflex Trigger ---
        reflex_filler = DualBrainReflexEngine.get_reflex_filler(clean_user_text, persona=self.persona_id)
        
        text_queue = asyncio.Queue()
        self.tts_task = asyncio.create_task(self._run_tts(text_queue, my_turn))
        chunker = ResponseChunker(first_min_chars=6, min_chars=16, max_chars=80)
        
        # If tool / complex query, immediately vocalize natural reflex filler
        if reflex_filler:
            await text_queue.put(reflex_filler)
            
        full_response = ""
        tool_call = None
        
        try:
            async for event in self.llm.stream(self.conversation, tools=list_tools_for_llm()):
                if self.cancelled or my_turn != self.active_turn:
                    break
                if event.type == LLMEventType.TOKEN:
                    if self.t3 is None:
                        self.t3 = event.t_event
                    full_response += event.text
                    for phrase in chunker.add(event.text):
                        clean_phrase = SpokenDialogueNormalizer.normalize_for_speech(phrase)
                        # Enterprise Policy Guardrails filter
                        guard = EnterpriseGuardrails.validate_and_filter(clean_phrase)
                        if guard.sanitized_text:
                            await text_queue.put(guard.sanitized_text)
                elif event.type == LLMEventType.TOOL_CALL:
                    tool_call = event
                    break
                elif event.type == LLMEventType.DONE:
                    for phrase in chunker.flush():
                        clean_phrase = SpokenDialogueNormalizer.normalize_for_speech(phrase)
                        guard = EnterpriseGuardrails.validate_and_filter(clean_phrase)
                        if guard.sanitized_text:
                            await text_queue.put(guard.sanitized_text)
                    await text_queue.put(None)
                    break
        except Exception:
            await text_queue.put(None)
            return

        # If a tool was triggered: Brain-2 resolves tool while Brain-1 filler finishes speaking
        if tool_call and tool_call.tool_name and my_turn == self.active_turn and not self.cancelled:
            result = await run_tool(
                tool_call.tool_name,
                tool_call.tool_args or {},
                tool_call_id=tool_call.tool_call_id,
            )
            
            self.conversation.append(ChatMessage(
                role="tool",
                content=json.dumps(result.data if result.ok else {"error": result.error}),
                name=result.name,
                tool_call_id=tool_call.tool_call_id,
            ))
            
            if my_turn == self.active_turn and not self.cancelled:
                chunker.reset()
                
                async for event in self.llm.stream(self.conversation, tools=list_tools_for_llm()):
                    if self.cancelled or my_turn != self.active_turn:
                        break
                    if event.type == LLMEventType.TOKEN:
                        full_response += event.text
                        for phrase in chunker.add(event.text):
                            clean_phrase = SpokenDialogueNormalizer.normalize_for_speech(phrase)
                            guard = EnterpriseGuardrails.validate_and_filter(clean_phrase)
                            if guard.sanitized_text:
                                await text_queue.put(guard.sanitized_text)
                    elif event.type == LLMEventType.DONE:
                        for phrase in chunker.flush():
                            clean_phrase = SpokenDialogueNormalizer.normalize_for_speech(phrase)
                            guard = EnterpriseGuardrails.validate_and_filter(clean_phrase)
                            if guard.sanitized_text:
                                await text_queue.put(guard.sanitized_text)
                        await text_queue.put(None)
                        break
        else:
            await text_queue.put(None)
                        
        if full_response and my_turn == self.active_turn:
            # Enforce guardrails on full stored message
            final_guard = EnterpriseGuardrails.validate_and_filter(full_response)
            self.conversation.append(ChatMessage(role="assistant", content=final_guard.sanitized_text))
            self.conversation[:] = cap_messages(self.conversation)
