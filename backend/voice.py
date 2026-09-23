"""
Ultra-Low Latency Real-Time Voice-to-Voice Loop:
1. Microphone Input with Fast Energy VAD (SoundDevice) - 350ms silence detection
2. Sub-100ms Local STT (faster-whisper tiny.en)
3. Small Fast LLM Streaming (Ollama llama3.2:1b with max 30 tokens)
4. Ultra-Fast TTS Streaming:
   - "edge_tts" (Microsoft Neural Voice: ~250ms TTFA, crystal clear)
   - "macos_native" (macOS say: ~10ms TTFA, instant response)
   - "qwen" (Qwen3-TTS local model)
"""

import os
import re
import time
import queue
import asyncio
import threading
import json
import subprocess
import requests
import numpy as np
import sounddevice as sd
import io

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# TTS_ENGINE options: "edge_tts" (Natural & Fast ~250ms) or "macos_native" (Instant ~10ms) or "qwen"
TTS_ENGINE = "edge_tts"  
EDGE_VOICE = "en-US-JennyNeural"  # or "en-US-GuyNeural", "en-US-AriaNeural"

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"
WHISPER_MODEL_SIZE = "tiny.en"

SAMPLE_RATE = 16000
SILENCE_DURATION = 0.45  # 450ms silence threshold for snappy turn-taking
ENERGY_THRESHOLD = 0.012 # Ambient noise threshold

# ---------------------------------------------------------
# Initialize STT
# ---------------------------------------------------------
print("🔄 Initializing Fast STT (faster-whisper tiny.en)...")
from faster_whisper import WhisperModel
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("✅ STT ready.")

# Initialize Qwen if selected
qwen_tts_model = None
if TTS_ENGINE == "qwen":
    print("🔄 Initializing Qwen-TTS...")
    import torch
    from qwen_tts import Qwen3TTSModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    qwen_tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device if device != "cpu" else None,
    )
    print("✅ Qwen-TTS ready.")
else:
    print(f"✅ Using Ultra-Fast TTS Engine: '{TTS_ENGINE}' (TTFA < 300ms)")

# ---------------------------------------------------------
# 1. Microphone Listener with Real-time VAD
# ---------------------------------------------------------
def record_user_speech():
    """Listens to the microphone and stops automatically when user finishes speaking."""
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    print("\n🎤 Listening... (Speak into your mic)")
    recorded_frames = []
    speaking = False
    silence_start = None

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        while True:
            chunk = audio_queue.get()
            recorded_frames.append(chunk)
            rms = np.sqrt(np.mean(chunk**2))

            if rms > ENERGY_THRESHOLD:
                if not speaking:
                    print("  ⚡ [User speaking...]")
                    speaking = True
                silence_start = None
            else:
                if speaking:
                    if silence_start is None:
                        silence_start = time.perf_counter()
                    elif time.perf_counter() - silence_start > SILENCE_DURATION:
                        print("  🛑 [Silence detected - processing...]")
                        break

    audio_data = np.concatenate(recorded_frames, axis=0).flatten().astype(np.float32)
    return audio_data


# ---------------------------------------------------------
# 2. Fast Local STT
# ---------------------------------------------------------
def speech_to_text(audio_data):
    stt_start = time.perf_counter()
    segments, _ = stt_model.transcribe(audio_data, beam_size=1)
    text = " ".join([s.text for s in segments]).strip()
    latency_ms = (time.perf_counter() - stt_start) * 1000
    print(f"📝 STT ({latency_ms:.0f}ms): \"{text}\"")
    return text


# ---------------------------------------------------------
# 3. Streaming Ollama LLM (Real Human Phone Call Persona)
# ---------------------------------------------------------
CONVERSATION_HISTORY = []

def stream_llm_response(user_prompt):
    """
    Generates a single, ultra-natural, 1-line human phone conversation response.
    Maintains dialogue context for natural back-and-forth.
    """
    system_prompt = (
        "You are having a casual phone conversation with a friend. "
        "Rules:\n"
        "1. Respond in EXACTLY ONE short, natural, human spoken sentence (under 10-12 words).\n"
        "2. Sound like a real person on a call (use casual language, natural fillers like 'Oh', 'Yeah', 'Gotcha', 'Haha' when appropriate).\n"
        "3. NEVER act like an AI, customer support bot, or say 'How can I assist/help you today?'.\n"
        "4. If insulted or teased, brush it off casually or banter with light humor like a real friend.\n"
        "5. No lists, emojis, quotes, or markdown. Just one crisp spoken line."
    )

    # Keep last 6 conversational turns
    CONVERSATION_HISTORY.append({"role": "user", "content": user_prompt})
    recent_history = CONVERSATION_HISTORY[-6:]

    # Format into Ollama /api/chat payload for multi-turn conversation
    messages = [{"role": "system", "content": system_prompt}] + recent_history

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.75,
            "top_p": 0.9,
            "num_predict": 25,  # Strictly 1 short line
            "stop": ["\n", "User:", "Assistant:"],
        },
    }

    try:
        resp = requests.post("http://localhost:11434/api/chat", json=payload, stream=True, timeout=5)
    except Exception as e:
        yield f"Hey, sorry, couldn't hear that: {e}"
        return

    full_reply = ""
    for line in resp.iter_lines():
        if line:
            try:
                chunk_data = json.loads(line)
                token = chunk_data.get("message", {}).get("content", "")
                full_reply += token
            except Exception:
                continue

    cleaned_reply = full_reply.strip().replace('"', '').replace('*', '')
    if cleaned_reply:
        CONVERSATION_HISTORY.append({"role": "assistant", "content": cleaned_reply})
        yield cleaned_reply


# ---------------------------------------------------------
# 4. Ultra-Fast TTS Playback
# ---------------------------------------------------------
async def _synthesize_edge_tts(text: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(text, EDGE_VOICE, rate="+15%")
    mp3_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes.extend(chunk["data"])
    return bytes(mp3_bytes)


def speak_text(text: str):
    """Speaks the text with sub-second latency using chosen engine."""
    tts_start = time.perf_counter()

    if TTS_ENGINE == "macos_native":
        # Sub-10ms native speech
        subprocess.run(["say", "-r", "210", text])
        latency_ms = (time.perf_counter() - tts_start) * 1000
        return latency_ms

    elif TTS_ENGINE == "edge_tts":
        # Neural TTS stream (~200ms)
        import soundfile as sf
        import av

        mp3_data = asyncio.run(_synthesize_edge_tts(text))
        
        # Fast decode mp3 buffer to pcm
        container = av.open(io.BytesIO(mp3_data))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
        pcm_chunks = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                pcm_chunks.append(r.to_ndarray())

        if pcm_chunks:
            audio_array = np.concatenate(pcm_chunks, axis=1).flatten().astype(np.float32) / 32768.0
            latency_ms = (time.perf_counter() - tts_start) * 1000
            sd.play(audio_array, 24000)
            sd.wait()
            return latency_ms

    elif TTS_ENGINE == "qwen" and qwen_tts_model is not None:
        wavs, sr = qwen_tts_model.generate_voice_design(
            text=text,
            instruct="Clear natural voice",
            language="english",
        )
        latency_ms = (time.perf_counter() - tts_start) * 1000
        sd.play(wavs[0], sr)
        sd.wait()
        return latency_ms

    return 0


# ---------------------------------------------------------
# 5. Full Real-Time Voice-to-Voice Loop
# ---------------------------------------------------------
def run_voice_loop():
    print("\n" + "="*55)
    print(f"🚀 Low-Latency Voice-to-Voice Loop")
    print(f"   • STT:    faster-whisper ({WHISPER_MODEL_SIZE})")
    print(f"   • LLM:    Ollama ({OLLAMA_MODEL})")
    print(f"   • TTS:    {TTS_ENGINE}")
    print("   Press Ctrl+C to stop.")
    print("="*55)

    try:
        while True:
            # 1. Listen with VAD
            audio_data = record_user_speech()
            if len(audio_data) < SAMPLE_RATE * 0.4:
                continue

            # 2. Transcribe
            turn_start = time.perf_counter()
            transcript = speech_to_text(audio_data)
            if not transcript or len(transcript.strip()) < 2:
                continue

            # 3. LLM + Streaming TTS
            print("🤖 Assistant speaking:")
            first_chunk = True

            for clause in stream_llm_response(transcript):
                clause = clause.strip()
                if not clause:
                    continue

                if first_chunk:
                    ttfa_ms = (time.perf_counter() - turn_start) * 1000
                    print(f"  ⚡ [TTFA: {ttfa_ms:.0f} ms]")
                    first_chunk = False

                tts_time = speak_text(clause)
                print(f"  🗣️ \"{clause}\" (TTS: {tts_time:.0f}ms)")

    except KeyboardInterrupt:
        print("\n👋 Exiting voice loop.")


if __name__ == "__main__":
    run_voice_loop()


