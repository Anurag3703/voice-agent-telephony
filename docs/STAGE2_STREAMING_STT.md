# Stage 2 – Streaming STT

## Goal
Prove the streaming STT path with real partials, persistent connection, client-side VAD, and measured **T0 → T1** latency.

## Components

### Protocol (`backend/stt_protocol.py`)
```python
class StreamingSTT(ABC):
    async def connect(self) -> None
    async def send_audio(self, chunk: bytes, sample_rate: int = 16000) -> None
    async def receive_events(self) -> AsyncIterator[TranscriptEvent]
    async def finalize(self) -> None
    async def close(self) -> None
    supports_partials: bool
```

Any real provider (Deepgram, AssemblyAI, Google, Azure, etc.) must implement this. Batch-only STT is not acceptable for the realtime path.

### Mock (`backend/mock_stt.py`)
- Emits progressive partials every ~110 ms
- On `finalize()` adds a realistic ~35 ms finalization delay
- Deterministic phrase for testing
- No API keys required

### Server (`backend/server.py`)
- FastAPI + WebSocket at `/ws/stt`
- Accepts binary PCM16 LE mono @ 16 kHz
- Control messages: `{"type": "finalize"}`, `{"type": "reset"}`
- Streams JSON transcript events back
- Serves the frontend

### Frontend
- Persistent WebSocket
- getUserMedia → ScriptProcessor → downsample to 16 kHz PCM16 → send
- Simple RMS energy VAD
- On silence duration → send finalize (T0)
- Displays partials live + final + T0→T1 latency

## Run

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/workdir/artifacts/voice-agent/backend
python3 server.py
# or: uvicorn server:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080

1. Connect WebSocket
2. Start Microphone (allow permission)
3. Speak for a couple of seconds, then stop
4. Watch partials appear, then final + latency numbers

## Latency target for this stage
STT finalization (T0 → T1) should be in the order of the mock's `final_delay_ms` (~35–50 ms) + network.  
Real providers will be higher; the mock lets us validate the rest of the pipeline without external variance.

## What is intentionally still missing (later stages)
- Real STT provider adapter
- Speculative LLM warming on partials
- Streaming LLM + TTS pipeline
- Barge-in cancellation
- Full end-to-end TTFA dashboard
