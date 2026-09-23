"""
Qwen3-TTS Provider Adapter for Real-Time Streaming Telephony & WebAudio.
Synthesizes natural speech via Qwen3-TTS-12Hz models and decodes directly to 24kHz PCM16.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import AsyncIterator, Optional, List
import numpy as np

from tts_protocol import StreamingTTS, TTSEvent, TTSEventType

try:
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    HAS_QWEN_TTS = True
except ImportError:
    HAS_QWEN_TTS = False


class QwenStreamingTTS(StreamingTTS):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker: str = "vivian",
        language: str = "english",
        sample_rate: int = 24000,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.speaker = speaker
        self.language = language
        self._sample_rate = sample_rate
        
        if device:
            self.device = device
        elif HAS_QWEN_TTS and torch.cuda.is_available():
            self.device = "cuda"
        elif HAS_QWEN_TTS and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._model: Optional[Qwen3TTSModel] = None
        self._connected = False
        self._cancelled = False
        self._closed = False

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def audio_format(self) -> str:
        return "pcm16"

    async def connect(self) -> None:
        if not HAS_QWEN_TTS:
            self._connected = False
            return
            
        if self._model is None:
            torch_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            # Load model in thread pool to prevent blocking event loop
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    dtype=torch_dtype,
                    device_map=self.device if self.device != "cpu" else None,
                )
            )
        self._connected = True
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        self._model = None

    async def cancel(self) -> None:
        self._cancelled = True

    def _synthesize_chunk(self, text: str) -> bytes:
        if self._model is None:
            return b""
        
        word_count = len(text.split())
        max_tokens = max(16, min(int(word_count * 3.8), 60))
        
        if hasattr(self._model, "generate_custom_voice") and self.model_id.endswith("CustomVoice"):
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language=self.language,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
            )
        else:
            wavs, sr = self._model.generate_voice_design(
                text=text,
                instruct="A natural conversational human voice, smooth and clear.",
                language=self.language,
                max_new_tokens=max_tokens,
            )
            
        audio_np = wavs[0]
        # Resample to 24kHz if needed or convert float32 array to int16 PCM bytes
        pcm16_array = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm16_array.tobytes()

    async def stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[TTSEvent]:
        if not HAS_QWEN_TTS or self._closed:
            yield TTSEvent(type=TTSEventType.ERROR, text="qwen-tts not available")
            return

        await self.connect()
        self._cancelled = False
        first_audio = True
        loop = asyncio.get_running_loop()

        try:
            async for phrase in text_stream:
                if self._cancelled:
                    break
                phrase = (phrase or "").strip()
                if not phrase:
                    continue

                t0 = time.perf_counter()
                pcm_bytes = await loop.run_in_executor(None, self._synthesize_chunk, phrase)
                latency_ms = (time.perf_counter() - t0) * 1000

                if pcm_bytes:
                    yield TTSEvent(
                        type=TTSEventType.AUDIO,
                        audio_data=pcm_bytes,
                        format="pcm16",
                        sample_rate=self._sample_rate,
                        is_first=first_audio,
                    )
                    first_audio = False

            yield TTSEvent(type=TTSEventType.DONE)
        except Exception as e:
            yield TTSEvent(type=TTSEventType.ERROR, text=str(e))
