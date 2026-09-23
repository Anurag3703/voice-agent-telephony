"""
High-Performance Telephony Audio, Codecs & Server-Side VAD Engine.

Designed for real-world PSTN / SIP / WebRTC / Twilio / Telnyx voice calls:
- Fast zero-dependency G.711 u-law (PCMU) and A-law (PCMA) <-> Linear PCM16 conversion
- Telephony 8kHz <-> 16kHz <-> 24kHz fast linear resampling
- Frame chunking (e.g. 20ms G.711 160-byte frames)
- Robust Server-Side Adaptive Hysteresis VAD with pre-speech audio ring-buffer
- Carrier Clear / Barge-in detection with zero speech loss
"""

from __future__ import annotations

import array
import collections
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Deque
from acoustic_prosodic_core import AcousticProsodicAnalyzer, AcousticFeatures


# ==============================================================================
# G.711 u-law / A-law Lookup Tables for Blazing Fast Conversions
# ==============================================================================

def _build_ulaw_tables():
    """Generate 8-bit ulaw <-> 16-bit linear PCM lookup tables."""
    u2linear = []
    BIAS = 0x84
    for u in range(256):
        u_inv = ~u & 0xFF
        sign = u_inv & 0x80
        exponent = (u_inv >> 4) & 0x07
        mantissa = u_inv & 0x0F
        sample = ((mantissa << 3) + BIAS) << exponent
        sample -= BIAS
        linear = -sample if sign != 0 else sample
        u2linear.append(linear)

    linear2u = []
    for linear in range(-32768, 32768):
        sign = 0
        if linear < 0:
            linear = -linear
            sign = 0x80
        linear = min(linear, 32767)
        linear = linear + BIAS
        if linear > 0x7FFF:
            linear = 0x7FFF
        
        # Find exponent
        exponent = 7
        for exp in range(7, -1, -1):
            if linear & (1 << (exp + 7)):
                exponent = exp
                break
        
        mantissa = (linear >> (exponent + 3)) & 0x0F
        u_val = ~(sign | (exponent << 4) | mantissa) & 0xFF
        linear2u.append(u_val)

    return u2linear, linear2u


_U2LINEAR, _LINEAR2U = _build_ulaw_tables()


def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """Decode 8kHz 8-bit G.711 u-law bytes into 16-bit Linear PCM bytes."""
    samples = [_U2LINEAR[b] for b in ulaw_bytes]
    return struct.pack(f"<{len(samples)}h", *samples)


def pcm16_to_ulaw(pcm16_bytes: bytes) -> bytes:
    """Encode 16-bit Linear PCM bytes into 8-bit G.711 u-law bytes."""
    num_samples = len(pcm16_bytes) // 2
    if num_samples == 0:
        return b""
    samples = struct.unpack(f"<{num_samples}h", pcm16_bytes)
    # Map each 16-bit signed int to 0..65535 index in _LINEAR2U table
    return bytes(_LINEAR2U[s + 32768] for s in samples)


def resample_pcm16(pcm16_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Fast linear interpolation resampling for 16-bit mono PCM."""
    if from_rate == to_rate or len(pcm16_bytes) < 2:
        return pcm16_bytes
    
    num_in = len(pcm16_bytes) // 2
    in_samples = struct.unpack(f"<{num_in}h", pcm16_bytes)
    
    ratio = from_rate / to_rate
    num_out = int(num_in / ratio)
    if num_out == 0:
        return b""
    
    out_samples = []
    for i in range(num_out):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx
        if idx >= num_in - 1:
            val = in_samples[-1]
        else:
            val = int(in_samples[idx] * (1.0 - frac) + in_samples[idx + 1] * frac)
        out_samples.append(max(-32768, min(32767, val)))
        
    return struct.pack(f"<{len(out_samples)}h", *out_samples)


def calculate_pcm_rms(pcm16_bytes: bytes) -> float:
    """Calculate normalized RMS energy (0.0 to 1.0) of 16-bit PCM mono audio."""
    num_samples = len(pcm16_bytes) // 2
    if num_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{num_samples}h", pcm16_bytes)
    sum_sq = sum((s / 32768.0) ** 2 for s in samples)
    return math.sqrt(sum_sq / num_samples)


# ==============================================================================
# Server-Side Adaptive VAD for Telephony Streams
# ==============================================================================

class VADState(str, Enum):
    SILENCE = "silence"
    POSSIBLE_SPEECH = "possible_speech"
    SPEECH = "speech"
    HANGOVER = "hangover"


@dataclass
class TelephonyVADConfig:
    sample_rate: int = 16000           # Internal VAD operates in 16kHz PCM16
    frame_ms: int = 20                 # 20ms standard telephony audio frame
    start_threshold: float = 0.020     # Minimum RMS above noise floor to start speech
    end_threshold: float = 0.012       # Minimum RMS to stay in speech
    min_speech_ms: int = 80            # Must sustain energy for >= 80ms to confirm speech
    hangover_ms: int = 400             # Trailing silence required to trigger speech end
    pre_speech_buffer_ms: int = 260    # Pre-speech audio buffer so first syllables aren't lost
    min_utterance_ms: int = 220        # Discard false clicks/coughs shorter than this
    noise_alpha: float = 0.04          # Background noise floor adaptation rate
    barge_in_margin: float = 2.4       # Multiplier during agent speech playback
    barge_in_confirm_ms: int = 50      # Faster confirmation for barge-in


class TelephonyVAD:
    """
    Robust Server-Side VAD for continuous streaming telephony audio.
    - Tracks dynamic background noise floor (handles cellular line hiss).
    - Maintains a pre-speech ring buffer so no initial phonemes are clipped.
    - Features echo-aware residual suppression during agent playback.
    """
    def __init__(self, config: Optional[TelephonyVADConfig] = None):
        self.cfg = config or TelephonyVADConfig()
        self.state = VADState.SILENCE
        self.noise_floor = 0.008
        self.audio_time_ms: float = 0.0
        self.prosodic_analyzer = AcousticProsodicAnalyzer(sample_rate=self.cfg.sample_rate)
        
        # Audio buffer ring (stores raw 16kHz PCM16 frames)
        max_pre_frames = int(self.cfg.pre_speech_buffer_ms / self.cfg.frame_ms) + 1
        self._ring_buffer: Deque[bytes] = collections.deque(maxlen=max_pre_frames)
        
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        self.utterance_start_time: Optional[float] = None
        self.agent_speaking = False
        self._current_utterance_pcm: bytearray = bytearray()
        self.latest_acoustic_features: Optional[AcousticFeatures] = None
        
    def reset(self):
        self.state = VADState.SILENCE
        self._ring_buffer.clear()
        self.audio_time_ms = 0.0
        self.speech_start_time = None
        self.silence_start_time = None
        self.utterance_start_time = None
        self._current_utterance_pcm.clear()
        self.prosodic_analyzer.reset()
        self.latest_acoustic_features = None
        
    def set_agent_speaking(self, speaking: bool):
        """Inform VAD if agent is currently playing TTS out to the caller."""
        self.agent_speaking = speaking
        if speaking:
            # Slightly elevate noise floor baseline to handle line echo
            self.noise_floor = max(0.008, self.noise_floor)

    def process_frame(
        self,
        pcm16_frame: bytes,
        now: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[bytes], float]:
        """
        Process a 20ms PCM16 audio frame.
        Returns:
            (event, audio_payload_if_applicable, rms_level)
            event: "speech_start", "speech_end", "speech_end_short", None
            audio_payload: On "speech_start", includes the buffered pre-speech audio!
        """
        frame_dur_ms = (len(pcm16_frame) / 2) / (self.cfg.sample_rate / 1000.0)
        self.audio_time_ms += frame_dur_ms
        t_now = self.audio_time_ms
            
        level = calculate_pcm_rms(pcm16_frame)
        self._ring_buffer.append(pcm16_frame)
        
        # Real-Time Acoustic Prosody Analysis (Pitch, Deepfake, Stress, Turn Probability)
        acoustics = self.prosodic_analyzer.analyze_frame(pcm16_frame)
        self.latest_acoustic_features = acoustics
        
        # Dynamic threshold calculation
        if self.agent_speaking:
            start_thresh = max(self.cfg.start_threshold * 1.2, self.noise_floor * self.cfg.barge_in_margin)
            end_thresh = max(self.cfg.end_threshold * 1.1, self.noise_floor * 1.5)
            min_speech_req = self.cfg.barge_in_confirm_ms
        else:
            start_thresh = max(self.cfg.start_threshold, self.noise_floor * 1.8)
            end_thresh = max(self.cfg.end_threshold, self.noise_floor * 1.2)
            min_speech_req = self.cfg.min_speech_ms

        # Background noise floor tracking when quiet or during non-vocal ambient playback
        if (level < start_thresh and self.state == VADState.SILENCE) or (self.agent_speaking and not acoustics.is_voiced_speech):
            self.noise_floor = self.noise_floor * (1.0 - self.cfg.noise_alpha) + level * self.cfg.noise_alpha

        # --- State Machine ---
        # Robust Noise Rejection: When agent is speaking, ignore non-vocal noise and ambient hum
        is_vocal_band = (200.0 <= acoustics.spectral_centroid_hz <= 3200.0)
        
        if self.agent_speaking and not acoustics.is_voiced_speech and (acoustics.is_transient_noise or not is_vocal_band):
            self.state = VADState.SILENCE
            self.speech_start_time = None
            return None, None, level

        if self.state == VADState.SILENCE:
            if level >= start_thresh:
                # If agent is speaking, strictly require voiced human speech or high-energy vocal band
                if not self.agent_speaking or acoustics.is_voiced_speech or (is_vocal_band and level > start_thresh * 2.0):
                    self.state = VADState.POSSIBLE_SPEECH
                    self.speech_start_time = t_now
            return None, None, level

        elif self.state == VADState.POSSIBLE_SPEECH:
            if level >= start_thresh:
                dur = t_now - (self.speech_start_time or t_now)
                # Voiced human speech confirms in 40ms, unvoiced vocal band requires 140ms
                required_confirm = min_speech_req if (not self.agent_speaking or acoustics.is_voiced_speech) else min_speech_req * 3.5
                if dur >= required_confirm:
                    self.state = VADState.SPEECH
                    self.utterance_start_time = self.speech_start_time or t_now
                    self.silence_start_time = None
                    self.speech_start_time = None
                    
                    # Consolidate pre-speech audio buffer so initial syllables are preserved
                    pre_audio = b"".join(self._ring_buffer)
                    self._current_utterance_pcm = bytearray(pre_audio)
                    return "speech_start", pre_audio, level
            else:
                self.state = VADState.SILENCE
                self.speech_start_time = None
            return None, None, level

        elif self.state == VADState.SPEECH:
            self._current_utterance_pcm.extend(pcm16_frame)
            if level < end_thresh:
                self.state = VADState.HANGOVER
                self.silence_start_time = t_now
            return None, None, level

        elif self.state == VADState.HANGOVER:
            self._current_utterance_pcm.extend(pcm16_frame)
            if level >= start_thresh:
                # Resumed talking before hangover expired
                self.state = VADState.SPEECH
                self.silence_start_time = None
                return None, None, level
            else:
                silence_dur = t_now - (self.silence_start_time or t_now)
                
                # --- PROSODIC PREDICTIVE ACCELERATION ---
                # If pitch declination and acoustic roll-off indicate a completed sentence,
                # adaptively reduce hangover requirement from 380ms to ~140ms for snappy handoff!
                effective_hangover_ms = self.cfg.hangover_ms
                if acoustics.turn_completion_probability >= 0.80:
                    effective_hangover_ms = min(self.cfg.hangover_ms, 140.0)
                elif acoustics.turn_completion_probability >= 0.60:
                    effective_hangover_ms = min(self.cfg.hangover_ms, 240.0)

                if silence_dur >= effective_hangover_ms:
                    total_dur = t_now - (self.utterance_start_time or t_now)
                    self.state = VADState.SILENCE
                    self.silence_start_time = None
                    self.utterance_start_time = None
                    
                    if total_dur < self.cfg.min_utterance_ms:
                        self._current_utterance_pcm.clear()
                        return "speech_end_short", None, level
                    
                    full_utterance = bytes(self._current_utterance_pcm)
                    self._current_utterance_pcm.clear()
                    return "speech_end", full_utterance, level

        return None, None, level
