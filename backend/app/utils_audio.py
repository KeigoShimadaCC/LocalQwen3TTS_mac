from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
import soundfile as sf

try:  # optional MP3 support via pydub
    from pydub import AudioSegment

    HAS_PYDUB = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PYDUB = False


def waveform_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def wav_bytes_to_mp3_bytes(wav_bytes: bytes) -> bytes:
    if not HAS_PYDUB:
        raise RuntimeError("MP3 conversion requires pydub + ffmpeg")
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)
    return mp3_buffer.read()


def waveform_duration(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(len(audio) / sample_rate)


def audio_bytes_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("ascii")


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise ValueError("Unsupported audio shape")


def normalize_waveform(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio
    speed = max(0.1, min(speed, 4.0))
    new_length = max(1, int(len(audio) / speed))
    old_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    new_indices = np.linspace(0, len(audio) - 1, num=new_length)
    return np.interp(new_indices, old_indices, audio).astype(audio.dtype)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    new_length = max(1, int(duration * target_sr))
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(audio.dtype)


def convert_audio(
    audio: np.ndarray, sample_rate: int, fmt: str
) -> Tuple[bytes, str, int]:
    wav_bytes = waveform_to_wav_bytes(audio, sample_rate)
    if fmt == "wav":
        return wav_bytes, "wav", sample_rate
    if fmt == "mp3":
        return wav_bytes_to_mp3_bytes(wav_bytes), "mp3", sample_rate
    raise ValueError(f"Unsupported format: {fmt}")
