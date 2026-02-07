from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import settings


class ModelName(str, Enum):
    SMALL = "qwen3-tts-0.6b"
    LARGE = "qwen3-tts-1.7b"


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


class Language(str, Enum):
    AUTO = "auto"
    EN = "en"
    JA = "ja"
    ZH = "zh"
    FR = "fr"
    ES = "es"
    DE = "de"
    KO = "ko"


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: ModelName = Field(default=ModelName.SMALL)
    voice: Optional[str] = None
    language: Language = Field(default=Language.AUTO)
    tone: Optional[str] = None
    format: AudioFormat = Field(default=AudioFormat.WAV)
    sample_rate: int = Field(default=settings.default_sample_rate, gt=0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    speed: float = Field(default=1.0, gt=0.1, le=4.0)
    request_id: Optional[str] = None
    stream: bool = False

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, value: str) -> str:
        if len(value) > settings.max_text_length:
            raise ValueError("text too long")
        return value


class TTSResponse(BaseModel):
    request_id: str
    audio_format: AudioFormat
    sample_rate: int
    duration_sec: float
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    device: str
    models_loaded: list[str]
    uptime_sec: float


class VoiceResponse(BaseModel):
    model: ModelName
    voices: list[str]
