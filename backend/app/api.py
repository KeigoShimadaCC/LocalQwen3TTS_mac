from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from .cache import AudioCache
from .config import settings
from .model_manager import QueueFullError, get_manager
from .schemas import (
    AudioFormat,
    HealthResponse,
    ModelName,
    TTSRequest,
    TTSResponse,
    VoiceResponse,
)
from .storage import storage
from .utils_audio import audio_bytes_base64, convert_audio, waveform_duration


router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    manager = get_manager()
    uptime = time.time() - settings.startup_time
    return HealthResponse(
        status="ok",
        device=manager.device.type,
        models_loaded=list(manager.models.keys()),
        uptime_sec=uptime,
    )


@router.get("/v1/voices", response_model=VoiceResponse)
async def list_voices(
    model: ModelName = ModelName.SMALL, refresh: bool = False
) -> VoiceResponse:
    manager = get_manager()
    voices = manager.available_voices(model.value, refresh=refresh)
    return VoiceResponse(model=model, voices=voices)


@router.post("/v1/tts", response_model=TTSResponse)
async def synthesize(request: TTSRequest) -> TTSResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming not implemented yet")
    manager = get_manager()
    req_id = request.request_id or uuid.uuid4().hex
    start_time = time.perf_counter()
    LOGGER.info(
        "[tts] request_id=%s model=%s voice=%s language=%s format=%s",
        req_id,
        request.model.value,
        request.voice,
        request.language.value,
        request.format.value,
    )
    try:
        cache_key = None
        if cache:
            cache_key = (
                request.model.value,
                request.text,
                request.voice,
                request.language.value,
                request.tone,
                request.speed,
                request.seed,
                request.sample_rate,
                request.format.value,
            )
            cached = cache.get(cache_key)
            if cached:
                cache_hits, cache_misses = cache.stats
                LOGGER.info(
                    "[tts] cache hit request_id=%s hits=%s misses=%s",
                    req_id,
                    cache_hits,
                    cache_misses,
                )
                audio_bytes, fmt, sr, duration = cached
                return _build_response(req_id, fmt, sr, duration, audio_bytes)

        audio, sample_rate = await manager.synthesize(
            model_name=request.model.value,
            text=request.text,
            voice=request.voice,
            language=request.language.value,
            tone=request.tone,
            seed=request.seed,
            speed=request.speed,
            sample_rate=request.sample_rate,
        )
    except QueueFullError:
        LOGGER.warning("[tts] queue full request_id=%s", req_id)
        raise HTTPException(status_code=429, detail="queue full")

    audio_bytes, fmt, sr = convert_audio(audio, sample_rate, request.format.value)
    duration = waveform_duration(audio, sr)

    if cache and cache_key:
        cache.put(cache_key, (audio_bytes, fmt, sr, duration))
        cache_hits, cache_misses = cache.stats
        LOGGER.info(
            "[tts] cache store request_id=%s hits=%s misses=%s",
            req_id,
            cache_hits,
            cache_misses,
        )

    elapsed = (time.perf_counter() - start_time) * 1000
    LOGGER.info(
        "[tts] completed request_id=%s duration_ms=%.2f sample_rate=%s",
        req_id,
        elapsed,
        sr,
    )
    return _build_response(req_id, fmt, sr, duration, audio_bytes)


def _build_response(
    request_id: str, fmt: str, sample_rate: int, duration: float, audio_bytes: bytes
) -> TTSResponse:
    if settings.output_mode == "base64":
        return TTSResponse(
            request_id=request_id,
            audio_format=AudioFormat(fmt),
            sample_rate=sample_rate,
            duration_sec=duration,
            audio_base64=audio_bytes_base64(audio_bytes),
        )

    file_id = storage.save(audio_bytes, fmt)
    return TTSResponse(
        request_id=request_id,
        audio_format=AudioFormat(fmt),
        sample_rate=sample_rate,
        duration_sec=duration,
        audio_url=f"/v1/audio/{file_id}",
    )


@router.get("/v1/audio/{file_name}")
async def get_audio(file_name: str):
    if settings.output_mode != "file":
        raise HTTPException(status_code=404, detail="file serving disabled")
    return storage.serve(file_name)


cache = AudioCache(settings.cache_size) if settings.enable_cache else None
