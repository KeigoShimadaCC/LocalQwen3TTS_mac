from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal


DeviceOption = Literal["auto", "mps", "cpu"]
DTypeOption = Literal["auto", "float16", "float32"]
OutputMode = Literal["base64", "file"]


def _get_bool(env_name: str, default: bool) -> bool:
    value = os.getenv(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    device_preference: DeviceOption = os.getenv("TTS_DEVICE", "auto")  # type: ignore
    dtype_preference: DTypeOption = os.getenv("TTS_DTYPE", "auto")  # type: ignore
    preload_models: bool = _get_bool("TTS_PRELOAD_MODELS", False)
    max_concurrency_per_model: int = int(
        os.getenv("TTS_MAX_CONCURRENCY_PER_MODEL", "1")
    )
    max_queue_size: int = int(os.getenv("TTS_MAX_QUEUE_SIZE", "32"))
    output_mode: OutputMode = os.getenv("TTS_OUTPUT_MODE", "base64")  # type: ignore
    output_dir: str = os.getenv("TTS_OUTPUT_DIR", "generated_audio")
    max_text_length: int = int(os.getenv("TTS_MAX_TEXT_LENGTH", "600"))
    enable_cache: bool = _get_bool("TTS_ENABLE_CACHE", False)
    cache_size: int = int(os.getenv("TTS_CACHE_SIZE", "32"))
    hf_home: str | None = os.getenv("HF_HOME")
    scalable_mode: bool = _get_bool("TTS_SCALABLE_MODE", True)
    default_sample_rate: int = int(os.getenv("TTS_DEFAULT_SAMPLE_RATE", "24000"))
    startup_time: float = time.time()


settings = Settings()
