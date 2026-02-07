from __future__ import annotations

import asyncio
import importlib
import logging
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .config import settings
from .devices import plan_device
from .utils_audio import ensure_mono, normalize_waveform
from .worker import ModelWorker

LOGGER = logging.getLogger(__name__)

VOICE_ALIAS_MAP: Dict[str, str] = {
    "custom_female": "vivian",
    "default_female": "vivian",
    "female": "vivian",
    "warm_female": "serena",
    "storyteller": "serena",
    "narrator": "serena",
    "custom_male": "aiden",
    "default_male": "aiden",
    "male": "aiden",
    "english_male": "ryan",
}

SUPPORTED_QWEN_SPEAKERS: List[str] = [
    "vivian",
    "serena",
    "uncle_fu",
    "dylan",
    "eric",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
]


def _to_numpy(audio: Any) -> np.ndarray:
    """Convert pipeline output (torch.Tensor or array-like) to float32 numpy."""
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    return arr


def canonical_voice_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = str(name).strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    mapped = VOICE_ALIAS_MAP.get(lowered, lowered)
    return mapped


def canonicalize_voice_list(voices: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for voice in voices:
        canonical = canonical_voice_name(voice)
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def augment_with_aliases(voices: List[str]) -> List[str]:
    canonical = [voice for voice in voices if voice]
    seen = set(canonical)
    aliases: List[str] = []
    for alias, target in VOICE_ALIAS_MAP.items():
        if target in seen and alias not in seen and alias not in aliases:
            aliases.append(alias)
    combined = canonical + aliases
    return combined


class QwenModelWrapper:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        fallback_voices: List[str] | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.fallback_voices = canonicalize_voice_list(fallback_voices or [])
        self.pipeline = self._load_pipeline()
        self._voice_cache: List[str] | None = None

    def _load_pipeline(self):  # pragma: no cover - depends on external repo
        qwen_module = importlib.import_module("qwen_tts")
        pipeline_cls = getattr(qwen_module, "Qwen3TTS", None)
        if pipeline_cls is None:
            pipeline_cls = getattr(qwen_module, "Qwen3TTSModel", None)
        if pipeline_cls is None:
            raise RuntimeError(
                "Official Qwen3-TTS pipeline not found. Install qwen_tts module."
            )
        kwargs = {}
        if settings.hf_home:
            kwargs["cache_dir"] = settings.hf_home
        if settings.hf_mirror:
            kwargs["mirror"] = settings.hf_mirror
        if pipeline_cls.__name__ == "Qwen3TTSModel":
            kwargs["torch_dtype"] = self.dtype
        pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)
        self._maybe_move_pipeline(pipeline)
        return pipeline

    def _maybe_move_pipeline(self, pipeline):  # pragma: no cover - device shim
        if pipeline is None:
            return
        if hasattr(pipeline, "to"):
            try:
                pipeline.to(self.device)
                return
            except Exception as exc:  # noqa: PERF203
                LOGGER.warning("Failed to move pipeline to %s: %s", self.device, exc)
        model = getattr(pipeline, "model", None)
        if hasattr(model, "to"):
            try:
                model.to(self.device)
            except Exception as exc:  # noqa: PERF203
                LOGGER.warning("Failed to move model to %s: %s", self.device, exc)

    async def synthesize(self, **kwargs) -> Tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._blocking_infer, kwargs)

    def _blocking_infer(self, kwargs: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        if hasattr(self.pipeline, "generate_custom_voice") or hasattr(
            self.pipeline, "generate_voice_design"
        ):
            wav, sample_rate = self._run_qwen3_model(kwargs)
            audio = ensure_mono(_to_numpy(wav))
            audio = normalize_waveform(audio)
            return audio, sample_rate

        result = self.pipeline.generate(
            text=kwargs["text"],
            voice=kwargs.get("voice"),
            language=kwargs.get("language"),
            tone=kwargs.get("tone"),
            seed=kwargs.get("seed"),
            speed=kwargs.get("speed", 1.0),
            sample_rate=kwargs.get("sample_rate"),
        )
        audio = ensure_mono(_to_numpy(result["audio"]))
        audio = normalize_waveform(audio)
        sample_rate = result.get("sample_rate", kwargs.get("sample_rate"))
        return audio, sample_rate

    def _run_qwen3_model(self, kwargs: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        model = getattr(self.pipeline, "model", None)
        model_type = getattr(model, "tts_model_type", "custom_voice")
        text = kwargs["text"]
        language = self._coerce_language(kwargs.get("language"))
        if model_type == "custom_voice":
            speaker = self._resolve_speaker(kwargs.get("voice"))
            wavs, sample_rate = self.pipeline.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                non_streaming_mode=True,
            )
            return wavs[0], sample_rate
        if model_type == "voice_design":
            instruct = kwargs.get("tone") or kwargs.get("voice") or ""
            wavs, sample_rate = self.pipeline.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                non_streaming_mode=True,
            )
            return wavs[0], sample_rate
        if model_type == "base":
            raise RuntimeError("Base voice-clone model is not supported yet")
        raise RuntimeError(f"Unsupported Qwen3-TTS model type {model_type}")

    def _resolve_speaker(self, requested: str | None) -> str:
        canonical = canonical_voice_name(requested)
        if canonical:
            return canonical
        if self._voice_cache:
            return self._voice_cache[0]
        if self.fallback_voices:
            return self.fallback_voices[0]
        return SUPPORTED_QWEN_SPEAKERS[0]

    @staticmethod
    def _coerce_language(language: Any) -> str:
        if not language:
            return "Auto"
        value = str(language).strip()
        if value.lower() == "auto":
            return "Auto"
        mapping = {
            "en": "English",
            "ja": "Japanese",
            "zh": "Chinese",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "ko": "Korean",
        }
        return mapping.get(value.lower(), value)

    def list_voices(self, refresh: bool = False) -> List[str]:
        if not refresh and self._voice_cache is not None:
            return self._voice_cache
        voices = canonicalize_voice_list(self._extract_voices())
        if not voices:
            voices = self.fallback_voices or SUPPORTED_QWEN_SPEAKERS
        voices = augment_with_aliases(voices)
        self._voice_cache = voices
        return voices

    def _extract_voices(self) -> List[str]:  # pragma: no cover - metadata path
        candidates: List[str] = []
        search_order = [
            getattr(self.pipeline, "list_voices", None),
            getattr(self.pipeline, "available_voices", None),
            getattr(self.pipeline, "get_available_voices", None),
            getattr(self.pipeline, "get_supported_speakers", None),
        ]
        pipeline_model = getattr(self.pipeline, "model", None)
        if pipeline_model is not None:
            search_order.append(getattr(pipeline_model, "get_supported_speakers", None))
        for entry in search_order:
            try:
                result = entry() if callable(entry) else entry
            except Exception as exc:  # noqa: PERF203 - optional metadata
                LOGGER.debug(
                    "Voice metadata probe failed for %s: %s", self.model_id, exc
                )
                continue
            voices = self._coerce_voice_payload(result)
            if voices:
                candidates = voices
                break
        if not candidates:
            config = getattr(self.pipeline, "config", None)
            config_voices = getattr(config, "voices", None)
            candidates = self._coerce_voice_payload(config_voices)
        return sorted(set(candidates))

    @staticmethod
    def _coerce_voice_payload(payload: Any) -> List[str]:
        if payload is None:
            return []
        if isinstance(payload, (list, tuple, set)):
            return [str(item) for item in payload if isinstance(item, (str, bytes))]
        if isinstance(payload, dict):
            keys = [str(key) for key in payload.keys()]
            if keys:
                return keys
            voices = payload.get("voices")
            if isinstance(voices, (list, tuple, set)):
                return [str(item) for item in voices]
        if isinstance(payload, str):
            return [payload]
        return []


class ModelManager:
    MODEL_IDS = {
        "qwen3-tts-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "qwen3-tts-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }
    DEFAULT_VOICES = {
        "qwen3-tts-0.6b": SUPPORTED_QWEN_SPEAKERS,
        "qwen3-tts-1.7b": SUPPORTED_QWEN_SPEAKERS,
    }

    def __init__(self):
        plan = plan_device(settings.device_preference, settings.dtype_preference)
        self.device = plan.device
        self.dtype = plan.dtype
        self.models: Dict[str, QwenModelWrapper] = {}
        self.workers: Dict[str, ModelWorker] = {}
        self._lock = Lock()
        LOGGER.info(
            "ModelManager initialized device=%s dtype=%s", self.device, self.dtype
        )

    def get_or_load(self, model: str) -> QwenModelWrapper:
        if model not in self.MODEL_IDS:
            raise ValueError(f"Unknown model {model}")
        if model not in self.models:
            with self._lock:
                if model not in self.models:
                    model_id = self.MODEL_IDS[model]
                    LOGGER.info("Loading model %s from %s", model, model_id)
                    self.models[model] = QwenModelWrapper(
                        model_id=model_id,
                        device=self.device,
                        dtype=self.dtype,
                        fallback_voices=self.DEFAULT_VOICES.get(model, []),
                    )
                    if settings.scalable_mode:
                        self.workers[model] = ModelWorker(
                            run_fn=self._worker_run,
                            max_queue=settings.max_queue_size,
                            workers=settings.max_concurrency_per_model,
                        )
        return self.models[model]

    async def ensure_workers(self) -> None:
        if not settings.scalable_mode:
            return
        await asyncio.gather(*[worker.start() for worker in self.workers.values()])

    async def synthesize(self, model_name: str, **kwargs) -> Tuple[np.ndarray, int]:
        wrapper = self.get_or_load(model_name)
        if settings.scalable_mode:
            worker = self.workers[model_name]
            await worker.start()
            if worker.queue_full():
                raise QueueFullError("queue full")
            return await worker.enqueue({"wrapper": wrapper, "kwargs": kwargs})
        return await wrapper.synthesize(**kwargs)

    async def _worker_run(self, payload: Dict[str, Any]):
        wrapper = payload["wrapper"]
        kwargs = payload["kwargs"]
        return await wrapper.synthesize(**kwargs)

    def preload_all(self):
        for model_name in self.MODEL_IDS:
            self.get_or_load(model_name)

    def available_voices(self, model_name: str, refresh: bool = False) -> List[str]:
        fallback = self.DEFAULT_VOICES.get(model_name, [])
        wrapper = self.models.get(model_name)
        if wrapper is None:
            if model_name not in self.MODEL_IDS:
                LOGGER.warning("Unknown model requested for voices: %s", model_name)
                return fallback
            try:
                wrapper = self.get_or_load(model_name)
            except Exception as exc:  # pragma: no cover - load failure path
                LOGGER.warning(
                    "Failed to load wrapper for voices %s: %s", model_name, exc
                )
                return fallback
        voices = wrapper.list_voices(refresh=refresh)
        if not voices:
            LOGGER.debug("No metadata voices found for %s, using fallback", model_name)
            return fallback
        return voices


class QueueFullError(Exception):
    pass


@lru_cache(maxsize=1)
def get_manager() -> ModelManager:
    manager = ModelManager()
    if settings.preload_models:
        manager.preload_all()
    return manager
