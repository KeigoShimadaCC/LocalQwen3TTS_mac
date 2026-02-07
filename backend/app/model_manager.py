from __future__ import annotations

import asyncio
import importlib
import logging
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .config import settings
from .utils_audio import ensure_mono, normalize_waveform
from .worker import ModelWorker

LOGGER = logging.getLogger(__name__)


class QwenModelWrapper:
    def __init__(self, model_id: str, device: torch.device, dtype: torch.dtype):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):  # pragma: no cover - depends on external repo
        qwen_module = importlib.import_module("qwen_tts")
        pipeline_cls = getattr(qwen_module, "Qwen3TTS", None)
        if pipeline_cls is None:
            raise RuntimeError(
                "Official Qwen3-TTS pipeline not found. Install qwen_tts module."
            )
        kwargs = {"device": self.device.type}
        if settings.hf_home:
            kwargs["cache_dir"] = settings.hf_home
        pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)
        pipeline.to(self.device.type)
        return pipeline

    async def synthesize(self, **kwargs) -> Tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._blocking_infer, kwargs)

    def _blocking_infer(self, kwargs: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        result = self.pipeline.generate(
            text=kwargs["text"],
            voice=kwargs.get("voice"),
            language=kwargs.get("language"),
            tone=kwargs.get("tone"),
            seed=kwargs.get("seed"),
            speed=kwargs.get("speed", 1.0),
            sample_rate=kwargs.get("sample_rate"),
        )
        audio = ensure_mono(np.array(result["audio"], dtype=np.float32))
        audio = normalize_waveform(audio)
        sample_rate = result.get("sample_rate", kwargs.get("sample_rate"))
        return audio, sample_rate


class ModelManager:
    MODEL_IDS = {
        "qwen3-tts-0.6b": "Qwen/Qwen3-TTS-0.6B-CustomVoice",
        "qwen3-tts-1.7b": "Qwen/Qwen3-TTS-1.7B-CustomVoice",
    }

    def __init__(self):
        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()
        self.models: Dict[str, QwenModelWrapper] = {}
        self.workers: Dict[str, ModelWorker] = {}
        self._lock = Lock()
        LOGGER.info(
            "ModelManager initialized device=%s dtype=%s", self.device, self.dtype
        )

    def _resolve_device(self) -> torch.device:
        pref = settings.device_preference
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self) -> torch.dtype:
        pref = settings.dtype_preference
        if pref == "float16":
            return torch.float16
        if pref == "float32":
            return torch.float32
        if self.device.type == "mps":
            return torch.float16
        return torch.float32

    def get_or_load(self, model: str) -> QwenModelWrapper:
        if model not in self.MODEL_IDS:
            raise ValueError(f"Unknown model {model}")
        if model not in self.models:
            with self._lock:
                if model not in self.models:
                    model_id = self.MODEL_IDS[model]
                    LOGGER.info("Loading model %s from %s", model, model_id)
                    self.models[model] = QwenModelWrapper(
                        model_id, self.device, self.dtype
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


class QueueFullError(Exception):
    pass


@lru_cache(maxsize=1)
def get_manager() -> ModelManager:
    manager = ModelManager()
    if settings.preload_models:
        manager.preload_all()
    return manager
