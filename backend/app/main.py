from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import api
from .config import settings
from .logging import setup_logging
from .model_manager import get_manager


def create_app() -> FastAPI:
    setup_logging(getattr(logging, settings.log_level.upper(), logging.INFO))
    app = FastAPI(title="Qwen3-TTS Service")
    app.include_router(api.router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        manager = get_manager()
        if settings.preload_models:
            manager.preload_all()
        await manager.ensure_workers()

    return app


app = create_app()
