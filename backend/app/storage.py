from __future__ import annotations

import os
import pathlib
import uuid
from typing import Literal

from fastapi import HTTPException
from starlette.responses import FileResponse

from .config import settings


class AudioStorage:
    def __init__(self, mode: Literal["base64", "file"], base_dir: str):
        self.mode = mode
        self.base_path = pathlib.Path(base_dir)
        if mode == "file":
            self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, audio_bytes: bytes, extension: str) -> str:
        file_id = f"{uuid.uuid4().hex}.{extension}"
        file_path = self.base_path / file_id
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        return file_id

    def serve(self, file_name: str) -> FileResponse:
        file_path = self.base_path / file_name
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio not found")
        return FileResponse(file_path)


storage = AudioStorage(settings.output_mode, settings.output_dir)
