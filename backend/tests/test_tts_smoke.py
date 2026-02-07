from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import api as api_module
from app.api import AudioCache
from app.main import app


client = TestClient(app)


@patch("app.model_manager.ModelManager.synthesize")
def test_tts_smoke(mock_synth):
    mock_synth.return_value = (np.zeros(24000), 24000)
    payload = {
        "text": "hello world",
        "model": "qwen3-tts-0.6b",
        "language": "en",
    }
    resp = client.post("/v1/tts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["audio_format"] == "wav"
    assert "audio_base64" in data


def test_tts_cache_hit(monkeypatch):
    test_cache = AudioCache(max_size=4)
    monkeypatch.setattr(api_module, "cache", test_cache)

    # Pre-populate cache with expected key payload
    cache_key = (
        "qwen3-tts-0.6b",
        "hello world",
        None,
        "en",
        None,
        1.0,
        None,
        24000,
        "wav",
    )
    wav_bytes = b"RIFF...."
    test_cache.put(cache_key, (wav_bytes, "wav", 24000, 1.0))

    payload = {"text": "hello world", "model": "qwen3-tts-0.6b", "language": "en"}
    resp = client.post("/v1/tts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["audio_format"] == "wav"
    assert data["audio_base64"] is not None


def test_list_voices_endpoint(monkeypatch):
    class DummyManager:
        def available_voices(self, model_name: str, refresh: bool = False):
            assert model_name == "qwen3-tts-0.6b"
            assert refresh is False
            return ["v1", "v2"]

    monkeypatch.setattr(api_module, "get_manager", lambda: DummyManager())
    resp = client.get("/v1/voices")
    assert resp.status_code == 200
    assert resp.json()["voices"] == ["v1", "v2"]
