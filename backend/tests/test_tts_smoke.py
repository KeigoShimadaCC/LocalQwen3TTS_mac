from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

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
