#!/usr/bin/env python3
"""
Check backend env, model availability, and a minimal TTS run.
Run from backend/ with:
  source .venv3.11/bin/activate
  set -a && source ../.env.local && set +a   # optional: load HF_TOKEN
  python scripts/check_backend.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env.local from repo root if present
repo_root = Path(__file__).resolve().parents[2]
env_local = repo_root / ".env.local"
if env_local.exists():
    for line in env_local.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def main() -> int:
    print("=== Backend environment check ===\n")

    # 1. Python & cwd
    print(f"Python: {sys.version.split()[0]}")
    print(f"CWD: {os.getcwd()}\n")

    # 2. HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    print(f"HF_TOKEN: {'set (' + str(len(hf_token)) + ' chars)' if hf_token else 'NOT SET'}")
    if not hf_token:
        print("  -> Optional for cached models; required for first download.\n")
    else:
        print()

    # 3. Config
    try:
        from app.config import settings
        print("Config (env):")
        print(f"  TTS_DEVICE: {settings.device_preference}")
        print(f"  TTS_DTYPE: {settings.dtype_preference}")
        print(f"  TTS_OUTPUT_MODE: {settings.output_mode}")
        print(f"  HF_HOME: {settings.hf_home or '(default)'}\n")
    except Exception as e:
        print(f"Config load failed: {e}\n")
        return 1

    # 4. Model cache (HF_HOME = cache root, hub/ is inside it)
    hf_root = Path(settings.hf_home or os.path.expanduser("~/.cache/huggingface"))
    hub_dir = hf_root / "hub"
    for name, model_id in [("0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"), ("1.7B", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")]:
        slug = "models--" + model_id.replace("/", "--")
        path = hub_dir / slug
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        print(f"Model {name} ({model_id}): {'cached' if exists else 'not in cache'}")
    print()

    # 5. Manager + health
    try:
        from app.model_manager import get_manager
        manager = get_manager()
        print("ModelManager:")
        print(f"  device: {manager.device}")
        print(f"  dtype: {manager.dtype}")
        print(f"  models_loaded: {list(manager.models.keys()) or '(none yet)'}\n")
    except Exception as e:
        print(f"ModelManager init failed: {e}\n")
        return 1

    # 6. Minimal TTS (load one model + synthesize short text)
    print("TTS smoke test (qwen3-tts-0.6b, 'Hi')...")
    try:
        import asyncio
        async def run():
            audio, sr = await manager.synthesize(
                model_name="qwen3-tts-0.6b",
                text="Hi",
                voice=None,
                language="en",
                tone=None,
                seed=None,
                speed=1.0,
                sample_rate=24000,
            )
            return audio, sr
        audio, sr = asyncio.run(run())
        print(f"  -> OK: audio shape={getattr(audio, 'shape', len(audio))}, sample_rate={sr}\n")
    except Exception as e:
        print(f"  -> FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    print("=== All checks passed ===\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
