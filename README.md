# Qwen3-TTS on macOS

Local-first Text-to-Speech playground that wraps the **official Qwen3-TTS CustomVoice models** with a production-shaped FastAPI backend and a React/Vite frontend optimized for Apple Silicon (MPS). The goal is to ship realistic infra pieces—queue-based workers, caching, flexible storage, observability, and a descriptive operator-friendly README—so you can adapt it directly for internal tools or demos.

## Architecture Snapshot

- **Backend (`backend/`)** – FastAPI + asyncio workers. `ModelManager` lazily loads Qwen pipelines, pins them to MPS/CPU depending on availability, and exposes `/v1/tts`, `/v1/voices`, `/v1/audio/{id}`, and `/health`. Supports base64 or file download responses, optional LRU cache, queue limits, and structured logging.
- **Frontend (`frontend/`)** – Vite + React UI nicknamed “Control Room”. Includes explanatory comments for each UI block, error/request-id surfacing, and playback that handles both inline base64 audio and file URLs served by the backend.
- **Scripts & Tooling** – `scripts/benchmark.py` for async load tests, pytest suite with model mocking, and configuration surfaces for caching, queueing, and HuggingFace mirrors.

## Requirements

- macOS on Apple Silicon strongly recommended (MPS acceleration). Works on Intel/macOS but without GPU.
- Python 3.11 (backend). The repository standardizes on `backend/.venv3.11` so the official `qwen-tts` wheels install cleanly.
- Node.js ≥ 18 (frontend).
- FFmpeg installed if you plan to output MP3 (needed by `pydub`).

## Backend Setup

```bash
cd backend
python3.11 -m venv .venv3.11
source .venv3.11/bin/activate
pip install -e '.[dev]'

# Install official Qwen3-TTS pipeline
pip install git+https://github.com/QwenLM/Qwen3-TTS.git

# Authenticate with Hugging Face before the first download
export HF_TOKEN=<hf_xxx>

uvicorn app.main:app --reload
```

> Tip: if you forget to activate the virtualenv, prefix commands with `backend/.venv3.11/bin/...`.

Key environment toggles (see `backend/app/config.py` for defaults):

| Variable | Description |
| --- | --- |
| `TTS_DEVICE` | `auto`\|`mps`\|`cpu`. Auto prefers MPS when available.
| `TTS_DTYPE` | `auto` chooses float16 on MPS, float32 otherwise.
| `TTS_PRELOAD_MODELS` | Load all models at startup to avoid first-request penalty.
| `TTS_SCALABLE_MODE` | Enable asyncio worker queues per model (default `true`).
| `TTS_MAX_QUEUE_SIZE`, `TTS_MAX_CONCURRENCY_PER_MODEL` | Backpressure + concurrency settings.
| `TTS_OUTPUT_MODE` | `base64` or `file`. When `file`, audio is written under `TTS_OUTPUT_DIR` and served via `/v1/audio/{id}`.
| `TTS_ENABLE_CACHE` / `TTS_CACHE_SIZE` | Simple LRU cache for deduped requests.
| `HF_TOKEN` | Hugging Face token required to download Qwen3 checkpoints.
| `HF_HOME`, `TTS_HF_MIRROR` | Control model weight cache/mirror paths.

### Qwen3-TTS Weights

The backend never ships custom checkpoints. Install the official repo (above) and authenticate with HuggingFace/ModelScope the same way you normally would. `ModelManager.MODEL_IDS` points at `Qwen/Qwen3-TTS-*-CustomVoice` by default; override the map if you need different SKUs.

## Frontend Setup

```bash
cd frontend
npm install
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

`VITE_API_BASE` defaults to `http://127.0.0.1:8000` but can target remote hosts when tunneling to another machine.

## Running the Stack

1. Start backend: `cd backend && source .venv3.11/bin/activate && HF_TOKEN=<token> uvicorn app.main:app --reload`.
2. Start frontend: `cd frontend && npm run dev`.
3. Visit the dev server (typically `http://127.0.0.1:5173`). The Control Room explains each field (text, model, voice, language, tone, output format, speed) and shows recent responses (request_id, errors, audio player).

When `TTS_OUTPUT_MODE=file`, the UI automatically switches to audio URLs sourced from `/v1/audio/{uuid}.ext` to avoid base64 overhead.

## Testing & Benchmarking

```bash
cd backend
source .venv/bin/activate
pytest  # unit + integration tests (models patched)

# lightweight latency check (uses httpx AsyncClient)
python scripts/benchmark.py --host http://127.0.0.1:8000 --requests 10 --concurrency 2
```

The smoke tests monkeypatch `ModelManager.synthesize` so they remain fast and avoid model downloads. Additional tests cover cache hits and the `/v1/voices` metadata path.

## Observability & Ops Notes

- Logging is initialized via `app/logging.py` and respects `TTS_LOG_LEVEL`. Requests log `request_id`, model, voice, format, cache stats, and completion latency.
- LRU cache exposes hit/miss counters in logs; tune `TTS_CACHE_SIZE` based on max concurrency.
- Worker queues emit `429 queue full` when saturated—monitor these logs to tweak `TTS_MAX_QUEUE_SIZE` or concurrency per model.
- `/health` reports uptime, device, and which models are currently loaded. `/v1/voices?refresh=true` forces a metadata refresh straight from the Qwen pipeline.

## Troubleshooting

- **401 Hugging Face download errors** – export `HF_TOKEN` (or run `huggingface-cli login`) before calling `uvicorn` so the weights can be fetched.
- **MP3 output fails** – Install `pydub` + FFmpeg (`brew install ffmpeg`). Without it, stick to WAV.
- **MPS dtype errors** – Some PyTorch versions need `PYTORCH_ENABLE_MPS_FALLBACK=1`. Set via env if you see “Metal device set to CPU fallback”.
- **Large output_dir growth** – If `TTS_OUTPUT_MODE=file`, ensure a cron/job cleans up `generated_audio/` or point `TTS_OUTPUT_DIR` to a tmpfs.
- **Voice metadata empty** – The backend falls back to a static voice list if the Qwen pipeline does not expose `list_voices`. Use `refresh=true` to force another probe after updating weights.

## Batch Testing

After the backend is up, replay any JSON array of `/v1/tts` payloads (e.g., `/tmp/tts_batch.json`) and capture the audio outputs:

```bash
cd backend
source .venv3.11/bin/activate
qwen3-tts-batch /tmp/tts_batch.json --out-dir /tmp/qwen_outputs
```

Each entry produces `request_id.<format>` files inside the chosen output directory, using the same request IDs defined in the batch file.

## What’s Next

- Expand documentation for deployment targets (Docker, uvicorn workers) and add CI that runs pytest + frontend typechecking.
- Add streaming support once Qwen exposes incremental audio.
- Layer structured logging (JSON) and tracing hooks for queue times.

Jump into `backend/README.md` and `frontend/README.md` for focused setup notes, or open `scripts/benchmark.py` to stress the service locally. Contributions and issue reports welcome.
