# Qwen3-TTS Backend

Production-shaped FastAPI service that orchestrates the official Qwen3-TTS pipelines on Apple Silicon hardware.

## Features at a Glance

- Lazy model loading with shared asyncio workers per model (`ModelManager`).
- Device/dtype planning that prefers MPS + float16, with logging of chosen plan.
- `/v1/tts` endpoint handles base64 or file output, queue limits, cache hits, and request IDs.
- `/v1/voices` dynamically queries the Qwen pipeline metadata (falls back to static voices when unavailable).
- `/health` reports uptime, device, and which models are loaded.
- Optional LRU cache with hit/miss stats, file storage backend, and structured logging hooks.
- Benchmark + pytest suite (model generation mocked for speed).

## Environment Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `TTS_DEVICE` | `auto` | `auto`, `mps`, or `cpu`. Auto prefers MPS. |
| `TTS_DTYPE` | `auto` | Float16 enforced on MPS unless overridden. |
| `TTS_PRELOAD_MODELS` | `false` | Load all models at startup. |
| `TTS_SCALABLE_MODE` | `true` | Enables queue/worker layer. |
| `TTS_MAX_QUEUE_SIZE` | `32` | Pending requests per model in queue. |
| `TTS_MAX_CONCURRENCY_PER_MODEL` | `1` | Worker count per model. |
| `TTS_OUTPUT_MODE` | `base64` | `file` writes WAV/MP3 to `TTS_OUTPUT_DIR`. |
| `TTS_ENABLE_CACHE` | `false` | Cache identical requests with LRU eviction. |
| `TTS_CACHE_SIZE` | `32` | Entries stored in cache. |
| `TTS_MAX_TEXT_LENGTH` | `600` | Enforced via Pydantic validator. |
| `TTS_DEFAULT_SAMPLE_RATE` | `24000` | Requested sample rate passed to pipeline. |
| `TTS_LOG_LEVEL` | `INFO` | Logging level for `setup_logging`. |
| `HF_TOKEN` | unset | Hugging Face token required for Qwen downloads. |
| `HF_HOME` | unset | Optional HuggingFace cache directory. |
| `TTS_HF_MIRROR` | unset | Mirror URL forwarded to Qwen loader. |

## Setup

```bash
cd backend
python3.11 -m venv .venv3.11
source .venv3.11/bin/activate
pip install -e '.[dev]'

# Install official Qwen3-TTS wrapper (requires Python ≥3.10 for latest deps)
pip install git+https://github.com/QwenLM/Qwen3-TTS.git

# If accelerate>=1.12.0 fails, install accelerate==1.10.1 manually
```

## Running Locally

```bash
source .venv3.11/bin/activate
export HF_TOKEN=<hf_xxx>
uvicorn app.main:app --reload
```

Key endpoints:

- `POST /v1/tts` – synthesize (accepts `stream=false` only today).
- `GET /v1/voices?model=qwen3-tts-0.6b` – voice metadata; `refresh=true` forces re-probe.
- `GET /v1/audio/{id}` – served files when `TTS_OUTPUT_MODE=file`.
- `GET /health` – uptime/device diagnostics.

## Testing & Benchmarking

```bash
source .venv3.11/bin/activate
pytest
python scripts/benchmark.py --host http://127.0.0.1:8000 --requests 10 --concurrency 2
```

Tests patch out `ModelManager.synthesize` and storage to avoid heavy downloads. Add additional cases under `backend/tests/` to exercise more queue/caching scenarios.

## Notes on Weights and Storage

- `ModelManager.MODEL_IDS` maps friendly names to `Qwen/Qwen3-TTS-*-CustomVoice`. Update if you maintain private weights.
- When `TTS_OUTPUT_MODE=file`, audio is stored under `generated_audio/` (gitignored). Use `TTS_OUTPUT_DIR` to relocate or mount to tmp storage; consider cleanup jobs for long-running instances.
- Voice metadata is best-effort: the code inspects `pipeline.list_voices`, `pipeline.available_voices`, or config metadata. Static fallbacks ensure the endpoint remains functional even when metadata is absent.

## Troubleshooting

- **Accelerate / transformers pinning** – Keep the upstream repo requirements satisfied. If your system python is <3.10, stick to the last versions compatible with 3.9 (e.g., `accelerate==1.10.1`).
- **MPS fallback logs** – Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if you see runtime warnings about CPU fallback.
- **QueueFull (429)** – Increase `TTS_MAX_QUEUE_SIZE` or concurrency per model, or disable scalable mode entirely for simplicity.

For design details inspect `app/model_manager.py`, `app/api.py`, and `app/worker.py`. Open issues/PRs welcome!
