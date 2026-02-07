# Qwen3-TTS on macOS

This repo bundles a FastAPI backend powered by the official Qwen3-TTS models and a React/Vite frontend for interactive synthesis.

## Requirements

- Python 3.11+
- Node.js 18+
- Apple Silicon with macOS (optimized for MPS)

## Backend Setup

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pip install git+https://github.com/QwenLM/Qwen3-TTS.git
uvicorn app.main:app --reload
```

Configure env vars like `TTS_DEVICE`, `TTS_OUTPUT_MODE`, `TTS_PRELOAD_MODELS`. See `backend/README.md` for details.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE` if backend runs on a different origin.

## First Run

1. Start backend with `uvicorn`.
2. Start frontend via `npm run dev`.
3. Visit the Vite dev server, input text, choose model, and click Generate.

## Tests & Benchmark

```bash
cd backend
pytest
python scripts/benchmark.py --requests 5 --concurrency 1
```
