# Qwen3-TTS Backend

## Setup

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Install the official Qwen3-TTS repo:

```bash
pip install git+https://github.com/QwenLM/Qwen3-TTS.git
```

Download weights via the instructions in the upstream repo (HF identifiers referenced in `model_manager.py`).

## Running

```bash
uvicorn app.main:app --reload
```

Configure via environment variables:

- `TTS_DEVICE` (`auto|mps|cpu`)
- `TTS_DTYPE` (`auto|float16|float32`)
- `TTS_PRELOAD_MODELS`
- `TTS_OUTPUT_MODE` (`base64|file`)
- `TTS_OUTPUT_DIR`
- `TTS_MAX_QUEUE_SIZE`
- `TTS_MAX_TEXT_LENGTH`

## Tests

```bash
pytest
```

## Benchmark

```bash
python scripts/benchmark.py --host http://127.0.0.1:8000 --requests 10 --concurrency 2
```
