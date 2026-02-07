# Qwen3-TTS Frontend

Vite + React single-page app that acts as an operator dashboard for the backend.

## Setup

```bash
cd frontend
npm install
```

## Running

```bash
# Optional override if backend is remote
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

## UI Overview

- **Hero banner** – Introduces the “Control Room” concept.
- **Form panel** – Textarea plus grid selectors for model, voice, language, tone, output format, and speed. Each block carries inline comments (see `src/App.tsx`).
- **Submission button** – Shows “Generating…” state while awaiting backend response.
- **Error + request ID** – Displays FastAPI errors (including 429 queue full) and exposes `request_id` for debugging.
- **Playback panel** – Renders `<audio>` element for base64 responses or links to `/v1/audio/{id}` when the backend serves files.
- **Examples table** – Click rows to load preset text/voice/language combos to speed up demos.

## Notes

- `src/api.ts` centralizes axios calls; it’s easy to swap for fetch if desired.
- When `TTS_OUTPUT_MODE=file`, the frontend concatenates `audio_url` with `VITE_API_BASE` to fetch audio bytes.
- State hooks track errors separately so logs don’t get lost once playback starts.
