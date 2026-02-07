"""Utility for replaying a batch of TTS requests."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import httpx


def _load_jobs(path: Path) -> Sequence[Mapping[str, object]]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid input file
        raise SystemExit(f"Failed to parse {path}: {exc}") from exc
    if not isinstance(data, list):  # pragma: no cover - invalid input file
        raise SystemExit(f"Expected a list of jobs in {path}")
    return data


def _write_audio(
    request_id: str, payload: Mapping[str, object], out_dir: Path, client: httpx.Client
) -> Path:
    audio_format = payload.get("audio_format")
    if not audio_format:
        raise RuntimeError(f"Response for {request_id} missing audio_format")
    out_path = out_dir / f"{request_id}.{audio_format}"
    encoded = payload.get("audio_base64")
    if encoded:
        out_path.write_bytes(base64.b64decode(str(encoded)))
        return out_path

    audio_url = payload.get("audio_url")
    if not audio_url:
        raise RuntimeError(f"Response for {request_id} missing audio payload")
    audio = client.get(str(audio_url))
    audio.raise_for_status()
    out_path.write_bytes(audio.content)
    return out_path


def run_batch(
    api_base: str,
    job_file: Path,
    output_dir: Path,
    timeout: float,
) -> Iterable[Path]:
    jobs = _load_jobs(job_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=api_base, timeout=timeout) as client:
        for job in jobs:
            req_id = str(job.get("request_id") or "")
            if not req_id:
                raise SystemExit("Each job must include request_id")
            response = client.post("/v1/tts", json=job)
            response.raise_for_status()
            out_path = _write_audio(req_id, response.json(), output_dir, client)
            yield out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a batch of TTS requests")
    parser.add_argument("batch", type=Path, help="Path to JSON array of TTS requests")
    parser.add_argument(
        "--api-base", default="http://127.0.0.1:8000", help="Backend base URL"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/qwen_batch_outputs"),
        help="Directory for synthesized audio",
    )
    parser.add_argument(
        "--timeout", type=float, default=300, help="Request timeout seconds"
    )
    args = parser.parse_args(argv)

    try:
        for path in run_batch(args.api_base, args.batch, args.out_dir, args.timeout):
            print(f"saved {path}")
        return 0
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - general failure
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
