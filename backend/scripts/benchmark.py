from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import List

import httpx


async def _run_request(client: httpx.AsyncClient, url: str, payload: dict) -> float:
    start = time.perf_counter()
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


async def run_benchmark(host: str, n: int, concurrency: int) -> None:
    url = f"{host}/v1/tts"
    payload = {
        "text": "Benchmarking Qwen3 TTS",
        "model": "qwen3-tts-0.6b",
        "language": "en",
    }
    latencies: List[float] = []
    limits = httpx.Limits(
        max_keepalive_connections=concurrency, max_connections=concurrency
    )
    timeout = httpx.Timeout(120.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for i in range(0, n, concurrency):
            batch = min(concurrency, n - i)
            tasks = [
                asyncio.create_task(_run_request(client, url, payload))
                for _ in range(batch)
            ]
            latencies.extend(await asyncio.gather(*tasks))
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies))]
    print(f"Benchmark against {url}")
    print(f"Requests: {n}, Concurrency: {concurrency}")
    print(f"p50: {p50:.2f} ms, p95: {p95:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTS backend")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.host, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
