from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict


@dataclass
class SynthesisTask:
    payload: Dict[str, Any]
    future: asyncio.Future


class ModelWorker:
    def __init__(
        self,
        run_fn: Callable[[Dict[str, Any]], Awaitable[Any]],
        max_queue: int,
        workers: int,
    ):
        self.queue: asyncio.Queue[SynthesisTask] = asyncio.Queue(maxsize=max_queue)
        self.run_fn = run_fn
        self.worker_tasks: list[asyncio.Task] = []
        self.num_workers = max(1, workers)

    async def start(self) -> None:
        if self.worker_tasks:
            return
        for _ in range(self.num_workers):
            self.worker_tasks.append(asyncio.create_task(self._worker_loop()))

    async def _worker_loop(self) -> None:
        while True:
            task = await self.queue.get()
            try:
                result = await self.run_fn(task.payload)
                task.future.set_result(result)
            except Exception as exc:  # pragma: no cover - runtime path
                task.future.set_exception(exc)
            finally:
                self.queue.task_done()

    async def enqueue(self, payload: Dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        task = SynthesisTask(payload=payload, future=future)
        self.queue.put_nowait(task)
        return await future

    def queue_full(self) -> bool:
        return self.queue.full()
