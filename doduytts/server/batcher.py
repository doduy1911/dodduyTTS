"""Dynamic micro-batcher: gathers concurrent TTS requests into batches.

A single worker loop pulls the first waiting request, then keeps collecting
until either ``max_batch_size`` requests are gathered or ``window_ms`` has
elapsed since the first one — then runs the whole batch through the engine
in one ``generate()`` call. Requests submitted while a batch is generating
queue up for the next batch, so the model is never called concurrently.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("doduytts.server")


@dataclass
class TTSRequest:
    text: str
    language: Optional[str] = None
    instruct: Optional[str] = None
    voice_clone_prompt: Any = None
    session: str = ""
    created_at: float = field(default_factory=time.perf_counter)


class MicroBatcher:
    def __init__(
        self,
        run_batch: Callable[[list[TTSRequest]], list[np.ndarray]],
        max_batch_size: int = 4,
        window_ms: float = 20.0,
    ):
        self._run_batch = run_batch
        self.max_batch_size = max_batch_size
        self.window_s = window_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        # simple stats for observability
        self.batches = 0
        self.requests = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._worker())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, request: TTSRequest) -> np.ndarray:
        """Queue one request and wait for its generated audio."""
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((request, fut))
        return await fut

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self.window_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(
                        await asyncio.wait_for(self._queue.get(), remaining)
                    )
                except asyncio.TimeoutError:
                    break

            requests = [req for req, _ in batch]
            t0 = time.perf_counter()
            try:
                results = await asyncio.to_thread(self._run_batch, requests)
            except Exception as exc:  # propagate to every waiter in the batch
                logger.exception("batch generation failed (%d items)", len(batch))
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(RuntimeError(str(exc)))
                continue
            dt = time.perf_counter() - t0
            self.batches += 1
            self.requests += len(batch)
            logger.info(
                "batch #%d: size=%d gen=%.2fs texts=%s",
                self.batches,
                len(batch),
                dt,
                [r.text[:30] for r in requests],
            )
            for (_, fut), audio in zip(batch, results):
                if not fut.done():
                    fut.set_result(audio)
