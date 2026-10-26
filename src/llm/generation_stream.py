"""Async generation consumption with bounded isolation for legacy producers."""

import asyncio
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any


class GenerationBusyError(RuntimeError):
    """A cancelled synchronous producer still owns its worker/backend."""


@dataclass
class _SyncRun:
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)
    cancelled: threading.Event = field(default_factory=threading.Event)
    chunks: deque = field(default_factory=deque)
    done: bool = False
    error: BaseException | None = None


class GuardedGeneration:
    """Share the original producer while checking privacy at every async boundary."""

    def __init__(self, source, available: Callable[[], bool]):
        self.source, self.available = source, available
        self.max_chars = source.max_chars

    @property
    def sync_busy(self) -> bool:
        """Retain the original producer's admission and actual work ownership."""
        return self.source.sync_busy

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[dict]:
        """Deny network admission and discard chunks from invalidated inputs."""
        if not self.available():
            raise asyncio.CancelledError
        async with aclosing(self.source.astream(prompt, admission=self.available, **kwargs)) as stream:
            async for chunk in stream:
                if not self.available():
                    raise asyncio.CancelledError
                yield chunk
        if not self.available():
            raise asyncio.CancelledError


class GenerationStream:
    """Prefer native async I/O; retain at most one legacy producer per owner.

    Cancelling a thread cannot interrupt a blocking model or subprocess call.
    Its run stays isolated, retains only a bounded queue, and rejects new sync
    generations until that producer actually exits. Late chunks are discarded.
    Consumers must close this async generator when leaving it before EOF.
    """

    def __init__(self, reasoner: Any, *, queue_capacity: int = 2, max_chars: int = 65536):
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError("invalid_generation_queue_capacity")
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("invalid_generation_limit")
        self.reasoner = reasoner
        self.queue_capacity = queue_capacity
        self.max_chars = max_chars
        self._producer_lock = threading.Lock()

    @property
    def sync_busy(self) -> bool:
        """Whether a synchronous producer still owns this backend."""
        return self._producer_lock.locked()

    async def astream(self, prompt: str, *, admission: Callable[[], bool] | None = None, **kwargs) -> AsyncIterator[dict]:
        """Yield text deltas without synchronous inference on the event loop."""
        if admission is not None and not admission():
            raise asyncio.CancelledError
        native = getattr(self.reasoner, "astream", None)
        admitted = getattr(self.reasoner, "astream_with_admission", None) if admission is not None else None
        if callable(admitted):
            native = admitted
            kwargs["admission"] = admission
        if callable(native):
            async with aclosing(native(prompt, max_chars=self.max_chars, **kwargs)) as source:
                async for chunk in source:
                    yield chunk
            return

        if not self._producer_lock.acquire(blocking=False):
            raise GenerationBusyError("previous_generation_still_running")
        run = _SyncRun()
        loop = asyncio.get_running_loop()

        def notify() -> None:
            try:
                loop.call_soon_threadsafe(run.wake.set)
            except RuntimeError:
                run.cancelled.set()

        def produce() -> None:
            source = None
            try:
                if admission is not None and not admission():
                    return
                streaming = getattr(self.reasoner, "generate_stream", None)
                if callable(streaming):
                    source = iter(streaming(prompt, **kwargs))
                else:
                    source = iter([{"token": self.reasoner.generate(prompt, **kwargs)}])
                total = 0
                while not run.cancelled.is_set() and (admission is None or admission()):
                    try:
                        chunk = next(source)
                    except StopIteration:
                        break
                    if run.cancelled.is_set():
                        break
                    token = chunk.get("token", "")
                    if type(token) is not str:
                        raise ValueError("invalid_stream_text")
                    total += len(token)
                    if total > self.max_chars:
                        raise ValueError("generation_text_limit")
                    # Do not retain provider objects or repeated accumulated text.
                    with run.condition:
                        while len(run.chunks) >= self.queue_capacity and not run.cancelled.is_set():
                            run.condition.wait()
                        if run.cancelled.is_set():
                            break
                        run.chunks.append({"token": token})
                    notify()
            except BaseException as error:
                run.error = error
            finally:
                try:
                    close = getattr(source, "close", None)
                    if callable(close):
                        close()
                except BaseException as error:
                    run.error = run.error or error
                finally:
                    with run.condition:
                        run.done = True
                    self._producer_lock.release()
                    notify()

        try:
            threading.Thread(target=produce, name="kza-generation", daemon=True).start()
        except BaseException:
            self._producer_lock.release()
            raise
        try:
            while True:
                run.wake.clear()
                with run.condition:
                    if run.chunks:
                        chunk = run.chunks.popleft()
                        run.condition.notify_all()
                    elif run.done:
                        if run.error is not None:
                            raise run.error
                        return
                    else:
                        chunk = None
                if chunk is None:
                    await run.wake.wait()
                else:
                    yield chunk
                    await asyncio.sleep(0)
        finally:
            run.cancelled.set()
            with run.condition:
                run.chunks.clear()
                run.condition.notify_all()
