"""Model slot base: one model instance, its own execution lane and explicit lifecycle.

A slot owns exactly one model object. Load, inference, unload and reload all run
on the slot's own standard ``ThreadPoolExecutor(max_workers=1)`` (or, for the CPU
embedding slot, through ``asyncio.to_thread`` serialised by a lock), so a model is
never touched by two threads. ``reload()`` drains in-flight work, unloads
(``unload_model()``/``close()`` + ``del`` + ``gc``) and recreates the model on that
same lane. A failed (re)load leaves the slot ``UNAVAILABLE`` until an explicit
``ReloadModel``: there is no lazy reload on the request path.
"""

import asyncio
import concurrent.futures
import gc
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SlotState(Enum):
    """Lifecycle state of a model slot."""

    LOADING = "loading"
    READY = "ready"
    RELOADING = "reloading"
    UNAVAILABLE = "unavailable"


class InvalidInputError(ValueError):
    """Request payload rejected (maps to ``INVALID_ARGUMENT``)."""


class SlotUnavailableError(RuntimeError):
    """Slot is not loaded / failed / reloading (maps to ``UNAVAILABLE``)."""


class ResourceExhaustedError(RuntimeError):
    """Queue full or device out of memory (maps to ``RESOURCE_EXHAUSTED``)."""


class ExecutorKind(Enum):
    """How a slot runs blocking model calls."""

    DEDICATED = "dedicated"  # ThreadPoolExecutor(max_workers=1) owned by the slot
    TO_THREAD = "to_thread"  # asyncio.to_thread, serialised by a threading.Lock


@dataclass(frozen=True)
class SlotSnapshot:
    """Point-in-time health view of a slot (no request data)."""

    slot_id: str
    model_name: str
    device: str
    state: SlotState
    consecutive_failures: int
    vram_mib: int


def is_out_of_memory(exc: BaseException) -> bool:
    """Duck-typed OOM detection (torch ``OutOfMemoryError``, CTranslate2/CUDA messages)."""
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text or "alloc" in text)


def _default_unload(model: Any) -> None:
    """Release a model's native resources before dropping the reference.

    CTranslate2 Whisper exposes ``model.model.unload_model()``; adapters may expose
    ``close()``. Both are optional.
    """
    inner = getattr(model, "model", None)
    unload = getattr(inner, "unload_model", None)
    if callable(unload):
        unload()
    close = getattr(model, "close", None)
    if callable(close):
        close()


class ModelSlot:
    """Owns one model instance and serialises every access to it.

    Args:
        slot_id: Stable identifier exposed in ``GetHealth`` and ``ReloadModel``.
        model_name: Human-readable model id reported to clients.
        device: Device string (``cuda:N``/``cpu``).
        loader: Zero-arg factory building the model; runs on the slot's lane.
        max_pending: Maximum queued+running jobs before ``ResourceExhaustedError``.
        executor_kind: Dedicated single-thread executor or ``asyncio.to_thread``.
        unloader: Releases native resources of a model before it is dropped.
        on_change: Callback (event-loop thread) fired when capacity/state changes.
    """

    def __init__(
        self,
        slot_id: str,
        model_name: str,
        device: str,
        loader: Callable[[], Any],
        *,
        max_pending: int = 8,
        executor_kind: ExecutorKind = ExecutorKind.DEDICATED,
        unloader: Callable[[Any], None] = _default_unload,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.slot_id = slot_id
        self.model_name = model_name
        self.device = device
        self._loader = loader
        self._unloader = unloader
        self._max_pending = max_pending
        self._kind = executor_kind
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        if executor_kind is ExecutorKind.DEDICATED:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"nexa-ml-{slot_id}"
            )
        self._thread_lock = threading.Lock()
        self._model: Any = None
        self._state = SlotState.LOADING
        self._failures = 0
        self._inflight = 0
        self._idle: asyncio.Event | None = None
        self.on_change = on_change

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> SlotState:
        """Current lifecycle state."""
        return self._state

    @property
    def inflight(self) -> int:
        """Jobs submitted and not yet finished on the worker thread."""
        return self._inflight

    def snapshot(self) -> SlotSnapshot:
        """Return a health snapshot without touching the model thread."""
        vram = 0
        model = self._model
        probe = getattr(model, "vram_mib", None)
        if callable(probe):
            try:
                vram = int(probe())
            except Exception:  # noqa: BLE001 - health must never fail on a probe
                vram = 0
        return SlotSnapshot(self.slot_id, self.model_name, self.device, self._state, self._failures, vram)

    def _idle_event(self) -> asyncio.Event:
        if self._idle is None:
            self._idle = asyncio.Event()
            if self._inflight == 0:
                self._idle.set()
        return self._idle

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    # --------------------------------------------------------------- lanes
    def _dispatch(self, call: Callable[[], Any]) -> asyncio.Future[Any]:
        """Run ``call`` on the slot lane; release accounting when the thread finishes."""
        loop = asyncio.get_running_loop()
        self._inflight += 1
        self._idle_event().clear()

        def release_threadsafe(_f: object) -> None:
            try:
                loop.call_soon_threadsafe(self._release)
            except RuntimeError:  # loop already closed during shutdown
                pass

        if self._executor is not None:
            cf = self._executor.submit(call)
            cf.add_done_callback(release_threadsafe)
            return asyncio.wrap_future(cf)

        def serialised() -> Any:
            with self._thread_lock:
                return call()

        task = asyncio.ensure_future(asyncio.to_thread(serialised))
        task.add_done_callback(lambda t: (self._release(), _consume(t)))
        # Shield: to_thread work cannot be interrupted; accounting follows the thread.
        return asyncio.ensure_future(asyncio.shield(task))

    def _release(self) -> None:
        self._inflight -= 1
        if self._inflight == 0:
            self._idle_event().set()
        self._changed()

    def submit(self, fn: Callable[[Any], Any]) -> asyncio.Future[Any]:
        """Schedule ``fn(model)`` on the slot lane without awaiting it.

        Capacity is reserved synchronously so callers (e.g. the ambient pool) can
        pick and reserve a slot atomically within one event-loop step.

        Args:
            fn: Blocking callable receiving the model; runs on the slot thread.

        Returns:
            Future resolving to ``fn``'s result.

        Raises:
            SlotUnavailableError: Slot not ``READY``.
            ResourceExhaustedError: ``max_pending`` reached.
        """
        if self._state is not SlotState.READY:
            raise SlotUnavailableError(f"slot {self.slot_id} is {self._state.value}")
        if self._inflight >= self._max_pending:
            raise ResourceExhaustedError(f"slot {self.slot_id} queue is full")
        model = self._model
        return self._dispatch(lambda: fn(model))

    async def run(self, fn: Callable[[Any], Any]) -> Any:
        """Run ``fn(model)`` on the slot lane and track success/failure.

        Raises:
            InvalidInputError: Propagated from ``fn`` (not counted as failure).
            ResourceExhaustedError: Queue full or device OOM.
            SlotUnavailableError: Slot not ready.
        """
        return await self.track(self.submit(fn))

    async def track(self, future: asyncio.Future[Any]) -> Any:
        """Await a submitted job and update the consecutive-failure counter."""
        try:
            result = await future
        except (InvalidInputError, asyncio.CancelledError):
            raise
        except Exception as exc:
            self._failures += 1
            if is_out_of_memory(exc):
                raise ResourceExhaustedError(f"slot {self.slot_id} out of memory") from exc
            raise
        self._failures = 0
        return result

    # ------------------------------------------------------------ lifecycle
    def _swap(self) -> None:
        """Unload the current model and build a new one (runs on the slot lane)."""
        old, self._model = self._model, None
        if old is not None:
            try:
                self._unloader(old)
            finally:
                del old
                gc.collect()
        self._model = self._loader()

    async def _run_lifecycle(self, verb: str) -> None:
        try:
            await self._dispatch(self._swap)
        except Exception as exc:
            self._state = SlotState.UNAVAILABLE
            self._changed()
            logger.error("Slot %s %s failed: %s", self.slot_id, verb, type(exc).__name__)
            raise SlotUnavailableError(f"slot {self.slot_id} {verb} failed") from exc
        self._state = SlotState.READY
        self._failures = 0
        self._changed()
        logger.info("Slot %s %s ok (%s on %s)", self.slot_id, verb, self.model_name, self.device)

    async def load(self) -> None:
        """Initial load. On failure the slot stays ``UNAVAILABLE`` (no retry).

        Raises:
            SlotUnavailableError: The loader raised.
        """
        self._state = SlotState.LOADING
        await self._run_lifecycle("load")

    async def reload(self) -> None:
        """Drain in-flight work, unload, ``del``+``gc`` and recreate on the slot lane.

        New requests are rejected with ``UNAVAILABLE`` while reloading.

        Raises:
            ResourceExhaustedError: A load/reload is already in progress.
            SlotUnavailableError: Recreating the model failed (slot stays unavailable).
        """
        if self._state in (SlotState.LOADING, SlotState.RELOADING):
            raise ResourceExhaustedError(f"slot {self.slot_id} is already {self._state.value}")
        self._state = SlotState.RELOADING
        self._changed()
        await self._idle_event().wait()
        await self._run_lifecycle("reload")

    def close(self) -> None:
        """Stop accepting work and release the model (best effort, non-blocking)."""
        self._state = SlotState.UNAVAILABLE
        if self._executor is not None:
            model, self._model = self._model, None

            def drop() -> None:
                if model is not None:
                    try:
                        self._unloader(model)
                    except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                        logger.warning("Slot %s unload failed: %s", self.slot_id, type(exc).__name__)

            self._executor.submit(drop)
            self._executor.shutdown(wait=False, cancel_futures=False)
        else:
            self._model = None


def _consume(task: asyncio.Future[Any]) -> None:
    """Mark a shielded task's exception as retrieved (avoid 'never retrieved' noise)."""
    if not task.cancelled():
        task.exception()
