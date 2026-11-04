"""CPython 3.13 executor with native worker ownership before Thread.start.

The ordinary default worker count and submit/shutdown protocol remain stdlib.
Only native start ownership and join receipts differ from ThreadPoolExecutor.
"""
from __future__ import annotations

import sys
import threading
import types
import weakref
from concurrent.futures import thread as _pool

_UNSUPPORTED = "native_executor_unsupported_runtime"


def _require_runtime() -> None:
    worker = getattr(_pool, "_worker", None)
    queues = getattr(_pool, "_threads_queues", None)
    submit = _pool.ThreadPoolExecutor.submit
    shutdown = _pool.ThreadPoolExecutor.shutdown
    handle_type = getattr(threading, "_ThreadHandle", None)
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13)
            or not isinstance(worker, types.FunctionType)
            or worker.__code__.co_argcount != 4
            or worker.__code__.co_kwonlyargcount != 0
            or worker.__code__.co_varnames[:4] != (
                "executor_reference", "work_queue", "initializer", "initargs")
            or type(queues) is not weakref.WeakKeyDictionary
            or not isinstance(handle_type, type)
            or not all(hasattr(handle_type, name) for name in ("ident", "join", "is_done"))
            or not hasattr(getattr(_pool, "_global_shutdown_lock", None), "__enter__")
            or not isinstance(submit, types.FunctionType)
            or not isinstance(shutdown, types.FunctionType)
            or not {"_shutdown_lock", "_global_shutdown_lock", "_adjust_thread_count"}
                <= set(submit.__code__.co_names)
            or not {"_shutdown_lock", "_threads", "join"} <= set(shutdown.__code__.co_names)):
        raise RuntimeError(_UNSUPPORTED)


class _NativeWorker(threading.Thread):
    def join(self, timeout: float | None = None) -> None:
        if not self._started.is_set():
            if self._handle.ident == 0:
                if self._handle.is_done():
                    # Positive terminal receipt from a failed native start.
                    return
                raise RuntimeError("native_executor_start_unconfirmed")
            if timeout is None:
                self._handle.join()
            else:
                self._handle.join(max(timeout, 0))
        else:
            super().join(timeout)
        if timeout is None and not self._handle.is_done():
            raise RuntimeError("native_executor_exit_unconfirmed")


class NativeThreadPoolExecutor(_pool.ThreadPoolExecutor):
    """Default-concurrency executor retaining every real native worker."""

    def __init__(self, *, thread_name_prefix: str = "") -> None:
        _require_runtime()
        # None preserves CPython 3.13's process_cpu_count-based default.
        super().__init__(max_workers=None, thread_name_prefix=thread_name_prefix)
        self._submission_local = threading.local()
        if (type(self._threads) is not set or type(self._max_workers) is not int
                or self._max_workers < 1
                or not callable(getattr(self._idle_semaphore, "acquire", None))
                or not callable(getattr(self._work_queue, "put", None))
                or not hasattr(self._shutdown_lock, "__enter__")):
            raise RuntimeError(_UNSUPPORTED)

    def submit(self, fn, /, *args, **kwargs):
        previous = getattr(self._submission_local, "active", False)
        try:
            # Only the synchronous native-admission stack is marked, never
            # the event-loop Task's await or detached queued operation.
            self._submission_local.active = True
            return super().submit(fn, *args, **kwargs)
        finally:
            self._submission_local.active = previous
            fn = args = kwargs = None

    def require_shutdown_caller(self) -> None:
        # Metadata only, without the executor's submission/start lock.
        if getattr(self._submission_local, "active", False):
            raise RuntimeError("native_executor_close_from_submission")
        if threading.current_thread() in tuple(self._threads):
            raise RuntimeError("native_executor_close_from_worker")

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.require_shutdown_caller()
        return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return
        if len(self._threads) >= self._max_workers:
            return

        def weakref_cb(_, queue=self._work_queue):
            queue.put(None)

        thread = _NativeWorker(
            name=f"{self._thread_name_prefix}_{len(self._threads)}", target=_pool._worker,
            args=(weakref.ref(self, weakref_cb), self._work_queue,
                  self._initializer, self._initargs),
        )
        handle = getattr(thread, "_handle", None)
        started = getattr(thread, "_started", None)
        handle_type = getattr(threading, "_ThreadHandle", None)
        if (handle_type is None or type(handle) is not handle_type
                or type(getattr(handle, "ident", None)) is not int or handle.ident != 0
                or not callable(getattr(handle, "join", None))
                or not callable(getattr(handle, "is_done", None))
                or not callable(getattr(started, "is_set", None))
                or not callable(getattr(started, "wait", None)) or started.is_set()):
            raise RuntimeError(_UNSUPPORTED)
        start_called = False
        try:
            self._threads.add(thread)
            _pool._threads_queues[thread] = self._work_queue
            start_called = True
            thread.start()
        except BaseException:
            if start_called:
                # ident publication follows OS admission in CPython. Neither
                # zero ident nor an unset Python Event proves no native start.
                self._broken = "native_executor_start_failed"
            if not start_called or (handle.ident == 0 and handle.is_done()):
                _pool._threads_queues.pop(thread, None)
                self._threads.discard(thread)
            # Otherwise retain Thread/handle/queue; shutdown's finite join
            # either observes actual native exit or reports unconfirmed start.
            raise
