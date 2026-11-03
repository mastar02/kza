"""Retain pinned workers across partial CPython ThreadPoolExecutor submission.

This adapter uses the CPython 3.13 private executor/ThreadHandle contract,
reviewed in 3.13.0 and 3.13.12 and checked before creating any worker.
The small worker-start sequence derives from Lib/concurrent/futures/thread.py
(PSF license, Copyright 2009 Brian Quinlan). It changes registration order only;
submission locks, queue execution and actual shutdown joins remain the SDK's.
"""

from __future__ import annotations

import sys
import threading
import types
import weakref
from concurrent.futures import thread as _pool
from typing import Callable


_UNSUPPORTED = 'pinned_executor_unsupported_runtime'


def _require_runtime() -> None:
    """Reject an unaudited private layout before any worker can be started."""
    worker = getattr(_pool, '_worker', None)
    queues = getattr(_pool, '_threads_queues', None)
    submit = _pool.ThreadPoolExecutor.submit
    shutdown = _pool.ThreadPoolExecutor.shutdown
    handle_type = getattr(threading, '_ThreadHandle', None)
    if (sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13)
            or not isinstance(worker, types.FunctionType)
            or worker.__code__.co_argcount != 4
            or worker.__code__.co_kwonlyargcount != 0
            or worker.__code__.co_varnames[:4] != (
                'executor_reference', 'work_queue', 'initializer', 'initargs')
            or type(queues) is not weakref.WeakKeyDictionary
            or not isinstance(handle_type, type)
            or not all(hasattr(handle_type, name) for name in ('ident', 'join', 'is_done'))
            or not hasattr(getattr(_pool, '_global_shutdown_lock', None), '__enter__')
            or not isinstance(submit, types.FunctionType)
            or not isinstance(shutdown, types.FunctionType)
            or not {'_shutdown_lock', '_global_shutdown_lock', '_adjust_thread_count'}
                <= set(submit.__code__.co_names)
            or not {'_shutdown_lock', '_threads', 'join'} <= set(shutdown.__code__.co_names)):
        raise RuntimeError(_UNSUPPORTED)


class _PinnedWorker(threading.Thread):
    """Join the native handle even if Python bootstrap has not published yet."""

    def join(self, timeout: float | None = None) -> None:
        if not self._started.is_set() and self._handle.ident != 0:
            if timeout is None:
                self._handle.join()
            else:
                self._handle.join(max(timeout, 0))
            return
        super().join(timeout)


class PinnedThreadPoolExecutor(_pool.ThreadPoolExecutor):
    """Own the sole real worker before a possibly partial Thread.start call."""

    def __init__(
        self, max_workers: int = 1, thread_name_prefix: str = '',
        initializer: Callable | None = None, initargs: tuple = (),
    ) -> None:
        """Create the audited one-worker executor without starting a worker.

        Args:
            max_workers: Must be exactly one for this pinned model adapter.
            thread_name_prefix: Name prefix forwarded to the SDK executor.
            initializer: Optional SDK worker initializer.
            initargs: Arguments for the optional initializer.

        Raises:
            RuntimeError: If the installed private SDK contract is unsupported.
            ValueError: If more than one worker or a noninteger is requested.
        """
        _require_runtime()
        if type(max_workers) is not int or max_workers != 1:
            raise ValueError('pinned_executor_requires_one_worker')
        super().__init__(max_workers=1, thread_name_prefix=thread_name_prefix,
                         initializer=initializer, initargs=initargs)
        if (type(self._threads) is not set or self._max_workers != 1
                or not callable(getattr(self._idle_semaphore, 'acquire', None))
                or not callable(getattr(self._work_queue, 'put', None))
                or not hasattr(self._shutdown_lock, '__enter__')):
            raise RuntimeError(_UNSUPPORTED)

    def _adjust_thread_count(self) -> None:
        # Inherited submit holds the SDK shutdown and global shutdown locks.
        # No caller/model lifecycle lock is held while Thread.start can wait.
        if self._idle_semaphore.acquire(timeout=0):
            return
        if len(self._threads) >= self._max_workers:
            return

        def weakref_cb(_, queue=self._work_queue):
            queue.put(None)

        thread = _PinnedWorker(
            name=f'{self._thread_name_prefix}_{len(self._threads)}', target=_pool._worker,
            args=(weakref.ref(self, weakref_cb), self._work_queue,
                  self._initializer, self._initargs),
        )
        handle = getattr(thread, '_handle', None)
        started = getattr(thread, '_started', None)
        handle_type = getattr(threading, '_ThreadHandle', None)
        if (handle_type is None or type(handle) is not handle_type
                or type(getattr(handle, 'ident', None)) is not int or handle.ident != 0
                or not callable(getattr(handle, 'join', None))
                or not callable(getattr(started, 'is_set', None))
                or not callable(getattr(started, 'wait', None)) or started.is_set()):
            raise RuntimeError(_UNSUPPORTED)
        try:
            # Both owners must exist before the SDK can create an OS thread.
            self._threads.add(thread)
            _pool._threads_queues[thread] = self._work_queue
            thread.start()
        except BaseException:
            # Thread.ident may still be None after native creation. The audited
            # native handle receives its nonzero ident before bootstrap runs.
            if handle.ident == 0 and not started.is_set():
                # This start admission has resolved without a native worker.
                _pool._threads_queues.pop(thread, None)
                self._threads.discard(thread)
            else:
                # Retain the native handle without waiting under the SDK's
                # submission locks. Its join observes native exit even before
                # Python bootstrap can publish the ordinary started receipt.
                pass
            raise
