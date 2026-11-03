"""Explicit standalone resource ownership with native retirement receipts.

This module owns metadata/retirement only. Callers keep measurements and SDK
conversions inside admitted scopes, detach payloads/errors before retirement,
and retain responsibility for source files and emitted reports.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import Context
from dataclasses import dataclass
from types import TracebackType
from typing import Any

_OWNERS: set = set()
_FAILED_OWNERS: set = set()
_OWNERS_LOCK = threading.Lock()

@dataclass(frozen=True)
class ResourceFailure:
    """Keep classification, never a native exception/traceback or payload."""

    stage: str
    kind: str
    exit_code: int | None = None

    @classmethod
    def capture(cls, stage: str, error: BaseException) -> ResourceFailure:
        if isinstance(error, KeyboardInterrupt):
            return cls(stage, "interrupt")
        if isinstance(error, SystemExit):
            code = error.code
            return cls(stage, "exit", code if code is None or type(code) in (int, bool) else 1)
        if isinstance(error, GeneratorExit):
            return cls(stage, "generator")
        if isinstance(error, asyncio.CancelledError):
            return cls(stage, "cancel")
        if not isinstance(error, Exception):
            return cls(stage, "control")
        return cls(stage, type(error).__name__)

    def exception(self) -> BaseException:
        if self.kind == "interrupt":
            return KeyboardInterrupt()
        if self.kind == "exit":
            return SystemExit(self.exit_code)
        if self.kind == "generator":
            return GeneratorExit()
        if self.kind == "cancel":
            return asyncio.CancelledError()
        if self.kind == "control":
            return BaseException(f"{self.stage}:control")
        return RuntimeError(f"{self.stage}:{self.kind}")

def raise_errors(errors: list[BaseException], *, message: str = "resource_retirement_failed") -> None:
    for error in errors:
        if not isinstance(error, Exception):
            raise error
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup(message, errors)

@dataclass(eq=False)
class OperationLease:
    owner: threading.Thread
    active: bool = True


@dataclass(eq=False)
class _StartAdmission:
    """Identify the sole native-start transaction and its originating caller."""

    owner: threading.Thread
    finished: bool = False

class ResourceOwner:
    """Own a resource and finite acquisition/work until actual native exit."""

    def __init__(
        self,
        resource: Any = None,
        *,
        owns_resource: bool = False,
        error_prefix: str = "resource",
        caller_guard: Callable[[], None] | None = None,
        thread_name: str | None = None,
    ) -> None:
        """Configure an owner before native acquisition or load.

        Args:
            resource: Explicit resource, or None until adopted before load.
            owns_resource: Whether this owner closes its resource.
            error_prefix: Fixed diagnostic prefix without SDK payloads.
            caller_guard: Optional metadata check on the calling thread before
                sealing or dispatching a closer with a fresh context.
            thread_name: Optional native closer name for diagnostics.
        """
        if type(owns_resource) is not bool:
            raise ValueError("resource_ownership_must_be_bool")
        self.resource, self.owns_resource = resource, owns_resource
        self.error_prefix = error_prefix
        self._caller_guard = caller_guard
        self._condition = threading.Condition()
        self._sealed = False
        self._operations: set[OperationLease] = set()
        self._start_called = False
        self._start_admission: _StartAdmission | None = None
        self._admission_done = threading.Event()
        self._target_exited = threading.Event()
        self._start_failure = self._close_failure = None
        self._confirmed = False
        self._closer = threading.Thread(
            target=Context().run, args=(self._retire,),
            name=thread_name or f"{error_prefix}-close", daemon=True,
        )
        self._handle = getattr(self._closer, "_handle", None)
        handle_type = getattr(threading, "_ThreadHandle", None)
        started = getattr(self._closer, "_started", None)
        # Audit boundary shared with the local K/D native-thread receipts; no
        # STT private helper import, and no actor starts before this guard.
        if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13)
                or handle_type is None or type(self._handle) is not handle_type
                or type(getattr(self._handle, "ident", None)) is not int or self._handle.ident != 0
                or not callable(getattr(self._handle, "join", None))
                or not callable(getattr(self._handle, "is_done", None))
                or not callable(getattr(started, "is_set", None)) or started.is_set()):
            raise RuntimeError(f"{error_prefix}_close_unsupported_runtime")
        if owns_resource:
            # The adapter is explicit and retained before load or warmup.
            with _OWNERS_LOCK:
                _OWNERS.add(self)

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Admit one identity lease, including reservation/lock-exit failures.

        Callers clear payload locals and leave SDK exception handlers before
        ending the admitted body or asking another owner to retire resources.
        """
        lease = OperationLease(threading.current_thread())
        try:
            with self._condition:
                nested = any(item.active and item.owner is lease.owner for item in self._operations)
                if self._sealed and not nested:
                    raise RuntimeError(f"{self.error_prefix}_engine_closed")
                self._operations.add(lease)
            yield
        finally:
            # Logical exit is plain metadata and precedes fallible notification.
            # A retained inactive token cannot become a phantom producer.
            lease.active = False
            with self._condition:
                self._operations.discard(lease)
                self._condition.notify_all()

    def adopt(
        self,
        resource: Any,
        *,
        owns_resource: bool = True,
        caller_guard: Callable[[], None] | None = None,
    ) -> None:
        """Attach before native load; in-flight acquisition may finish after seal."""
        if type(owns_resource) is not bool:
            raise ValueError("resource_ownership_must_be_bool")
        if owns_resource:
            with _OWNERS_LOCK:
                _OWNERS.add(self)
        caller = threading.current_thread()
        with self._condition:
            if self.resource is not None and self.resource is not resource:
                raise RuntimeError("resource_owner_already_bound")
            active = any(item.active and item.owner is caller for item in self._operations)
            if self._sealed and not active:
                raise RuntimeError(f"{self.error_prefix}_engine_closed")
            self.resource, self.owns_resource = resource, owns_resource
            self._caller_guard = caller_guard

    def require_close_caller(self) -> None:
        """Reject closure originating in this owner's operation or closer."""
        caller = threading.current_thread()
        with self._condition:
            if any(lease.active and lease.owner is caller for lease in self._operations):
                raise RuntimeError(f"{self.error_prefix}_close_from_operation")
            if caller is self._closer:
                raise RuntimeError(f"{self.error_prefix}_close_from_owner")
            admission = self._start_admission
            if admission is not None and admission.owner is caller and not admission.finished:
                raise RuntimeError(f"{self.error_prefix}_close_from_admission")
            resource, guard = self.resource, self._caller_guard
        # FastWhisper dispatches a caller's load/transcribe onto its own pinned
        # Thread. Check that actual Thread before creating a fresh-context
        # closer, which would otherwise hide its close-from-worker origin.
        # These are adapter metadata only; no model or private helper is called.
        try:
            worker = vars(resource).get("_worker_thread") if resource is not None else None
        except TypeError:
            worker = None
        if caller is worker:
            raise RuntimeError(f"{self.error_prefix}_close_from_operation")
        if guard is not None:
            guard()

    def request_close(self) -> None:
        """Seal only this callable; already admitted load/warmup may finish."""
        with self._condition:
            self._sealed = True
            self._condition.notify_all()

    def begin_close(self) -> threading.Thread:
        """Resolve one native-start admission even if pre-dispatch metadata fails."""
        self.require_close_caller()
        self.request_close()
        admission = None
        try:
            with self._condition:
                if self._start_admission is None:
                    admission = _StartAdmission(threading.current_thread())
                    # Owner identity is part of the single published token.
                    # A follower never acquires responsibility for this start.
                    self._start_admission = admission
                    self._start_called = True
            if admission is not None:
                with _OWNERS_LOCK:
                    _OWNERS.add(self)
                # No lifecycle lock across SDK/native thread admission.
                self._closer.start()
        except BaseException as error:
            if admission is None or self._start_admission is not admission:
                raise
            self._start_failure = ResourceFailure.capture(f"{self.error_prefix}_close_start_failed", error)
        finally:
            if admission is not None and self._start_admission is admission:
                try:
                    self._admission_done.set()
                finally:
                    # This plain receipt also covers notification interruption.
                    # It does not claim that Thread.start dispatched or exited.
                    admission.finished = True
        return self._closer

    def _retire(self) -> None:
        resource = None
        try:
            with self._condition:
                while any(lease.active for lease in self._operations):
                    # Operation notification is advisory, including cleanup faults.
                    self._condition.wait(.05)
                resource = self.resource
            if self.owns_resource and resource is not None:
                resource.close()
        except BaseException as error:
            self._close_failure = ResourceFailure.capture(
                f"{self.error_prefix}_engine_close_failed", error)
        finally:
            resource = None
            self._target_exited.set()

    def wait_closed(self) -> None:
        """Observe actual native exit through repeated observer interruption."""
        self.require_close_caller()
        closer = self._closer
        controls = []
        begin_failure = None
        failure = None
        try:
            self.begin_close()
        except BaseException as error:
            # Direct close has the same receipt guarantee as session closure:
            # notification failure does not abandon an already started actor.
            begin_failure = ResourceFailure.capture(
                f"{self.error_prefix}_close_begin_failed", error)
        admission = self._start_admission
        while admission is not None and not admission.finished:
            try:
                # Notification is advisory: its publisher may be interrupted.
                # Never wait indefinitely on an unset Event after resolution.
                self._admission_done.wait(.05)
                if not admission.finished and self._admission_done.is_set():
                    time.sleep(0)
            except BaseException as error:
                item = ResourceFailure.capture(f"{self.error_prefix}_close_wait_failed", error)
                if isinstance(error, Exception):
                    failure = item
                    break
                controls.append(item)
        if failure is None:
            if self._handle.ident == 0:
                failure = ResourceFailure(f"{self.error_prefix}_close_unconfirmed", "start")
            else:
                while True:
                    try:
                        if not closer._started.is_set():
                            self._handle.join()
                        else:
                            closer.join()
                        if not self._handle.is_done() or not self._target_exited.is_set():
                            failure = ResourceFailure(f"{self.error_prefix}_close_unconfirmed", "exit")
                        break
                    except BaseException as error:
                        item = ResourceFailure.capture(f"{self.error_prefix}_close_wait_failed", error)
                        if isinstance(error, Exception):
                            failure = item
                            break
                        controls.append(item)
        if failure is None and self._close_failure is None:
            with self._condition:
                self.resource = None
                self._caller_guard = None
                self._operations.clear()
                self._confirmed = True
            with _OWNERS_LOCK:
                _OWNERS.discard(self)
        else:
            with _OWNERS_LOCK:
                _FAILED_OWNERS.add(self)
        errors = [item.exception() for item in
                  (self._start_failure, self._close_failure, begin_failure,
                   failure, *controls) if item is not None]
        raise_errors(errors)

    def close(self) -> None:
        """Retire this callable; only an owning handle retires its adapter."""
        self.wait_closed()

class ResourceSession:
    """Own only the distinct engines constructed for this script invocation."""

    def __init__(self, *, error_prefix: str = "resource") -> None:
        self.error_prefix = error_prefix
        self._condition = threading.Condition()
        self._engines: dict[int, ResourceOwner] = {}
        self._sealed = False

    def register(self, engine: ResourceOwner) -> None:
        with self._condition:
            if self._sealed:
                raise RuntimeError(f"{self.error_prefix}_session_closed")
            self._engines[id(engine)] = engine

    def __enter__(self) -> ResourceSession:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        prior_error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        # Every guard is metadata-only, before any seal or context-empty child.
        with self._condition:
            engines = tuple(self._engines.values())
            for engine in engines:
                engine.require_close_caller()
            self._sealed = True
        errors = []
        for engine in engines:
            try:
                engine.request_close()
            except BaseException as error:
                errors.append(error)
        for engine in engines:
            try:
                engine.begin_close()
            except BaseException as error:
                errors.append(error)
        for engine in engines:
            try:
                engine.wait_closed()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise_errors(([prior_error] if prior_error is not None else []) + errors,
                         message=f"{self.error_prefix}_retirement_failed")
        return False
