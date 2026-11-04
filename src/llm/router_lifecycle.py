"""Ownership for FastRouter's retained sync clients and native async work."""
from __future__ import annotations

import asyncio
import threading
from contextvars import copy_context

from src.core.adapter_lifecycle import _AdapterLifecycle
from src.core.native_executor import NativeThreadPoolExecutor
from src.core.resource_retirement import ResourceFailure, ResourceOwner, ResourceSession, raise_errors


_ROUTERS: set[object] = set()
_ROUTERS_LOCK = threading.Lock()


class _RouterClients:
    """An owned SDK client and the explicitly supplied underlying HTTP client."""

    def __init__(self, router) -> None:
        self.router = router
        self.http = None
        self.sdk = None

    def close(self) -> None:
        errors = []
        sdk = http = None
        try:
            sdk, http = self.sdk, self.http
            if sdk is not None:
                try:
                    sdk.close()
                except BaseException as error:
                    errors.append(ResourceFailure.capture("router_sdk_close_failed", error))
            if http is not None:
                try:
                    # SDK.close normally closes this same HTTPX client. Its
                    # actual return precedes this flag check; flag alone is
                    # never the SDK close receipt.
                    if sdk is None or getattr(http, "is_closed", False) is not True:
                        http.close()
                except BaseException as error:
                    errors.append(ResourceFailure.capture("router_http_close_failed", error))
            if not errors:
                self.sdk = self.http = self.router = None
        finally:
            sdk = http = None
        raise_errors([item.exception() for item in errors], message="router_clients_retirement_failed")


class _RouterOutcome:
    def __init__(self) -> None:
        self.value = None
        self.error = None


class _RouterWork:
    """Pre-dispatch identity, payload and mutable observer result.

    A submit failure can occur after native dispatch. Only a pending call can
    be discarded; a started call always retains its lease until its own finally.
    """

    def __init__(self, router, admission, lease, context, inputs) -> None:
        self._lock = threading.Lock()
        self.router = router
        self.admission = admission
        self.lease = lease
        self.context = context
        self.inputs = inputs
        self.started = self.discarded = self.finished = False
        self._native_entered = False
        self._finish_admission = None
        self._finish_token = object()
        self.observer = True
        self.outcome = _RouterOutcome()

    def abandon(self) -> None:
        # Monotone observer state; final clearing does not depend on a
        # potentially failing bookkeeping Lock.
        self.observer = False
        try:
            with self._lock:
                self.outcome.value = self.outcome.error = None
        finally:
            self.outcome.value = self.outcome.error = None

    def discard_pending(self) -> None:
        try:
            with self._lock:
                self.observer = False
                self.outcome.value = self.outcome.error = None
                if not self.started and not self._native_entered:
                    self.discarded = True
        finally:
            if self.discarded:
                self._finish()

    def deliver(self):
        failure = value = None
        try:
            with self._lock:
                value, failure = self.outcome.value, self.outcome.error
                self.outcome.value = self.outcome.error = None
                self.observer = False
            if failure is not None:
                raise failure
            return value
        finally:
            failure = value = None

    def _store(self, *, value=None, error=None) -> None:
        try:
            with self._lock:
                if self.observer:
                    self.outcome.value, self.outcome.error = value, error
        finally:
            # A failed Lock exit must not restore an abandoned payload.
            if not self.observer:
                self.outcome.value = self.outcome.error = None
            value = error = None

    def _metadata_failure(self, router, error: BaseException) -> None:
        try:
            if router is not None:
                router._router_work_failure = True
            # This path cannot use _store: the failure may be that Lock's
            # own exit, while an SDK exception is active as __context__.
            failure = ResourceFailure.capture("router_work_retirement_failed", error)
            self.outcome.value = self.outcome.error = None
            if self.observer:
                self.outcome.error = failure.exception()
            if not self.observer:
                self.outcome.error = None
        finally:
            router = error = failure = None

    def _finish(self) -> None:
        router = admission = lease = None
        try:
            # run/discard_pending select the sole finisher under _lock before
            # any fallible disposal. A discarded queued run never finishes it.
            if self._finish_admission is None:
                self._finish_admission = self._finish_token
        finally:
            if self._finish_admission is self._finish_token and not self.finished:
                router, admission, lease = self.router, self.admission, self.lease
                self.finished = True
                self.router = self.admission = self.lease = None
                self.context = self.inputs = None
                if lease is not None:
                    lease.active = False
                try:
                    if admission is not None:
                        admission.__exit__(None, None, None)
                except BaseException as error:
                    self._metadata_failure(router, error)
                finally:
                    if router is not None:
                        try:
                            with router._adapter_condition:
                                router._router_work.discard(self)
                        except BaseException as error:
                            self._metadata_failure(router, error)
                    router = admission = lease = None

    def run(self):
        router = context = inputs = lease = result = None
        discarded = False
        try:
            # Record actual callable entry even if the first Lock fails.
            # It then owns its finally; discard_pending cannot also finish it.
            self._native_entered = True
            with self._lock:
                discarded = self.discarded
                if not discarded:
                    self.started = True
                    router, context, inputs, lease = self.router, self.context, self.inputs, self.lease
            if not discarded:
                with router._adapter_entered(lease):
                    try:
                        result = context.run(router._complete_native, *inputs)
                    except BaseException as error:
                        # Only the actual request exception is an original
                        # caller loan. Infrastructure errors use a safe DTO.
                        self._store(error=error)
                    else:
                        self._store(value=result)
        except BaseException as error:
            self._metadata_failure(router or self.router, error)
        finally:
            context = inputs = result = None
            try:
                if not discarded:
                    self._finish()
            except BaseException as error:
                self._metadata_failure(router or self.router, error)
                # SDK body has exited. End only the logical lease after
                # clearing payloads; pool shutdown still proves native exit.
                # Keep unresolved admission/owner metadata strongly retained.
                self.context = self.inputs = None
                failed_lease = lease or self.lease
                if failed_lease is not None:
                    failed_lease.active = False
                failed_lease = None
            finally:
                router = lease = None
        return self.outcome


def _observe_future(future) -> None:
    if not future.cancelled():
        future.exception()


class _FastRouterLifecycle(_AdapterLifecycle):
    def _router_initialize(self) -> None:
        self._adapter_initialize("router")
        self._router_session = ResourceSession(error_prefix="router")
        self._router_executor = None
        self._router_work: set[_RouterWork] = set()
        self._router_work_failure = False
        self._router_native_futures = set()
        self._router_dispatch_uncertain = False

    def _router_new_clients(self):
        # The global ResourceOwner registry retains this bundle and router
        # before HTTPX/OpenAI acquisition, including partial load failures.
        with _ROUTERS_LOCK:
            _ROUTERS.add(self)
        clients = _RouterClients(self)
        owner = ResourceOwner(clients, owns_resource=True, error_prefix="router_clients",
                              thread_name="router-http-close")
        try:
            self._router_session.register(owner)
        except BaseException:
            owner.close()
            raise
        return clients, owner

    def _adapter_require_close_caller(self) -> None:
        super()._adapter_require_close_caller()
        # SDK/HTTP close executes in cohort ResourceOwner Threads, distinct
        # from this adapter's close-admission Thread. Reject their recursive
        # self-close before an offload can hide that origin and wait in a cycle.
        caller = threading.current_thread()
        for owner in tuple(self._router_session._engines.values()):
            if caller is owner._closer:
                raise RuntimeError("router_close_from_owner")
        executor = self._router_executor
        if executor is not None:
            executor.require_shutdown_caller()

    def _router_require_close_caller(self) -> None:
        """Metadata guard for ResourceOwner before its fresh-context closer."""
        self._adapter_require_close_caller()

    def _router_get_executor(self):
        with self._adapter_lock:
            if self._router_executor is None:
                with _ROUTERS_LOCK:
                    _ROUTERS.add(self)
                self._router_executor = NativeThreadPoolExecutor(thread_name_prefix="fast-router")
            return self._router_executor

    async def _router_complete(self, prompt, max_tokens, temperature, stop):
        admission = lease = work = native = wrapped = executor = None
        entered = False
        try:
            admission = self._adapter_stream_admit()
            lease = admission.__enter__()
            entered = True
            # Detached queued work belongs to no event-loop Thread. Only the
            # actual callable's _adapter_entered activates a Thread/TLS guard.
            lease.thread = None
            lease.on_stack = False
            work = _RouterWork(self, admission, lease, copy_context(),
                               (prompt, max_tokens, temperature, stop))
            with self._adapter_condition:
                self._router_work.add(work)
            executor = self._router_get_executor()
            # No adapter/lifecycle lock crosses Thread.start/native admission.
            try:
                native = executor.submit(work.run)
                wrapped = asyncio.wrap_future(native)
            except BaseException:
                # A submit exception may follow dispatch without returning its
                # Future. Logical lease retirement cannot prove native return.
                with self._adapter_condition:
                    self._router_dispatch_uncertain = True
                raise
            with self._adapter_condition:
                self._router_native_futures.add(native)

            def finished(done, owner=self):
                try:
                    with owner._adapter_condition:
                        owner._router_native_futures.discard(done)
                except BaseException:
                    owner._router_work_failure = True
                finally:
                    owner = done = None

            try:
                native.add_done_callback(finished)
            except BaseException:
                with self._adapter_condition:
                    self._router_dispatch_uncertain = True
                raise
            wrapped.add_done_callback(_observe_future)
            await asyncio.shield(wrapped)
            return work.deliver()
        finally:
            try:
                if work is not None:
                    # As with cancellation of the original to_thread future,
                    # queued work may be discarded; started work must drain.
                    # No returned Future is not proof of no native call.
                    try:
                        work.discard_pending()
                    finally:
                        work.abandon()
                elif entered:
                    admission.__exit__(None, None, None)
            finally:
                self = prompt = stop = admission = lease = work = native = wrapped = executor = None

    def _adapter_release_resources(self) -> None:
        executor = self._router_executor
        try:
            if executor is not None:
                # Operation-finally is not a Thread-exit receipt. Join every
                # owned executor worker before any HTTP/SDK resource closes.
                executor.shutdown(wait=True)
            if self._router_work_failure:
                # Native pool joined, but failed infrastructure did not give
                # a successful resource-retirement receipt. Keep SDK owners.
                raise RuntimeError("router_work_retirement_failed")
            self._router_session.__exit__(None, None, None)
            self._client = None
            self._router_executor = None
            with self._adapter_condition:
                self._router_work.clear()
                self._router_native_futures.clear()
                self._router_dispatch_uncertain = False
            with _ROUTERS_LOCK:
                _ROUTERS.discard(self)
        finally:
            executor = None
