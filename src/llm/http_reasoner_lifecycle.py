"""Instance ownership for HttpReasoner; no SDK factories or private transports."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from contextvars import Context
from dataclasses import dataclass, field
import weakref

from src.core.adapter_lifecycle import _AdapterLease, _AdapterLifecycle
from src.core.resource_retirement import ResourceFailure
from src.llm.http_generation import CloseTaskFactory, HTTPGenerationClient, owned_resource

_UNCERTAIN_HTTP_OWNERS = set()


def _capture(stage, error):
    if isinstance(error, BaseExceptionGroup):
        return [failure for leaf in error.exceptions for failure in _capture(stage, leaf)]
    return [ResourceFailure.capture(stage, error)]


def _raise_failures(failures, original=None):
    errors = ([original] if original is not None and not isinstance(original, GeneratorExit) else [])
    errors.extend(item.exception() for item in failures)
    if len(errors) == 1:
        raise errors[0] from None
    if errors:
        raise BaseExceptionGroup("http_reasoner_operation_failed", errors) from None
    if original is not None:
        raise original


@dataclass(eq=False)
class _TaskEntry:
    task: object
    active: bool = True


@dataclass(eq=False)
class _Reservation:
    task: object = None
    active: bool = True
    started: bool = False
    finished: bool = False
    awaitable: object = None


@dataclass(eq=False)
class _TaskLease:
    owner: object
    thread: object = None
    on_stack: bool = False
    body_active: bool = True
    entries: set = field(default_factory=set)
    latest: object = None
    streaming: bool = False
    failed_close: bool = False

    @property
    def active(self):
        return self.body_active or any(item.active for item in self.entries)


class _HttpClients:
    def __init__(self):
        self.http = self.sdk = None
        self.close_started = self.close_finished = False
        self.failures = ()

    def close(self):
        # All callers hold an admitted producer or the elected disposal owner.
        if self.close_started:
            if not self.close_finished:
                raise RuntimeError("http_reasoner_clients_close_unconfirmed")
            _raise_failures(self.failures)
            return
        self.close_started = True
        sdk = http = None
        failures = []
        try:
            sdk, http = self.sdk, self.http
            if sdk is not None:
                # Installed OpenAI.close cascades to the supplied HTTPX client.
                # A thrown cascade is uncertain even if HTTPX set CLOSED first.
                sdk.close()
                if http is not None and getattr(http, "is_closed", False) is not True:
                    # Compatibility for a successful non-cascading SDK double;
                    # never retry a leaf after a failed cascade.
                    http.close()
            elif http is not None:
                http.close()
            self.sdk = self.http = None
        except BaseException as error:
            failures.extend(_capture("http_reasoner_clients_close_failed", error))
        finally:
            self.failures = tuple(failures)
            self.close_finished = True
            sdk = http = None
        _raise_failures(self.failures)


class _PreparationFailure:
    def __init__(self, error):
        self.client = None
        self.error = error


class _PreparedClient:
    def __init__(self):
        self.client = self.error = None


class _AsyncCloser:
    def __init__(self, function):
        self.function = function

    async def close(self):
        await self.function()


class _StreamSources:
    def __init__(self):
        self.source = self.iterator = None


class _ReadControl:
    def __init__(self, error):
        self.error = error


class _Initialization:
    def __init__(self, owner, lease, function, args):
        self.owner, self.lease, self.function, self.args = owner, lease, function, args
        self.prepared_slot = _PreparedClient()
        self.resources = self.prepared_slot
        self.native_finished = False
        self.reservation = self.awaitable = self.close_work = None
        self.discard_reservation = self.discard_awaitable = None

    def adopt_client(self, client):
        self.prepared_slot.client = client
        return self.owner.start_close

    def _error(self, error):
        if self.resources is None:
            self.resources = _PreparationFailure(error)
        elif self.resources.error is None:
            self.resources.error = error
        else:
            self.resources.error = BaseExceptionGroup(
                "http_reasoner_prepare_failed", [self.resources.error, error])

    def run_native(self):
        try:
            with self.owner.native_entered(self.lease, initialization=self):
                # Publish returned handle before fallible TLS/condition exit.
                self.resources = self.function(*self.args)
        except BaseException as error:
            self._error(error)
        finally:
            self.native_finished = True
            self.function = None
            self.args = ()

    def prepare(self):
        self.reservation = self.owner.reserve(self.lease)
        self.awaitable = self._prepare(self.reservation)
        self.reservation.awaitable = self.awaitable
        return self.awaitable

    async def _prepare(self, reservation):
        try:
            with self.owner.reserved(self.lease, reservation):
                if self.owner._dispatch_uncertain:
                    self._error(RuntimeError("http_reasoner_prepare_unconfirmed"))
                    self.native_finished = True
                    return None
                await asyncio.to_thread(self.run_native)
        except BaseException as error:
            # Internal direct cancellation is unsupported; keep this holder if
            # native work has not returned rather than inventing a receipt.
            self._error(error)
        return None


class _CloseWork:
    def __init__(self, owner, lease, resource):
        self.owner, self.lease, self.resource = owner, lease, resource
        self.failures = ()
        self.closed_receipt = False
        self.reservation = self.awaitable = None

    def close(self):
        self.reservation = self.owner.reserve(self.lease)
        self.awaitable = self._close(self.reservation)
        self.reservation.awaitable = self.awaitable
        return self.awaitable

    async def _close(self, reservation):
        try:
            with self.owner.reserved(self.lease, reservation):
                try:
                    if self.owner._dispatch_uncertain or self.lease.failed_close:
                        raise RuntimeError("http_reasoner_resource_close_unconfirmed")
                    await self.resource.close()
                    self.closed_receipt = True
                finally:
                    if self.closed_receipt:
                        self.resource = None
        except BaseException as error:
            self.failures = tuple(_capture("http_reasoner_async_close_failed", error))
            self.lease.failed_close = True
        finally:
            if not self.failures:
                self.resource = self.lease = self.owner = None


class HttpReasonerLifecycle(_AdapterLifecycle):
    """Composed owner: Thread scopes for sync, explicit Task scopes for async."""

    def __init__(self, reasoner: Any) -> None:
        """Capture ownership metadata without acquiring SDK resources."""
        self._adapter_initialize("http_reasoner")
        self._reasoner = weakref.ref(reasoner)
        self._bundles = []
        self._streams = {}
        self._async_streams = {}
        self._initializing = set()
        self._close_work = set()
        self._resource_failures = []
        self._finish_failures = ()
        self._dispatch_uncertain = False
        self._native_reservation = self._native_pending = None
        self._close_dispatch_started = False
        self._outer_pending = None

    def sync(self) -> AbstractContextManager[_AdapterLease]:
        """Admit a synchronous operation under its actual Thread."""
        return self._adapter_admit()

    @contextmanager
    def stream(self) -> Iterator[_AdapterLease]:
        """Retain a lazy stream until its source and iterator retire."""
        with self._adapter_stream_admit() as lease:
            self._streams[lease] = _StreamSources()
            try:
                yield lease
            finally:
                sources = self._streams.get(lease)
                if sources is not None and sources.source is None and sources.iterator is None:
                    self._streams.pop(lease, None)

    def entered(self, lease: _AdapterLease) -> AbstractContextManager[Any]:
        """Mark only an actual synchronous pull or cleanup call."""
        return self._adapter_entered(lease)

    def track(self, source: Any) -> Any:
        """Retain a source before a later fallible operation."""
        self._adapter_track_iterator(source)
        return source

    def attach_source(self, lease: _AdapterLease, source: Any) -> None:
        """Capture the SDK source before asking it for an iterator."""
        self._streams[lease].source = source
        self._adapter_track_iterator(source)

    def attach_iterator(self, lease: _AdapterLease, iterator: Any) -> None:
        """Capture the distinct iterator before its first pull."""
        self._streams[lease].iterator = iterator
        self._adapter_track_iterator(iterator)

    def attach_async_stream(self, lease: _TaskLease, stream: Any) -> None:
        """Retain the inner async stream through actual cleanup."""
        self._async_streams[lease] = stream

    def retired_async_stream(self, lease: _TaskLease) -> None:
        """Release the stream only after its cleanup returned."""
        self._async_streams.pop(lease, None)

    def initialization(self, lease: _TaskLease, function: Callable[..., Any], *args: Any) -> _Initialization:
        """Publish a constructor holder before scheduling acquisition."""
        work = _Initialization(self, lease, function, args)
        self._initializing.add(work)
        return work

    def release_initialization(self, work: _Initialization) -> None:
        """Release only a transferred or actually retired acquisition."""
        resources = work.resources
        transferred = work.close_work is not None and work.close_work in self._close_work
        retired = work.close_work is not None and work.close_work.closed_receipt
        no_handle = (work.native_finished and work.reservation is not None
                     and work.reservation.finished and resources is not None
                     and resources.client is None)
        if transferred or retired or no_handle:
            self._initializing.discard(work)
            work.resources = work.owner = work.lease = None
        else:
            self.uncertain(work.reservation, work.awaitable,
                           "http_reasoner_initialization_unconfirmed")

    def reserve(self, lease: _TaskLease) -> _Reservation:
        """Reserve a child lifetime before invoking a Task factory."""
        reservation = _Reservation()
        try:
            with self._adapter_condition:
                if lease.owner is not self or not lease.active:
                    raise RuntimeError("http_reasoner_closed")
                lease.entries.add(reservation)
            return reservation
        except BaseException:
            # Factory has not yet been called: positive pre-dispatch absence.
            reservation.active = False
            reservation.finished = True
            raise

    @contextmanager
    def reserved(self, lease: _TaskLease, reservation: _Reservation) -> Iterator[None]:
        """Register the actual child Task while its body executes."""
        try:
            with self._adapter_condition:
                if not reservation.active or lease.owner is not self or not lease.active:
                    raise RuntimeError("http_reasoner_closed")
                reservation.task = asyncio.current_task()
                reservation.started = True
            yield
        finally:
            reservation.active = False
            reservation.finished = True
            reservation.task = reservation.awaitable = None
            with self._adapter_condition:
                lease.entries.discard(reservation)
                if not lease.active:
                    self._adapter_leases.discard(lease)
                self._adapter_condition.notify_all()

    def uncertain(self, reservation: _Reservation | None, awaitable: Awaitable[Any] | None, stage: str) -> None:
        """Seal and retain ownership when dispatch lacks a receipt."""
        self._dispatch_uncertain = self._adapter_sealed = True
        if reservation is not None and not reservation.finished:
            reservation.awaitable = awaitable
        self._resource_failures.append(ResourceFailure(stage, "RuntimeError"))
        _UNCERTAIN_HTTP_OWNERS.add(self)

    def record_failure(self, stage: str, error: BaseException) -> list[ResourceFailure]:
        """Retain safe failure details and reject new operations."""
        failures = _capture(stage, error)
        self._resource_failures.extend(failures)
        self._adapter_sealed = True
        _UNCERTAIN_HTTP_OWNERS.add(self)
        return failures

    def async_client_adopter(self) -> Callable[[HTTPGenerationClient], CloseTaskFactory] | None:
        """Return the current constructor's early-adoption callback."""
        current = self._adapter_current()
        initialization = getattr(current, "initialization", None)
        return initialization.adopt_client if initialization is not None else None

    def start_close(self, function: Callable[[], Awaitable[None]]) -> asyncio.Task[None]:
        """Own an internal HTTP close Task before it can dispatch."""
        lease = self.current_task_lease()
        if lease is None:
            raise RuntimeError("http_reasoner_close_task_not_owned")
        work = _CloseWork(self, lease, _AsyncCloser(function))
        self._close_work.add(work)
        closing = work.close()  # reserve BEFORE create_task can dispatch
        pending = self._close_task(work, closing)
        try:
            return asyncio.create_task(pending)
        except BaseException:
            self.uncertain(work.reservation, pending,
                           "http_reasoner_inner_close_dispatch_unconfirmed")
            raise

    async def _close_task(self, work, closing):
        try:
            await closing
        finally:
            if work.failures:
                self._resource_failures.extend(work.failures)
                self._adapter_sealed = True
                _UNCERTAIN_HTTP_OWNERS.add(self)
            elif work.closed_receipt:
                self._close_work.discard(work)
            else:
                self.uncertain(work.reservation, closing,
                               "http_reasoner_inner_close_unconfirmed")
            closing = None
        if work.failures or not work.closed_receipt:
            # Generic HTTP finish_owned_close must reject without delivering
            # KI/SystemExit from a child Task. Controls remain in safe DTOs.
            raise RuntimeError("http_reasoner_inner_close_failed") from None

    def new_bundle(self) -> _HttpClients:
        """Register an SDK/HTTP holder before either acquisition."""
        bundle = _HttpClients()
        with self._adapter_condition:
            self._bundles.append(bundle)
        return bundle

    def retire_attempt(self, bundle: _HttpClients) -> list[ResourceFailure]:
        """Retire a failed acquisition once and retain uncertainty."""
        failures = []
        try:
            bundle.close()
        except BaseException as error:
            failures.extend(_capture("http_reasoner_attempt_close_failed", error))
        if failures:
            self._resource_failures.extend(failures)
            self._adapter_sealed = True
        else:
            with self._adapter_condition:
                self._bundles.remove(bundle)
        return failures

    def close_sources(self, lease: _AdapterLease, source: Any, iterator: Any, *, exhausted: bool = False) -> list[ResourceFailure]:
        """Close distinct iterator/source handles without retrying failure."""
        failures = []
        seen = set()
        for item in (iterator, source):
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            try:
                self._adapter_close_iterator(item, exhausted=exhausted)
            except BaseException as error:
                failures.extend(_capture("http_reasoner_iterator_close_failed", error))
        if failures:
            self._resource_failures.extend(failures)
        else:
            sources = self._streams.get(lease)
            if sources is not None:
                sources.source = sources.iterator = None
                self._streams.pop(lease, None)
        return failures

    def current_task_lease(self) -> _TaskLease | None:
        """Find ownership from actual registered Task identities."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return None
        with self._adapter_condition:
            return next((lease for lease in self._adapter_leases
                         if isinstance(lease, _TaskLease) and lease.active
                         and any(entry.active and entry.task is task for entry in lease.entries)), None)

    @contextmanager
    def task(self, *, streaming: bool = False) -> Iterator[_TaskLease]:
        """Admit an async operation without retaining Thread-local state."""
        parent = self.current_task_lease()
        lease = _TaskLease(self, streaming=streaming)
        try:
            with self._adapter_condition:
                if self._adapter_sealed and parent is None:
                    raise RuntimeError("http_reasoner_closed")
                self._adapter_leases.add(lease)
            yield lease
        finally:
            # Entries may still own actual children after a metadata failure.
            lease.body_active = False
            lease.latest = None
            with self._adapter_condition:
                if not lease.active:
                    self._adapter_leases.discard(lease)
                self._adapter_condition.notify_all()

    @contextmanager
    def task_entered(self, lease: _TaskLease, *, consumer: bool = True) -> Iterator[None]:
        """Track the actual Task executing this admitted operation."""
        task = asyncio.current_task()
        entry = _TaskEntry(task)
        try:
            with self._adapter_condition:
                if lease.owner is not self or not lease.active:
                    raise RuntimeError("http_reasoner_closed")
                lease.entries.add(entry)
                if consumer and lease.streaming:
                    lease.latest = weakref.ref(task)
            yield
        finally:
            entry.active = False
            entry.task = None
            with self._adapter_condition:
                lease.entries.discard(entry)
                if not lease.active:
                    self._adapter_leases.discard(lease)
                self._adapter_condition.notify_all()

    @contextmanager
    def session(self) -> Iterator[_TaskLease]:
        """Borrow the current Task lease or own a direct session."""
        parent = self.current_task_lease()
        if parent is not None:
            yield parent
        else:
            # Private direct session use still has a lifetime/consumer receipt.
            with self.task(streaming=True) as lease:
                with self.task_entered(lease):
                    yield lease

    @contextmanager
    def native_entered(self, parent: _TaskLease, *, initialization: _Initialization | None = None) -> Iterator[None]:
        """Loan ownership to the actual native constructor call."""
        previous = getattr(self._adapter_local, "lease", None)
        lease = _AdapterLease(self)
        lease.initialization = initialization
        try:
            with self._adapter_condition:
                if parent.owner is not self or not parent.active:
                    raise RuntimeError("http_reasoner_closed")
                self._adapter_leases.add(lease)
            self._adapter_local.lease = lease
            yield
        finally:
            lease.active = lease.on_stack = False
            try:
                self._adapter_local.lease = previous
            finally:
                with self._adapter_condition:
                    self._adapter_leases.discard(lease)
                    self._adapter_condition.notify_all()

    def prepare(self, lease: _TaskLease, function: Callable[..., Any], *args: Any) -> Any:
        """Run native preparation while preserving control failures."""
        try:
            with self.native_entered(lease):
                return function(*args)
        except BaseException as error:
            # A native KI must not become an unobserved asyncio.Task exception.
            return _PreparationFailure(error)
        finally:
            function = None
            args = ()

    async def run_task(self, lease: _TaskLease, awaitable: Awaitable[Any], *, consumer: bool = False) -> Any:
        """Run an admitted coroutine under its actual Task identity."""
        started = False
        try:
            with self.task_entered(lease, consumer=consumer):
                started = True
                return await awaitable
        finally:
            if not started and callable(getattr(awaitable, "close", None)):
                awaitable.close()
            awaitable = None

    async def _read_task(self, lease, reservation, iterator):
        with self.reserved(lease, reservation):
            try:
                return await anext(iterator)
            except (KeyboardInterrupt, SystemExit, GeneratorExit) as error:
                return _ReadControl(error)

    async def read_next(self, lease: _TaskLease, iterator: AsyncIterator[Any], timeout: float) -> Any:
        """Retain an explicit read Task through deadline and cleanup."""
        reservation = self.reserve(lease)
        pending = self._read_task(lease, reservation, iterator)
        reservation.awaitable = pending
        reading = None
        try:
            # CPython 3.13 wait_for may await a coroutine inline. Retain an
            # explicit Task so every admitted read has an observable receipt.
            reading = asyncio.create_task(pending)
            result = await asyncio.wait_for(reading, timeout)
        except BaseException:
            if (reading is not None and reading.done() and reading.cancelled()
                    and not reservation.started and not reservation.finished):
                # This exact owned Task terminated before entering the reader.
                reservation.active = False
                reservation.finished = True
                reservation.task = reservation.awaitable = None
                with self._adapter_condition:
                    lease.entries.discard(reservation)
                    self._adapter_condition.notify_all()
            if not reservation.finished:
                self.uncertain(reservation, pending, "http_reasoner_read_dispatch_unconfirmed")
            raise
        finally:
            pending = reading = None
        if isinstance(result, _ReadControl):
            error, result.error = result.error, None
            try:
                raise error
            finally:
                error = result = None
        return result

    @asynccontextmanager
    async def owned(self, lease: _TaskLease, resource: Any, *, initialization: _Initialization | None = None) -> AsyncIterator[Any]:
        """Keep request-local ownership through actual resource cleanup."""
        work = _CloseWork(self, lease, resource)
        if initialization is not None:
            initialization.close_work = work
        self._close_work.add(work)  # before the existing helper can create a Task
        original = None
        try:
            async with owned_resource(work):
                yield resource
        except BaseException as error:
            original = error
        finally:
            if work.failures:
                self._resource_failures.extend(work.failures)
                self._adapter_sealed = True
                # Failed work retains its actual resource; no close retry.
            elif work.closed_receipt:
                self._close_work.discard(work)
            else:
                self.uncertain(work.reservation, work.awaitable,
                               "http_reasoner_close_dispatch_unconfirmed")
                work.failures = (ResourceFailure(
                    "http_reasoner_close_dispatch_unconfirmed", "RuntimeError"),)
            resource = None
        try:
            _raise_failures(work.failures, original)
        finally:
            original = work = None

    def discard(self, lease: _TaskLease, initializing: asyncio.Task, initialization: _Initialization) -> Awaitable[list[ResourceFailure]]:
        """Reserve cancellation cleanup before its Task can dispatch."""
        reservation = self.reserve(lease)
        pending = self._discard(lease, reservation, initializing, initialization)
        reservation.awaitable = pending
        initialization.discard_reservation = reservation
        initialization.discard_awaitable = pending
        return pending

    async def _discard(self, lease, reservation, initializing, initialization):
        resources = work = None
        failures = []
        try:
            with self.reserved(lease, reservation):
                await initializing
                resources = initialization.resources
                if resources.client is not None:
                    work = _CloseWork(self, lease, resources.client)
                    initialization.close_work = work
                    self._close_work.add(work)
                    await work.close()  # same pre-existing discard Task
                    failures.extend(work.failures)
                    if work.failures:
                        self._resource_failures.extend(work.failures)
                        self._adapter_sealed = True
                    elif work.closed_receipt:
                        self._close_work.discard(work)
                    else:
                        self.uncertain(work.reservation, work.awaitable,
                                       "http_reasoner_discard_close_unconfirmed")
                resources.error = None
        except BaseException as error:
            failures.extend(_capture("http_reasoner_async_discard_failed", error))
        finally:
            resources = work = initializing = initialization = None
        return failures

    def _adapter_require_close_caller(self):
        super()._adapter_require_close_caller()
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None:
            return
        native = self._native_reservation
        if native is not None and native.active and native.task is task:
            raise RuntimeError("http_reasoner_close_from_owner")
        with self._adapter_condition:
            for lease in self._adapter_leases:
                if not isinstance(lease, _TaskLease) or not lease.active:
                    continue
                if (any(entry.active and entry.task is task for entry in lease.entries)
                        or (lease.streaming and lease.latest is not None and lease.latest() is task)):
                    raise RuntimeError("http_reasoner_close_from_worker")

    def require_close_caller(self) -> None:
        """Reject a caller that would await its own resource exit."""
        self._adapter_require_close_caller()

    @staticmethod
    def _require_off_loop():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError("http_reasoner_close_requires_async")

    def _adapter_release_resources(self):
        if (self._resource_failures or self._close_work or self._async_streams or self._initializing
                or any(item.source is not None or item.iterator is not None
                       for item in self._streams.values())):
            raise RuntimeError("http_reasoner_resource_retirement_failed")
        reasoner = None
        clients = ()
        failures = []
        try:
            clients = tuple(bundle.sdk for bundle in self._bundles if bundle.sdk is not None)
            for bundle in self._bundles:
                try:
                    bundle.close()
                except BaseException as error:
                    failures.extend(bundle.failures or _capture("http_reasoner_clients_close_failed", error))
            if failures:
                self._resource_failures.extend(failures)
                raise RuntimeError("http_reasoner_resource_retirement_failed")
            reasoner = self._reasoner()
            if reasoner is not None and any(reasoner._client is client for client in clients):
                reasoner._client = None
                reasoner._resolved_model = reasoner._resolved_base_url = None
            self._bundles.clear()
        finally:
            reasoner = None
            clients = ()

    def _finish_native(self):
        if self._dispatch_uncertain:
            return
        try:
            self._adapter_wait_closed()
        except BaseException as error:
            if not self._resource_failures:
                self._finish_failures = tuple(_capture("http_reasoner_close_failed", error))
        finally:
            if self._resource_failures or self._finish_failures:
                _UNCERTAIN_HTTP_OWNERS.add(self)

    def _native_close_call(self, reservation):
        try:
            self._finish_native()
        finally:
            reservation.native_finished = True

    async def _native_close_run(self, reservation):
        reservation.task = asyncio.current_task()
        reservation.started = True
        try:
            await asyncio.to_thread(self._native_close_call, reservation)
        finally:
            if reservation.native_finished:
                reservation.active = False
                reservation.finished = True
                reservation.task = reservation.awaitable = None
                self._native_pending = None

    def wait_closed(self) -> None:
        """Observe actual retirement from a synchronous off-loop caller."""
        self._require_off_loop()  # before sealing, even for an empty instance
        self.require_close_caller()
        self.request_close()
        self._finish_native()
        _raise_failures([*self._resource_failures, *self._finish_failures])

    def close(self) -> None:
        """Retire this instance from a synchronous caller."""
        self.wait_closed()

    def begin_close(self) -> asyncio.Task[None]:
        """Seal this instance and retain one actual retirement Task."""
        self.require_close_caller()  # original Task/Thread before fresh Context
        loop = asyncio.get_running_loop()
        self.request_close()
        with self._adapter_condition:
            if self._adapter_closing is None:
                if self._close_dispatch_started:
                    raise RuntimeError("http_reasoner_close_dispatch_unconfirmed")
                async def finish():
                    native = None
                    try:
                        await asyncio.sleep(0)
                        while True:
                            if self._dispatch_uncertain:
                                return
                            with self._adapter_condition:
                                admission = self._adapter_close_admission
                                active = any(lease.active for lease in self._adapter_leases)
                            if admission is not None:
                                # An actual native closer already owns disposal.
                                # Observe its receipt without occupying a worker.
                                if admission.finished:
                                    if admission.failure is not None and not self._resource_failures:
                                        self._finish_failures = (ResourceFailure(
                                            "http_reasoner_close_failed", "RuntimeError"),)
                                    return
                            elif not active:
                                break
                            await asyncio.sleep(.005)
                        # Only actual disposal uses the pool; no Condition wait
                        # can run ahead of the admitted cleanup in its queue.
                        reservation = _Reservation()
                        reservation.native_finished = False
                        self._native_reservation = reservation
                        pending = self._native_close_run(reservation)
                        reservation.awaitable = self._native_pending = pending
                        try:
                            native = asyncio.create_task(pending, context=Context())
                        except BaseException:
                            self.uncertain(reservation, pending,
                                           "http_reasoner_native_dispatch_unconfirmed")
                            raise
                        while not native.done():
                            try:
                                await asyncio.shield(native)
                            except asyncio.CancelledError:
                                continue
                        native.result()
                    except BaseException as error:
                        self._finish_failures = tuple(_capture("http_reasoner_close_failed", error))
                        _UNCERTAIN_HTTP_OWNERS.add(self)
                    finally:
                        native = None
                        self._outer_pending = None
                        # Graph borrowers await begin_close() directly. A safe
                        # failed receipt must reject there, never return quiet;
                        # controls are reconstructed only by aclose observers.
                        if (self._resource_failures or self._finish_failures
                                or self._dispatch_uncertain):
                            raise RuntimeError("http_reasoner_close_failed") from None
                pending = finish()
                self._outer_pending = pending
                self._close_dispatch_started = True
                try:
                    self._adapter_closing = loop.create_task(pending, context=Context())
                except BaseException:
                    self.uncertain(None, pending, "http_reasoner_close_dispatch_unconfirmed")
                    raise
            return self._adapter_closing

    async def aclose(self, closing: asyncio.Task | None = None) -> None:
        """Observe the selected public close Task through cancellation."""
        closing = self.begin_close() if closing is None else closing
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                if not closing.done():
                    raise
        try:
            closing.result()
        except Exception as error:
            if not self._resource_failures and not self._finish_failures:
                self._finish_failures = tuple(_capture("http_reasoner_close_failed", error))
        _raise_failures([*self._resource_failures, *self._finish_failures],
                        asyncio.CancelledError() if cancelled else None)
