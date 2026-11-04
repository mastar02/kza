"""Identity admission for selected synchronous adapters and their streams.

The adapter owns backend references, not caller outputs or a global executor.
Native iterator retirement must be acknowledged before backend reference release.
"""
import asyncio
from contextlib import contextmanager
from contextvars import Context
from dataclasses import dataclass, field
import threading


@dataclass(eq=False)
class _AdapterLease:
    owner: object
    thread: threading.Thread = field(default_factory=threading.current_thread)
    active: bool = True
    on_stack: bool = True


class _AdapterLifecycle:
    """Retain this adapter's real operations without owning its callers' data."""

    @dataclass(eq=False)
    class _CloseAdmission:
        """Resolve one close attempt without retrying uncertain disposal."""

        owner: threading.Thread
        failure: str | None = "adapter_close_failed"
        finished: bool = False

    def _adapter_initialize(self, prefix: str) -> None:
        self._adapter_prefix = prefix
        self._adapter_lock = threading.RLock()
        self._adapter_condition = threading.Condition()
        self._adapter_local = threading.local()
        self._adapter_leases: set[_AdapterLease] = set()
        self._adapter_sealed = False
        self._adapter_close_started = False
        self._adapter_close_owner = None
        self._adapter_close_admission: _AdapterLifecycle._CloseAdmission | None = None
        self._adapter_close_done = threading.Event()
        self._adapter_failure = None
        self._adapter_closing = None
        self._adapter_retained_iterators: dict[int, object] = {}

    @property
    def _adapter_active(self) -> int:
        with self._adapter_condition:
            return sum(lease.active for lease in self._adapter_leases)

    def _adapter_current(self):
        lease = getattr(self._adapter_local, "lease", None)
        return (lease if lease is not None and lease.owner is self and lease.active
                and lease.on_stack and lease.thread is threading.current_thread() else None)

    @contextmanager
    def _adapter_admit(self):
        previous = getattr(self._adapter_local, "lease", None)
        lease = self._adapter_current()
        owned = lease is None
        if owned:
            lease = _AdapterLease(self)
        try:
            if owned:
                with self._adapter_condition:
                    if self._adapter_sealed:
                        raise RuntimeError(self._adapter_prefix + "_closed")
                    self._adapter_leases.add(lease)
            self._adapter_local.lease = lease
            yield lease
        finally:
            if owned:
                lease.active = False
                lease.on_stack = False
            try:
                self._adapter_local.lease = previous
            finally:
                if owned:
                    with self._adapter_condition:
                        self._adapter_leases.discard(lease)
                        self._adapter_condition.notify_all()

    @contextmanager
    def _adapter_stream_admit(self):
        # This context spans yields without leaving TLS installed in its caller.
        current = self._adapter_current()
        lease = _AdapterLease(self)
        try:
            with self._adapter_condition:
                if self._adapter_sealed and current is None:
                    raise RuntimeError(self._adapter_prefix + "_closed")
                self._adapter_leases.add(lease)
            lease.on_stack = False
            yield lease
        finally:
            lease.active = False
            lease.on_stack = False
            with self._adapter_condition:
                self._adapter_leases.discard(lease)
                self._adapter_condition.notify_all()

    @contextmanager
    def _adapter_entered(self, lease):
        previous = getattr(self._adapter_local, "lease", None)
        previous_stack = lease.on_stack
        try:
            if lease.owner is not self or not lease.active:
                raise RuntimeError(self._adapter_prefix + "_closed")
            # Keep the last real consumer after suspension, including migration.
            lease.thread = threading.current_thread()
            lease.on_stack = True
            self._adapter_local.lease = lease
            yield
        finally:
            lease.on_stack = previous_stack
            self._adapter_local.lease = previous

    def _adapter_track_iterator(self, source):
        # Install before pulling/closing: failed metadata cannot erase ownership.
        self._adapter_retained_iterators[id(source)] = source

    def _adapter_close_iterator(self, source, *, exhausted: bool = False) -> None:
        if source is None:
            return
        self._adapter_track_iterator(source)
        try:
            close = getattr(source, "close", None)
            if close is not None:
                close()
            elif not exhausted:
                raise RuntimeError(self._adapter_prefix + "_iterator_close_failed")
        except BaseException:
            self._adapter_sealed = True
            raise
        else:
            self._adapter_retained_iterators.pop(id(source), None)

    def _adapter_require_close_caller(self) -> None:
        if self._adapter_current() is not None:
            raise RuntimeError(self._adapter_prefix + "_close_from_worker")
        caller = threading.current_thread()
        with self._adapter_condition:
            # Publication can precede TLS. A suspended stream also depends on
            # its latest consumer, which must not wait for itself here.
            if any(lease.active and lease.thread is caller for lease in self._adapter_leases):
                raise RuntimeError(self._adapter_prefix + "_close_from_worker")
            admission = self._adapter_close_admission
            if admission is not None and admission.owner is caller and not admission.finished:
                raise RuntimeError(self._adapter_prefix + "_close_from_owner")

    def request_close(self) -> None:
        """Seal admission using memory only, without waiting for model locks."""
        with self._adapter_condition:
            self._adapter_sealed = True
            self._adapter_condition.notify_all()

    def _adapter_close_failure_code(self, _phase: str, _error: BaseException) -> str:
        """Map a trusted close stage to a fixed code, never SDK error text."""
        return self._adapter_prefix + '_close_failed'

    def wait_closed(self) -> None:
        """Retire once, keeping metadata/SDK failure terminal and observable."""
        self._adapter_require_close_caller()
        self._adapter_wait_closed()

    def _adapter_wait_closed(self) -> None:
        """Drain after the original caller has passed the public close guard."""
        admission = self._CloseAdmission(
            threading.current_thread(), self._adapter_prefix + "_close_failed")
        phase = "admission"
        try:
            self.request_close()
            with self._adapter_condition:
                if self._adapter_close_admission is None:
                    # Identity alone elects disposal; publication and lock exit
                    # are covered before compatibility metadata can fail.
                    self._adapter_close_admission = admission
                    self._adapter_close_started = True
                    self._adapter_close_owner = admission.owner
            if self._adapter_close_admission is admission:
                phase = "drain"
                with self._adapter_condition:
                    while any(lease.active for lease in self._adapter_leases):
                        # Notifications are advisory, including cleanup faults.
                        self._adapter_condition.wait(.05)
                phase = "iterators"
                if self._adapter_retained_iterators:
                    raise RuntimeError(self._adapter_prefix + "_iterator_close_failed")
                phase = "resource"
                self._adapter_release_resources()
                admission.failure = None
        except BaseException as error:
            if self._adapter_close_admission is not admission:
                raise
            # Preserve the established fixed close-error contract. An uncertain
            # metadata/disposal failure never admits a second resource release.
            admission.failure = self._adapter_close_failure_code(phase, error)
        finally:
            if self._adapter_close_admission is admission:
                try:
                    self._adapter_failure = admission.failure
                    self._adapter_close_owner = None
                    self._adapter_close_done.set()
                except BaseException:
                    admission.failure = self._adapter_prefix + "_close_failed"
                    self._adapter_failure = admission.failure
                finally:
                    # This is attempt resolution, including notify failure;
                    # it is not an acknowledgement of successful disposal.
                    admission.finished = True
        current = self._adapter_close_admission
        if current is None:
            raise RuntimeError(self._adapter_prefix + "_close_failed")
        while not current.finished:
            with self._adapter_condition:
                if not current.finished:
                    self._adapter_condition.wait(.05)
        if current.failure is not None:
            raise RuntimeError(current.failure)

    def close(self) -> None:
        """Synchronous retirement; async callers use begin_close or aclose."""
        self.wait_closed()

    def begin_close(self) -> asyncio.Task:
        """Return the sole retained owner; never cancel this owner directly."""
        self._adapter_require_close_caller()
        loop = asyncio.get_running_loop()
        self.request_close()
        with self._adapter_condition:
            if self._adapter_closing is None:
                async def finish():
                    # Publish the task outside the condition even with eager tasks.
                    await asyncio.sleep(0)
                    if (
                        type(self).wait_closed is _AdapterLifecycle.wait_closed
                        and "wait_closed" not in vars(self)
                    ):
                        await asyncio.to_thread(self._adapter_wait_closed)
                    else:
                        await asyncio.to_thread(self.wait_closed)

                self._adapter_closing = loop.create_task(finish(), context=Context())
                self._adapter_closing.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception(),
                    context=Context(),
                )
            return self._adapter_closing

    async def aclose(self) -> None:
        """Keep the real retirement owner through repeated observer cancellation."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError
