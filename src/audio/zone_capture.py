"""Local ownership of one explicitly authorized zone-setup capture."""

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Callable


_OWNERS: set["ZoneCaptureOwner"] = set()
_OWNERS_LOCK = threading.Lock()


class ZoneCaptureStopped(RuntimeError):
    """The current invocation may no longer admit live input or output."""


@dataclass(eq=False)
class _Work:
    thread: threading.Thread
    kind: str
    active: bool = True


@dataclass(eq=False)
class _Stream:
    stream: Any = None
    start_attempted: bool = False
    close_attempted: bool = False
    closed: bool = False


def _failure(error: BaseException) -> tuple[str, int | None]:
    if isinstance(error, KeyboardInterrupt):
        return ("interrupt", None)
    if isinstance(error, SystemExit):
        return ("exit", error.code if error.code is None or type(error.code) is int else 1)
    return ("error" if isinstance(error, Exception) else "control", None)

def _capture_failures(error: BaseException) -> list[tuple[str, int | None]]:
    if isinstance(error, BaseExceptionGroup):
        result = []
        for child in error.exceptions:
            result.extend(_capture_failures(child))
        return result
    return [_failure(error)]


def _raise_failures(failures: list[tuple[str, int | None]]) -> None:
    errors: list[BaseException] = []
    for kind, code in failures:
        if kind == "interrupt":
            errors.append(KeyboardInterrupt())
        elif kind == "exit":
            errors.append(SystemExit(code))
        elif kind == "control":
            errors.append(BaseException("zone_capture_cleanup_failed"))
        else:
            errors.append(RuntimeError("zone_capture_cleanup_failed"))
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("zone_capture_cleanup_failed", errors)


class ZoneCaptureOwner:
    """Keep known streams and real callback returns until one retirement."""

    def __init__(self, authority_factory: Callable | None) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._work: set[_Work] = set()
        self._streams: dict[str, _Stream] = {}
        self._failures: list[tuple[str, int | None]] = []
        self._sealed = False
        self._uncertain = False
        self._finisher: threading.Thread | None = None
        self._result: tuple[bool, tuple] | None = None
        self.authority: Any = None
        self._authority_factory = authority_factory
        self._stop = threading.Event()
        self.retired = False
        with _OWNERS_LOCK:
            _OWNERS.add(self)

    def initialize(self) -> None:
        # Import lazily: constructing a metadata-only MultiMicCapture must not
        # acquire a session, microphone or native monitor.
        from src.audio.native_capture_authority import ZoneSetupAuthorityClient

        with self.operation("startup", require_authority=False):
            factory, self._authority_factory = self._authority_factory, None
            if factory is None:
                raise RuntimeError("zone_capture_authority_required")
            self.authority = factory()
            if not isinstance(self.authority, ZoneSetupAuthorityClient):
                # A foreign object is borrowed: never call its cleanup methods.
                self.authority = None
                raise RuntimeError("zone_capture_authority_required")
            self.authority.register()
            self.authority.activate()
            self.authority.start_monitor(self.seal)
            self.require_current()

    def seal(self) -> None:
        self._stop.set()
        with self._condition:
            self._sealed = True
            self._condition.notify_all()

    def require_current(self) -> None:
        if self._stop.is_set() or self._sealed:
            raise ZoneCaptureStopped("zone_capture_stopped")
        authority = self.authority
        if authority is None or not authority.current():
            self.seal()
            raise ZoneCaptureStopped("zone_capture_stopped")

    @contextmanager
    def operation(self, kind: str, *, require_authority: bool = True):
        lease = _Work(threading.current_thread(), kind)
        try:
            with self._condition:
                if self._sealed:
                    raise ZoneCaptureStopped("zone_capture_stopped")
                if require_authority:
                    self.require_current()
                self._work.add(lease)
            yield
        finally:
            # Inactive is the receipt; notification is advisory.
            lease.active = False
            with self._condition:
                self._work.discard(lease)
                self._condition.notify_all()

    def publish(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        result = None
        try:
            with self.operation("output"):
                try:
                    result = function(*args, **kwargs)
                    self.require_current()
                    return result
                finally:
                    function = result = None
                    args = ()
                    kwargs = {}
        finally:
            function = result = None
            args = ()
            kwargs = {}

    def pause(self, seconds: float) -> None:
        self.require_current()
        self._stop.wait(seconds)
        self.require_current()

    def callback(self, function: Callable) -> Callable:
        def receive(*args: Any) -> None:
            try:
                with self.operation("callback"):
                    try:
                        function(*args)
                    except ZoneCaptureStopped:
                        pass
                    except BaseException as error:
                        descriptors = _capture_failures(error)
                        with self._condition:
                            self._failures.extend(descriptors)
                        self.seal()
                    finally:
                        args = ()
            except ZoneCaptureStopped:
                return
            finally:
                args = ()
        return receive

    def record_failure(self, error: BaseException) -> None:
        """Retain descriptors only, never a caller/SDK traceback."""
        descriptors = _capture_failures(error)
        with self._condition:
            self._failures.extend(descriptors)
        self.seal()

    def open_stream(self, key: str, factory: Callable, owned: Callable) -> Any:
        stream = None
        with self.operation("startup"):
            try:
                slot = _Stream()
                with self._condition:
                    if key in self._streams:
                        raise RuntimeError("zone_capture_stream_still_owned")
                    self._streams[key] = slot
                self.require_current()
                try:
                    stream = factory()
                except BaseException:
                    self._uncertain = True
                    raise
                slot.stream = stream
                if stream is None:
                    self._uncertain = True
                    raise RuntimeError("zone_capture_stream_unconfirmed")
                owned(stream)
                self.require_current()
                slot.start_attempted = True
                stream.start()
                self.require_current()
                return stream
            finally:
                stream = factory = owned = None

    def _wait(self, kinds: set[str] | None, failures: list) -> None:
        while True:
            try:
                with self._condition:
                    pending = any(work.active and (kinds is None or work.kind in kinds)
                                  for work in self._work)
                    if not pending:
                        return
                    self._condition.wait(.05)
            except BaseException as error:
                failures.extend(_capture_failures(error))

    def finish(self) -> None:
        current = threading.current_thread()
        with self._condition:
            if any(work.active and work.thread is current for work in self._work):
                raise RuntimeError("zone_capture_self_drain")
            if self._finisher is current and self._result is None:
                raise RuntimeError("zone_capture_self_drain")
        self.seal()
        elected = False
        failures: list[tuple[str, int | None]] = []
        try:
            with self._condition:
                if self._result is None and self._finisher is None:
                    self._finisher = current
                elected = self._finisher is current and self._result is None
            if elected:
                def attempt(function: Callable, *, uncertain: bool = False) -> bool:
                    try:
                        function()
                        return True
                    except BaseException as error:
                        failures.extend(_capture_failures(error))
                        self._uncertain |= uncertain
                        return False

                self._wait({"startup"}, failures)
                for slot in self._streams.values():
                    stream = slot.stream
                    if stream is None or slot.closed:
                        continue
                    if slot.close_attempted:
                        self._uncertain = True
                        continue
                    slot.close_attempted = True
                    if slot.start_attempted:
                        attempt(lambda: stream.stop(ignore_errors=False))
                    if attempt(lambda: stream.close(ignore_errors=False), uncertain=True):
                        slot.closed = True
                        slot.stream = None
                    stream = None
                self._wait(None, failures)
                with self._condition:
                    failures.extend(self._failures)
                    self._failures.clear()
                authority = self.authority
                monitor_done = authority is None or attempt(
                    authority.stop_monitor, uncertain=True
                )
                if not self._uncertain and monitor_done:
                    self.retired = authority is None or attempt(
                        authority.retire, uncertain=True
                    )
                if authority is not None:
                    attempt(authority.close, uncertain=True)
                if self._uncertain:
                    self.retired = False
                if self.retired:
                    self.authority = None
        except BaseException as error:
            failures.extend(_capture_failures(error))
            if self._finisher is current and self._result is None:
                self._uncertain = True
                self.retired = False
        finally:
            if self._finisher is current and self._result is None:
                elected = True
                try:
                    with self._condition:
                        self._condition.notify_all()
                except BaseException as error:
                    failures.extend(_capture_failures(error))
                    self._uncertain = True
                    self.retired = False
                finally:
                    # One immutable result follows all fallible notification.
                    if not self.retired and not failures:
                        failures.append(("error", None))
                    self._result = (self.retired, tuple(failures))
        if not elected:
            while self._result is None:
                try:
                    with self._condition:
                        if self._result is None:
                            self._condition.wait(.05)
                except BaseException as error:
                    failures.extend(_capture_failures(error))
            failures.extend(self._result[1])
            if not self._result[0] and not failures:
                failures.append(("error", None))
        elif self.retired:
            with _OWNERS_LOCK:
                _OWNERS.discard(self)
        _raise_failures(failures)
