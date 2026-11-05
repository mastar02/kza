"""Owned, policy-bound capture for the two legacy training aliases."""

from dataclasses import dataclass
import os
import queue
import select
import sys
import threading
from typing import Any, Callable

from src.audio.native_capture_authority import WakewordTrainingAuthorityClient
from src.audio.zone_capture import (
    ZoneCaptureOwner,
    ZoneCaptureStopped,
    _capture_failures,
    _raise_failures,
)


class LegacyCaptureStopped(ZoneCaptureStopped):
    """The invocation cannot admit further live work."""


def fresh_error(error: BaseException) -> BaseException:
    """Reconstruct failure/control information without retaining SDK frames."""
    if isinstance(error, BaseExceptionGroup):
        return BaseExceptionGroup(
            "legacy_wakeword_capture_failed",
            [fresh_error(child) for child in error.exceptions],
        )
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt()
    if isinstance(error, SystemExit):
        code = error.code
        return SystemExit(code if code is None or type(code) is int else 1)
    if isinstance(error, ZoneCaptureStopped):
        return LegacyCaptureStopped("legacy_wakeword_capture_stopped")
    if (type(error) is RuntimeError and len(error.args) == 1
            and type(error.args[0]) is str and error.args[0] in {
                "legacy_wakeword_authority_required",
                "legacy_wakeword_self_drain",
            }):
        return RuntimeError(error.args[0])
    if isinstance(error, Exception):
        return RuntimeError("legacy_wakeword_capture_failed")
    return BaseException("legacy_wakeword_capture_failed")


@dataclass(eq=False)
class _Detector:
    value: Any = None
    entered: bool = False
    close_attempted: bool = False
    closed: bool = False
    closer: threading.Thread | None = None


@dataclass(eq=False)
class _Retirement:
    owner: threading.Thread
    result: tuple | None = None


class LegacyWakewordCapture(ZoneCaptureOwner):
    """Keep a whole legacy command admitted until its real work is retired."""

    def __init__(self, authority_factory: Callable | None):
        super().__init__(authority_factory)
        self._detector = _Detector()
        self._owned_queues: list[queue.Queue] = []
        self._legacy_retirement: _Retirement | None = None
        self._sequence = 0

    def operation(self, kind: str = "main", *, require_authority: bool = True):
        return super().operation(kind, require_authority=require_authority)

    def start(self) -> None:
        with self.operation("startup", require_authority=False):
            factory, self._authority_factory = self._authority_factory, None
            if factory is None:
                raise RuntimeError("legacy_wakeword_authority_required")
            self.authority = factory()
            if not isinstance(self.authority, WakewordTrainingAuthorityClient):
                # A foreign object remains borrowed; never close it.
                self.authority = None
                raise RuntimeError("legacy_wakeword_authority_required")
            self.authority.register()
            self.authority.activate()
            self.authority.start_monitor(self.seal)
            self.require_current()

    def require_current(self) -> None:
        if self._stop.is_set() or self._sealed:
            raise LegacyCaptureStopped("legacy_wakeword_capture_stopped")
        if self.authority is None or not self.authority.current():
            self.seal()
            raise LegacyCaptureStopped("legacy_wakeword_capture_stopped")

    def open_stream(self, factory: Callable):
        self._sequence += 1
        key = "legacy_stream_" + str(self._sequence)
        try:
            stream = super().open_stream(key, factory, lambda value: None)
        except BaseException as error:
            try:
                slot = self._streams.get(key)
                if slot is not None and slot.stream is not None:
                    self.close_stream(key)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "legacy_wakeword_start_failed",
                    [fresh_error(error), fresh_error(cleanup)],
                ) from None
            raise
        finally:
            factory = None
        return key, stream

    def close_stream(self, key: str) -> None:
        slot = self._streams[key]
        if slot.closed:
            return
        if slot.close_attempted:
            self._uncertain = True
            raise RuntimeError("legacy_wakeword_stream_close_unconfirmed")
        slot.close_attempted = True
        failures = []
        stream = slot.stream
        if slot.start_attempted:
            try:
                stream.stop(ignore_errors=False)
            except BaseException as error:
                failures.extend(_capture_failures(error))
        try:
            stream.close(ignore_errors=False)
        except BaseException as error:
            self._uncertain = True
            failures.extend(_capture_failures(error))
        else:
            slot.closed = True
            slot.stream = None
        finally:
            stream = None
        if failures:
            with self._condition:
                self._failures.extend(failures)
            self.seal()
            # Callers can combine this fresh cleanup with a read control.
            _raise_failures(failures)

    def own_detector(self, factory: Callable):
        with self.operation("detector"):
            slot = self._detector
            self.require_current()
            with self._condition:
                if self._sealed:
                    raise LegacyCaptureStopped("legacy_wakeword_capture_stopped")
                if slot.entered:
                    raise RuntimeError("legacy_wakeword_detector_still_owned")
                slot.entered = True
            try:
                slot.value = factory()
            except BaseException:
                self._uncertain = True
                raise
            finally:
                factory = None
            if slot.value is None:
                self._uncertain = True
                raise RuntimeError("legacy_wakeword_detector_unconfirmed")
            self.require_current()
            slot.value.load()
            self.require_current()
            return slot.value

    def release_detector(self) -> None:
        slot = self._detector
        if slot.closed:
            return
        if slot.value is None:
            if slot.entered:
                self._uncertain = True
            else:
                slot.closed = True
            return
        if slot.close_attempted:
            self._uncertain = True
            return
        slot.close_attempted = True
        slot.closer = threading.current_thread()
        try:
            # WakeWordDetector has a reviewed public synchronous close.
            # No private OpenWakeWord disposal method is inferred.
            slot.value.close()
            slot.value = None
            slot.closed = True
        except BaseException as error:
            self._uncertain = True
            with self._condition:
                self._failures.extend(_capture_failures(error))
        finally:
            slot.closer = None

    def own_queue(self, value: queue.Queue) -> queue.Queue:
        with self._condition:
            if self._sealed:
                raise LegacyCaptureStopped("legacy_wakeword_capture_stopped")
            if not any(existing is value for existing in self._owned_queues):
                self._owned_queues.append(value)
        return value

    def next_chunk(self, value: queue.Queue):
        with self.operation("queue"):
            while True:
                self.require_current()
                try:
                    chunk = value.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    self.require_current()
                    return chunk
                finally:
                    chunk = None

    def wait_for_enter(self, prompt: str) -> bool:
        with self.operation("prompt"):
            self.publish(sys.stdout.write, prompt)
            self.publish(sys.stdout.flush)
            fd = sys.stdin.fileno()
            while True:
                self.require_current()
                ready, _, _ = select.select([fd], [], [], .05)
                self.require_current()
                if not ready:
                    continue
                try:
                    value = os.read(fd, 1)
                except BlockingIOError:
                    continue
                if not value:
                    return False
                if value in (b"\n", b"\r"):
                    return True

    def _clear_owned_queues(self) -> None:
        for value in self._owned_queues:
            while True:
                try:
                    value.get_nowait()
                except queue.Empty:
                    break
        self._owned_queues.clear()

    def finish(self) -> None:
        current = threading.current_thread()
        candidate = _Retirement(current)
        with self._condition:
            if (any(work.active and work.thread is current for work in self._work)
                    or self._detector.closer is current
                    or self._legacy_retirement is not None
                    and self._legacy_retirement.owner is current
                    and self._legacy_retirement.result is None):
                raise RuntimeError("legacy_wakeword_self_drain")
        self.seal()
        failures = []
        try:
            with self._condition:
                if self._legacy_retirement is None:
                    self._legacy_retirement = candidate
            if self._legacy_retirement is candidate:
                # No more startup may publish; stop each known stream before
                # waiting for admitted callbacks to return.
                self._wait({"startup"}, failures)
                for key, slot in tuple(self._streams.items()):
                    if slot.stream is None or slot.closed or slot.close_attempted:
                        continue
                    try:
                        self.close_stream(key)
                    except BaseException as error:
                        failures.extend(_capture_failures(error))
                self._wait(None, failures)
                for release in (self._clear_owned_queues, self.release_detector):
                    try:
                        release()
                    except BaseException as error:
                        failures.extend(_capture_failures(error))
                        self._uncertain = True
                # Inherited retirement observes native monitor exit and only
                # acknowledges when no resource remains uncertain.
                super().finish()
        except BaseException as error:
            failures.extend(_capture_failures(error))
        finally:
            if self._legacy_retirement is candidate:
                try:
                    with self._condition:
                        self._condition.notify_all()
                except BaseException as error:
                    failures.extend(_capture_failures(error))
                finally:
                    candidate.result = tuple(failures)
        if self._legacy_retirement is not candidate:
            while self._legacy_retirement.result is None:
                try:
                    with self._condition:
                        if self._legacy_retirement.result is None:
                            self._condition.wait(.05)
                except BaseException as error:
                    failures.extend(_capture_failures(error))
            failures.extend(self._legacy_retirement.result)
        if not self.retired and not failures:
            raise RuntimeError("legacy_wakeword_retirement_unconfirmed")
        _raise_failures(failures)
