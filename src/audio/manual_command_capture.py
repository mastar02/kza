"""Invocation ownership for the manual voice-command diagnostic."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import select
import sys
import threading
from typing import Any, Callable

from src.audio.native_capture_authority import ManualVoiceCommandAuthorityClient
from src.audio.zone_capture import (
    ZoneCaptureOwner,
    ZoneCaptureStopped,
    _capture_failures,
    _raise_failures,
)


class ManualCommandStopped(ZoneCaptureStopped):
    """No more live work may enter this invocation."""


@dataclass(eq=False)
class _Model:
    close: bool
    value: Any = None
    entered: bool = False
    close_attempted: bool = False
    released: bool = False
    closer: threading.Thread | None = None


@dataclass(eq=False)
class _Iterator:
    value: Any
    exhausted: bool = False
    close_attempted: bool = False
    closed: bool = False

@dataclass(eq=False, slots=True)
class _SegmentAcquisition:
    source: _Iterator
    iterator: _Iterator
    wrapper: Any = None
    result: Any = None
    model: Any = None
    audio: Any = None
    entered: bool = False
    failed_without_source: bool = False
    iterator_entered: bool = False
    uncertain: bool = False


@dataclass(eq=False)
class _Close:
    owner: threading.Thread
    result: tuple | None = None


class _Segments:
    def __init__(self, owner, resource, source):
        self.owner, self.resource, self.source = owner, resource, source

    def __iter__(self):
        return self

    def __next__(self):
        with self.owner.operation("segment"):
            try:
                return next(self.resource.value)
            except StopIteration:
                self.resource.exhausted = self.source.exhausted = True
                raise


class ManualCommandCapture(ZoneCaptureOwner):
    """Reuse capture receipts, with a distinct authority and owned SDK scope.

    No model close is inferred: only resources explicitly registered with
    close=True have their public close called. Other model cleanup releases
    this invocation's Python references after its actual work has returned.
    """

    def __init__(self, authority_factory: Callable | None):
        super().__init__(authority_factory)
        self._models: dict[str, _Model] = {}
        self._iterators: dict[int, _Iterator] = {}
        self._segment_acquisitions: list[_SegmentAcquisition] = []
        self._manual_close: _Close | None = None
        self._sequence = 0

    def start(self) -> None:
        with self.operation("startup", require_authority=False):
            factory, self._authority_factory = self._authority_factory, None
            if factory is None:
                raise RuntimeError("manual_command_authority_required")
            self.authority = factory()
            if not isinstance(self.authority, ManualVoiceCommandAuthorityClient):
                self.authority = None
                raise RuntimeError("manual_command_authority_required")
            self.authority.register()
            self.authority.activate()
            self.authority.start_monitor(self.seal)
            self.require_current()

    def require_current(self) -> None:
        if self._stop.is_set() or self._sealed:
            raise ManualCommandStopped("manual_command_stopped")
        if self.authority is None or not self.authority.current():
            self.seal()
            raise ManualCommandStopped("manual_command_stopped")

    @contextmanager
    def operation(self, kind: str = "main", *, require_authority: bool = True):
        # The reviewed base owns an identity lease through the caller's finally.
        with super().operation(kind, require_authority=require_authority):
            yield

    def construct(self, name: str, factory: Callable, *, close: bool = False):
        slot = _Model(close)
        with self.operation("constructor"):
            with self._condition:
                if name in self._models:
                    raise RuntimeError("manual_command_resource_still_owned")
                self._models[name] = slot
            self.require_current()
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
                raise RuntimeError("manual_command_constructor_unconfirmed")
            self.require_current()
            return slot.value

    def open_record_stream(self, factory: Callable):
        self._sequence += 1
        key = "record_" + str(self._sequence)
        try:
            stream = self.open_stream(key, factory, lambda value: None)
        except BaseException as error:
            try:
                slot = self._streams.get(key)
                if slot is not None and slot.stream is not None:
                    self.close_record_stream(key)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "manual_command_stream_start_failed", [error, cleanup],
                ) from None
            raise
        finally:
            factory = None
        return key, stream

    def close_record_stream(self, key: str) -> None:
        slot = self._streams[key]
        if slot.closed:
            return
        if slot.close_attempted:
            self._uncertain = True
            raise RuntimeError("manual_command_stream_close_unconfirmed")
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
            raise RuntimeError("manual_command_stream_close_failed")

    def _remember_iterator(self, value) -> _Iterator:
        """Accept an already-created source without claiming pre-call ownership."""
        for resource in self._iterators.values():
            if resource.value is value:
                return resource
        try:
            acquisition = self._reserve_segments()
        except BaseException:
            self._uncertain = True
            raise
        acquisition.result = value
        acquisition.entered = True
        acquisition.source.value = value
        return acquisition.source

    def _reserve_segments(self) -> _SegmentAcquisition:
        # Every allocation/publication precedes the SDK acquisition.
        source = _Iterator(None)
        iterator = _Iterator(None)
        acquisition = _SegmentAcquisition(source, iterator)
        acquisition.wrapper = _Segments(self, iterator, source)
        self._segment_acquisitions.append(acquisition)
        self._iterators[id(source)] = source
        self._iterators[id(iterator)] = iterator
        return acquisition

    def _prepare_segments(self, acquisition: _SegmentAcquisition):
        self.require_current()
        acquisition.iterator_entered = True
        try:
            acquisition.iterator.value = iter(acquisition.source.value)
        except BaseException:
            acquisition.uncertain = self._uncertain = True
            raise
        self.require_current()
        return acquisition.wrapper

    def transcribe(self, model, audio, **kwargs):
        with self.operation("segments"):
            acquisition = self._reserve_segments()
            acquisition.model = model
            acquisition.audio = audio
            self.require_current()
            acquisition.entered = True
            try:
                # Own the complete result before indexing or iterator metadata.
                acquisition.result = model.transcribe(audio, **kwargs)
            except Exception:
                # An ordinary synchronous failure completes without a source.
                acquisition.failed_without_source = True
                raise
            except BaseException:
                acquisition.uncertain = self._uncertain = True
                raise
            try:
                acquisition.source.value = acquisition.result[0]
                info = acquisition.result[1]
            except BaseException:
                acquisition.uncertain = self._uncertain = True
                raise
            return self._prepare_segments(acquisition), info

    def _close_segment_resource(self, resource: _Iterator, *, acquired: bool) -> bool:
        if resource.closed:
            return True
        if resource.value is None:
            if acquired:
                self._uncertain = True
                return False
            resource.closed = True
            return True
        if resource.close_attempted:
            self._uncertain = True
            return False
        resource.close_attempted = True
        try:
            close = getattr(resource.value, "close", None)
            if callable(close):
                close()
            elif not resource.exhausted:
                raise RuntimeError("manual_command_iterator_close_unconfirmed")
            resource.value = None
            resource.closed = True
            return True
        except BaseException as error:
            self._uncertain = True
            with self._condition:
                self._failures.extend(_capture_failures(error))
            return False
        finally:
            close = None

    def own_segments(self, source):
        # Explicit transfer of a source created by an external caller.
        try:
            acquisition = self._reserve_segments()
        except BaseException:
            self._uncertain = True
            raise
        acquisition.result = source
        acquisition.entered = True
        acquisition.source.value = source
        with self.operation("segments"):
            return self._prepare_segments(acquisition)

    def close_segments(self) -> None:
        failed = False
        for acquisition in reversed(tuple(self._segment_acquisitions)):
            source, iterator = acquisition.source, acquisition.iterator
            same = source.value is not None and source.value is iterator.value
            if same:
                closed = self._close_segment_resource(iterator, acquired=True)
                if closed:
                    source.value = None
                    source.closed = True
                else:
                    failed = True
            else:
                iterator_closed = self._close_segment_resource(
                    iterator, acquired=acquisition.iterator_entered,
                )
                source_closed = self._close_segment_resource(
                    source, acquired=(acquisition.entered
                                      and not acquisition.failed_without_source),
                )
                failed |= not iterator_closed or not source_closed
            if source.closed and iterator.closed and not acquisition.uncertain:
                acquisition.result = None
                acquisition.model = acquisition.audio = acquisition.wrapper = None
        self._iterators = {
            key: resource for key, resource in self._iterators.items()
            if not resource.closed
        }
        self._segment_acquisitions = [
            acquisition for acquisition in self._segment_acquisitions
            if not (acquisition.source.closed and acquisition.iterator.closed
                    and not acquisition.uncertain)
        ]
        if failed:
            self.seal()

    def release_models(self) -> None:
        for slot in reversed(tuple(self._models.values())):
            if slot.released:
                continue
            if slot.close_attempted:
                self._uncertain = True
                continue
            if slot.value is None:
                if slot.entered:
                    self._uncertain = True
                else:
                    slot.released = True
                continue
            slot.closer = threading.current_thread()
            try:
                if slot.close:
                    slot.close_attempted = True
                    close = getattr(slot.value, "close", None)
                    if not callable(close):
                        raise RuntimeError("manual_command_close_unsupported")
                    close()
                    close = None
                slot.value = None
                slot.released = True
            except BaseException as error:
                self._uncertain = True
                with self._condition:
                    self._failures.extend(_capture_failures(error))
            finally:
                close = None
                slot.closer = None

    def wait_for_enter(self, prompt: str) -> bool:
        """Read borrowed stdin cooperatively; never spawn a reader Thread."""
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

    def finish(self) -> None:
        current = threading.current_thread()
        candidate = _Close(current)
        with self._condition:
            if (any(work.active and work.thread is current for work in self._work)
                    or any(slot.closer is current for slot in self._models.values())
                    or self._manual_close is not None
                    and self._manual_close.owner is current
                    and self._manual_close.result is None):
                raise RuntimeError("manual_command_self_drain")
        self.seal()
        failures = []
        try:
            with self._condition:
                if self._manual_close is None:
                    self._manual_close = candidate
            if self._manual_close is candidate:
                # The caller clears all payload aliases inside its MAIN lease.
                # A blocked native read stays owned; seal is never exit proof.
                self._wait(None, failures)
                for release in (self.close_segments, self.release_models):
                    try:
                        release()
                    except BaseException as error:
                        failures.extend(_capture_failures(error))
                        self._uncertain = True
                super().finish()
        except BaseException as error:
            failures.extend(_capture_failures(error))
        finally:
            if self._manual_close is candidate:
                try:
                    with self._condition:
                        self._condition.notify_all()
                except BaseException as error:
                    failures.extend(_capture_failures(error))
                finally:
                    candidate.result = tuple(failures)
        if self._manual_close is not candidate:
            while self._manual_close.result is None:
                try:
                    with self._condition:
                        if self._manual_close.result is None:
                            self._condition.wait(.05)
                except BaseException as error:
                    failures.extend(_capture_failures(error))
            failures.extend(self._manual_close.result)
        if not self.retired and not failures:
            raise RuntimeError("manual_command_retirement_unconfirmed")
        _raise_failures(failures)
