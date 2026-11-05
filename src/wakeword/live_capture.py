"""Own a single explicitly authorized wakeword recording or live test."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from src.audio.native_capture_authority import (
    WakewordDiagnosticsAuthorityClient,
    WakewordTrainingAuthorityClient,
)

_ACTIVE: set[LiveWakewordCapture] = set()
_FAILED: set[LiveWakewordCapture] = set()
_OWNERS_LOCK = threading.Lock()


class CapturePolicyStopped(RuntimeError):
    """The invocation cannot admit more capture or publication."""


class LiveWakewordCapture:
    """Retain native resources until their actual retirement has returned.

    A control-plane stop seals admission immediately. The retire-only receipt
    waits for this invocation's stream, callbacks, output and monitor to exit.
    """

    def __init__(self, authority: WakewordTrainingAuthorityClient) -> None:
        self._initialize_authority(authority, WakewordTrainingAuthorityClient)

    def _initialize_authority(self, authority: Any, expected_type: type) -> None:
        if not isinstance(authority, expected_type):
            raise RuntimeError("wakeword_capture_authority_required")
        self.authority = authority
        self.stop = threading.Event()
        self._condition = threading.Condition(threading.RLock())
        self._callbacks: dict[int, int] = {}
        self._outputs: dict[int, int] = {}
        self._callback_failures: list[BaseException] = []
        self._stream: Any = None
        self._start_attempted = False
        self._close_attempted = False
        self._detector: Any = None
        self._uncertain = False
        self._finished = False

    def seal(self) -> None:
        """Permanently reject new work without claiming native retirement."""
        with self._condition:
            self.stop.set()
            self._condition.notify_all()

    def require_current(self) -> None:
        """Check the short local control fence before admitting work."""
        if self.stop.is_set() or not self.authority.current():
            self.seal()
            raise CapturePolicyStopped("wakeword_capture_stopped")

    def __enter__(self) -> LiveWakewordCapture:
        with _OWNERS_LOCK:
            _ACTIVE.add(self)
        try:
            self.authority.register()
            self.authority.activate()
            self.authority.start_monitor(self.seal)
            self.require_current()
            return self
        except BaseException as error:
            self._finish(error)
            raise

    def pause(self, seconds: float) -> None:
        """Wait between prompts while remaining responsive to a policy stop."""
        self.require_current()
        self.stop.wait(seconds)
        self.require_current()

    def publish(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Fence output admission and retain its actual return as work."""
        ident = threading.get_ident()
        with self._condition:
            self.require_current()
            self._outputs[ident] = self._outputs.get(ident, 0) + 1
        try:
            result = function(*args, **kwargs)
            self.require_current()
            return result
        finally:
            with self._condition:
                self._outputs[ident] -= 1
                if not self._outputs[ident]:
                    del self._outputs[ident]
                self._condition.notify_all()

    def callback(self, function: Callable) -> Callable:
        """Wrap SDK callbacks with local admission and a retained drain."""
        def receive(*args: Any) -> None:
            ident = threading.get_ident()
            with self._condition:
                try:
                    self.require_current()
                except CapturePolicyStopped:
                    return
                self._callbacks[ident] = self._callbacks.get(ident, 0) + 1
            try:
                function(*args)
            except BaseException as error:
                # Retained owners must not retain SDK frames or PCM through
                # a callback exception's traceback/context/arguments.
                if isinstance(error, Exception):
                    failure = RuntimeError("wakeword_capture_callback_failed")
                elif isinstance(error, KeyboardInterrupt):
                    failure = KeyboardInterrupt()
                elif isinstance(error, SystemExit):
                    code = error.code
                    failure = SystemExit(code if code is None or type(code) is int else 1)
                else:
                    failure = BaseException("wakeword_capture_callback_failed")
                with self._condition:
                    self._callback_failures.append(failure)
                self.seal()
            finally:
                args = ()
                with self._condition:
                    self._callbacks[ident] -= 1
                    if not self._callbacks[ident]:
                        del self._callbacks[ident]
                    self._condition.notify_all()
        return receive

    def open_stream(self, factory: Callable[[], Any]) -> Any:
        """Own a late constructor result and start only while still allowed."""
        self.require_current()
        if self._stream is not None or self._uncertain:
            raise RuntimeError("wakeword_capture_stream_still_owned")
        self._start_attempted = self._close_attempted = False
        try:
            self._stream = factory()
        except BaseException:
            self._uncertain = True
            raise
        if self._stream is None:
            self._uncertain = True
            raise RuntimeError("wakeword_capture_stream_unconfirmed")
        self.require_current()
        self._start_attempted = True
        self._stream.start()
        self.require_current()
        return self._stream

    def close_stream(self) -> None:
        """Observe one strict stop/close before another sample can start."""
        if self._stream is None:
            return
        if self._close_attempted:
            raise RuntimeError("wakeword_capture_stream_close_unconfirmed")
        self._close_attempted = True
        failures: list[BaseException] = []
        if self._start_attempted:
            try:
                self._stream.stop(ignore_errors=False)
            except BaseException as error:
                failures.append(error)
        try:
            self._stream.close(ignore_errors=False)
            self._stream = None
        except BaseException as error:
            self._uncertain = True
            failures.append(error)
        if failures:
            if len(failures) == 1:
                raise failures[0]
            raise BaseExceptionGroup("wakeword_capture_stream_close_failed", failures)

    def own_detector(self, factory: Callable[[], Any]) -> Any:
        """Retain even a partially loaded detector for actual close."""
        self.require_current()
        try:
            self._detector = factory()
        except BaseException:
            self._uncertain = True
            raise
        if self._detector is None:
            self._uncertain = True
            raise RuntimeError("wakeword_capture_detector_unconfirmed")
        self.require_current()
        self._detector.load()
        self.require_current()
        return self._detector

    def _release_owned_resources(self) -> None:
        """Release any additional owned Python references after native drains."""

    def _finish(self, prior_error: BaseException | None) -> None:
        with self._condition:
            self.seal()
            ident = threading.get_ident()
            if ident in self._callbacks or ident in self._outputs:
                raise RuntimeError("wakeword_capture_self_drain")
            if self._finished:
                raise RuntimeError("wakeword_capture_retirement_already_attempted")
            self._finished = True
        failures: list[BaseException] = []

        def attempt(function: Callable, *, uncertain: bool = False) -> bool:
            try:
                function()
                return True
            except BaseException as error:
                failures.append(error)
                self._uncertain |= uncertain
                return False

        if self._stream is not None and not self._close_attempted:
            attempt(self.close_stream)
        with self._condition:
            while self._callbacks or self._outputs:
                attempt(self._condition.wait)
        if not self._callbacks and self._detector is not None:
            if attempt(self._detector.close, uncertain=True):
                self._detector = None
        attempt(self._release_owned_resources, uncertain=True)
        failures.extend(self._callback_failures)
        self._callback_failures.clear()
        monitor_done = attempt(self.authority.stop_monitor)
        if not self._uncertain and monitor_done:
            attempt(self.authority.retire)
        attempt(self.authority.close)
        with _OWNERS_LOCK:
            if self._uncertain or not monitor_done or failures:
                _FAILED.add(self)
            else:
                _ACTIVE.discard(self)
        if failures:
            errors = ([prior_error] if prior_error is not None else []) + failures
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("wakeword_capture_cleanup_failed", errors)

    def __exit__(self, exc_type: Any, error: BaseException | None, tb: Any) -> bool:
        self._finish(error)
        return False


class LiveWakewordDiagnosticsCapture(LiveWakewordCapture):
    """Own diagnostic calls and Python model references without an SDK wipe claim."""

    def __init__(self, authority: WakewordDiagnosticsAuthorityClient) -> None:
        self._initialize_authority(authority, WakewordDiagnosticsAuthorityClient)
        self._raw_model: Any = None
        self._model_release_entered = False

    def own_python_model(self, factory: Callable[[], Any]) -> Any:
        """Retain even a late model result; an unreturned constructor is unknown."""
        self.require_current()
        if self._raw_model is not None or self._uncertain:
            raise RuntimeError("wakeword_diagnostic_model_still_owned")
        self._model_release_entered = False
        try:
            self._raw_model = factory()
        except BaseException:
            self._uncertain = True
            raise
        if self._raw_model is None:
            self._uncertain = True
            raise RuntimeError("wakeword_diagnostic_model_unconfirmed")
        self.require_current()
        return self._raw_model

    def release_python_model(self) -> None:
        """Observe owned Python reference release, without inventing Model.close."""
        if self._model_release_entered:
            if self._raw_model is not None:
                raise RuntimeError("wakeword_diagnostic_release_unconfirmed")
            return
        if self._raw_model is None:
            return
        self._model_release_entered = True
        try:
            self._raw_model = None
        except BaseException:
            self._uncertain = True
            raise

    def _release_owned_resources(self) -> None:
        self.release_python_model()

    @staticmethod
    def model_failure(error: BaseException) -> BaseException:
        """Retain no native traceback, PCM or model through a failed SDK call."""
        if isinstance(error, Exception):
            return RuntimeError("wakeword_diagnostic_model_failed")
        if isinstance(error, KeyboardInterrupt):
            return KeyboardInterrupt()
        if isinstance(error, SystemExit):
            code = error.code
            return SystemExit(code if code is None or type(code) is int else 1)
        return BaseException("wakeword_diagnostic_model_failed")
