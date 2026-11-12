"""Stop and readiness contract of the managed process entry (U3b-0d1).

Stdlib only. The managed entry arms this module before the event loop exists:
SIGTERM and SIGINT are unblocked (a parent's blocked mask is inherited through
fork and exec), and an early SIGTERM is recorded instead of killing the
process. ``main()`` then routes SIGTERM through the loop. A stop never cancels
construction: the graph finishes building, start-up work is skipped, the owned
stack closes and the admission receipt is written as on any other exit.

Readiness goes to a private descriptor passed by the parent in
``KZA_NOTIFY_FD`` (a pipe or socket). Without one, notifications are no-ops.
The controller relay to systemd is a later unit; this module never sees
``NOTIFY_SOCKET``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from enum import Enum
import logging
import os
import signal
import stat
import sys
import time

logger = logging.getLogger(__name__)

NOTIFY_FD_ENV = "KZA_NOTIFY_FD"
WATCHDOG_INTERVAL_ENV = "KZA_WATCHDOG_INTERVAL_USEC"
STOP_SIGNALS = frozenset({signal.SIGTERM, signal.SIGINT})
_STATUS_LIMIT = 200


class StopPhase(Enum):
    """Content-free process phases, logged when a stop arrives."""

    ENTRY = "entry"
    CONSTRUCTION = "construction"
    WARMUP = "warmup"
    SERVICES = "services"
    PIPELINE = "pipeline"
    READY = "ready"
    CAPTURE_BLOCKED = "capture_blocked"
    CLOSING = "closing"


class StopRequested(Exception):
    """Raised at a start-up checkpoint once a stop was requested (a normal exit)."""


class StopSignal:
    """Process-wide, idempotent stop request shared by main, graph and pipeline.

    The first request records the phase, sets the event and notifies the
    listeners. Every later request (a second SIGTERM, a re-raised signal) only
    increments a counter.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._event = asyncio.Event()
        self._requested = False
        self._requests = 0
        self._phase = StopPhase.ENTRY
        self._requested_phase: StopPhase | None = None
        self._requested_at: float | None = None
        self._source: str | None = None
        self._announced = False
        self._installed = False
        self._listeners: list[Callable[[], None]] = []

    @property
    def requested(self) -> bool:
        """Whether a stop has been requested."""
        return self._requested

    @property
    def requests(self) -> int:
        """Number of stop requests received, including ignored repeats."""
        return self._requests

    @property
    def phase(self) -> StopPhase:
        """The current process phase."""
        return self._phase

    @property
    def requested_phase(self) -> StopPhase | None:
        """The phase that was current when the first stop arrived."""
        return self._requested_phase

    def enter(self, phase: StopPhase) -> None:
        """Record a phase transition (content-free)."""
        self._phase = phase

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Call ``listener`` once, on the first stop request."""
        self._listeners.append(listener)

    def request(self, source: str = "SIGTERM") -> None:
        """Request a stop; idempotent."""
        self._requests += 1
        if self._requested:
            logger.info("[Stop] repeated stop request ignored (count=%d)", self._requests)
            return
        self._mark(source)
        self._announce()

    def _early_signal(self, signum: int, frame: object) -> None:
        # Python-level handler before the loop exists: no logging, no listeners.
        self._requests += 1
        if not self._requested:
            self._mark(signal.Signals(signum).name)

    def _mark(self, source: str) -> None:
        self._requested = True
        self._requested_phase = self._phase
        self._requested_at = self._clock()
        self._source = source
        self._event.set()

    def _announce(self) -> None:
        if self._announced:
            return
        self._announced = True
        logger.info("[Stop] stop requested by %s during phase=%s", self._source,
                    self._requested_phase.value)
        for listener in self._listeners:
            try:
                listener()
            except Exception as error:
                logger.warning("[Stop] stop listener failed: %s", type(error).__name__)

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        """Route SIGTERM through ``loop``; it stays installed until process exit."""
        loop.add_signal_handler(signal.SIGTERM, self.request, "SIGTERM")
        self._installed = True
        if self._requested:
            self._announce()

    async def wait(self) -> None:
        """Wait until a stop is requested."""
        await self._event.wait()

    def checkpoint(self, step: str) -> None:
        """Raise StopRequested before ``step`` when a stop was requested."""
        if self._requested:
            logger.info("[Stop] start-up stopped before %s", step)
            raise StopRequested(step)

    def elapsed(self) -> float | None:
        """Seconds since the first stop request, or None."""
        return None if self._requested_at is None else self._clock() - self._requested_at


class ReadinessNotifier:
    """Write notification lines to a private pipe passed in by the parent.

    Allowed lines: READY=1, STOPPING=1 and STATUS=<ascii, at most 200>. The
    descriptor is non-blocking: a full pipe drops the line instead of blocking
    the event loop.
    """

    def __init__(self, fd: int | None = None, watchdog_usec: int | None = None) -> None:
        self._fd = fd
        self.watchdog_usec = watchdog_usec
        self._sent: list[str] = []
        self._dropped = 0

    @classmethod
    def from_environment(cls, environ: dict | None = None) -> ReadinessNotifier:
        """Take the descriptor and watchdog interval from the environment.

        Both variables are removed so subprocesses never inherit them, and the
        descriptor is made non-inheritable. A malformed value disables
        notifications with a warning.
        """
        environ = os.environ if environ is None else environ
        raw = environ.pop(NOTIFY_FD_ENV, None)
        watchdog = _positive_int(environ.pop(WATCHDOG_INTERVAL_ENV, None))
        if raw is None:
            return cls(None, watchdog)
        fd = _positive_int(raw)
        try:
            if fd is None or fd < 3:
                raise ValueError("notify_fd_invalid")
            mode = os.fstat(fd).st_mode
            if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)):
                raise ValueError("notify_fd_not_pipe")
            os.set_inheritable(fd, False)
            os.set_blocking(fd, False)
        except (OSError, ValueError) as error:
            logger.warning("[Notify] notifications disabled: %s", getattr(error, "args", ("",))[0]
                           if isinstance(error, ValueError) else type(error).__name__)
            return cls(None, watchdog)
        return cls(fd, watchdog)

    @property
    def enabled(self) -> bool:
        """Whether a descriptor is attached."""
        return self._fd is not None

    @property
    def sent(self) -> tuple[str, ...]:
        """Lines written so far (content-free protocol lines)."""
        return tuple(self._sent)

    def ready(self) -> None:
        """Send READY=1 once."""
        self._send_once("READY=1")

    def stopping(self) -> None:
        """Send STOPPING=1 once."""
        self._send_once("STOPPING=1")

    def status(self, text: str) -> None:
        """Send a short ASCII status line."""
        if (not text or len(text) > _STATUS_LIMIT or not text.isascii()
                or not text.isprintable()):
            raise ValueError("notify_status_invalid")
        self._send("STATUS=" + text)

    def _send_once(self, line: str) -> None:
        if line not in self._sent:
            self._send(line)

    def _send(self, line: str) -> None:
        if self._fd is None:
            return
        try:
            os.write(self._fd, (line + "\n").encode("ascii"))
        except OSError as error:
            self._dropped += 1
            if self._dropped == 1:
                logger.warning("[Notify] notification dropped: %s", type(error).__name__)
            return
        self._sent.append(line)


class ProcessLifecycle:
    """The stop signal, readiness notifier and receipt hook of one process.

    Args:
        stop: The process-wide stop signal.
        notifier: Readiness notifications to the parent.
    """

    def __init__(self, stop: StopSignal | None = None, notifier: ReadinessNotifier | None = None) -> None:
        self.stop = stop if stop is not None else StopSignal()
        self.notifier = notifier if notifier is not None else ReadinessNotifier()
        self._admission = None
        self.stop.add_listener(self.notifier.stopping)

    def ready(self) -> None:
        """Called by the pipeline right after 'System ready'.

        Raises:
            StopRequested: a stop arrived first; READY is never sent while stopping.
        """
        self.stop.checkpoint("ready")
        self.stop.enter(StopPhase.READY)
        self.notifier.ready()

    def capture_blocked(self) -> None:
        """Announce a degraded but serving process: capture is blocked.

        The dashboard, routines and HA sync still run, so the service is
        ready (READY=1) with a STATUS line instead of never becoming ready.

        Raises:
            StopRequested: a stop arrived first; nothing is announced.
        """
        self.stop.checkpoint("capture_blocked")
        self.stop.enter(StopPhase.CAPTURE_BLOCKED)
        self.notifier.ready()
        self.notifier.status("capture blocked")

    def attach_admission(self, admission: object) -> None:
        """Bind the managed admission; logging closes before its receipt."""
        if self._admission is not None:
            raise RuntimeError("process_lifecycle_admission_already_attached")
        self._admission = admission
        admission.set_receipt_preparation(self.before_receipt)

    @property
    def receipt_written(self) -> bool:
        """Whether the managed admission wrote its closure receipt."""
        return self._admission is not None and self._admission.receipt_written is True

    def before_receipt(self) -> None:
        """Log the receipt intent, then close logging (review D1)."""
        logger.info("[Stop] closure receipt next; closing logging (phase=%s, elapsed=%s)",
                    self.stop.phase.value, _seconds(self.stop.elapsed()))
        close_logging()

    def flush_for_exit(self) -> None:
        """Flush everything an ``os._exit`` would otherwise discard."""
        close_logging()
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass


def arm_process_entry(environ: dict | None = None) -> ProcessLifecycle:
    """First statement of the managed entry.

    Unblocks SIGTERM/SIGINT (the mask survives exec) and records a SIGTERM
    that arrives before the loop installs its own handler.

    Returns:
        The lifecycle that ``main()`` uses for this process.
    """
    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
    lifecycle = ProcessLifecycle(StopSignal(), ReadinessNotifier.from_environment(environ))
    signal.signal(signal.SIGTERM, lifecycle.stop._early_signal)
    return lifecycle


async def run_until_stopped(work: Coroutine, stop: StopSignal) -> None:
    """Run ``work`` until it returns or a stop is requested, then cancel it.

    Work failures propagate; a cancellation caused by the stop is a normal
    return. Outer cancellation still propagates.
    """
    if stop.requested:
        work.close()
        return
    running = asyncio.create_task(work, name="stop-bounded-work")
    waiting = asyncio.create_task(stop.wait(), name="stop-wait")
    try:
        await asyncio.wait({running, waiting}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiting.cancel()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, waiting, return_exceptions=True)
    if running.cancelled() and stop.requested:
        logger.info("[Stop] %s cancelled by stop request", running.get_name())
        return
    running.result()


def blocked_stop_signals() -> frozenset[int]:
    """The stop signals currently blocked in the calling thread."""
    return frozenset(signal.pthread_sigmask(signal.SIG_BLOCK, ())) & STOP_SIGNALS


def close_logging() -> None:
    """Flush, close and detach every logging handler (idempotent).

    Detaching matters: a closed FileHandler reopens its file on the next emit.
    Later records fall back to logging.lastResort (stderr, WARNING and up).
    """
    loggers = [logging.getLogger()]
    loggers.extend(item for item in logging.Logger.manager.loggerDict.values()
                   if isinstance(item, logging.Logger))
    for owner in loggers:
        for handler in list(owner.handlers):
            try:
                handler.flush()
                handler.close()
            except (OSError, ValueError):
                pass
            owner.removeHandler(handler)


def _positive_int(raw: str | None) -> int | None:
    if raw is None or not raw.isascii() or not raw.isdigit() or len(raw) > 18:
        return None
    value = int(raw)
    return value if value > 0 else None


def _seconds(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}s"
