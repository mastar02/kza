"""Opt-in, read-only direction telemetry isolated from audio processing."""

from collections.abc import Callable, Collection
from dataclasses import dataclass, fields, replace
from datetime import datetime
import math
from threading import TIMEOUT_MAX, Event, Lock, Thread, current_thread
from uuid import uuid4
from weakref import WeakKeyDictionary, ref

from src.audio.microphone_status import DirectionSnapshot, DirectionState, DirectionSupport
from src.audio.xvf_controller import XvfController, XvfReadOutcome, XvfReadResult


@dataclass(frozen=True)
class DirectionObservationConfig:
    """Bound polling and transfer budgets; disabled unless explicitly enabled."""

    enabled: bool = False
    poll_interval_s: float = 0.5
    stale_after_s: float = 2.0
    read_timeout_ms: int = 50
    max_attempts: int = 2
    error_backoff_initial_s: float = 2.0
    error_backoff_max_s: float = 30.0


@dataclass(frozen=True)
class DirectionCalibration:
    """Room transform bound to a specific microphone and physical USB port."""

    offset_deg: float
    rotation_sign: int
    calibration_id: str
    mic_id: str
    mic_usb_port: str


def _finite_number(value: object) -> bool:
    """Accept exact built-in numbers representable as finite consumer floats."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # Python integers are unbounded; float conversion used by consumers is not.
        return False


def _valid_config(config: DirectionObservationConfig) -> bool:
    waits = (config.poll_interval_s, config.error_backoff_initial_s, config.error_backoff_max_s)
    if not (
        type(config.enabled) is bool
        and all(_finite_number(value) and 0 < value <= TIMEOUT_MAX for value in waits)
        and type(config.read_timeout_ms) is int and 1 <= config.read_timeout_ms <= 50
        and type(config.max_attempts) is int and 1 <= config.max_attempts <= 2
        and config.error_backoff_max_s >= config.error_backoff_initial_s
        and _finite_number(config.stale_after_s) and config.stale_after_s > 0
    ):
        return False
    # Validate the actual seconds-to-ms operation before int() in the snapshot.
    # Keep a positive TTL in its published millisecond unit; never round to zero.
    stale_after_ms = config.stale_after_s * 1000
    return _finite_number(stale_after_ms) and stale_after_ms >= 1


def parse_direction_observation(
    raw: object, known_room_ids: Collection[str],
) -> tuple[DirectionObservationConfig, tuple[str, ...], str | None]:
    """Validate opt-in and explicit room selection without blocking voice startup.

    Returns:
        Observer configuration, selected room IDs, and an optional refusal reason.
        Invalid settings disable only the direction pilot, with no fallback polling.
    """
    if raw is None:
        return DirectionObservationConfig(), (), None
    allowed = {item.name for item in fields(DirectionObservationConfig)}
    if not isinstance(raw, dict) or set(raw) - allowed - {'room_ids'}:
        return DirectionObservationConfig(), (), 'invalid_config'
    config = DirectionObservationConfig(**{key: value for key, value in raw.items() if key in allowed})
    if not _valid_config(config):
        return DirectionObservationConfig(), (), 'invalid_config'
    selected = raw.get('room_ids', [])
    if (not isinstance(selected, list) or any(type(room) is not str for room in selected)
            or len(set(selected)) != len(selected) or any(room not in known_room_ids for room in selected)):
        return replace(config, enabled=False), (), 'invalid_room_ids'
    return config, tuple(selected) if config.enabled else (), None


def project_direction(
    snapshot: DirectionSnapshot, last_sample_monotonic: float | None, now_monotonic: float,
) -> DirectionSnapshot:
    """Expire an immutable sample using elapsed time, without transport access."""
    if last_sample_monotonic is None:
        return snapshot
    age_ms = max(0, int((now_monotonic - last_sample_monotonic) * 1000))
    current = replace(snapshot, age_ms=age_ms)
    if age_ms >= current.stale_after_ms and current.state in {
        DirectionState.OBSERVED, DirectionState.NO_SPEECH, DirectionState.UNKNOWN,
    }:
        return replace(current, state=DirectionState.STALE, relative_angle_deg=None,
                       calibrated_room_angle=None, speech_detected=None)
    return current


class DirectionObserver:
    """Own one bounded worker and publish atomic direction snapshots in memory.

    Args:
        controller: Existing controller; only read_param_result is used.
        room_id: Room owning the observation.
        usb_port: Current physical port for calibration identity checks.
        mic_id: Current unambiguous microphone identity.
        config: Explicit opt-in and bounded polling configuration.
        calibration: Optional room transform tied to the current identity.
        monotonic: Elapsed-time clock, independent from wall-clock corrections.
        utc_now: UTC observation clock used only for valid samples.
    """

    _owners_lock = Lock()
    _owners: WeakKeyDictionary = WeakKeyDictionary()

    def __init__(
        self, controller: XvfController, room_id: str, usb_port: str, mic_id: str,
        config: DirectionObservationConfig, calibration: DirectionCalibration | None,
        monotonic: Callable[[], float], utc_now: Callable[[], datetime],
    ) -> None:
        self._controller = controller
        self._room_id = room_id
        self._usb_port = usb_port
        self._mic_id = mic_id
        self._config = config
        self._calibration = calibration
        self._monotonic = monotonic
        self._utc_now = utc_now
        self._memory_lock = Lock()
        self._lifecycle_lock = Lock()
        self._poll_lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._valid = _valid_config(config)
        self._enabled = self._valid and config.enabled
        self._stopped = False
        self._version_requested = False
        self._last_sample_monotonic: float | None = None
        self._backoff_s = config.error_backoff_initial_s if self._valid else 0.0
        self._next_wait_s = config.poll_interval_s if self._valid else 0.0
        self._snapshot = self._initial_snapshot()

    def _initial_snapshot(self) -> DirectionSnapshot:
        return DirectionSnapshot(
            state=DirectionState.UNKNOWN if self._enabled else DirectionState.DISABLED,
            generation=uuid4().hex if self._enabled else None,
            stale_after_ms=int(self._config.stale_after_s * 1000) if self._valid else 2000,
            reason='invalid_config' if not self._valid else None,
        )

    def start(self) -> bool:
        """Start the sole worker, refusing disabled or still-running instances."""
        with self._lifecycle_lock:
            if not self._enabled or (self._thread is not None and self._thread.is_alive()):
                return False
            with self._owners_lock:
                owner_ref = self._owners.get(self._controller)
                owner = owner_ref() if owner_ref else None
                if owner is not None and owner is not self:
                    return False
                self._owners[self._controller] = ref(self)
            with self._memory_lock:
                if self._stopped:
                    self._snapshot = self._initial_snapshot()
                    self._last_sample_monotonic = None
                    self._version_requested = False
                self._stopped = False
                self._stop_event.clear()
                self._backoff_s = self._config.error_backoff_initial_s
            self._thread = Thread(target=self._run, name=f'direction-{self._room_id}', daemon=True)
            self._thread.start()
            return True

    def invalidate(self) -> None:
        """Retire the generation immediately using only memory and a stop signal."""
        self._stop_event.set()
        with self._memory_lock:
            self._stopped = True
            self._snapshot = replace(self._initial_snapshot(), state=DirectionState.DISABLED,
                                     generation=None, reason='stopped')
            self._last_sample_monotonic = None

    def stop(self) -> None:
        """Invalidate immediately and interrupt waits; retain a still-live thread."""
        with self._lifecycle_lock:
            self.invalidate()
            thread = self._thread
            if thread is not None and thread is not current_thread():
                thread.join(timeout=0.25)

    def wait_stopped(self) -> None:
        """Drain the owned worker; async lifecycle callers must use to_thread."""
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()

    def snapshot(self) -> DirectionSnapshot:
        """Read memory and project elapsed age; never poll the controller."""
        with self._memory_lock:
            return project_direction(self._snapshot, self._last_sample_monotonic, self._monotonic())

    def _read(self, name: str) -> XvfReadResult:
        try:
            return self._controller.read_param_result(
                name, timeout_ms=self._config.read_timeout_ms,
                max_attempts=self._config.max_attempts, skip_if_busy=True,
            )
        except Exception:
            # Isolate an unexpected controller failure from audio capture and lifecycle.
            return XvfReadResult(XvfReadOutcome.USB_ERROR)

    def _calibrated_angle(self, angle: float) -> tuple[float | None, str | None, str | None]:
        calibration = self._calibration
        if calibration is None:
            return None, None, None
        identity = self._mic_id
        known_identity = isinstance(identity, str) and (
            (identity.startswith('serial:') and bool(identity[7:].strip()))
            or (identity.startswith('usb:') and '@' in identity
                and identity.split('@', 1)[0] != 'usb:unknown'
                and identity.endswith(f'@{self._usb_port}'))
        )
        if not (
            known_identity and bool(self._usb_port)
            and calibration.mic_id == identity and calibration.mic_usb_port == self._usb_port
            and _finite_number(calibration.offset_deg)
            and type(calibration.rotation_sign) is int and calibration.rotation_sign in (-1, 1)
            and isinstance(calibration.calibration_id, str) and calibration.calibration_id.strip()
        ):
            return None, None, 'invalid_calibration'
        return ((angle * calibration.rotation_sign + calibration.offset_deg) % 360,
                calibration.calibration_id, None)

    def poll_once(self) -> None:
        """Perform one synchronous cycle without queueing concurrent requests.

        Called by the observer worker; direct calls permit deterministic synthetic tests.
        """
        if not self._enabled or self._stopped:
            return
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not current_thread():
            return
        if not self._poll_lock.acquire(blocking=False):
            return
        try:
            with self._memory_lock:
                generation = self._snapshot.generation
                version_requested = self._version_requested
                self._version_requested = True
                previous = self._snapshot
            if not version_requested:
                version_result = self._read('VERSION')
                version = version_result.values
                if (version_result.outcome == XvfReadOutcome.OK and isinstance(version, tuple)
                        and len(version) == 3 and all(type(v) is int and 0 <= v <= 255 for v in version)):
                    previous = replace(previous, version=version)
            if self._stop_event.is_set():
                return
            result = self._read('DOA_VALUE')
            values = result.values
            valid = (result.outcome == XvfReadOutcome.OK and isinstance(values, tuple)
                     and len(values) == 2 and _finite_number(values[0])
                     and 0 <= values[0] < 360 and type(values[1]) is int and values[1] in (0, 1))
            sample_time = None
            if valid:
                angle, speech = float(values[0]), bool(values[1])
                room_angle, calibration_id, reason = self._calibrated_angle(angle)
                current = replace(
                    previous, state=DirectionState.OBSERVED if speech else DirectionState.NO_SPEECH,
                    relative_angle_deg=angle if speech else None,
                    calibrated_room_angle=room_angle if speech else None,
                    speech_detected=speech, sample_seq=previous.sample_seq + 1,
                    support=DirectionSupport.SUPPORTED, observed_at=self._utc_now().timestamp(),
                    age_ms=0, reason=reason, calibration_id=calibration_id,
                )
                sample_time = self._monotonic()
                self._backoff_s = self._config.error_backoff_initial_s
                self._next_wait_s = self._config.poll_interval_s
            else:
                outcome = result.outcome if result.outcome != XvfReadOutcome.OK else XvfReadOutcome.INVALID_PAYLOAD
                current = replace(previous, state=DirectionState.UNKNOWN,
                                  relative_angle_deg=None, calibrated_room_angle=None,
                                  speech_detected=None, reason=outcome.value)
                if outcome == XvfReadOutcome.BUSY:
                    self._next_wait_s = self._config.poll_interval_s
                else:
                    self._next_wait_s = self._backoff_s
                    self._backoff_s = min(self._backoff_s * 2, self._config.error_backoff_max_s)
            with self._memory_lock:
                if not self._stopped and self._snapshot.generation == generation:
                    self._snapshot = current
                    if sample_time is not None:
                        self._last_sample_monotonic = sample_time
        finally:
            self._poll_lock.release()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self.poll_once()
                if self._stop_event.wait(self._next_wait_s):
                    break
        finally:
            with self._owners_lock:
                owner_ref = self._owners.get(self._controller)
                if owner_ref and owner_ref() is self:
                    del self._owners[self._controller]
