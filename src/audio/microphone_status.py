"""Read-only USB identity inventory and immutable microphone telemetry.

Only the producer reads sysfs. Snapshot projection is pure and keeps internal
monotonic clocks out of the public DTOs consumed by the dashboard.
"""

from __future__ import annotations

import stat
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path


class ConnectionState(StrEnum):
    """Connection evidence from the most recent USB inventory."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"


class BindingState(StrEnum):
    """Agreement between configured identity and the opened stream."""

    MATCHED = "matched"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    UNASSIGNED = "unassigned"


class CaptureState(StrEnum):
    """Host callback activity, independent of nonzero signal or speech."""

    UNAVAILABLE = "unavailable"
    CLOSED = "closed"
    WAITING = "waiting"
    RECEIVING = "receiving"
    STALE = "stale"


class VerificationState(StrEnum):
    """Claims for which this telemetry producer has no physical evidence."""

    UNVERIFIED = "unverified"
    PENDING = "pending"


@dataclass(frozen=True)
class MicrophoneRoom:
    """Configuration adapter; callers may include rooms without a stream."""

    room_id: str
    name: str
    configured_usb_port: str | None = None
    configured_serial: str | None = None
    device_index: int | None = None


@dataclass(frozen=True)
class UsbMicrophone:
    """Identity observed from known USB attributes and sound/card links."""

    usb_port: str
    serial: str | None = None
    product: str | None = None
    alsa_card: int | None = None
    observed_at: float | None = None


@dataclass(frozen=True)
class UsbInventory:
    """Internal inventory with clocks and retained identities for removals."""

    observed_at: float
    observed_monotonic: float
    success: bool
    devices: tuple[UsbMicrophone, ...] = ()
    previous_devices: tuple[UsbMicrophone, ...] = ()


@dataclass(frozen=True)
class CaptureObservation:
    """Internal observations for one opening; never serialize this DTO."""

    generation: int = 0
    opened: bool = False
    usb_port: str | None = None
    serial: str | None = None
    alsa_card: int | None = None
    device_index: int | None = None
    frames_received: int = 0
    last_callback_at: float | None = None
    last_callback_monotonic: float | None = None
    last_nonzero_at: float | None = None
    nonzero_seen: bool = False
    sample_rate_hz: float | None = None
    channels: int | None = None
    selected_channel: int | None = None
    host_dtype: str | None = None


@dataclass(frozen=True)
class CaptureSnapshot:
    """Public capture activity and the actual opened host format."""

    state: CaptureState
    generation: int = 0
    frames_received: int = 0
    last_callback_at: float | None = None
    last_nonzero_at: float | None = None
    nonzero_seen: bool = False
    sample_rate_hz: float | None = None
    channels: int | None = None
    selected_channel: int | None = None
    host_dtype: str | None = None


@dataclass(frozen=True)
class VoiceSnapshot:
    """Nonzero samples do not establish speech presence."""

    state: VerificationState = VerificationState.UNVERIFIED
    detected: bool | None = None


class DirectionState(StrEnum):
    """Direction telemetry states, independent from voice and physical acceptance."""

    UNVERIFIED = "unverified"
    DISABLED = "disabled"
    UNKNOWN = "unknown"
    OBSERVED = "observed"
    NO_SPEECH = "no_speech"
    STALE = "stale"


class DirectionSupport(StrEnum):
    """Firmware capability demonstrated only by a valid direction response."""

    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class DirectionSnapshot:
    """Immutable direction telemetry; the base producer performs no DSP reads."""

    state: DirectionState = DirectionState.UNVERIFIED
    relative_angle_deg: float | None = None
    calibrated_room_angle: float | None = None
    observed_at: float | None = None
    speech_detected: bool | None = None
    sample_seq: int = 0
    generation: str | None = None
    support: DirectionSupport = DirectionSupport.UNKNOWN
    version: tuple[int, int, int] | None = None
    age_ms: int | None = None
    stale_after_ms: int = 2000
    reason: str | None = None
    calibration_id: str | None = None


@dataclass(frozen=True)
class PhysicalValidationSnapshot:
    """Physical acceptance requires separate evidence."""

    state: VerificationState = VerificationState.PENDING


@dataclass(frozen=True)
class MicSnapshot:
    """Public microphone identity, room binding, and capture evidence."""

    mic_id: str
    room_id: str | None
    room_name: str | None
    configured_usb_port: str | None
    usb_port: str | None
    serial: str | None
    product: str | None
    alsa_card: int | None
    portaudio_index: int | None
    identity_observed_at: float | None
    connection: ConnectionState
    binding: BindingState
    capture: CaptureSnapshot
    voice: VoiceSnapshot = field(default_factory=VoiceSnapshot)
    direction: DirectionSnapshot = field(default_factory=DirectionSnapshot)
    physical_validation: PhysicalValidationSnapshot = field(default_factory=PhysicalValidationSnapshot)


@dataclass(frozen=True)
class MicrophonesSnapshot:
    """Public immutable projection with explicit cache freshness."""

    observed_at: float
    stale_after_s: float
    stale: bool
    microphones: tuple[MicSnapshot, ...]
    unconfigured_rooms: tuple[str, ...]
    schema_version: int = 1


def _attribute(device: Path, name: str) -> str | None:
    """Read one known, regular sysfs attribute without following special files."""
    path = device / name
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise OSError("Unsupported USB attribute type")
    return path.read_text(encoding="utf-8").strip() or None


def read_usb_microphones(usb_root: Path) -> UsbInventory:
    """Inventory USB audio devices using known attributes and sound/cardN.

    Args:
        usb_root: Sysfs USB devices directory, injectable for synthetic tests.

    Returns:
        An immutable successful inventory or an explicit failed observation.
        No PortAudio, PCM, USB control transfer, or model is accessed.
    """
    devices = []
    success = True
    try:
        entries = tuple(usb_root.iterdir())
        physical = {entry.name: entry for entry in entries if ":" not in entry.name}
        cards: dict[str, list[int]] = {}
        for interface in entries:
            port, separator, _ = interface.name.partition(":")
            if not separator or port not in physical:
                continue
            try:
                sound_entries = tuple((interface / "sound").iterdir())
            except FileNotFoundError:
                continue
            for card in sound_entries:
                if card.name.startswith("card") and card.name[4:].isdigit():
                    cards.setdefault(port, []).append(int(card.name[4:]))
        for port, port_cards in sorted(cards.items()):
            device = physical[port]
            # Multiple sound cards under one physical USB device cannot be
            # assigned by arbitrarily taking the first directory entry.
            card = port_cards[0] if len(set(port_cards)) == 1 else None
            devices.append(UsbMicrophone(
                usb_port=port, serial=_attribute(device, "serial"),
                product=_attribute(device, "product"), alsa_card=card,
            ))
    except OSError:
        success = False
        devices = []
    observed_at = time.time()
    return UsbInventory(
        observed_at=observed_at, observed_monotonic=time.monotonic(),
        success=success, devices=tuple(replace(device, observed_at=observed_at) for device in devices),
    )


def _capture_snapshot(observation: CaptureObservation, now: float, stale_after_s: float) -> CaptureSnapshot:
    if not observation.opened:
        state = CaptureState.CLOSED if observation.generation else CaptureState.UNAVAILABLE
    elif observation.last_callback_monotonic is None:
        state = CaptureState.WAITING
    elif now - observation.last_callback_monotonic > stale_after_s:
        state = CaptureState.STALE
    else:
        state = CaptureState.RECEIVING
    return CaptureSnapshot(
        state=state, generation=observation.generation,
        frames_received=observation.frames_received,
        last_callback_at=observation.last_callback_at,
        last_nonzero_at=observation.last_nonzero_at,
        nonzero_seen=observation.nonzero_seen,
        sample_rate_hz=observation.sample_rate_hz, channels=observation.channels,
        selected_channel=observation.selected_channel, host_dtype=observation.host_dtype,
    )


def build_microphone_snapshot(
    *, rooms: Sequence[MicrophoneRoom], inventory: UsbInventory,
    captures: Mapping[str, CaptureObservation], now_monotonic: float,
    now_wall: float, stale_after_s: float,
) -> MicrophonesSnapshot:
    """Project cached observations without I/O or mutations.

    Args:
        rooms: Configured room adapters, including rooms without microphones.
        inventory: Most recent inventory plus retained removed identities.
        captures: Observations from each currently selected stream generation.
        now_monotonic: Clock used for freshness, independent of wall changes.
        now_wall: Current wall clock, retained for caller clock symmetry.
        stale_after_s: Maximum age of USB and callback evidence in seconds.

    Returns:
        Frozen public telemetry containing no internal monotonic timestamps.
    """
    stale = not inventory.success or now_monotonic - inventory.observed_monotonic > stale_after_s
    current = {device.usb_port: device for device in inventory.devices}
    known = {device.usb_port: device for device in inventory.previous_devices}
    known.update(current)
    serial_counts = Counter(device.serial for device in current.values() if device.serial)
    used: set[str] = set()
    output: list[MicSnapshot] = []
    unconfigured = []
    assignments: dict[str, list[str]] = {}
    for room in rooms:
        if room.configured_usb_port:
            assignments.setdefault(room.configured_usb_port, []).append(room.room_id)

    def emit(room: MicrophoneRoom | None, device: UsbMicrophone | None, ambiguous: bool = False) -> None:
        observation = captures.get(room.room_id, CaptureObservation()) if room else CaptureObservation()
        present = device is not None and device.usb_port in current
        duplicate = bool(device and device.serial and serial_counts[device.serial] > 1)
        connection = ConnectionState.UNKNOWN if stale or ambiguous else (
            ConnectionState.CONNECTED if present else ConnectionState.DISCONNECTED
        )
        if ambiguous or duplicate:
            binding = BindingState.AMBIGUOUS
        elif room is None:
            binding = BindingState.UNASSIGNED
        elif (stale or not present or not observation.opened or observation.usb_port is None
              or observation.serial is None or observation.alsa_card is None):
            binding = BindingState.UNKNOWN
        elif (device.serial is not None and observation.usb_port == device.usb_port
              and observation.serial == device.serial and observation.alsa_card == device.alsa_card
              and device.alsa_card is not None
              and (not room.configured_serial or room.configured_serial == device.serial)):
            binding = BindingState.MATCHED
        else:
            binding = BindingState.MISMATCH
        output.append(MicSnapshot(
            mic_id=f"usb:{device.serial or 'unknown'}@{device.usb_port}" if device else f"room:{room.room_id}",
            room_id=room.room_id if room else None, room_name=room.name if room else None,
            configured_usb_port=room.configured_usb_port if room else None,
            usb_port=device.usb_port if device else None,
            serial=device.serial if device else None, product=device.product if device else None,
            alsa_card=device.alsa_card if device else None,
            portaudio_index=observation.device_index,
            identity_observed_at=device.observed_at if device else None,
            connection=connection, binding=binding,
            capture=_capture_snapshot(observation, now_monotonic, stale_after_s),
        ))

    for room in rooms:
        capture = captures.get(room.room_id)
        port = room.configured_usb_port
        if not port and room.configured_serial:
            candidates = [device for device in current.values() if device.serial == room.configured_serial]
            if not candidates:
                # Retained ports describe absence, not concurrent devices.
                # A unique current identity must win after a USB port change.
                candidates = [device for device in known.values() if device.serial == room.configured_serial]
            if len(candidates) > 1:
                emit(room, None, ambiguous=True)
                continue
            port = candidates[0].usb_port if candidates else None
        if not port and capture and capture.usb_port:
            port = capture.usb_port
        if not port and not room.configured_serial and room.device_index is None:
            unconfigured.append(room.room_id)
            continue
        if port and (len(assignments.get(port, ())) > 1 or port in used):
            emit(room, None, ambiguous=True)
            continue
        device = known.get(port) if port else None
        emit(room, device)
        if device:
            used.add(device.usb_port)
    for port, device in sorted(known.items()):
        if port not in used:
            emit(None, device)
    return MicrophonesSnapshot(
        observed_at=inventory.observed_at, stale_after_s=stale_after_s,
        stale=stale, microphones=tuple(output), unconfigured_rooms=tuple(unconfigured),
    )
