"""XVF3800 DSP tuning, USB reboot recovery, microphone inventory, and DoA telemetry.

Extracted from ``src/pipeline/room_stream_manager.py`` as part of the Phase 1
audio capture rearchitecture so hardware DSP tuning, reboot policy, and
dashboard/health telemetry live in a dedicated cohesive module.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import deque
from contextvars import Context
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from threading import Event, Lock
from pathlib import Path
from typing import Callable, Awaitable, Optional
from uuid import uuid4

import numpy as np
import sounddevice as sd

from src.ambient.tap import MultiChannelTap
from src.audio.input_group import InputGroup
from src.audio.owned_input import InputNotAdmitted
from src.audio.xvf_controller import XvfController, XvfRebootOutcome
from src.audio.xvf_recovery import (
    REBOOT_DEFERRAL_ALERT_S, XvfRebootBudget, XvfRebootDecision, XvfRebootPolicy,
)
from src.pipeline.ambient_guard import AmbientGuard, GuardState, classify_outcome
from src.pipeline.command_event import CapturedTranscript, CommandEvent, TranscriptSource
from src.wakeword.wake_clip_writer import WakeClipWriter
from src.pipeline.capture_room import CaptureRoom
from src.pipeline.post_wake_endpoint import (
    CaptureTrigger, EndpointDecision, EndpointState, NoiseFloorTracker, PostWakeConfig, frame_rms,
)
from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.core.capture_opening import CaptureOpening, coherent_opening, current_capture_opening
from src.evaluation.provenance import CaptureLane
from src.memory.producer_context import PRODUCER_CONTEXT, model_admission, producer_context
from src.memory.producer_scopes import ProducerScopes, current_producer_scope
from src.memory.prompt_context import SourcesInvalidated, run_private
from src.wakeword.detector import WakeWordDetector
from src.audio.echo_suppressor import EchoSuppressor
from src.audio.direction_observer import (
    DirectionCalibration, DirectionObserver, parse_direction_observation,
)
from src.audio import microphone_status
from src.audio.microphone_status import (
    CaptureObservation, MicrophoneRoom, MicrophonesSnapshot, UsbInventory,
    build_microphone_snapshot,
)
from src.conversation.command_turns import CommandTurns
from src.conversation.follow_up_mode import FollowUpMode
from src.nlu.command_grammar import PartialCommand, parse_partial_command
from src.rooms.room_context import resolve_mic_usb_port, usb_port_to_alsa_card

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1280

# Warn after this observation budget without cancelling native publication.
# _audio_health_work retains one real write until it exits, even after timeout.
AUDIO_HEALTH_WRITE_TIMEOUT_S = 1.0


import sys as _sys


def _facade():
    return _sys.modules.get("src.pipeline.multi_room_audio_loop")


class _CaptureHolder:
    """One callback-owned observation pointer; retired holders stay isolated."""

    def __init__(self, observation: CaptureObservation):
        self.observation = observation
        self.opening: CaptureOpening | None = None


@dataclass(frozen=True)
class XvfRebootReport:
    """What the native part of one XVF firmware reboot observed (T048-U1).

    Attributes:
        outcome: Result of the REBOOT control transfer.
        reenumerated: The device came back on its port within the settle wait.
        alsa_card: ALSA card it exposes again, or None if none appeared in time.
        devnum: Kernel device number it came back with (sysfs), when readable.
    """

    outcome: XvfRebootOutcome
    reenumerated: bool = False
    alsa_card: int | None = None
    devnum: int | None = None


@dataclass
class _RebootDeferral:
    """A due firmware reboot waiting for an activity to end (T048-U1-RV1 F3).

    Attributes:
        reason: Blocker last logged (``command``, ``conversation``, ``response``).
        since: Microphone-clock time the deferral started.
        escalated: The long-deferral error was already logged.
    """

    reason: str
    since: float
    escalated: bool = False


class XvfTelemetryMixin:
    """XVF3800 tuning, reboot recovery, USB inventory, and direction observers."""

    def _apply_xvf_tuning(self) -> None:
        """Aplica el tuning configurado al DSP (EN RAM). Fail-open por param.

        Un nombre/valor inválido en el yaml NO tira el servicio: se loguea y
        se continúa con el resto. Loguea valor previo → nuevo para auditoría.
        """
        if not self._xvf_tuning.get("apply_on_start", False):
            return
        if not self._xvf_by_room:
            logger.warning(
                "[XVF-tuning] xvf_tuning.apply_on_start=true pero no hay "
                "XvfController — tuning NO aplicado (¿mic XVF3800 presente?)"
            )
            return
        # El tuning se aplica al DSP de CADA mic: son chips distintos y cada uno
        # arranca con el preset persistido de fábrica al re-enchufarse.
        for room_id, ctrl in self._xvf_by_room.items():
            self._apply_xvf_tuning_room(room_id, ctrl)

    def _xvf_tuning_enabled(self) -> bool:
        """Whether the configured tuning is written to the DSP (``apply_on_start``)."""
        return bool(self._xvf_tuning.get("apply_on_start", False)
                    and self._xvf_tuning.get("params"))

    def _apply_xvf_tuning_room(self, room_id: str, ctrl) -> None:
        """Write the configured tuning to one room's DSP (RAM). Fail-open per param.

        Blocking USB I/O: runs on the ``_xvf_native`` owner, never on the loop.

        Args:
            room_id: Room whose controller is written (for the audit log).
            ctrl: That room's port-bound XVF controller.
        """
        if not self._xvf_tuning.get("apply_on_start", False):
            return
        params = self._xvf_tuning.get("params") or {}
        for name, values in params.items():
            try:
                before = ctrl.read_param(name)
            except ValueError:
                before = None
            try:
                ok = ctrl.write_param(name, values)
            except ValueError as e:  # typo en el yaml: log y seguir
                logger.error(f"[XVF-tuning] parámetro inválido en config: {e}")
                continue
            except Exception as e:  # cualquier otra cosa: fail-open
                logger.warning(f"[XVF-tuning] {room_id}: {name} no aplicado: {e}")
                continue
            if ok:
                logger.info(
                    f"[XVF-tuning] {room_id}: {name}: {before} → {values} (RAM)"
                )
            else:
                logger.warning(
                    f"[XVF-tuning] {room_id}: {name} no aplicado (write fail-open)"
                )

    def _reapply_xvf_tuning(self, room_id: str, ctrl) -> None:
        """Reapply the tuning after the room's device re-enumerated (native).

        The controller may still cache the handle of the device that left the
        bus, so it is dropped first and the next transfer finds the new one.

        Args:
            room_id: Room whose device re-enumerated.
            ctrl: That room's port-bound XVF controller.
        """
        invalidate = getattr(ctrl, "invalidate_device", None)
        if callable(invalidate):
            invalidate()
        self._apply_xvf_tuning_room(room_id, ctrl)

    def _reboot_xvf_room(self, ctrl, usb_port: str, epoch: int) -> XvfRebootReport:
        """Send the firmware reboot and wait for the device and its ALSA card (native).

        Runs on the ``_xvf_native`` owner. Both waits share the policy's
        settle bound and stop early when a pause or close supersedes the
        recovery epoch.

        Args:
            ctrl: The room's port-bound XVF controller.
            usb_port: The room's mic port, where the card must reappear.
            epoch: Recovery epoch that admitted this reboot.

        Returns:
            What was observed; never raises for USB problems (fail-open).
        """
        deadline = getattr(_facade(), 'time', time).monotonic() + self._xvf_reboot_budget.policy.settle_s

        def superseded() -> bool:
            return not self._input_epoch_current(epoch)

        result = ctrl.reboot()
        if result.outcome is XvfRebootOutcome.NO_DEVICE:
            return XvfRebootReport(result.outcome)
        # A transfer error may be the device resetting before it answered.
        if not ctrl.wait_reenumerated(result.previous_address, deadline - getattr(_facade(), 'time', time).monotonic(),
                                      abort=superseded):
            return XvfRebootReport(result.outcome)
        card = None
        while not superseded():
            card = getattr(_facade(), 'usb_port_to_alsa_card', usb_port_to_alsa_card)(usb_port, str(self._microphone_usb_root))
            if card is not None or getattr(_facade(), 'time', time).monotonic() >= deadline:
                break
            getattr(_facade(), 'time', time).sleep(0.05)
        try:
            devnum_text = (self._microphone_usb_root / usb_port / "devnum").read_text().strip()
        except OSError:
            devnum_text = ""
        devnum = int(devnum_text) if devnum_text.isdigit() else None
        return XvfRebootReport(result.outcome, reenumerated=True, alsa_card=card, devnum=devnum)

    async def _recover_with_xvf_reboot(self, room_id: str) -> None:
        """Firmware-reboot one room's stuck XVF3800 and bring its capture back.

        Incident 2026-11-19: the isochronous endpoint of every XVF3800 kept
        sending oversized packets after a warm host reboot. The stream opened
        and never delivered audio, and only the firmware REBOOT recovered it.
        Sequence: retire this room's stream, reboot its device (located by
        its mic port), wait for the re-enumeration, reapply the RAM tuning
        (the reboot restores the factory preset), then reopen; when PortAudio
        does not list the new card yet, refresh its catalog (T025-U2 path).
        The other rooms keep capturing unless that refresh is needed.

        Args:
            room_id: Room whose ordinary recovery failed the configured times.
        """
        epoch = self._recovery_epoch
        if not self._input_epoch_current(epoch):
            return
        reason = self._catalog_refresh_blocker()
        if reason is not None:
            # A wake or a response may have started since admission.
            self._note_xvf_reboot_deferred(room_id, reason)
            return
        ctrl = self._xvf_reboot_controller(room_id)
        if ctrl is None:
            return
        rs = self.room_streams[room_id]
        port = self._xvf_room_port(room_id)
        budget = self._xvf_reboot_budget
        failures = budget.failures(room_id)
        attempt = budget.charge(room_id)
        self._xvf_reboot_deferred.pop(room_id, None)
        logger.error(
            f"[audio-watchdog] {room_id}: isochronous capture dead after {failures} reopens; "
            f"XVF firmware reboot {attempt}/{budget.policy.max_per_hour} (USB port {port})"
        )
        report = None
        self._xvf_rebooting.add(room_id)
        try:
            await self._inputs.retire(room_id)
            if not self._input_epoch_current(epoch):
                return
            try:
                report = await self._xvf_native.run(self._reboot_xvf_room, ctrl, port, epoch)
            except Exception as error:
                logger.warning(f"[audio-watchdog] {room_id}: XVF firmware reboot failed: {type(error).__name__}")
            if report is not None and report.reenumerated:
                logger.info(
                    f"[audio-watchdog] {room_id}: XVF3800 re-enumerated at {port} "
                    f"(ALSA card {report.alsa_card}); reapplying the tuning"
                )
                try:
                    await self._xvf_native.run(self._apply_xvf_tuning_room, room_id, ctrl)
                except Exception as error:
                    logger.warning(f"[XVF-tuning] {room_id}: tuning not reapplied: {type(error).__name__}")
                if room_id in self._xvf_port_seen and report.alsa_card is not None:
                    # This reboot already retuned the new device: record it as
                    # the baseline so a later inventory poll does not retune
                    # it a second time (T048-U1-RV1 F4).
                    self._xvf_port_seen[room_id] = (report.devnum, report.alsa_card)
            elif report is not None and report.outcome is not XvfRebootOutcome.NO_DEVICE:
                logger.warning(
                    f"[audio-watchdog] {room_id}: XVF3800 at {port} did not re-enumerate within "
                    f"{budget.policy.settle_s:.0f}s after the firmware reboot"
                )
        finally:
            self._xvf_rebooting.discard(room_id)
        if not self._input_epoch_current(epoch):
            return
        if await self._try_reopen_once(rs):
            logger.info(
                f"[audio-watchdog] {room_id}: stream reopened after the XVF firmware reboot "
                f"(device={rs.device_index}); awaiting audio frames"
            )
            return
        if not self._input_epoch_current(epoch):
            return
        if report is not None and report.alsa_card is not None:
            # The card is back in sysfs but PortAudio still lists its old
            # catalog: refresh it (brief cut in every room). The reboot took
            # seconds, so a wake or a response may have started meanwhile:
            # check again and never cut the other rooms under it
            # (T048-U1-RV1 F1). The room's reopen waiter asks for the refresh
            # through _request_catalog_refresh once the activity ends.
            blocker = self._catalog_refresh_blocker()
            if blocker is None:
                await self._recover_streams([room_id], catalog_refresh=True)
                return
            logger.info(
                f"[audio-watchdog] {room_id}: PortAudio catalog refresh after the XVF firmware "
                f"reboot deferred ({blocker}); waiting for ALSA card {report.alsa_card}"
            )
        self._schedule_reopen(rs)

    def _update_audio_anchors(self, now_mono: float, now_wall: float) -> None:
        """Refrescar las anclas de salud que la reapertura no puede resetear.

        ⚠️ Por qué existe esto en vez de leer `rs.last_frame_ts`/`rs.opened_ts`
        directo: `_try_reopen_once` pone `opened_ts = now` y
        `last_frame_ts = 0.0` en CADA reapertura exitosa. Un XVF3800 con el
        endpoint isócrono muerto —el modo de falla de los incidentes de 27h y
        7h— abre perfecto en PortAudio y entrega ceros, así que el watchdog lo
        detecta a los ~180s, "recupera", y el reset deja `ever=False` con
        `age_s` en 0. Bucle eterno de ~182s en el que `age_s` solo supera el
        umbral ~2s por vuelta: la alerta quedaba MÁS DÉBIL justo en el caso
        primario (simulado: mediana 1,67h hasta detectar, máximo 15,98h,
        1/200 sin detectar en 24h).

        ⚠️ Y el fix intuitivo empeora las cosas: subir `first_frame_grace_s`
        del poller hace que NUNCA detecte, porque `age_s` está acotado por el
        ciclo de recovery. La salida es esta: un ancla que la reapertura no
        toca, con la que la detección pasa a ser determinística a los
        `deaf_after_s`.

        Args:
            now_mono: getattr(_facade(), 'time', time).monotonic() del ciclo actual.
            now_wall: getattr(_facade(), 'time', time).time() del ciclo actual.
        """
        for room_id, rs in self.room_streams.items():
            self._room_since_wall.setdefault(room_id, now_wall)
            if rs.last_frame_ts <= 0.0:
                continue  # todavía no entregó (o lo acaban de resetear)
            # Convertir el monotónico del último frame a wall clock. Exacto:
            # no depende del período del watchdog. El max() es lo que hace el
            # ancla monótona — una reapertura que pone last_frame_ts=0 cae en
            # el `continue` de arriba y no puede hacerla retroceder.
            frame_wall = now_wall - (now_mono - rs.last_frame_ts)
            previo = self._last_audio_wall.get(room_id)
            if previo is None or frame_wall > previo:
                self._last_audio_wall[room_id] = frame_wall

    def _audio_health_rooms(self) -> dict[str, tuple[float, bool]]:
        """(ancla_wall, ever) por room, tal como lo espera write_audio_health."""
        return {
            room_id: (
                self._last_audio_wall.get(
                    room_id, self._room_since_wall.get(room_id, 0.0)
                ),
                room_id in self._last_audio_wall,
            )
            for room_id in self.room_streams
        }

    def _publish_microphone_inventory(self, inventory: UsbInventory) -> None:
        """Retain disappeared identities without treating old data as present."""
        previous = self._microphone_inventory
        known = {}
        if previous is not None:
            known.update((device.usb_port, device) for device in previous.previous_devices)
            known.update((device.usb_port, device) for device in previous.devices)
        self._microphone_inventory = replace(inventory, previous_devices=tuple(known.values()))
        self._microphone_inventory_ready.set()
        snapshot = self._base_microphone_snapshot()
        if snapshot is not None:
            current = {mic.room_id: mic for mic in snapshot.microphones}
            with self._direction_lock:
                changed = [room for room in self._direction_observers
                           if self._direction_key(current.get(room)) != self._direction_keys.get(room)]
            for room in changed:
                self._invalidate_direction(room)

    async def _refresh_microphone_inventory(self) -> None:
        """Share one actual read; reject late publication and reconciliation."""
        epoch = self._auxiliary_epoch
        if not self._auxiliary_allowed(epoch):
            return
        worker = self._microphone_read_task
        if worker is None:
            reader = self._microphone_reader or getattr(_facade(), 'microphone_status', microphone_status).read_usb_microphones
            worker = asyncio.create_task(
                asyncio.to_thread(reader, self._microphone_usb_root), context=Context())
            self._microphone_read_task = worker
        try:
            inventory = await asyncio.shield(worker)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning('Microphone inventory failed: %s', type(error).__name__)
            inventory = UsbInventory(
                observed_at=self._microphone_wall_clock(),
                observed_monotonic=self._microphone_monotonic(), success=False)
        if not self._auxiliary_allowed(epoch) or self._microphone_read_task is not worker:
            return
        self._microphone_read_task = None
        self._publish_microphone_inventory(inventory)
        reenumerated = self._note_xvf_reenumerations(inventory)
        if self._auxiliary_allowed(epoch):
            await self._sync_direction_observers()
        for room_id in reenumerated:
            if not self._auxiliary_allowed(epoch):
                return
            await self._retune_reenumerated(room_id)

    def _xvf_room_port(self, room_id: str) -> str | None:
        """USB port the room's XVF controller itself is bound to, or None.

        Deliberately no fallback to the room's ``mic_usb_port``: a controller
        without its own port discovers whichever XVF3800 comes first on the
        bus, so rebooting or retuning through it could hit another room's
        microphone (T048-U1-RV1 F2).
        """
        port = getattr(self._xvf_by_room.get(room_id), "usb_port", None)
        return port if isinstance(port, str) and port else None

    def _note_xvf_reenumerations(self, inventory: UsbInventory) -> tuple[str, ...]:
        """Find rooms whose XVF3800 re-enumerated since the previous inventory.

        A re-enumeration (physical replug, firmware reboot, bus reset) resets
        the DSP to its factory preset. It shows up as the device appearing at
        the room's port, or with another kernel device number or ALSA card.
        The first observation of a room only records a baseline: the start
        already applied the tuning.

        Args:
            inventory: Inventory just published.

        Returns:
            Rooms whose tuning must be reapplied (none while tuning is off,
            and never one whose own firmware reboot is in flight).
        """
        if not inventory.success or not self._xvf_tuning_enabled():
            return ()
        by_port = {device.usb_port: (device.devnum, device.alsa_card)
                   for device in inventory.devices}
        changed = []
        for room_id in self._xvf_by_room:
            port = self._xvf_room_port(room_id)
            if not port:
                continue
            current = by_port.get(port)
            known = room_id in self._xvf_port_seen
            previous = self._xvf_port_seen.get(room_id)
            self._xvf_port_seen[room_id] = current
            if (known and current is not None and current != previous
                    and room_id not in self._xvf_rebooting):
                changed.append(room_id)
        return tuple(changed)

    async def _retune_reenumerated(self, room_id: str) -> None:
        """Reapply the tuning to one re-enumerated room; never raises.

        Args:
            room_id: Room reported by ``_note_xvf_reenumerations``.
        """
        ctrl = self._xvf_by_room.get(room_id)
        if ctrl is None:
            return
        logger.info(
            f"[XVF-tuning] {room_id}: device re-enumerated at {self._xvf_room_port(room_id)}; "
            "reapplying the tuning"
        )
        try:
            await self._xvf_native.run(self._reapply_xvf_tuning, room_id, ctrl)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(f"[XVF-tuning] {room_id}: tuning not reapplied: {type(error).__name__}")

    async def _microphone_telemetry_loop(self) -> None:
        """Poll only while capture auxiliaries may admit native work."""
        while self._running and self._auxiliary_allowed():
            await self._refresh_microphone_inventory()
            if not self._auxiliary_allowed():
                return
            await asyncio.sleep(self._microphone_interval_s)

    async def _stop_microphone_telemetry(self) -> None:
        """Join a retained stop owner without cancelling the actual read."""
        if self._microphone_stop_task is None:
            async def finish():
                task = self._microphone_telemetry_task
                if task is not None:
                    task.cancel()
                    await asyncio.gather(asyncio.shield(task), return_exceptions=True)
                worker = self._microphone_read_task
                if worker is not None:
                    await asyncio.shield(worker)
                    if worker.cancelled():
                        raise RuntimeError('microphone_read_unconfirmed')
                    if self._microphone_read_task is worker:
                        self._microphone_read_task = None
                if self._microphone_telemetry_task is task:
                    self._microphone_telemetry_task = None
            self._microphone_stop_task = asyncio.create_task(finish(), context=Context())
            self._microphone_stop_task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(self._microphone_stop_task)

    def _retire_microphone_capture(self, room_id: str) -> None:
        self._invalidate_direction(room_id)
        room = self._capture_rooms.get(room_id)
        if room is not None:
            room.retire_stream(self._microphone_wall_clock())
        holder = self._microphone_captures.get(room_id)
        if holder is not None:
            # Swap holders rather than mutate the callback-owned one. Even an
            # already executing old callback can only update retired memory.
            self._microphone_captures[room_id] = _CaptureHolder(replace(holder.observation, opened=False))

    def _base_microphone_snapshot(self) -> MicrophonesSnapshot | None:
        """Project the existing capture contract without direction lifecycle work."""
        inventory = self._microphone_inventory
        if inventory is None:
            return None
        return getattr(_facade(), 'build_microphone_snapshot', build_microphone_snapshot)(
            rooms=self._microphone_rooms, inventory=inventory,
            captures={room: holder.observation for room, holder in self._microphone_captures.items()},
            now_monotonic=self._microphone_monotonic(), now_wall=self._microphone_wall_clock(),
            stale_after_s=self._microphone_stale_after_s,
        )

    def _direction_key(self, mic) -> tuple | None:
        if (mic is None or mic.room_id not in self._direction_room_ids
                or mic.room_id in self._direction_opening):
            return None
        controller = self._xvf_by_room.get(mic.room_id)
        if (mic.connection != 'connected' or mic.binding != 'matched'
                or not mic.usb_port or mic.usb_port != mic.configured_usb_port
                or mic.usb_port != getattr(controller, 'usb_port', None)):
            return None
        return mic.mic_id, mic.usb_port, mic.capture.generation

    def _invalidate_direction(self, room_id: str) -> None:
        """Immediately hide an old generation; defer joins to async lifecycle."""
        with self._direction_lock:
            observer = self._direction_observers.pop(room_id, None)
            self._direction_keys.pop(room_id, None)
            if observer is not None:
                observer.invalidate()
                self._direction_retired.append(observer)
                self._direction_needs_reset.add(room_id)

    @staticmethod
    def _drain_direction_observer(observer: DirectionObserver) -> None:
        observer.stop()
        observer.wait_stopped()

    def _room_direction_calibration(self, room_id: str) -> DirectionCalibration | None:
        raw = self._direction_calibrations.get(room_id) if isinstance(self._direction_calibrations, dict) else False
        if raw is None or isinstance(raw, DirectionCalibration):
            return raw
        try:
            return DirectionCalibration(**raw)
        except (TypeError, ValueError):
            # Preserve the distinction between no calibration and an invalid one.
            return DirectionCalibration(0, 0, '', '', '')

    async def _sync_direction_observers(self) -> None:
        """Reconcile identity-bound workers outside GET and the audio callback."""
        epoch = self._auxiliary_epoch
        if not self._direction_started or not self._direction_room_ids or not self._auxiliary_allowed(epoch):
            return
        async with self._direction_lifecycle_lock:
            if not self._direction_started or not self._auxiliary_allowed(epoch):
                return
            snapshot = self._base_microphone_snapshot()
            current = {mic.room_id: mic for mic in snapshot.microphones} if snapshot else {}
            with self._direction_lock:
                changed = [room for room in self._direction_observers
                           if self._direction_key(current.get(room)) != self._direction_keys.get(room)]
            for room in changed:
                self._invalidate_direction(room)
            await self._drain_retired_directions()
            if not self._direction_started or not self._auxiliary_allowed(epoch):
                return
            for room_id in self._direction_room_ids:
                if not self._direction_started or not self._auxiliary_allowed(epoch):
                    return
                with self._direction_lock:
                    needs_reset = room_id in self._direction_needs_reset
                if needs_reset:
                    await self._xvf_native.run(self._xvf_by_room[room_id].invalidate_device)
                    if not self._direction_started or not self._auxiliary_allowed(epoch):
                        return
                    with self._direction_lock:
                        self._direction_needs_reset.discard(room_id)
                # Identity/generation can change while an old worker drains.
                snapshot = self._base_microphone_snapshot()
                mic = next((mic for mic in snapshot.microphones if mic.room_id == room_id), None) if snapshot else None
                key = self._direction_key(mic)
                if key is None:
                    continue
                with self._direction_lock:
                    fresh = self._base_microphone_snapshot()
                    fresh_mic = next((mic for mic in fresh.microphones if mic.room_id == room_id), None) if fresh else None
                    if (not self._direction_started or not self._auxiliary_allowed(epoch)
                            or room_id in self._direction_observers
                            or self._direction_key(fresh_mic) != key):
                        continue
                    observer = getattr(_facade(), 'DirectionObserver', DirectionObserver)(
                        controller=self._xvf_by_room[room_id], room_id=room_id, usb_port=mic.usb_port,
                        mic_id=mic.mic_id, config=self._direction_config,
                        calibration=self._room_direction_calibration(room_id),
                        monotonic=self._microphone_monotonic,
                        utc_now=lambda: datetime.fromtimestamp(self._microphone_wall_clock(), timezone.utc),
                    )
                    self._direction_observers[room_id] = observer
                    self._direction_keys[room_id] = key
                    observer.start()

    async def _drain_retired_directions(self) -> None:
        """Keep observers and shared real join tasks until successful return."""
        with self._direction_lock:
            retired = tuple(self._direction_retired)
        for observer in retired:
            if observer not in self._direction_join_tasks:
                self._direction_join_tasks[observer] = asyncio.create_task(
                    asyncio.to_thread(self._drain_direction_observer, observer), context=Context())
        tasks = tuple(self._direction_join_tasks[observer] for observer in retired)
        results = await asyncio.gather(*(asyncio.shield(task) for task in tasks),
                                       return_exceptions=True)
        errors = []
        for observer, result in zip(retired, results):
            if isinstance(result, BaseException):
                errors.append(result)
                continue
            with self._direction_lock:
                if observer in self._direction_retired:
                    self._direction_retired.remove(observer)
            self._direction_join_tasks.pop(observer, None)
        if errors:
            raise errors[0]

    async def _stop_direction_observers(self) -> None:
        """Invalidate all generations before awaiting complete worker drainage."""
        self._direction_started = False
        with self._direction_lock:
            room_ids = tuple(self._direction_observers)
        for room_id in room_ids:
            self._invalidate_direction(room_id)
        async with self._direction_lifecycle_lock:
            await self._drain_retired_directions()

    def get_microphone_snapshot(self) -> MicrophonesSnapshot | None:
        """Project cached capture and direction snapshots without any I/O."""
        snapshot = self._base_microphone_snapshot()
        if snapshot is None:
            return None
        microphones = []
        with self._direction_lock:
            for mic in snapshot.microphones:
                direction = getattr(_facade(), 'microphone_status', microphone_status).DirectionSnapshot(
                    state=getattr(_facade(), 'microphone_status', microphone_status).DirectionState.DISABLED,
                    reason=('capture_retirement_failed' if self._auxiliary_phase == 'failed'
                            else 'capture_retiring' if self._auxiliary_phase in ('retiring', 'closing')
                            else 'capture_paused') if self._paused or self._closed else self._direction_reason,
                )
                if mic.room_id in self._direction_room_ids and self._direction_started:
                    observer = self._direction_observers.get(mic.room_id)
                    key = self._direction_key(mic)
                    if observer is not None and key is not None and key == self._direction_keys.get(mic.room_id):
                        direction = observer.snapshot()
                    else:
                        direction = replace(direction, state=getattr(_facade(), 'microphone_status', microphone_status).DirectionState.UNKNOWN,
                                            reason='identity_unavailable')
                microphones.append(replace(mic, direction=direction))
        return replace(snapshot, microphones=tuple(microphones))

    def get_audio_health(self) -> dict:
        """Return room frame health without file I/O or audio content."""
        now_wall = getattr(_facade(), 'time', time).time()
        self._update_audio_anchors(now_mono=getattr(_facade(), 'time', time).monotonic(), now_wall=now_wall)
        return {
            "rooms": {
                room: {"age_s": max(0.0, now_wall - anchor), "ever": ever}
                for room, (anchor, ever) in self._audio_health_rooms().items()
            },
            "deaf_after_s": self._watchdog_timeout_s,
            "first_frame_grace_s": self._watchdog_first_frame_grace_s,
        }

    def _publish_audio_health(self) -> None:
        """Refresh anchors and admit at most one retained native publication."""
        now = getattr(_facade(), 'time', time).monotonic()
        now_wall = getattr(_facade(), 'time', time).time()
        self._update_audio_anchors(now_mono=now, now_wall=now_wall)
        task = self._audio_health_task
        if (task is not None and not task.done()) or self._audio_health_work._tasks:
            if not self._audio_health_timed_out and now - self._audio_health_started >= AUDIO_HEALTH_WRITE_TIMEOUT_S:
                self._audio_health_timed_out = True
                logger.warning('Audio health publication pending: native write still owned')
            return
        if not self._auxiliary_allowed():
            return
        from src.monitoring.audio_health import write_audio_health

        self._audio_health_started = now
        self._audio_health_timed_out = False
        task = self._audio_health_task = asyncio.create_task(
            self._audio_health_work.run(write_audio_health, self._audio_health_path,
                self._audio_health_rooms(), now_wall=now_wall), context=Context())

        def finished(done):
            if done.cancelled():
                logger.warning('Audio health publication observer cancelled')
            elif done.exception() is not None:
                logger.warning('Audio health publication failed: %s', type(done.exception()).__name__)
            if not done.cancelled() and self._audio_health_task is done:
                self._audio_health_task = None

        task.add_done_callback(finished, context=Context())

    def _xvf_reboot_controller(self, room_id: str):
        """The room's controller when it can send a firmware reboot, else None."""
        ctrl = self._xvf_by_room.get(room_id)
        if (ctrl is None or not callable(getattr(ctrl, "reboot", None))
                or not callable(getattr(ctrl, "wait_reenumerated", None))
                or not self._xvf_room_port(room_id)):
            return None
        return ctrl

    def _note_xvf_reboot_deferred(self, room_id: str, reason: str) -> None:
        """Log once per blocker that a due firmware reboot waits for it.

        A deferral longer than ``REBOOT_DEFERRAL_ALERT_S`` is logged once as
        an error: an activity flag that never clears would otherwise keep the
        reboot away forever with a single info line (T048-U1-RV1 F3).
        """
        now = self._microphone_monotonic()
        deferral = self._xvf_reboot_deferred.get(room_id)
        if deferral is None:
            deferral = self._xvf_reboot_deferred[room_id] = _RebootDeferral(reason, now)
            logger.info(f"[audio-watchdog] {room_id}: XVF firmware reboot deferred ({reason})")
        elif deferral.reason != reason:
            deferral.reason = reason
            logger.info(f"[audio-watchdog] {room_id}: XVF firmware reboot deferred ({reason})")
        if not deferral.escalated and now - deferral.since >= REBOOT_DEFERRAL_ALERT_S:
            deferral.escalated = True
            logger.error(
                f"[audio-watchdog] {room_id}: XVF firmware reboot deferred for "
                f"{now - deferral.since:.0f}s ({reason}); capture is still dead and only the "
                "ordinary reopen runs; check whether that activity is stuck"
            )
