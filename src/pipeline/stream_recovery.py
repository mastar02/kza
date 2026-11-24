"""PortAudio catalog refresh, stream reopen backoff, watchdog, and dead-stream recovery.

Extracted from ``src/pipeline/room_stream_manager.py`` as part of the Phase 1
audio capture rearchitecture so recovery/reopen state machines live in a
dedicated cohesive module.
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
from src.wakeword.whisper_wake import _text_likely_truncated
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


def detect_stale_streams(
    states: list[tuple[str, float, float]],
    now: float,
    timeout_s: float,
    first_frame_grace_s: float,
) -> list[str]:
    """Return room_ids whose audio stream is not delivering audio.

    Dos regímenes distintos, porque "todavía no arrancó" no es lo mismo que
    "dejó de entregar":

    - Ya entregó audio alguna vez (``last_frame_ts > 0``): stale si pasaron
      más de ``timeout_s`` desde el último frame con señal.
    - Abierto pero sin entregar nunca (``last_frame_ts == 0``): stale recién
      pasado ``first_frame_grace_s`` desde la apertura. Medido sobre arranques
      reales (2026-07-29/30): lo normal son 1.5-2s, pero se observó uno de
      135s. Sin esta gracia el watchdog dispararía recovery sobre un mic que
      todavía está despertando, y como el recovery cierra TODOS los streams,
      se llevaría puestos a los sanos.
    - Nunca abierto (``opened_ts == 0``): se ignora, no hay nada que recuperar.

    Args:
        states: list of (room_id, last_frame_ts, opened_ts), monotónicos.
        now: current monotonic time.
        timeout_s: segundos sin audio, tras haber entregado, para darlo por muerto.
        first_frame_grace_s: margen para el PRIMER frame tras abrir el stream.
    """
    stale = []
    for room_id, last_frame_ts, opened_ts in states:
        if last_frame_ts > 0.0:
            if (now - last_frame_ts) > timeout_s:
                stale.append(room_id)
        elif opened_ts > 0.0 and (now - opened_ts) > first_frame_grace_s:
            stale.append(room_id)
    return stale


class PortAudioReset(Enum):
    """Outcome of a PortAudio terminate+initialize (T025-U2-RV3-5)."""

    # Terminated and initialized again: the device list was re-read.
    RESCANNED = "rescanned"
    # Terminate failed while still initialized: the old device list is kept.
    UNCHANGED = "unchanged"
    # Initialize failed: no stream can open until a later reset.
    UNINITIALIZED = "uninitialized"


class StreamRecoveryMixin:
    """PortAudio reinitialization, room reopen backoff, and watchdog recovery."""

    def _reinit_portaudio(self) -> PortAudioReset:
        """Force PortAudio to re-scan the device list (sync; fail-open).

        PortAudio snapshots devices at Pa_Initialize and never re-scans. After a
        USB re-enumeration the cached indices point to dead devices, so we must
        terminate+initialize to see the new ones. This is GLOBAL: it invalidates
        every open stream, which is why _recover_streams reopens all of them.

        Terminate and initialize fail separately. A failed initialize leaves
        PortAudio uninitialized, and sounddevice then refuses to terminate it
        ('PortAudio not initialized'); initialize must still run on the next
        reset or no microphone could ever open again (T025-U2-RV2-1).

        Returns:
            ``RESCANNED`` when the device list was re-read; ``UNCHANGED``
            when terminate failed while PortAudio stayed initialized, so its
            old list is kept (no re-scan happened); ``UNINITIALIZED`` when it
            was left uninitialized and no stream can open until a later reset.
        """
        try:
            getattr(_facade(), 'sd', sd)._terminate()
        except Exception as e:
            logger.warning(f"[audio-watchdog] PortAudio terminate failed: {e}")
            initialized = getattr(getattr(_facade(), 'sd', sd), '_initialized', None)
            if type(initialized) is int and initialized > 0:
                # Still initialized with its old catalog: initializing again
                # would only nest sounddevice's count, and a later terminate
                # would then never re-scan.
                return PortAudioReset.UNCHANGED
        try:
            getattr(_facade(), 'sd', sd)._initialize()
        except Exception as e:
            logger.warning(f"[audio-watchdog] PortAudio reinit failed: {e}")
            return PortAudioReset.UNINITIALIZED
        return PortAudioReset.RESCANNED

    def _note_catalog_gap(self, rs: RoomStream) -> None:
        """Record whether a configured card is in sysfs but not in PortAudio.

        Runs in the native open owner right after the port was resolved, so
        the sysfs read never blocks the event loop. An unresolved port whose
        card is also absent from sysfs (an unplugged microphone) is not a
        catalog gap: refreshing PortAudio cannot bring it back.

        Args:
            rs: Room whose ``device_index`` was just resolved from its port.
        """
        if self._catalog_refresh_interval_s <= 0:
            return
        card = None
        if rs.device_index is None:
            card = getattr(_facade(), 'usb_port_to_alsa_card', usb_port_to_alsa_card)(rs.mic_usb_port, str(self._microphone_usb_root))
        if card is None:
            self._catalog_missing.pop(rs.room_id, None)
            self._catalog_gap_seen.pop(rs.room_id, None)
        else:
            self._catalog_missing[rs.room_id] = card
            seen = self._catalog_gap_seen.get(rs.room_id)
            if seen is None or seen[0] != card:
                self._catalog_gap_seen[rs.room_id] = (card, getattr(_facade(), 'time', time).monotonic())

    def _catalog_refresh_blocker(self) -> str | None:
        """Name the activity a brief all-room capture cut would interrupt.

        Returns:
            ``'command'`` while a command is captured or dispatched in some
            room, ``'conversation'`` while the follow-up conversation is
            active, ``'response'`` while voice output is owned (a PortAudio
            reset would also stop playback), or None when a cut is allowed.
        """
        # Probe directly: CommandTurns.active would mark the turn as observed
        # and notify its end listeners for a check that is not a consumer.
        if self._command_turn_busy():
            return 'command'
        if bool(getattr(self.follow_up, 'is_active', False)):
            return 'conversation'
        handler = self._response_handler
        if handler is not None and bool(getattr(handler, 'is_responding', False)):
            return 'response'
        return None

    def _request_catalog_refresh(self, rs: RoomStream) -> bool:
        """Ask the global recovery to refresh PortAudio for a late microphone.

        Called by a room's reopen waiter after a failed open. The refresh is
        admitted only when that room's card is in sysfs but missing from
        PortAudio, that gap was first seen at least the reopen backoff
        ceiling ago (``settling``: a card that just enumerated may not be
        openable yet), no command, conversation or response is active, and
        the previous refresh is older than the configured interval.

        Args:
            rs: Room still waiting for its microphone.

        Returns:
            True when a global catalog recovery was scheduled.
        """
        if self._catalog_refresh_interval_s <= 0:
            return False
        card = self._catalog_missing.get(rs.room_id)
        futile = self._catalog_futile.get(rs.room_id)
        if futile is not None and futile[0] != card:
            # The card left sysfs or came back with another number: start over.
            del self._catalog_futile[rs.room_id]
        if card is None:
            return False
        reason = self._catalog_refresh_blocker()
        seen = self._catalog_gap_seen.get(rs.room_id)
        if reason is None and (seen is None or seen[0] != card
                               or getattr(_facade(), 'time', time).monotonic() - seen[1] < self._watchdog_backoff_max_s):
            reason = 'settling'
        last = self._catalog_refreshed_at
        if (reason is None and last is not None
                and self._microphone_monotonic() - last < self._catalog_refresh_wait_s(rs.room_id)):
            reason = 'rate_limit'
        if reason is not None:
            if self._catalog_refresh_deferred != reason:
                self._catalog_refresh_deferred = reason
                logger.info(
                    f"[audio-watchdog] {rs.room_id}: PortAudio catalog refresh deferred "
                    f"({reason}); ALSA card {card} is present but not listed"
                )
            return False
        return self._schedule_recovery([rs.room_id], catalog_refresh=True)

    def _hold_output(self) -> tuple[object | None, bool]:
        """Stop voice output from opening a stream while PortAudio is reset.

        A timer, reminder notice or alert speaks on its own schedule and
        would open an output stream on another thread in the middle of
        ``Pa_Terminate``/``Pa_Initialize`` (T025-U2-RV2-2). Only output that
        goes through the attached ResponseHandler is held.

        Returns:
            ``(token, output_busy)``: the hold token (None when nothing was
            held) and whether a response owns the output right now, so the
            reset would cut it.
        """
        handler = self._response_handler
        hold = getattr(handler, 'try_hold_output', None) if handler is not None else None
        if not callable(hold):
            return None, False
        token = hold()
        return token, token is None

    def _release_output(self, token: object | None) -> None:
        """Release an output hold from ``_hold_output``; safe from any thread."""
        if token is None:
            return
        release = getattr(self._response_handler, 'release_output_hold', None)
        if callable(release):
            release(token)

    def _capture_working(self) -> bool:
        """Whether a brief all-room capture cut would interrupt a microphone.

        Returns:
            True when some room has an open stream that delivered audio within
            the watchdog timeout, or was opened and is still inside its
            first-frame grace; False when every room is deaf.
        """
        now = getattr(_facade(), 'time', time).monotonic()
        for room_id, rs in self.room_streams.items():
            if room_id not in self._streams:
                continue
            if rs.last_frame_ts > 0.0:
                if now - rs.last_frame_ts <= self._watchdog_timeout_s:
                    return True
            elif rs.opened_ts > 0.0 and now - rs.opened_ts <= self._watchdog_first_frame_grace_s:
                return True
        return False

    def _catalog_short_wait_s(self, attempts: int) -> float:
        """Reopen backoff ceiling doubled per consecutive attempt, capped at the interval."""
        exponent = min(max(attempts, 1) - 1, 16)
        return min(self._watchdog_backoff_max_s * 2 ** exponent, self._catalog_refresh_interval_s)

    def _catalog_futile_wait_s(self, room_id: str | None) -> float:
        """The interval, doubled per refresh that left this room's card unlisted, up to the ceiling."""
        interval = self._catalog_refresh_interval_s
        futile = self._catalog_futile.get(room_id) if room_id is not None else None
        if futile is None:
            return interval
        return min(interval * 2 ** min(futile[1] - 1, 16), self._catalog_refresh_ceiling_s)

    def _catalog_refresh_wait_s(self, room_id: str | None = None) -> float:
        """Minimum time between two catalog refresh cuts.

        The interval and the futile backoff only protect working rooms from
        the cut. When nothing captures, a refresh cuts nothing and the house
        stays deaf until one succeeds.

        Args:
            room_id: Room asking for the refresh; its card may have survived
                earlier refreshes unlisted.

        Returns:
            A short backoff (the reopen backoff ceiling, doubling per
            consecutive failure, capped at the interval) while the last reset
            left PortAudio uninitialized (RV2-1), or while no room has a
            working stream (RV3-1, doubling per refresh that ran deaf and left
            a card unlisted). Otherwise the configured interval, doubled per
            further refresh that left this room's card unlisted, up to the
            ceiling.
        """
        if self._catalog_reinit_failures:
            return self._catalog_short_wait_s(self._catalog_reinit_failures)
        if not self._capture_working():
            return self._catalog_short_wait_s(self._catalog_deaf_refreshes)
        return self._catalog_futile_wait_s(room_id)

    def _note_catalog_refresh_result(self, before: dict[str, int], *, deaf: bool = False) -> None:
        """Count refreshes after which PortAudio still did not list a card.

        Called once every room was reopened after a successful reset, so
        ``_catalog_missing`` holds what the refreshed catalog lists.

        Args:
            before: ``_catalog_missing`` as it was when the refresh started.
            deaf: No room was capturing when the refresh started, so the cut
                interrupted nothing: count it toward the short deaf backoff
                instead of the per-room futile streak.
        """
        unlisted = {room_id: card for room_id, card in before.items()
                    if self._catalog_missing.get(room_id) == card}
        self._catalog_deaf_refreshes = self._catalog_deaf_refreshes + 1 if deaf and unlisted else 0
        if self._catalog_deaf_refreshes == 2:
            logger.warning(
                f"[audio-watchdog] PortAudio still does not list ALSA cards {unlisted} after "
                f"{self._catalog_deaf_refreshes} catalog refreshes with no microphone capturing; "
                f"retrying the refresh in {self._catalog_short_wait_s(self._catalog_deaf_refreshes):.1f}s, "
                f"doubling up to {self._catalog_refresh_interval_s:.0f}s"
            )
        for room_id, card in before.items():
            if self._catalog_missing.get(room_id) != card:
                self._catalog_futile.pop(room_id, None)
                continue
            if deaf:
                continue
            previous = self._catalog_futile.get(room_id)
            streak = previous[1] + 1 if previous is not None and previous[0] == card else 1
            self._catalog_futile[room_id] = (card, streak)
            if streak == 2:
                logger.warning(
                    f"[audio-watchdog] {room_id}: PortAudio still does not list ALSA card {card} "
                    f"after {streak} catalog refreshes; backing off the next refresh to "
                    f"{self._catalog_futile_wait_s(room_id):.0f}s, doubling up to "
                    f"{self._catalog_refresh_ceiling_s:.0f}s"
                )
        for room_id, (card, _streak) in tuple(self._catalog_futile.items()):
            if self._catalog_missing.get(room_id) != card:
                del self._catalog_futile[room_id]

    def _note_catalog_reinit(self, ready: bool) -> None:
        """Record whether the last PortAudio reset left it initialized.

        Args:
            ready: Whether ``_reinit_portaudio`` left PortAudio initialized.
        """
        if ready:
            self._catalog_reinit_failures = 0
            return
        self._catalog_reinit_failures += 1
        if self._catalog_refresh_interval_s > 0:
            logger.error(
                "[audio-watchdog] PortAudio is not initialized after the catalog reset; "
                f"no microphone can open; retrying the refresh in {self._catalog_refresh_wait_s():.1f}s"
            )
        else:
            logger.error(
                "[audio-watchdog] PortAudio is not initialized after the catalog reset; "
                "no microphone can open and the catalog refresh is disabled"
            )

    def _input_epoch_current(self, epoch: int) -> bool:
        return not self._closed and self._inputs.accepting and epoch == self._recovery_epoch

    async def _try_reopen_once(self, rs: "RoomStream") -> bool:
        """Join one retained native open without crossing an input pause epoch."""
        epoch = self._recovery_epoch
        if not self._input_epoch_current(epoch) or self._global_recovery:
            return False
        stream = await self._inputs.open(rs.room_id)
        if stream is None or not self._input_epoch_current(epoch) or self._global_recovery:
            return False
        await self._sync_direction_observers()
        return self._input_epoch_current(epoch) and not self._global_recovery

    async def _reopen_room(self, rs: "RoomStream", *, epoch: int | None = None) -> None:
        """Reintentar la reapertura con backoff hasta que el device vuelva.

        Waits indefinitely (while self._running) for the device to reappear in
        sysfs — the service stays alive; only this mic waits. Never raises.

        ⚠️ Por eso NUNCA debe awaitearse en serie sobre varias rooms: ver
        `_recover_streams`.
        """
        epoch = self._recovery_epoch if epoch is None else epoch
        backoff = self._watchdog_backoff_min_s
        while self._running and self._input_epoch_current(epoch) and not self._global_recovery:
            if await self._try_reopen_once(rs):
                if not self._input_epoch_current(epoch):
                    return
                logger.info(
                    f"[audio-watchdog] {rs.room_id}: recovered "
                    f"(device={rs.device_index})"
                )
                return
            if not self._input_epoch_current(epoch) or self._global_recovery:
                return
            # An admitted global recovery cancels this waiter and reopens every
            # room after the refresh. Keep waiting meanwhile: the recovery may
            # still decline the cut if a command starts before it runs.
            self._request_catalog_refresh(rs)
            logger.warning(
                f"[audio-watchdog] {rs.room_id}: device {rs.mic_usb_port} "
                f"absent, retry in {backoff:.1f}s"
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._watchdog_backoff_max_s)

    def _cancel_reopen_tasks(self) -> list:
        """Cancelar las esperas de reapertura en curso (shutdown).

        `cancel()` solo la SOLICITA: la task no muere hasta que el loop la
        procesa. Devuelve las tasks para que quien pueda awaitear (`stop`)
        confirme que murieron antes de seguir.
        """
        tasks = list(self._reopen_tasks.values())
        for task in tasks:
            task.cancel()
        self._reopen_tasks.clear()
        return tasks

    def _is_awaiting_reopen(self, room_id: str) -> bool:
        """True si `room_id` ya tiene una espera de reapertura en curso."""
        task = self._reopen_tasks.get(room_id)
        return task is not None and not task.done()

    def _schedule_reopen(self, rs: "RoomStream") -> None:
        """Dejar a `rs` esperando su device en una task propia (idempotente)."""
        if (self._closed or not self._inputs.accepting or self._global_recovery
                or self._is_awaiting_reopen(rs.room_id)):
            return  # ya hay una espera viva para esta room
        logger.warning(
            f"[audio-watchdog] {rs.room_id}: device {rs.mic_usb_port} ausente "
            f"— esperando en background (las demás rooms siguen operativas)"
        )
        task = asyncio.create_task(self._reopen_room(rs, epoch=self._recovery_epoch), context=Context())
        self._reopen_tasks[rs.room_id] = task
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception(), context=Context())

    def _schedule_recovery(self, trigger_room_ids: list, *, catalog_refresh: bool = False,
                           xvf_reboot: bool = False) -> bool:
        """Admit one finite controller; native effects remain owned by InputGroup.

        Args:
            trigger_room_ids: Rooms that need recovery.
            catalog_refresh: Refresh the PortAudio catalog for a late
                microphone even while other rooms are healthy; re-checked
                against active commands and conversations when it starts.
            xvf_reboot: Firmware-reboot the single trigger room's XVF3800
                (T048-U1); re-checked against activity when it starts.

        Returns:
            True when a recovery controller was admitted.
        """
        if (not self._running or not self._recovery_ready or self._closed
                or not self._inputs.accepting
                or self._recovery_task is not None and not self._recovery_task.done()):
            return False
        epoch = self._recovery_epoch
        rooms = tuple(room for room in self.room_streams if room in trigger_room_ids)
        if not rooms:
            return False

        async def recover():
            if not (self._input_epoch_current(epoch) and self._running and self._recovery_ready):
                return
            if xvf_reboot:
                await self._recover_with_xvf_reboot(rooms[0])
            elif catalog_refresh:
                # A wake or a response may have started since admission.
                if self._catalog_refresh_blocker() is not None:
                    return
                await self._recover_streams(list(rooms), catalog_refresh=True)
            else:
                await self._recover_streams(list(rooms))

        self._recovery_task = asyncio.create_task(recover(), context=Context())

        def finished(done):
            if done.cancelled():
                # A native admission may survive unexpected owner cancellation.
                # Do not admit replacement recovery until a real pause drains it.
                if self._recovery_task is done:
                    self._recovery_ready = False
                logger.warning('Audio recovery owner cancelled; admission sealed')
            elif done.exception() is not None:
                logger.warning('Audio recovery failed: %s', type(done.exception()).__name__)
            if not done.cancelled() and self._recovery_task is done:
                # Native work was awaited to its real exit. Drop exception
                # frames and RoomStream references instead of caching a task.
                self._recovery_task = None

        self._recovery_task.add_done_callback(finished, context=Context())
        return True

    async def _recover_streams(self, trigger_room_ids: list, *, catalog_refresh: bool = False) -> None:
        """Recover failed rooms without interrupting rooms still receiving audio.

        Reinitialize the shared PortAudio catalog only when no other stream is
        healthy, or when ``catalog_refresh`` asks for it because a configured
        card is in sysfs but missing from PortAudio's device list. Native
        stop/open calls must not block the asyncio event loop.

        Args:
            trigger_room_ids: Rooms that stopped delivering or never opened.
            catalog_refresh: Force the global path: retire every room,
                refresh the catalog and reopen every room.
        """
        epoch = self._recovery_epoch
        if not self._input_epoch_current(epoch):
            return
        if catalog_refresh:
            blocker = self._catalog_refresh_blocker()
            if blocker is not None:
                # Check before anything is retired: the cut itself is what
                # must not happen under a command, conversation or response
                # (T048-U1-RV1 F1). The trigger rooms keep (or get) their
                # reopen waiter, which asks for the refresh again later.
                logger.info(
                    f"[audio-watchdog] PortAudio catalog refresh for {list(trigger_room_ids)} "
                    f"deferred ({blocker}) before any capture cut"
                )
                for room_id in trigger_room_ids:
                    rs = self.room_streams.get(room_id)
                    if rs is not None and self._streams.get(room_id) is None:
                        self._schedule_reopen(rs)
                return
        now = getattr(_facade(), 'time', time).monotonic()
        healthy_other = not catalog_refresh and any(
            room_id not in trigger_room_ids and rs.last_frame_ts > 0
            and now - rs.last_frame_ts <= self._watchdog_timeout_s
            for room_id, rs in self.room_streams.items()
        )
        affected = trigger_room_ids if healthy_other else list(self.room_streams)
        missing_before = dict(self._catalog_missing)
        # Whether the cut interrupts a working microphone (RV3-1).
        deaf_before = catalog_refresh and not self._capture_working()
        refreshed = False
        if catalog_refresh:
            cards = {room: self._catalog_missing.get(room) for room in trigger_room_ids}
            logger.warning(
                f"[audio-watchdog] PortAudio does not list ALSA cards {cards} present in sysfs; "
                f"refreshing the catalog with a brief capture cut in {affected}"
            )
        else:
            logger.error(f"[audio-watchdog] streams {trigger_room_ids} stopped delivering audio; recovering {affected}")
        if not healthy_other:
            # Catalog reset invalidates every native handle. Quiesce retry
            # controllers before the first await, then drain their retained
            # InputGroup opens as well as already visible native owners.
            self._global_recovery = True
        try:
            if not healthy_other:
                await asyncio.gather(*self._cancel_reopen_tasks(), return_exceptions=True)
                if not self._input_epoch_current(epoch):
                    return
            for room_id in affected:
                if not self._input_epoch_current(epoch):
                    return
                await self._inputs.retire(room_id)
            if not self._input_epoch_current(epoch):
                return
            if catalog_refresh:
                # Charge the rate limit only once the cut really happened: a
                # pause that superseded this recovery cut nothing on its behalf.
                self._catalog_refreshed_at = self._microphone_monotonic()
                self._catalog_refresh_deferred = None
            if not healthy_other:
                # A retained open may not yet have constructed its owner when
                # the first retirement pass ran. No new retry can enter now.
                await asyncio.gather(*(asyncio.shield(task) for task in self._inputs.snapshot_operations()),
                                     return_exceptions=True)
                if not self._input_epoch_current(epoch):
                    return
                for room_id in affected:
                    await self._inputs.retire(room_id)
                    if not self._input_epoch_current(epoch):
                        return
                blocker = self._catalog_refresh_blocker() if catalog_refresh else None
                hold = None
                if blocker is None:
                    hold, output_busy = self._hold_output()
                    if output_busy and catalog_refresh:
                        blocker = 'response'
                if blocker is None:
                    outcome: list[PortAudioReset] = []

                    def reset_catalog() -> None:
                        try:
                            outcome.append(self._reinit_portaudio())
                        finally:
                            self._release_output(hold)

                    try:
                        await self._inputs.catalog_call(reset_catalog)
                    finally:
                        # Also when the reset was not admitted; idempotent.
                        self._release_output(hold)
                    if outcome:
                        self._note_catalog_reinit(outcome[0] is not PortAudioReset.UNINITIALIZED)
                        # Only a real re-scan can tell whether a card is
                        # still unlisted (RV3-5).
                        refreshed = catalog_refresh and outcome[0] is PortAudioReset.RESCANNED
                else:
                    # Never reset PortAudio under a starting command or a
                    # playing response; the cut already happened, so reopen.
                    logger.warning(
                        f"[audio-watchdog] PortAudio catalog refresh skipped ({blocker}); reopening {affected}"
                    )
                if not self._input_epoch_current(epoch):
                    return
        finally:
            if not healthy_other:
                self._global_recovery = False
        for room_id in affected:
            if not self._input_epoch_current(epoch):
                return
            rs = self.room_streams[room_id]
            if await self._try_reopen_once(rs):
                logger.info(f"[audio-watchdog] {room_id}: stream reopened (device={rs.device_index}); awaiting audio frames")
            elif self._input_epoch_current(epoch):
                self._schedule_reopen(rs)
        if refreshed:
            self._note_catalog_refresh_result(missing_before, deaf=deaf_before)

    async def _stream_watchdog(self) -> None:
        """Publish health independently of native I/O and one recovery owner."""
        while self._running:
            await asyncio.sleep(self._watchdog_check_interval_s)
            if not self._running:
                break
            try:
                self._publish_audio_health()
            except Exception as error:
                logger.warning('Audio health publication failed: %s', type(error).__name__)
            if not self._recovery_ready or not self._inputs.accepting:
                continue
            now = getattr(_facade(), 'time', time).monotonic()
            states = []
            for room_id, rs in self.room_streams.items():
                if rs.last_frame_ts > 0.0 and now - rs.last_frame_ts <= self._watchdog_timeout_s:
                    # Real audio ends the dead-capture streak (T048-U1).
                    self._xvf_reboot_budget.note_audio(room_id)
                    self._xvf_reboot_deferred.pop(room_id, None)
                if not self._is_awaiting_reopen(room_id):
                    states.append((room_id, rs.last_frame_ts, rs.opened_ts))
            stale = getattr(_facade(), 'detect_stale_streams', detect_stale_streams)(states, now,
                self._watchdog_timeout_s, self._watchdog_first_frame_grace_s)
            if stale:
                self._recover_dead_streams(stale)

    def _recover_dead_streams(self, stale: list[str]) -> None:
        """Route rooms whose stream delivers no audio to a reboot or the recovery.

        A room whose ordinary recovery already failed the configured number
        of times gets a firmware reboot of its own XVF3800 instead, unless a
        command, conversation or response is active: then the reboot waits
        and the room keeps the ordinary recovery it had before T048-U1.
        Rooms over the hourly cap, or without a usable controller, keep the
        ordinary recovery too.

        Args:
            stale: Rooms the watchdog found without audio.
        """
        budget = self._xvf_reboot_budget
        ordinary = []
        for room_id in stale:
            decision = budget.decide(room_id)
            if decision is XvfRebootDecision.REBOOT:
                if self._xvf_reboot_controller(room_id) is None:
                    if room_id not in self._xvf_reboot_unavailable:
                        self._xvf_reboot_unavailable.add(room_id)
                        logger.warning(
                            f"[audio-watchdog] {room_id}: capture dead after "
                            f"{budget.failures(room_id)} reopens but the room has no XVF "
                            "controller bound to its mic port; firmware reboot unavailable"
                        )
                    ordinary.append(room_id)
                    continue
                reason = self._catalog_refresh_blocker()
                if reason is not None:
                    # Keep the ordinary recovery meanwhile, exactly as before
                    # T048-U1: the room is never left worse off than without
                    # the reboot, even if the activity flag sticks (RV1 F3).
                    self._note_xvf_reboot_deferred(room_id, reason)
                    ordinary.append(room_id)
                    continue
                # One recovery owner at a time: a busy owner retries next check.
                self._schedule_recovery([room_id], xvf_reboot=True)
                continue
            if decision is XvfRebootDecision.CAPPED and budget.report_cap(room_id):
                logger.error(
                    f"[audio-watchdog] {room_id}: isochronous capture still dead after "
                    f"{budget.reboots_in_window(room_id)} XVF firmware reboots in the last hour "
                    f"(cap {budget.policy.max_per_hour}); no more reboots for now: physically "
                    f"unplug and replug the microphone at USB port "
                    f"{self._xvf_room_port(room_id) or self.room_streams[room_id].mic_usb_port}"
                )
            ordinary.append(room_id)
        if not ordinary:
            return
        openings = {room_id: self.room_streams[room_id].opened_ts for room_id in ordinary}
        if self._schedule_recovery(ordinary):
            for room_id, opening in openings.items():
                budget.note_recovery(room_id, opening)
