"""Intercom announcements through the application's local voice output."""

import asyncio
import logging
import re
import unicodedata
import uuid
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any

from src.core.household_clock import ZONE_PREFIX
from src.core.task_outcome import CURRENT_REQUEST_ID, DispatchState, EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.core.voice_delivery import ResponseResult, VoiceDeliveryError, invoke_voice_callback
from src.nlu.household_intents import INTERCOM_BROADCAST_TARGETS, intercom_slots

logger = logging.getLogger(__name__)

_ARTICLE = re.compile(r"^(?:el|la|los|las)\s+")
_VOICE_RECEIPT_LIMIT = 256


def _room_name(text: str) -> str:
    """Fold case, accents and a leading article in a spoken room label."""
    folded = "".join(char for char in unicodedata.normalize("NFD", text.casefold())
                     if unicodedata.category(char) != "Mn")
    return _ARTICLE.sub("", " ".join(folded.split()))


class AnnouncementPriority(StrEnum):
    """Priority labels for programmatic producers."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


class ZoneDelivery(StrEnum):
    """Evidence at the local voice output boundary for one destination."""

    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    NOT_SENT = "not_sent"


@dataclass
class Announcement:
    """Un anuncio para reproducir"""
    announcement_id: str
    message: str = field(repr=False)
    target_zones: list[str]           # Zonas destino ("all" = todas)
    priority: AnnouncementPriority = AnnouncementPriority.NORMAL

    # Metadatos
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str | None = None  # user_id
    source_zone: str | None = None # Zona de origen

    # Estado
    delivered: bool = False
    delivered_at: datetime | None = None
    failed_zones: list[str] = field(default_factory=list)
    zone_receipts: dict[str, ZoneDelivery] = field(default_factory=dict)

    # Configuración
    repeat: int = 1                    # Veces a repetir
    delay_between_repeats: float = 2.0 # Segundos entre repeticiones
    _settled: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False, compare=False)

    async def wait_for_delivery(self) -> None:
        """Wait until playback settles or the queue stops before delivery."""
        await self._settled.wait()

    @property
    def delivery_confirmed(self) -> bool:
        """Whether every requested zone has a confirmed output receipt."""
        return bool(self.zone_receipts) and all(
            state is ZoneDelivery.CONFIRMED for state in self.zone_receipts.values()
        )

    def receipt(self) -> dict:
        """Project the delivery evidence without retaining the message."""
        receipt = {
            "announcement_id": self.announcement_id,
            "zones": {zone: state.value for zone, state in self.zone_receipts.items()},
        }
        if not self._settled.is_set():
            receipt["in_flight"] = True
        return receipt

    def _receipt_copy(self) -> "Announcement":
        """Keep a settled receipt without copying the spoken message into the ledger."""
        receipt = replace(self, message="", target_zones=self.target_zones.copy(),
                          failed_zones=self.failed_zones.copy(), zone_receipts=self.zone_receipts.copy())
        receipt._settled.set()
        return receipt


@dataclass(frozen=True)
class _VoiceAttempt:
    fingerprint: bytes
    announcement: Announcement


class IntercomSystem:
    """Deliver announcements to explicit local outputs.

    Args:
        tts_callback: The shared voice owner, returning a playback receipt.
        zone_manager: Configured audio routes, used to load available outputs.
        room_catalog: Configured transport ids and names for spoken destinations.
            Programmatic producers use the configured audio routes.
    """

    def __init__(
        self,
        tts_callback: Callable[[str, str], Any] | None = None,
        zone_manager=None,
        *,
        room_catalog: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.tts = tts_callback
        self.zone_manager = zone_manager
        self._room_names = {
            zone: {_room_name(label) for label in labels} for zone, labels in (room_catalog or {}).items()
        }

        self._announcement_queue: asyncio.Queue[Announcement] = asyncio.Queue()
        self._announcement_history: list[Announcement] = []
        self._voice_in_flight: dict[str, _VoiceAttempt] = {}
        self._voice_receipts: OrderedDict[str, _VoiceAttempt] = OrderedDict()
        self._running = False
        self._process_task: asyncio.Task | None = None
        self._zones: dict[str, dict] = {}
        self._on_announcement_delivered: Callable | None = None
        self._on_announcement_failed: Callable | None = None

    async def start(self):
        """Start announcement processing and load the configured outputs."""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_queue())
        await self._load_zones()
        logger.info("Intercom started with %d audio zones", len(self._zones))

    async def stop(self):
        """Stop processing announcements."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        while not self._announcement_queue.empty():
            announcement = self._announcement_queue.get_nowait()
            announcement._settled.set()
            self._announcement_queue.task_done()

    async def _load_zones(self):
        """Load only explicitly configured audio routes."""
        if self.zone_manager:
            for zone_id, zone in self.zone_manager.get_all_zones().items():
                self._zones[zone_id] = {
                    "name": zone.name,
                    "media_player": zone.media_player_entity,
                    "speaker": zone.speaker_entity,
                    "tts_target": zone.tts_target,
                }

    def register_zone(self, zone_id: str, name: str, media_player: str = None, tts_target: str = None):
        """Register an explicit route for programmatic producers."""
        self._zones[zone_id] = {
            "name": name,
            "media_player": media_player,
            "speaker": None,
            "tts_target": tts_target or media_player,
        }

    def _parse_announcement_command(self, complement: str) -> tuple[str, list[str]]:
        """Resolve only a leading destination slot, keeping message text intact."""
        target, message_start = intercom_slots(complement)
        message = complement[message_start:].strip()
        if target is None or target.casefold() in INTERCOM_BROADCAST_TARGETS:
            return message, list(self._room_names)
        label = _room_name(target)
        matches = [zone for zone, names in self._room_names.items() if label and label in names]
        return message, matches if len(matches) == 1 else []

    def _voice_attempt(self, request_id: str) -> _VoiceAttempt | None:
        return self._voice_in_flight.get(request_id) or self._voice_receipts.get(request_id)

    def is_voice_replay(self, request_id: str, complement: str) -> bool:
        """Check the playback owner's ID and exact voice payload before barge-in."""
        attempt = self._voice_attempt(request_id)
        return attempt is not None and attempt.fingerprint == sha256(complement.encode("utf-8")).digest()

    async def announce_from_voice(self, complement: str) -> tuple[TaskOutcome, str, Announcement | None]:
        """Deliver an admitted P3-F2 request at the ordinary configured volume."""
        request_id = CURRENT_REQUEST_ID.get()
        fingerprint = sha256(complement.encode("utf-8")).digest() if request_id is not None else b""
        if request_id is not None:
            previous = self._voice_attempt(request_id)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    return TaskOutcome(TaskStatus.FAILED, persistence=PersistenceState.NOT_WRITTEN,
                                       dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED), (
                        "Esa solicitud ya corresponde a otro anuncio."
                    ), None
                receipt = previous.announcement
                if receipt._settled.is_set():
                    receipt = receipt._receipt_copy()
                return self._voice_result(receipt)
        if not self._running or not self._room_names or self.tts is None:
            return TaskOutcome(TaskStatus.UNSUPPORTED, persistence=PersistenceState.NOT_WRITTEN,
                               dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED), (
                "El intercomunicador no está disponible ahora."
            ), None
        message, zones = self._parse_announcement_command(complement)
        if not zones:
            return TaskOutcome(TaskStatus.NEEDS_INPUT, persistence=PersistenceState.NOT_WRITTEN,
                               dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED), (
                "Decime el nombre de una sala configurada o pedime anunciar en toda la casa."
            ), None
        if not any(char.isalnum() for char in message):
            return TaskOutcome(TaskStatus.NEEDS_INPUT, persistence=PersistenceState.NOT_WRITTEN,
                               dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED), (
                "Decime qué querés anunciar."
            ), None

        announcement = Announcement(f"ann_{uuid.uuid4().hex[:8]}", message, zones)
        if request_id is not None:
            self._voice_in_flight[request_id] = _VoiceAttempt(fingerprint, announcement)
        try:
            await self._deliver_announcement(announcement)
        finally:
            if request_id is not None:
                self._voice_receipts[request_id] = _VoiceAttempt(fingerprint, announcement._receipt_copy())
                self._voice_in_flight.pop(request_id)
                if len(self._voice_receipts) > _VOICE_RECEIPT_LIMIT:
                    self._voice_receipts.popitem(last=False)
        return self._voice_result(announcement)

    @staticmethod
    def _voice_result(announcement: Announcement) -> tuple[TaskOutcome, str, Announcement]:
        zones = announcement.target_zones
        confirmed = sum(state is ZoneDelivery.CONFIRMED for state in announcement.zone_receipts.values())
        if announcement.delivery_confirmed:
            return TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.VOLATILE,
                               dispatch=DispatchState.ACCEPTED, effect=EffectState.OBSERVED,
                               observation_scope="software_state"), "Anuncio reproducido.", announcement
        if confirmed:
            return TaskOutcome(TaskStatus.UNCERTAIN, persistence=PersistenceState.VOLATILE,
                               dispatch=DispatchState.PARTIAL, effect=EffectState.PARTIAL), (
                f"El anuncio se reprodujo en {confirmed} de {len(zones)} salas."
            ), announcement
        if not announcement._settled.is_set() or ZoneDelivery.UNCERTAIN in announcement.zone_receipts.values():
            return TaskOutcome(TaskStatus.UNCERTAIN, persistence=PersistenceState.VOLATILE,
                               dispatch=DispatchState.UNCERTAIN, effect=EffectState.UNCERTAIN), (
                "No pude confirmar la reproducción del anuncio."
            ), announcement
        return TaskOutcome(TaskStatus.FAILED, persistence=PersistenceState.VOLATILE,
                           dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED), (
            "No pude reproducir el anuncio."
        ), announcement

    async def _process_queue(self):
        """Process announcements admitted by programmatic producers."""
        while self._running:
            try:
                announcement = await self._announcement_queue.get()
                try:
                    await self._deliver_announcement(announcement)
                finally:
                    self._announcement_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as error:
                logger.error("Announcement processing failed: %s", type(error).__name__)

    async def _deliver_announcement(self, announcement: Announcement) -> None:
        """Keep playback evidence for each zone, including an interrupted delivery."""
        try:
            targets = self._targets(announcement)
            announcement.zone_receipts = dict.fromkeys(targets, ZoneDelivery.NOT_SENT)
            outputs: dict[Hashable, list[str]] = {}
            for zone in announcement.zone_receipts:
                key = self._playback_output_key(zone)
                if key is None:
                    announcement.zone_receipts[zone] = ZoneDelivery.FAILED
                else:
                    outputs.setdefault(key, []).append(zone)
            # One attempt per physical channel in a broadcast, including failures.
            repeats = announcement.repeat if len(targets) == 1 and "all" not in announcement.target_zones else 1
            for index in range(repeats):
                if index:
                    await asyncio.sleep(announcement.delay_between_repeats)
                for zones in outputs.values():
                    previous = announcement.zone_receipts[zones[0]]
                    # Until the output returns, cancellation or a replay can only
                    # confirm earlier zones. Later zones remain not_sent.
                    if previous is not ZoneDelivery.CONFIRMED:
                        announcement.zone_receipts.update(dict.fromkeys(zones, ZoneDelivery.UNCERTAIN))
                    delivery = await self._announce_to_zone(zones[0], announcement.message)
                    if previous is ZoneDelivery.CONFIRMED or (
                        previous is ZoneDelivery.UNCERTAIN and delivery is not ZoneDelivery.CONFIRMED
                    ):
                        delivery = previous
                    announcement.zone_receipts.update(dict.fromkeys(zones, delivery))
        finally:
            # Preserve the legacy summary fields alongside the individual receipts.
            announcement.failed_zones = [zone for zone, state in announcement.zone_receipts.items()
                                         if state is not ZoneDelivery.CONFIRMED]
            announcement.delivered = ZoneDelivery.CONFIRMED in announcement.zone_receipts.values()
            announcement.delivered_at = datetime.now()
            self._announcement_history.append(announcement)
            if len(self._announcement_history) > 100:
                self._announcement_history = self._announcement_history[-50:]
            announcement._settled.set()

        try:
            if announcement.delivered and self._on_announcement_delivered:
                self._on_announcement_delivered(announcement)
            elif not announcement.delivered and self._on_announcement_failed:
                self._on_announcement_failed(announcement, "No zone confirmed delivery")
        except Exception as error:
            logger.warning("Announcement listener failed: %s", type(error).__name__)

    def _targets(self, announcement: Announcement) -> list[str]:
        return list(self._zones) if "all" in announcement.target_zones else announcement.target_zones

    def _playback_output_key(self, zone_id: str) -> Hashable | None:
        if self.zone_manager is not None:
            return self.zone_manager.playback_output_key(zone_id)
        room_id = zone_id.removeprefix(ZONE_PREFIX)
        if room_id in self._zones or ZONE_PREFIX + room_id in self._zones:
            return ("registered", room_id)
        return None

    async def _announce_to_zone(self, zone_id: str, message: str) -> ZoneDelivery:
        """Deliver through the shared local owner to one configured route."""
        try:
            result = await invoke_voice_callback(self.tts, message, zone_id)
        except VoiceDeliveryError as error:
            if error.result is not None and error.result.output_started:
                return ZoneDelivery.UNCERTAIN
            return ZoneDelivery.FAILED
        except TimeoutError:
            return ZoneDelivery.UNCERTAIN
        except Exception as error:
            logger.warning("Announcement output failed: %s", type(error).__name__)
            return ZoneDelivery.FAILED
        return ZoneDelivery.CONFIRMED if isinstance(result, ResponseResult) else ZoneDelivery.UNCERTAIN

    async def announce(
        self,
        message: str,
        zones: list[str] = None,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
        user_id: str = None,
    ) -> Announcement:
        """Enqueue an announcement for a programmatic producer."""
        announcement = Announcement(
            announcement_id=f"ann_{uuid.uuid4().hex[:8]}",
            message=message,
            target_zones=["all"] if zones is None else list(zones),
            priority=priority,
            created_by=user_id,
        )
        announcement.zone_receipts = dict.fromkeys(self._targets(announcement), ZoneDelivery.NOT_SENT)
        if self._running and self.tts is not None:
            try:
                self._announcement_queue.put_nowait(announcement)
                return announcement
            except Exception as error:
                logger.warning("Announcement could not be queued: %s", type(error).__name__)
        announcement._settled.set()
        return announcement

    async def announce_emergency(self, message: str) -> Announcement:
        """Enqueue a programmatic emergency announcement for all zones."""
        return await self.announce(message, zones=["all"], priority=AnnouncementPriority.EMERGENCY)

    async def call_zone(self, from_zone: str, to_zone: str, message: str = None) -> Announcement:
        """Enqueue a one-way message or a call notice in the destination zone."""
        if message:
            return await self.announce(message, zones=[to_zone])
        source = self._zones.get(from_zone, {}).get("name", from_zone)
        return await self.announce(f"Llamada desde {source}", zones=[to_zone], priority=AnnouncementPriority.HIGH)

    def on_announcement_delivered(self, callback: Callable[[Announcement], None]):
        """Register a callback when at least one zone confirms playback."""
        self._on_announcement_delivered = callback

    def on_announcement_failed(self, callback: Callable[[Announcement, str], None]):
        """Register a callback when no zone confirms playback."""
        self._on_announcement_failed = callback

    def get_zones(self) -> dict[str, str]:
        """Get configured audio routes and their labels."""
        return {zone_id: info["name"] for zone_id, info in self._zones.items()}

    def get_history(self, limit: int = 10) -> list[Announcement]:
        """Get the recent announcements and their delivery receipts."""
        return self._announcement_history[-limit:]

    def get_status(self) -> dict:
        """Get the state of the announcement processor."""
        return {
            "running": self._running,
            "zones_count": len(self._zones),
            "queue_size": self._announcement_queue.qsize(),
            "total_announcements": len(self._announcement_history),
            "zones": list(self._zones),
        }
