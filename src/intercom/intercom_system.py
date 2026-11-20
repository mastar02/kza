"""Intercom announcements through the application's local voice output."""

import asyncio
import logging
import re
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from src.core.household_clock import ZONE_PREFIX
from src.core.task_outcome import DispatchState, EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.core.voice_delivery import ResponseResult, VoiceDeliveryError, invoke_voice_callback

logger = logging.getLogger(__name__)

_ALL_ROOMS = frozenset({"toda la casa", "todas las salas", "todas partes", "todos lados", "todos", "parlantes",
                        "todos los parlantes"})
_DIRECTION = re.compile(r"^(?:en|a|al|por|para)(?:\s+|$)", re.IGNORECASE)
_ARTICLE = re.compile(r"^(?:el|la|los|las)\s+")
_MESSAGE_MARKER = re.compile(r"\bque\b|:", re.IGNORECASE)


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


class IntercomSystem:
    """Deliver announcements to explicit local outputs.

    Args:
        tts_callback: The shared voice owner, returning a playback receipt.
        zone_manager: Configured audio routes, used to load available outputs.
        room_catalog: Configured transport ids and spoken names. With no catalog
            supplied, only standalone programmatic calls use explicit audio routes.
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
        self._room_names = None if room_catalog is None else {
            zone: {_room_name(label) for label in labels} for zone, labels in room_catalog.items()
        }

        self._announcement_queue: asyncio.Queue[Announcement] = asyncio.Queue()
        self._announcement_history: list[Announcement] = []
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
        text = complement.strip()
        room_names = self._room_names or {}
        direction = _DIRECTION.match(text)
        marker = _MESSAGE_MARKER.search(text)
        target = text[direction.end() if direction else 0:marker.start() if marker else len(text)]
        target = target.strip(" ,")
        if direction or target.casefold() in _ALL_ROOMS:
            if target.casefold() in _ALL_ROOMS:
                zones = list(room_names)
            else:
                label = _room_name(target)
                matches = [zone for zone, names in room_names.items() if label and label in names]
                zones = matches if len(matches) == 1 else []
            return (text[marker.end():].strip() if marker else ""), zones
        if marker and marker.start() == 0:
            text = text[marker.end():].strip()
        return text, list(room_names)

    async def announce_from_voice(self, complement: str) -> tuple[TaskOutcome, str, Announcement | None]:
        """Deliver an admitted P3-F2 request at the ordinary configured volume."""
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
        await self._deliver_announcement(announcement)
        confirmed = sum(state is ZoneDelivery.CONFIRMED for state in announcement.zone_receipts.values())
        if confirmed == len(zones):
            return TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.VOLATILE,
                               dispatch=DispatchState.ACCEPTED, effect=EffectState.OBSERVED,
                               observation_scope="software_state"), "Anuncio reproducido.", announcement
        if confirmed:
            return TaskOutcome(TaskStatus.UNCERTAIN, persistence=PersistenceState.VOLATILE,
                               dispatch=DispatchState.PARTIAL, effect=EffectState.PARTIAL), (
                f"El anuncio se reprodujo en {confirmed} de {len(zones)} salas."
            ), announcement
        if ZoneDelivery.UNCERTAIN in announcement.zone_receipts.values():
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
                await self._deliver_announcement(announcement)
            except asyncio.CancelledError:
                break
            except Exception as error:
                logger.error("Announcement processing failed: %s", type(error).__name__)

    async def _deliver_announcement(self, announcement: Announcement) -> None:
        """Keep playback evidence for each zone, including an interrupted delivery."""
        catalog = self._zones if self._room_names is None else self._room_names
        targets = list(catalog) if "all" in announcement.target_zones else announcement.target_zones
        announcement.zone_receipts = dict.fromkeys(targets, ZoneDelivery.NOT_SENT)
        try:
            for index in range(announcement.repeat):
                if index:
                    await asyncio.sleep(announcement.delay_between_repeats)
                for zone in announcement.zone_receipts:
                    try:
                        delivery = await self._announce_to_zone(zone, announcement.message)
                    except asyncio.CancelledError:
                        if announcement.zone_receipts[zone] is not ZoneDelivery.CONFIRMED:
                            announcement.zone_receipts[zone] = ZoneDelivery.UNCERTAIN
                        raise
                    previous = announcement.zone_receipts[zone]
                    if previous is not ZoneDelivery.CONFIRMED and (
                        previous is not ZoneDelivery.UNCERTAIN or delivery is ZoneDelivery.CONFIRMED
                    ):
                        announcement.zone_receipts[zone] = delivery
        finally:
            # Preserve the legacy summary fields alongside the individual receipts.
            announcement.failed_zones = [zone for zone, state in announcement.zone_receipts.items()
                                         if state is not ZoneDelivery.CONFIRMED]
            announcement.delivered = ZoneDelivery.CONFIRMED in announcement.zone_receipts.values()
            announcement.delivered_at = datetime.now()
            self._announcement_history.append(announcement)
            if len(self._announcement_history) > 100:
                self._announcement_history = self._announcement_history[-50:]

        try:
            if announcement.delivered and self._on_announcement_delivered:
                self._on_announcement_delivered(announcement)
            elif not announcement.delivered and self._on_announcement_failed:
                self._on_announcement_failed(announcement, "No zone confirmed delivery")
        except Exception as error:
            logger.warning("Announcement listener failed: %s", type(error).__name__)

    async def _announce_to_zone(self, zone_id: str, message: str) -> ZoneDelivery:
        """Deliver through the shared local owner to one configured route."""
        room_id = zone_id.removeprefix(ZONE_PREFIX)
        canonical = ZONE_PREFIX + room_id
        if self._room_names is not None and canonical not in self._room_names:
            return ZoneDelivery.FAILED
        if room_id not in self._zones and canonical not in self._zones:
            return ZoneDelivery.FAILED
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
            target_zones=zones or ["all"],
            priority=priority,
            created_by=user_id,
        )
        await self._announcement_queue.put(announcement)
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
