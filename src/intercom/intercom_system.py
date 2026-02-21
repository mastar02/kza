"""
Intercom System - Sistema de Anuncios e Intercomunicador
Permite anuncios multi-room y comunicación entre zonas.

"Anuncia que la cena está lista"
"Dile a los niños que bajen"
"Anuncia en toda la casa que llegó el delivery"
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class AnnouncementPriority(Enum):
    LOW = "low"           # No interrumpe música
    NORMAL = "normal"     # Baja volumen temporalmente
    HIGH = "high"         # Interrumpe todo
    EMERGENCY = "emergency"  # Máximo volumen, todas las zonas


@dataclass
class Announcement:
    """Un anuncio para reproducir"""
    announcement_id: str
    message: str
    target_zones: list[str]           # Zonas destino ("all" = todas)
    priority: AnnouncementPriority = AnnouncementPriority.NORMAL

    # Metadatos
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None  # user_id
    source_zone: Optional[str] = None # Zona de origen

    # Estado
    delivered: bool = False
    delivered_at: Optional[datetime] = None
    failed_zones: list[str] = field(default_factory=list)

    # Configuración
    repeat: int = 1                    # Veces a repetir
    delay_between_repeats: float = 2.0 # Segundos entre repeticiones
    chime_before: bool = True          # Sonido de atención antes


class IntercomSystem:
    """
    Sistema de anuncios e intercomunicador.

    Características:
    - Anuncios a zonas específicas o toda la casa
    - Prioridades (no interrumpir vs urgente)
    - Intercomunicador bidireccional
    - Comandos de voz naturales
    - Integración con media players de HA
    """

    # Patrones de voz
    ANNOUNCE_PATTERNS = [
        (r"anuncia(?:r)?(?:\s+que)?\s+(.+)", None),  # "Anuncia que la cena está lista"
        (r"avisa(?:r)?(?:\s+que)?\s+(.+)", None),    # "Avisa que llegué"
        (r"di(?:le)?(?:s)?(?:\s+(?:a\s+)?(?:los|las|el|la|todo[s]?))?(?:\s+que)?\s+(.+)", None),  # "Diles que bajen"
        (r"comunica(?:r)?(?:\s+que)?\s+(.+)", None), # "Comunica que hay visitas"
    ]

    # Zonas comunes
    ZONE_ALIASES = {
        "toda la casa": "all",
        "todas partes": "all",
        "everywhere": "all",
        "cocina": "kitchen",
        "sala": "living_room",
        "living": "living_room",
        "comedor": "dining_room",
        "cuarto principal": "master_bedroom",
        "habitación principal": "master_bedroom",
        "cuarto de niños": "kids_room",
        "cuarto de los niños": "kids_room",
        "baño": "bathroom",
        "garage": "garage",
        "jardín": "garden",
        "patio": "garden",
        "oficina": "office",
        "estudio": "office",
    }

    # Destinatarios especiales
    RECIPIENT_PATTERNS = {
        r"(?:a\s+)?(?:los\s+)?niños": ["kids_room"],
        r"(?:a\s+)?(?:mi\s+)?(?:esposo|esposa|pareja)": ["master_bedroom", "living_room"],
        r"(?:a\s+)?todos": ["all"],
        r"(?:a\s+)?arriba": ["upstairs"],  # Grupo de zonas
        r"(?:a\s+)?abajo": ["downstairs"],
    }

    def __init__(
        self,
        tts_callback: Callable[[str, str], Any] = None,
        media_player_callback: Callable[[str, str, dict], Any] = None,
        zone_manager = None,
        ha_client = None,
        default_volume: float = 0.7,
        announcement_chime: str = "chime.mp3",
        emergency_volume: float = 1.0
    ):
        self.tts = tts_callback
        self.media_player = media_player_callback
        self.zone_manager = zone_manager
        self.ha = ha_client

        self.default_volume = default_volume
        self.emergency_volume = emergency_volume
        self.announcement_chime = announcement_chime

        # Estado
        self._announcement_queue: asyncio.Queue = asyncio.Queue()
        self._announcement_history: list[Announcement] = []
        self._running = False
        self._process_task: Optional[asyncio.Task] = None

        # Zonas disponibles
        self._zones: dict[str, dict] = {}  # zone_id -> {name, media_player, speaker}

        # Callbacks
        self._on_announcement_delivered: Optional[Callable] = None
        self._on_announcement_failed: Optional[Callable] = None

    async def start(self):
        """Iniciar sistema de intercomunicador"""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_queue())

        # Cargar zonas disponibles
        await self._load_zones()

        logger.info(f"Intercom iniciado con {len(self._zones)} zonas")

    async def stop(self):
        """Detener sistema"""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

    async def _load_zones(self):
        """Cargar zonas desde Home Assistant o zone_manager"""
        if self.zone_manager:
            for zone_id, zone in self.zone_manager.get_all_zones().items():
                self._zones[zone_id] = {
                    "name": zone.name,
                    "media_player": zone.media_player_entity,
                    "speaker": zone.speaker_entity,
                    "tts_target": zone.tts_target
                }
        elif self.ha:
            # Descubrir media_players
            try:
                entities = await self.ha.get_entities("media_player")
                for entity in entities:
                    entity_id = entity["entity_id"]
                    name = entity.get("attributes", {}).get("friendly_name", entity_id)

                    # Inferir zona del nombre
                    zone_id = entity_id.split(".")[-1]

                    self._zones[zone_id] = {
                        "name": name,
                        "media_player": entity_id,
                        "speaker": None,
                        "tts_target": entity_id
                    }
            except Exception as e:
                logger.error(f"Error cargando zonas: {e}")

    def register_zone(self, zone_id: str, name: str, media_player: str = None, tts_target: str = None):
        """Registrar zona manualmente"""
        self._zones[zone_id] = {
            "name": name,
            "media_player": media_player,
            "speaker": None,
            "tts_target": tts_target or media_player
        }

    # ==================== Comandos de Voz ====================

    def handle_voice_command(
        self,
        text: str,
        user_id: str = None,
        source_zone: str = None
    ) -> dict:
        """
        Procesar comando de voz relacionado con anuncios.

        Returns:
            {
                "handled": bool,
                "response": str,
                "announcement": Announcement o None
            }
        """
        text_lower = text.lower().strip()

        # Verificar si es comando de anuncio
        message, target_zones = self._parse_announcement_command(text_lower)

        if message:
            # Crear y encolar anuncio
            announcement = Announcement(
                announcement_id=f"ann_{uuid.uuid4().hex[:8]}",
                message=message,
                target_zones=target_zones,
                priority=self._infer_priority(text_lower),
                created_by=user_id,
                source_zone=source_zone
            )

            # Encolar
            self._announcement_queue.put_nowait(announcement)

            # Respuesta
            if target_zones == ["all"]:
                zones_text = "toda la casa"
            else:
                zones_text = ", ".join(self._get_zone_name(z) for z in target_zones)

            response = f"Anunciando en {zones_text}"

            return {
                "handled": True,
                "response": response,
                "announcement": announcement
            }

        return {"handled": False, "response": "", "announcement": None}

    def _parse_announcement_command(self, text: str) -> tuple[str, list[str]]:
        """
        Parsear comando de anuncio.

        Returns:
            (mensaje, lista_de_zonas)
        """
        message = None
        target_zones = ["all"]  # Por defecto toda la casa

        # Buscar patrón de anuncio
        for pattern, _ in self.ANNOUNCE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                message = match.group(1).strip()
                break

        if not message:
            return None, []

        # Buscar zona destino en el texto
        for zone_text, zone_id in self.ZONE_ALIASES.items():
            if zone_text in text:
                target_zones = [zone_id] if zone_id != "all" else ["all"]
                # Limpiar zona del mensaje
                message = message.replace(zone_text, "").strip()
                break

        # Buscar destinatarios especiales
        for pattern, zones in self.RECIPIENT_PATTERNS.items():
            if re.search(pattern, text):
                target_zones = zones
                break

        # Buscar "en [zona]" al final
        match = re.search(r"(?:en\s+(?:la\s+|el\s+)?)([\w\s]+)$", message)
        if match:
            zone_text = match.group(1).strip()
            if zone_text in self.ZONE_ALIASES:
                target_zones = [self.ZONE_ALIASES[zone_text]]
                message = message[:match.start()].strip()

        return message, target_zones

    def _infer_priority(self, text: str) -> AnnouncementPriority:
        """Inferir prioridad del mensaje"""
        urgent_keywords = ["urgente", "emergencia", "importante", "ahora mismo"]
        if any(kw in text for kw in urgent_keywords):
            return AnnouncementPriority.HIGH

        emergency_keywords = ["fuego", "incendio", "ayuda", "peligro", "evacuación"]
        if any(kw in text for kw in emergency_keywords):
            return AnnouncementPriority.EMERGENCY

        return AnnouncementPriority.NORMAL

    def _get_zone_name(self, zone_id: str) -> str:
        """Obtener nombre legible de zona"""
        if zone_id in self._zones:
            return self._zones[zone_id]["name"]
        return zone_id.replace("_", " ").title()

    # ==================== Procesamiento ====================

    async def _process_queue(self):
        """Procesar cola de anuncios"""
        while self._running:
            try:
                # Obtener siguiente anuncio
                announcement = await asyncio.wait_for(
                    self._announcement_queue.get(),
                    timeout=1.0
                )

                await self._deliver_announcement(announcement)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error procesando anuncio: {e}")

    async def _deliver_announcement(self, announcement: Announcement):
        """Entregar un anuncio a las zonas destino"""
        logger.info(
            f"📢 Anuncio [{announcement.priority.value}]: "
            f"'{announcement.message}' -> {announcement.target_zones}"
        )

        # Resolver "all" a todas las zonas
        if "all" in announcement.target_zones:
            zones = list(self._zones.keys())
        else:
            zones = announcement.target_zones

        # Determinar volumen según prioridad
        volume = self.default_volume
        if announcement.priority == AnnouncementPriority.HIGH:
            volume = min(1.0, self.default_volume + 0.2)
        elif announcement.priority == AnnouncementPriority.EMERGENCY:
            volume = self.emergency_volume

        # Entregar a cada zona
        success_zones = []
        failed_zones = []

        for _ in range(announcement.repeat):
            for zone_id in zones:
                try:
                    await self._announce_to_zone(
                        zone_id,
                        announcement.message,
                        volume=volume,
                        chime=announcement.chime_before and _ == 0  # Chime solo primera vez
                    )
                    success_zones.append(zone_id)
                except Exception as e:
                    logger.error(f"Error anunciando en {zone_id}: {e}")
                    failed_zones.append(zone_id)

            if announcement.repeat > 1:
                await asyncio.sleep(announcement.delay_between_repeats)

        # Actualizar estado
        announcement.delivered = len(success_zones) > 0
        announcement.delivered_at = datetime.now()
        announcement.failed_zones = failed_zones

        # Guardar en historial
        self._announcement_history.append(announcement)
        if len(self._announcement_history) > 100:
            self._announcement_history = self._announcement_history[-50:]

        # Callbacks
        if announcement.delivered and self._on_announcement_delivered:
            self._on_announcement_delivered(announcement)
        elif not announcement.delivered and self._on_announcement_failed:
            self._on_announcement_failed(announcement, "All zones failed")

    async def _announce_to_zone(
        self,
        zone_id: str,
        message: str,
        volume: float = 0.7,
        chime: bool = True
    ):
        """Anunciar mensaje en una zona específica"""
        zone = self._zones.get(zone_id)
        if not zone:
            logger.warning(f"Zona desconocida: {zone_id}")
            return

        # Reproducir chime primero (si está configurado)
        if chime and self.announcement_chime and self.ha:
            try:
                await self.ha.call_service(
                    "media_player",
                    "play_media",
                    zone["media_player"],
                    {
                        "media_content_id": f"/local/sounds/{self.announcement_chime}",
                        "media_content_type": "music"
                    }
                )
                await asyncio.sleep(0.5)  # Esperar a que termine el chime
            except Exception:
                pass  # Chime es opcional

        # Anunciar con TTS
        if self.tts:
            self.tts(message, zone_id)
        elif self.ha and zone.get("tts_target"):
            # Usar TTS de Home Assistant directamente
            await self.ha.call_service(
                "tts",
                "speak",
                zone["tts_target"],
                {
                    "message": message,
                    "cache": False
                }
            )

    # ==================== API Programática ====================

    async def announce(
        self,
        message: str,
        zones: list[str] = None,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
        user_id: str = None
    ) -> Announcement:
        """
        Hacer un anuncio programáticamente.

        Args:
            message: Texto del anuncio
            zones: Lista de zone_ids (None = todas)
            priority: Prioridad del anuncio
            user_id: ID del usuario que hace el anuncio
        """
        announcement = Announcement(
            announcement_id=f"ann_{uuid.uuid4().hex[:8]}",
            message=message,
            target_zones=zones or ["all"],
            priority=priority,
            created_by=user_id
        )

        await self._announcement_queue.put(announcement)
        return announcement

    async def announce_emergency(self, message: str):
        """Anuncio de emergencia (todas las zonas, máximo volumen)"""
        return await self.announce(
            message,
            zones=["all"],
            priority=AnnouncementPriority.EMERGENCY
        )

    async def call_zone(self, from_zone: str, to_zone: str, message: str = None):
        """
        Iniciar llamada entre zonas (intercomunicador).

        Si no hay mensaje, activa modo escucha bidireccional.
        """
        if message:
            # Solo enviar mensaje
            return await self.announce(
                message,
                zones=[to_zone],
                priority=AnnouncementPriority.NORMAL
            )
        else:
            # TODO: Implementar modo escucha bidireccional
            logger.info(f"Intercom: {from_zone} -> {to_zone}")
            return await self.announce(
                f"Llamada desde {self._get_zone_name(from_zone)}",
                zones=[to_zone],
                priority=AnnouncementPriority.HIGH
            )

    # ==================== Callbacks ====================

    def on_announcement_delivered(self, callback: Callable[[Announcement], None]):
        """Registrar callback para anuncio entregado"""
        self._on_announcement_delivered = callback

    def on_announcement_failed(self, callback: Callable[[Announcement, str], None]):
        """Registrar callback para anuncio fallido"""
        self._on_announcement_failed = callback

    # ==================== Estado ====================

    def get_zones(self) -> dict:
        """Obtener zonas disponibles"""
        return {z: info["name"] for z, info in self._zones.items()}

    def get_history(self, limit: int = 10) -> list[Announcement]:
        """Obtener historial de anuncios"""
        return self._announcement_history[-limit:]

    def get_status(self) -> dict:
        """Obtener estado del sistema"""
        return {
            "running": self._running,
            "zones_count": len(self._zones),
            "queue_size": self._announcement_queue.qsize(),
            "total_announcements": len(self._announcement_history),
            "zones": list(self._zones.keys())
        }
