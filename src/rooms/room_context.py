"""
Room Context Module
Fusiona información de micrófono + Bluetooth para determinar automáticamente
la habitación desde donde se habla y proveer contexto al pipeline.

Cada habitación tiene:
- 1x XVF3800 (micrófono USB sobre RJ45)
- 1x Dongle BT (presencia USB sobre RJ45)
- Ambos llegan al servidor como dispositivos USB locales

El módulo resuelve automáticamente:
- "Apagá la luz" → ¿de qué habitación? → la del mic que detectó la voz
- Confirmación cruzada con BT (el celular del usuario está en esa zona)
- Contexto de Home Assistant por habitación
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable
from enum import StrEnum

logger = logging.getLogger(__name__)


class ContextSource(StrEnum):
    """Fuente de la detección de habitación"""
    MICROPHONE = "microphone"       # Detectado por audio (wake word en mic)
    BLUETOOTH = "bluetooth"         # Detectado por BLE
    BOTH = "both"                   # Confirmado por ambos
    MOTION_SENSOR = "motion"        # Sensor de movimiento HA
    MANUAL = "manual"               # Especificado por el usuario


@dataclass
class RoomConfig:
    """Configuración de una habitación"""
    room_id: str
    name: str
    display_name: str  # Para TTS: "el living", "la cocina"

    # Hardware USB
    mic_device_index: int | None = None      # sounddevice index del XVF3800
    mic_device_name: str | None = None       # Nombre USB para auto-detect
    bt_adapter: str | None = None            # MAC del adaptador BT (estable). Acepta también "hciN" como override explícito.
    bt_hci: str | None = None                # hciN resuelto al iniciar (no setear manualmente; lo llena el manager).

    # MA1260 output
    ma1260_zone: int | None = None          # MA1260 zone number (1-6)
    output_mode: str = "mono"                  # "stereo" or "mono"
    default_volume: int = 50                   # Default volume (0-100)
    noise_floor: float = 0.01                  # Noise floor for VAD

    # Home Assistant entities por defecto de esta habitación
    default_light: str | None = None         # light.living
    default_climate: str | None = None       # climate.living
    default_cover: str | None = None         # cover.living
    default_media_player: str | None = None  # media_player.living
    default_fan: str | None = None           # fan.living

    # Sensores HA
    motion_sensor: str | None = None         # binary_sensor.motion_living
    temperature_sensor: str | None = None    # sensor.temp_living
    humidity_sensor: str | None = None       # sensor.humidity_living

    # Aliases para comandos de voz
    aliases: list[str] = field(default_factory=list)  # ["living", "sala", "salón"]

    # Speaker para TTS de esta habitación
    tts_speaker: str | None = None           # media_player o device de audio


@dataclass
class RoomContext:
    """Contexto actual de una interacción por habitación"""
    room_id: str
    room_name: str
    display_name: str
    source: ContextSource
    confidence: float  # 0.0 - 1.0
    timestamp: float

    # Usuario detectado (si hay BT match)
    user_id: str | None = None
    user_name: str | None = None

    # Entities de HA para esta habitación
    entities: dict[str, str] = field(default_factory=dict)

    # Estado actual de la habitación
    is_occupied: bool = False
    people_count: int = 0
    temperature: float | None = None
    humidity: float | None = None

    @property
    def is_high_confidence(self) -> bool:
        """¿Es alta confianza? (confirmado por mic + BT)"""
        return self.confidence >= 0.8

    def get_entity(self, domain: str) -> str | None:
        """Obtener entity de HA por dominio (light, climate, etc)"""
        return self.entities.get(domain)


class RoomContextManager:
    """
    Gestor de contexto por habitación.

    Combina datos de MultiMicCapture + PresenceDetector para determinar
    automáticamente desde qué habitación habla el usuario y resolver
    comandos ambiguos como "apagá la luz" → light.cocina.

    Uso:
        manager = RoomContextManager()
        manager.add_room(RoomConfig(
            room_id="living",
            name="Living",
            display_name="el living",
            mic_device_index=2,
            bt_adapter="hci0",
            default_light="light.living",
            default_climate="climate.living_ac",
            aliases=["living", "sala", "salón"]
        ))

        # Cuando llega audio con wake word del mic 2:
        context = manager.resolve_room(mic_zone_id="living", user_id="mastar")
        # → RoomContext(room_id="living", confidence=1.0, source=BOTH)
    """

    def __init__(
        self,
        presence_detector=None,
        ha_client=None,
        cross_validation: bool = True,
        fallback_room: str | None = None
    ):
        """
        Args:
            presence_detector: PresenceDetector para datos BLE
            ha_client: HomeAssistant client para estados de entities
            cross_validation: Si True, usa BT para confirmar habitación del mic
            fallback_room: Habitación por defecto si no se puede determinar
        """
        self.presence_detector = presence_detector
        self.ha_client = ha_client
        self.cross_validation = cross_validation
        self.fallback_room = fallback_room

        # Habitaciones configuradas
        self._rooms: dict[str, RoomConfig] = {}

        # Mapeo mic_device_index → room_id
        self._mic_to_room: dict[int, str] = {}

        # Mapeo bt_adapter → room_id
        self._bt_to_room: dict[str, str] = {}

        # Mapeo alias → room_id
        self._alias_to_room: dict[str, str] = {}

        # Último contexto resuelto (cache)
        self._last_context: RoomContext | None = None
        self._last_context_time: float = 0

        # Historial de contextos por usuario
        self._user_room_history: dict[str, list[tuple[str, float]]] = {}

        # Callbacks
        self._on_room_changed: Callable | None = None

        logger.info("RoomContextManager inicializado")

    def add_room(self, config: RoomConfig):
        """Agregar una habitación al sistema.

        Si bt_adapter es una MAC, intenta resolverla a hciN leyendo sysfs.
        El hciN resuelto queda en config.bt_hci para que el BLE scanner lo use.
        """
        self._rooms[config.room_id] = config

        if config.mic_device_index is not None:
            self._mic_to_room[config.mic_device_index] = config.room_id

        if config.bt_adapter and not config.bt_hci:
            config.bt_hci = resolve_bt_adapter(config.bt_adapter)

        if config.bt_hci:
            self._bt_to_room[config.bt_hci] = config.room_id

        for alias in config.aliases:
            self._alias_to_room[alias.lower()] = config.room_id
        self._alias_to_room[config.room_id.lower()] = config.room_id
        self._alias_to_room[config.name.lower()] = config.room_id

        logger.info(
            f"Habitación agregada: {config.name} "
            f"(mic: {config.mic_device_index}, bt: {config.bt_adapter} → {config.bt_hci})"
        )

    def resolve_room(
        self,
        mic_zone_id: str | None = None,
        mic_device_index: int | None = None,
        user_id: str | None = None,
        spoken_room: str | None = None
    ) -> RoomContext | None:
        """
        Resolver habitación desde donde se habla.

        Prioridad:
        1. Si el usuario mencionó una habitación explícitamente → usar esa
        2. Si hay mic_zone_id → habitación del micrófono que capturó
        3. Cross-validate con BT si está disponible
        4. Fallback a última habitación conocida del usuario

        Args:
            mic_zone_id: Zone ID del micrófono que capturó el wake word
            mic_device_index: Device index alternativo del micrófono
            user_id: ID del usuario (para cross-validate con BT)
            spoken_room: Habitación mencionada en el comando ("la luz de la cocina")

        Returns:
            RoomContext con toda la info de la habitación
        """
        now = time.time()
        room_id = None
        source = ContextSource.MICROPHONE
        confidence = 0.5

        # 1. Si el usuario dijo explícitamente la habitación
        if spoken_room:
            resolved = self._resolve_alias(spoken_room)
            if resolved:
                room_id = resolved
                source = ContextSource.MANUAL
                confidence = 1.0
                logger.debug(f"Habitación explícita: {spoken_room} → {room_id}")

        # 2. Resolver por micrófono
        if not room_id and mic_zone_id:
            if mic_zone_id in self._rooms:
                room_id = mic_zone_id
                source = ContextSource.MICROPHONE
                confidence = 0.7

        if not room_id and mic_device_index is not None:
            room_id = self._mic_to_room.get(mic_device_index)
            if room_id:
                source = ContextSource.MICROPHONE
                confidence = 0.7

        # 3. Cross-validate con BT
        if room_id and user_id and self.cross_validation and self.presence_detector:
            bt_room = self._get_user_bt_room(user_id)
            if bt_room:
                if bt_room == room_id:
                    # Mic y BT coinciden → alta confianza
                    source = ContextSource.BOTH
                    confidence = 1.0
                    logger.debug(f"Confirmado mic+BT: {room_id}")
                else:
                    # No coinciden → confiar en el mic (el usuario habló ahí)
                    # pero registrar discrepancia
                    logger.debug(
                        f"Discrepancia mic({room_id}) vs BT({bt_room}), "
                        f"confiando en micrófono"
                    )
                    confidence = 0.7

        # 4. Si no se pudo resolver por mic, intentar solo BT
        if not room_id and user_id and self.presence_detector:
            bt_room = self._get_user_bt_room(user_id)
            if bt_room:
                room_id = bt_room
                source = ContextSource.BLUETOOTH
                confidence = 0.6

        # 5. Fallback a última habitación del usuario
        if not room_id and user_id:
            last_room = self._get_last_user_room(user_id)
            if last_room and (now - last_room[1]) < 300:  # < 5 min
                room_id = last_room[0]
                confidence = 0.4
                logger.debug(f"Usando última habitación de {user_id}: {room_id}")

        # 6. Fallback global
        if not room_id and self.fallback_room:
            room_id = self.fallback_room
            confidence = 0.2

        if not room_id:
            return None

        # Construir contexto
        config = self._rooms.get(room_id)
        if not config:
            return None

        context = RoomContext(
            room_id=room_id,
            room_name=config.name,
            display_name=config.display_name,
            source=source,
            confidence=confidence,
            timestamp=now,
            user_id=user_id,
            entities=self._build_entity_map(config)
        )

        # Enriquecer con datos de HA y presencia
        self._enrich_context(context, config)

        # Guardar en historial
        if user_id:
            if user_id not in self._user_room_history:
                self._user_room_history[user_id] = []
            self._user_room_history[user_id].append((room_id, now))
            # Mantener solo últimas 50 entradas
            if len(self._user_room_history[user_id]) > 50:
                self._user_room_history[user_id] = self._user_room_history[user_id][-50:]

        # Detectar cambio de habitación
        if self._last_context and self._last_context.room_id != room_id:
            if self._on_room_changed:
                self._on_room_changed(self._last_context.room_id, room_id, user_id)

        self._last_context = context
        self._last_context_time = now

        logger.info(
            f"Contexto resuelto: {config.name} "
            f"(source={source.value}, confidence={confidence:.1f}, user={user_id})"
        )

        return context

    def resolve_entity(
        self,
        domain: str,
        room_context: RoomContext | None = None,
        spoken_room: str | None = None
    ) -> str | None:
        """
        Resolver entity de HA dado un dominio y contexto.

        Ejemplo:
            "apagá la luz" + contexto=cocina → "light.cocina"
            "poné el aire a 22" + contexto=living → "climate.living_ac"

        Args:
            domain: Dominio de HA (light, climate, cover, fan, media_player)
            room_context: Contexto de habitación (si ya fue resuelto)
            spoken_room: Habitación mencionada explícitamente

        Returns:
            Entity ID de Home Assistant o None
        """
        # Si mencionó habitación explícita, resolver esa
        if spoken_room:
            resolved_room_id = self._resolve_alias(spoken_room)
            if resolved_room_id and resolved_room_id in self._rooms:
                config = self._rooms[resolved_room_id]
                return self._get_room_entity(config, domain)

        # Usar contexto actual
        ctx = room_context or self._last_context
        if ctx:
            return ctx.get_entity(domain)

        return None

    def get_room_config(self, room_id: str) -> RoomConfig | None:
        """Obtener configuración de una habitación"""
        return self._rooms.get(room_id)

    def get_all_rooms(self) -> dict[str, RoomConfig]:
        """Obtener todas las habitaciones"""
        return self._rooms.copy()

    def get_room_by_alias(self, alias: str) -> RoomConfig | None:
        """Buscar habitación por alias"""
        room_id = self._resolve_alias(alias)
        if room_id:
            return self._rooms.get(room_id)
        return None

    def get_tts_speaker(self, room_context: RoomContext) -> str | None:
        """Obtener speaker de TTS para responder en la habitación correcta"""
        config = self._rooms.get(room_context.room_id)
        if config:
            return config.tts_speaker
        return None

    def get_last_context(self) -> RoomContext | None:
        """Obtener último contexto resuelto"""
        return self._last_context

    def get_user_room(self, user_id: str) -> str | None:
        """Obtener habitación actual de un usuario"""
        # Primero BT
        if self.presence_detector:
            bt_room = self._get_user_bt_room(user_id)
            if bt_room:
                return bt_room

        # Fallback a historial
        last = self._get_last_user_room(user_id)
        if last and (time.time() - last[1]) < 300:
            return last[0]

        return None

    def get_room_summary(self) -> dict[str, dict]:
        """Resumen de todas las habitaciones"""
        summary = {}
        for room_id, config in self._rooms.items():
            occupancy = None
            if self.presence_detector:
                occupancy = self.presence_detector.get_zone_occupancy(room_id)

            summary[room_id] = {
                "name": config.name,
                "display_name": config.display_name,
                "mic_active": config.mic_device_index is not None,
                "bt_active": config.bt_adapter is not None,
                "occupied": occupancy.is_occupied if occupancy else False,
                "people": occupancy.estimated_people if occupancy else 0,
                "entities": self._build_entity_map(config)
            }
        return summary

    def on_room_changed(self, callback: Callable[[str, str, str | None], None]):
        """Callback cuando un usuario cambia de habitación (old_room, new_room, user_id)"""
        self._on_room_changed = callback

    # =========================================================================
    # Private methods
    # =========================================================================

    def _resolve_alias(self, text: str) -> str | None:
        """Resolver alias de habitación desde texto"""
        text_lower = text.lower().strip()

        # Match directo
        if text_lower in self._alias_to_room:
            return self._alias_to_room[text_lower]

        # Match parcial
        for alias, room_id in self._alias_to_room.items():
            if alias in text_lower or text_lower in alias:
                return room_id

        return None

    def _get_user_bt_room(self, user_id: str) -> str | None:
        """Obtener habitación del usuario por BLE"""
        if not self.presence_detector:
            return None

        zone = self.presence_detector.get_user_zone(user_id)
        if zone and zone in self._rooms:
            return zone
        return None

    def _get_last_user_room(self, user_id: str) -> tuple[str, float] | None:
        """Obtener última habitación conocida del usuario"""
        history = self._user_room_history.get(user_id, [])
        if history:
            return history[-1]
        return None

    def _build_entity_map(self, config: RoomConfig) -> dict[str, str]:
        """Construir mapa de entities para una habitación"""
        entities = {}
        if config.default_light:
            entities["light"] = config.default_light
        if config.default_climate:
            entities["climate"] = config.default_climate
        if config.default_cover:
            entities["cover"] = config.default_cover
        if config.default_media_player:
            entities["media_player"] = config.default_media_player
        if config.default_fan:
            entities["fan"] = config.default_fan
        return entities

    def _get_room_entity(self, config: RoomConfig, domain: str) -> str | None:
        """Obtener entity de una habitación por dominio"""
        mapping = {
            "light": config.default_light,
            "climate": config.default_climate,
            "cover": config.default_cover,
            "media_player": config.default_media_player,
            "fan": config.default_fan,
        }
        return mapping.get(domain)

    def _enrich_context(self, context: RoomContext, config: RoomConfig):
        """Enriquecer contexto con datos de HA y presencia"""
        # Ocupación por BLE
        if self.presence_detector:
            occupancy = self.presence_detector.get_zone_occupancy(config.room_id)
            if occupancy:
                context.is_occupied = occupancy.is_occupied
                context.people_count = occupancy.estimated_people

        # TODO: Obtener temperatura/humedad de HA de forma async
        # Por ahora se deja None, se puede enriquecer después


def create_default_rooms() -> list[RoomConfig]:
    """
    Crear configuración por defecto para las 5 habitaciones de KZA.

    Hardware por habitación:
    - 1x ReSpeaker XVF3800 (mic USB sobre extensor RJ45)
    - 1x Dongle BT 5.0 (USB sobre extensor RJ45)
    - Ambos llegan al servidor como dispositivos USB

    bt_adapter es la MAC del dongle BT de cada habitación (estable por hardware).
    El sistema resuelve MAC → hciN al iniciar leyendo /sys/class/bluetooth/.
    Ejecutar `python -m src.rooms.room_context --detect` para listar MACs disponibles.
    """
    rooms = [
        RoomConfig(
            room_id="living",
            name="Living",
            display_name="el living",
            mic_device_index=None,      # Auto-detect al iniciar
            mic_device_name="XVF3800",  # Buscar por nombre USB
            bt_adapter="F4:4E:FC:CF:BF:3F",  # UGREEN BT 5.3 living (instalado 2026-05-04)
            default_light="light.living",
            default_climate="climate.living_ac",
            default_cover="cover.living_persiana",
            default_media_player="media_player.living_tv",
            motion_sensor="binary_sensor.motion_living",
            temperature_sensor="sensor.temperature_living",
            aliases=["living", "sala", "salón", "el living"],
            tts_speaker="media_player.living_speaker"
        ),
        RoomConfig(
            room_id="escritorio",
            name="Escritorio",
            display_name="el escritorio",
            mic_device_index=None,
            mic_device_name="XVF3800",
            bt_adapter="F4:4E:FC:21:0D:66",  # UGREEN BT 5.3 escritorio
            default_light="light.escritorio",
            default_climate="climate.escritorio_ac",
            default_media_player="media_player.escritorio_monitor",
            motion_sensor="binary_sensor.motion_escritorio",
            temperature_sensor="sensor.temperature_escritorio",
            aliases=["escritorio", "oficina", "estudio", "el escritorio"],
            tts_speaker="media_player.escritorio_speaker"
        ),
        RoomConfig(
            room_id="hall",
            name="Hall",
            display_name="el hall",
            mic_device_index=None,
            mic_device_name="XVF3800",
            bt_adapter=None,  # TODO: setear MAC cuando se instale el dongle BT
            default_light="light.hall",
            motion_sensor="binary_sensor.motion_hall",
            aliases=["hall", "pasillo", "entrada", "el hall"],
            tts_speaker="media_player.hall_speaker"
        ),
        RoomConfig(
            room_id="cocina",
            name="Cocina",
            display_name="la cocina",
            mic_device_index=None,
            mic_device_name="XVF3800",
            bt_adapter=None,  # TODO: setear MAC cuando se instale el dongle BT
            default_light="light.cocina",
            default_fan="fan.cocina_extractor",
            motion_sensor="binary_sensor.motion_cocina",
            temperature_sensor="sensor.temperature_cocina",
            humidity_sensor="sensor.humidity_cocina",
            aliases=["cocina", "la cocina", "kitchen"],
            tts_speaker="media_player.cocina_speaker"
        ),
        RoomConfig(
            room_id="bano",
            name="Baño",
            display_name="el baño",
            mic_device_index=None,
            mic_device_name="XVF3800",
            bt_adapter=None,  # TODO: setear MAC cuando se instale el dongle BT
            default_light="light.bano",
            default_fan="fan.bano_extractor",
            motion_sensor="binary_sensor.motion_bano",
            humidity_sensor="sensor.humidity_bano",
            aliases=["baño", "el baño", "bathroom"],
            tts_speaker="media_player.bano_speaker"
        ),
    ]
    return rooms


async def auto_detect_microphones(rooms: list[RoomConfig]) -> list[RoomConfig]:
    """
    Auto-detectar XVF3800 en los dispositivos USB del sistema.

    Busca dispositivos de audio cuyo nombre contenga "XVF3800" o "ReSpeaker"
    y los asigna a las habitaciones en orden.

    Returns:
        Lista de RoomConfig con mic_device_index actualizado
    """
    try:
        import sounddevice as sd

        # Buscar todos los dispositivos de input
        xvf_devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                name = dev['name'].lower()
                if 'xvf3800' in name or 'respeaker' in name or 'xmos' in name:
                    xvf_devices.append({
                        "index": i,
                        "name": dev['name'],
                        "channels": dev['max_input_channels'],
                        "sample_rate": dev['default_samplerate']
                    })

        if not xvf_devices:
            logger.warning("No se encontraron dispositivos XVF3800. Buscando todos los USB mics...")
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0 and 'usb' in dev['name'].lower():
                    xvf_devices.append({
                        "index": i,
                        "name": dev['name'],
                        "channels": dev['max_input_channels'],
                        "sample_rate": dev['default_samplerate']
                    })

        # Asignar a habitaciones en orden
        rooms_needing_mic = [r for r in rooms if r.mic_device_index is None]

        for i, room in enumerate(rooms_needing_mic):
            if i < len(xvf_devices):
                room.mic_device_index = xvf_devices[i]['index']
                logger.info(
                    f"Auto-detect: {room.name} → mic device {xvf_devices[i]['index']} "
                    f"({xvf_devices[i]['name']})"
                )
            else:
                logger.warning(f"No hay mic disponible para {room.name}")

        return rooms

    except ImportError:
        logger.error("sounddevice no instalado")
        return rooms


_MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$")


def _looks_like_mac(value: str) -> bool:
    return bool(_MAC_RE.match(value))


def _list_bt_adapters_sysfs(bt_root: str) -> dict[str, str]:
    """Leer MACs desde /sys/class/bluetooth/hciN/address (kernels que lo exponen)."""
    import os

    if not os.path.exists(bt_root):
        return {}
    result: dict[str, str] = {}
    for entry in sorted(os.listdir(bt_root)):
        if not entry.startswith("hci"):
            continue
        addr_path = os.path.join(bt_root, entry, "address")
        if not os.path.exists(addr_path):
            continue
        with open(addr_path) as f:
            mac = f.read().strip().upper()
        if mac:
            result[entry] = mac
    return result


def _list_bt_adapters_hciconfig() -> dict[str, str]:
    """Fallback: parsear `hciconfig` cuando sysfs no expone address (BlueZ moderno)."""
    import subprocess

    try:
        out = subprocess.run(
            ["hciconfig"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    result: dict[str, str] = {}
    current_hci: str | None = None
    for line in out.splitlines():
        m = re.match(r"^(hci\d+):", line)
        if m:
            current_hci = m.group(1)
            continue
        if current_hci:
            mac_match = re.search(r"BD Address:\s*([0-9A-Fa-f:]{17})", line)
            if mac_match:
                result[current_hci] = mac_match.group(1).upper()
                current_hci = None
    return result


def list_bt_adapters(bt_root: str = "/sys/class/bluetooth") -> dict[str, str]:
    """
    Listar adaptadores BT del sistema como mapping hciN → MAC (uppercase).

    Intenta primero sysfs (`/sys/class/bluetooth/hciN/address`), después fallback
    a `hciconfig`. Tests inyectan un `bt_root` con archivos `address` simulados.

    Returns:
        {"hci0": "F4:4E:FC:21:0D:66", ...}. Vacío si no hay forma de leerlo.
    """
    result = _list_bt_adapters_sysfs(bt_root)
    if result:
        return result
    return _list_bt_adapters_hciconfig()


def resolve_bt_adapter(
    value: str | None,
    bt_root: str = "/sys/class/bluetooth"
) -> str | None:
    """
    Resolver una referencia de adaptador BT a un nombre hciN.

    Acepta:
    - MAC ("F4:4E:FC:21:0D:66"): busca en sysfs y devuelve el hciN correspondiente.
    - "hciN" literal: se devuelve tal cual (override explícito).
    - None / vacío: devuelve None.

    Si la MAC no matchea ningún adaptador presente, devuelve None y loguea warning
    (el caller decide si seguir o abortar).
    """
    if not value:
        return None

    if value.startswith("hci"):
        return value

    if _looks_like_mac(value):
        target = value.upper()
        adapters = list_bt_adapters(bt_root)
        for hci, mac in adapters.items():
            if mac == target:
                return hci
        logger.warning(
            f"BT adapter MAC {value} no encontrada en {bt_root} "
            f"(disponibles: {adapters or 'ninguno'})"
        )
        return None

    logger.warning(f"bt_adapter '{value}' no es ni MAC ni hciN — ignorado")
    return None


async def auto_detect_bt_adapters(rooms: list[RoomConfig]) -> list[RoomConfig]:
    """
    Auto-detectar adaptadores Bluetooth del sistema.

    Para rooms sin bt_adapter asignado, asigna la MAC de un hciN libre (en orden).
    Después resuelve bt_hci para todos los rooms con bt_adapter definido.
    """
    adapters = list_bt_adapters()
    if not adapters:
        logger.warning("No se encontraron adaptadores Bluetooth en /sys/class/bluetooth")
        return rooms

    used_macs = {r.bt_adapter.upper() for r in rooms if r.bt_adapter and _looks_like_mac(r.bt_adapter)}
    free = [(hci, mac) for hci, mac in adapters.items() if mac not in used_macs]

    rooms_needing_bt = [r for r in rooms if r.bt_adapter is None]
    for room, (hci, mac) in zip(rooms_needing_bt, free):
        room.bt_adapter = mac
        room.bt_hci = hci
        logger.info(f"Auto-detect: {room.name} → {hci} ({mac})")

    leftover = len(rooms_needing_bt) - len(free)
    if leftover > 0:
        logger.warning(f"{leftover} habitaciones sin BT adapter disponible")

    for room in rooms:
        if room.bt_adapter and not room.bt_hci:
            room.bt_hci = resolve_bt_adapter(room.bt_adapter)

    return rooms


if __name__ == "__main__":
    """Herramienta de diagnóstico para detectar hardware"""
    import sys
    import asyncio

    async def detect():
        print("=" * 60)
        print("KZA Room Hardware Detection")
        print("=" * 60)

        # Detectar micrófonos
        print("\n--- Micrófonos USB detectados ---")
        try:
            import sounddevice as sd
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    marker = " ← XVF3800" if any(
                        x in dev['name'].lower()
                        for x in ['xvf3800', 'respeaker', 'xmos']
                    ) else ""
                    print(f"  [{i}] {dev['name']} "
                          f"({dev['max_input_channels']}ch, "
                          f"{int(dev['default_samplerate'])}Hz)"
                          f"{marker}")
        except ImportError:
            print("  ⚠ sounddevice no instalado")

        # Detectar BT adapters
        print("\n--- Adaptadores Bluetooth detectados ---")
        adapters = list_bt_adapters()
        if adapters:
            for hci, mac in adapters.items():
                print(f"  {hci}: {mac}")
        else:
            print("  ⚠ No se encontraron adaptadores Bluetooth")

        # Validar mapeo MAC → hci de la config actual
        print("\n--- Validación config actual ---")
        rooms = create_default_rooms()
        rooms = await auto_detect_microphones(rooms)
        rooms = await auto_detect_bt_adapters(rooms)

        for room in rooms:
            mic = room.mic_device_index if room.mic_device_index is not None else "—"
            mac = room.bt_adapter or "—"
            hci = room.bt_hci or "—"
            status = "OK" if (room.bt_adapter and room.bt_hci) else (
                "sin BT" if not room.bt_adapter else "MAC no encontrada"
            )
            print(f"\n  {room.name}:")
            print(f"    mic_device_index: {mic}")
            print(f"    bt_adapter (MAC): {mac}")
            print(f"    bt_hci (resuelto): {hci}  [{status}]")

    if "--detect" in sys.argv:
        asyncio.run(detect())
    else:
        print("Uso: python -m src.rooms.room_context --detect")
