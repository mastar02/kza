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
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ContextSource(Enum):
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
    mic_device_index: Optional[int] = None      # sounddevice index del XVF3800
    mic_device_name: Optional[str] = None       # Nombre USB para auto-detect
    bt_adapter: Optional[str] = None            # hci0, hci1, etc.

    # Home Assistant entities por defecto de esta habitación
    default_light: Optional[str] = None         # light.living
    default_climate: Optional[str] = None       # climate.living
    default_cover: Optional[str] = None         # cover.living
    default_media_player: Optional[str] = None  # media_player.living
    default_fan: Optional[str] = None           # fan.living

    # Sensores HA
    motion_sensor: Optional[str] = None         # binary_sensor.motion_living
    temperature_sensor: Optional[str] = None    # sensor.temp_living
    humidity_sensor: Optional[str] = None       # sensor.humidity_living

    # Aliases para comandos de voz
    aliases: List[str] = field(default_factory=list)  # ["living", "sala", "salón"]

    # Speaker para TTS de esta habitación
    tts_speaker: Optional[str] = None           # media_player o device de audio


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
    user_id: Optional[str] = None
    user_name: Optional[str] = None

    # Entities de HA para esta habitación
    entities: Dict[str, str] = field(default_factory=dict)

    # Estado actual de la habitación
    is_occupied: bool = False
    people_count: int = 0
    temperature: Optional[float] = None
    humidity: Optional[float] = None

    @property
    def is_high_confidence(self) -> bool:
        """¿Es alta confianza? (confirmado por mic + BT)"""
        return self.confidence >= 0.8

    def get_entity(self, domain: str) -> Optional[str]:
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
        fallback_room: Optional[str] = None
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
        self._rooms: Dict[str, RoomConfig] = {}

        # Mapeo mic_device_index → room_id
        self._mic_to_room: Dict[int, str] = {}

        # Mapeo bt_adapter → room_id
        self._bt_to_room: Dict[str, str] = {}

        # Mapeo alias → room_id
        self._alias_to_room: Dict[str, str] = {}

        # Último contexto resuelto (cache)
        self._last_context: Optional[RoomContext] = None
        self._last_context_time: float = 0

        # Historial de contextos por usuario
        self._user_room_history: Dict[str, List[Tuple[str, float]]] = {}

        # Callbacks
        self._on_room_changed: Optional[Callable] = None

        logger.info("RoomContextManager inicializado")

    def add_room(self, config: RoomConfig):
        """Agregar una habitación al sistema"""
        self._rooms[config.room_id] = config

        # Mapeos
        if config.mic_device_index is not None:
            self._mic_to_room[config.mic_device_index] = config.room_id

        if config.bt_adapter:
            self._bt_to_room[config.bt_adapter] = config.room_id

        # Aliases
        for alias in config.aliases:
            self._alias_to_room[alias.lower()] = config.room_id
        self._alias_to_room[config.room_id.lower()] = config.room_id
        self._alias_to_room[config.name.lower()] = config.room_id

        logger.info(
            f"Habitación agregada: {config.name} "
            f"(mic: {config.mic_device_index}, bt: {config.bt_adapter})"
        )

    def resolve_room(
        self,
        mic_zone_id: Optional[str] = None,
        mic_device_index: Optional[int] = None,
        user_id: Optional[str] = None,
        spoken_room: Optional[str] = None
    ) -> Optional[RoomContext]:
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
        room_context: Optional[RoomContext] = None,
        spoken_room: Optional[str] = None
    ) -> Optional[str]:
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

    def get_room_config(self, room_id: str) -> Optional[RoomConfig]:
        """Obtener configuración de una habitación"""
        return self._rooms.get(room_id)

    def get_all_rooms(self) -> Dict[str, RoomConfig]:
        """Obtener todas las habitaciones"""
        return self._rooms.copy()

    def get_room_by_alias(self, alias: str) -> Optional[RoomConfig]:
        """Buscar habitación por alias"""
        room_id = self._resolve_alias(alias)
        if room_id:
            return self._rooms.get(room_id)
        return None

    def get_tts_speaker(self, room_context: RoomContext) -> Optional[str]:
        """Obtener speaker de TTS para responder en la habitación correcta"""
        config = self._rooms.get(room_context.room_id)
        if config:
            return config.tts_speaker
        return None

    def get_last_context(self) -> Optional[RoomContext]:
        """Obtener último contexto resuelto"""
        return self._last_context

    def get_user_room(self, user_id: str) -> Optional[str]:
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

    def get_room_summary(self) -> Dict[str, dict]:
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

    def on_room_changed(self, callback: Callable[[str, str, Optional[str]], None]):
        """Callback cuando un usuario cambia de habitación (old_room, new_room, user_id)"""
        self._on_room_changed = callback

    # =========================================================================
    # Private methods
    # =========================================================================

    def _resolve_alias(self, text: str) -> Optional[str]:
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

    def _get_user_bt_room(self, user_id: str) -> Optional[str]:
        """Obtener habitación del usuario por BLE"""
        if not self.presence_detector:
            return None

        zone = self.presence_detector.get_user_zone(user_id)
        if zone and zone in self._rooms:
            return zone
        return None

    def _get_last_user_room(self, user_id: str) -> Optional[Tuple[str, float]]:
        """Obtener última habitación conocida del usuario"""
        history = self._user_room_history.get(user_id, [])
        if history:
            return history[-1]
        return None

    def _build_entity_map(self, config: RoomConfig) -> Dict[str, str]:
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

    def _get_room_entity(self, config: RoomConfig, domain: str) -> Optional[str]:
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


def create_default_rooms() -> List[RoomConfig]:
    """
    Crear configuración por defecto para las 5 habitaciones de KZA.

    Hardware por habitación:
    - 1x ReSpeaker XVF3800 (mic USB sobre extensor RJ45)
    - 1x Dongle BT 5.0 (USB sobre extensor RJ45)
    - Ambos llegan al servidor como dispositivos USB

    Los device_index y hci adapters se deben ajustar según
    el orden en que Linux detecta los USB al bootear.
    Ejecutar `python -m src.rooms.room_context --detect` para auto-detectar.
    """
    rooms = [
        RoomConfig(
            room_id="living",
            name="Living",
            display_name="el living",
            mic_device_index=None,      # Auto-detect al iniciar
            mic_device_name="XVF3800",  # Buscar por nombre USB
            bt_adapter="hci0",
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
            bt_adapter="hci1",
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
            bt_adapter="hci2",
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
            bt_adapter="hci3",
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
            bt_adapter="hci4",
            default_light="light.bano",
            default_fan="fan.bano_extractor",
            motion_sensor="binary_sensor.motion_bano",
            humidity_sensor="sensor.humidity_bano",
            aliases=["baño", "el baño", "bathroom"],
            tts_speaker="media_player.bano_speaker"
        ),
    ]
    return rooms


async def auto_detect_microphones(rooms: List[RoomConfig]) -> List[RoomConfig]:
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


async def auto_detect_bt_adapters(rooms: List[RoomConfig]) -> List[RoomConfig]:
    """
    Auto-detectar adaptadores Bluetooth del sistema.

    Busca /sys/class/bluetooth/hciN y asigna en orden.
    """
    import os

    bt_path = "/sys/class/bluetooth"
    if not os.path.exists(bt_path):
        logger.warning("No se encontró /sys/class/bluetooth")
        return rooms

    adapters = sorted([
        d for d in os.listdir(bt_path)
        if d.startswith("hci")
    ])

    rooms_needing_bt = [r for r in rooms if r.bt_adapter is None]

    for i, room in enumerate(rooms_needing_bt):
        if i < len(adapters):
            room.bt_adapter = adapters[i]
            logger.info(f"Auto-detect: {room.name} → BT adapter {adapters[i]}")
        else:
            logger.warning(f"No hay BT adapter disponible para {room.name}")

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
        import os
        bt_path = "/sys/class/bluetooth"
        if os.path.exists(bt_path):
            for adapter in sorted(os.listdir(bt_path)):
                if adapter.startswith("hci"):
                    addr_path = os.path.join(bt_path, adapter, "address")
                    addr = "unknown"
                    if os.path.exists(addr_path):
                        with open(addr_path) as f:
                            addr = f.read().strip()
                    print(f"  {adapter}: {addr}")
        else:
            print("  ⚠ No se encontraron adaptadores Bluetooth")

        # Generar config sugerida
        print("\n--- Configuración sugerida ---")
        rooms = create_default_rooms()
        rooms = await auto_detect_microphones(rooms)
        rooms = await auto_detect_bt_adapters(rooms)

        for room in rooms:
            print(f"\n  {room.name}:")
            print(f"    mic_device_index: {room.mic_device_index}")
            print(f"    bt_adapter: {room.bt_adapter}")

    if "--detect" in sys.argv:
        asyncio.run(detect())
    else:
        print("Uso: python -m src.rooms.room_context --detect")
