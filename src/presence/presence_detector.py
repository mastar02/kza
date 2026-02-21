"""
Presence Detector Module
Detector de presencia de alto nivel que combina múltiples fuentes.

Soporta:
- BLE scanning (teléfonos, relojes)
- Integración con Home Assistant (sensores de movimiento)
- Tracking por zona/habitación
- Asociación con usuarios registrados
- Eventos de entrada/salida
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from collections import defaultdict

from src.presence.ble_scanner import BLEScanner, BLEDevice, DeviceType

logger = logging.getLogger(__name__)


class PresenceState(Enum):
    """Estado de presencia de un usuario"""
    UNKNOWN = "unknown"
    HOME = "home"           # En casa (detectado)
    AWAY = "away"           # Fuera de casa
    JUST_ARRIVED = "just_arrived"  # Recién llegó (< 5 min)
    JUST_LEFT = "just_left"        # Recién salió (< 5 min)
    SLEEPING = "sleeping"          # Probablemente durmiendo (noche + sin movimiento)


@dataclass
class UserPresence:
    """Estado de presencia de un usuario"""
    user_id: str
    state: PresenceState = PresenceState.UNKNOWN
    current_zone: Optional[str] = None
    last_zone: Optional[str] = None

    # Dispositivos asociados
    ble_devices: list[str] = field(default_factory=list)  # MAC addresses

    # Timestamps
    last_seen: float = 0
    last_state_change: float = 0
    arrival_time: Optional[float] = None
    departure_time: Optional[float] = None

    # Confianza
    confidence: float = 0.0

    @property
    def is_home(self) -> bool:
        return self.state in [PresenceState.HOME, PresenceState.JUST_ARRIVED]

    @property
    def minutes_since_seen(self) -> float:
        if self.last_seen == 0:
            return float('inf')
        return (time.time() - self.last_seen) / 60

    @property
    def time_home_str(self) -> str:
        """Tiempo en casa como string legible"""
        if not self.arrival_time:
            return "desconocido"
        minutes = (time.time() - self.arrival_time) / 60
        if minutes < 60:
            return f"{int(minutes)} minutos"
        hours = minutes / 60
        if hours < 24:
            return f"{hours:.1f} horas"
        days = hours / 24
        return f"{days:.1f} días"


@dataclass
class RoomOccupancy:
    """Ocupación de una habitación/zona"""
    zone_id: str
    zone_name: str

    # Conteos
    estimated_people: int = 0
    known_users: list[str] = field(default_factory=list)  # user_ids
    unknown_devices: int = 0

    # BLE devices en la zona
    devices: list[BLEDevice] = field(default_factory=list)

    # Timestamps
    last_motion: float = 0  # Último movimiento detectado (sensor HA)
    last_presence: float = 0  # Última presencia BLE

    @property
    def is_occupied(self) -> bool:
        return self.estimated_people > 0

    @property
    def minutes_since_motion(self) -> float:
        if self.last_motion == 0:
            return float('inf')
        return (time.time() - self.last_motion) / 60


class PresenceDetector:
    """
    Detector de presencia multi-zona.

    Combina:
    - BLE scanning por zona (múltiples adaptadores)
    - Sensores de movimiento de Home Assistant
    - Tracking de usuarios registrados

    Uso:
        detector = PresenceDetector(user_manager=user_manager, ha_client=ha_client)
        detector.add_zone("living", "Living Room", ble_adapter="hci0")
        detector.add_zone("dormitorio", "Dormitorio Principal", ble_adapter="hci1")

        await detector.start()

        # Obtener presencia
        print(detector.get_user_presence("user123"))
        print(detector.get_zone_occupancy("living"))
        print(detector.who_is_home())
    """

    def __init__(
        self,
        user_manager=None,
        ha_client=None,
        away_timeout: float = 300,  # 5 minutos sin ver = away
        just_arrived_duration: float = 300,  # 5 minutos como "recién llegado"
    ):
        """
        Args:
            user_manager: UserManager para asociar dispositivos con usuarios
            ha_client: HomeAssistant client para sensores de movimiento
            away_timeout: Segundos sin detectar para marcar como away
            just_arrived_duration: Duración del estado "recién llegado"
        """
        self.user_manager = user_manager
        self.ha_client = ha_client
        self.away_timeout = away_timeout
        self.just_arrived_duration = just_arrived_duration

        # Zonas configuradas
        self._zones: dict[str, dict] = {}  # zone_id -> config

        # Scanners BLE por zona
        self._scanners: dict[str, BLEScanner] = {}

        # Estado de presencia
        self._user_presence: dict[str, UserPresence] = {}  # user_id -> presence
        self._zone_occupancy: dict[str, RoomOccupancy] = {}  # zone_id -> occupancy

        # Dispositivos registrados (BLE address -> user_id)
        self._device_to_user: dict[str, str] = {}

        # Callbacks
        self._on_user_arrived: Optional[Callable] = None
        self._on_user_left: Optional[Callable] = None
        self._on_zone_occupied: Optional[Callable] = None
        self._on_zone_empty: Optional[Callable] = None

        # Estado
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Dispositivos desconocidos ya vistos (para detectar "nuevos")
        self._seen_unknown_devices: set[str] = set()

    def add_zone(
        self,
        zone_id: str,
        zone_name: str,
        ble_adapter: str = None,
        motion_sensor_entity: str = None
    ):
        """
        Agregar zona para tracking.

        Args:
            zone_id: ID único de la zona
            zone_name: Nombre legible
            ble_adapter: Adaptador BLE para esta zona (hci0, hci1, etc.)
            motion_sensor_entity: Entity ID del sensor de movimiento en HA
        """
        self._zones[zone_id] = {
            "name": zone_name,
            "ble_adapter": ble_adapter,
            "motion_sensor": motion_sensor_entity
        }

        self._zone_occupancy[zone_id] = RoomOccupancy(
            zone_id=zone_id,
            zone_name=zone_name
        )

        # Crear scanner BLE si hay adaptador
        if ble_adapter:
            scanner = BLEScanner(
                adapter=ble_adapter,
                zone_id=zone_id
            )
            scanner.on_registered_device(self._on_ble_device_detected)
            scanner.on_device_lost(self._on_ble_device_lost)
            self._scanners[zone_id] = scanner

        logger.info(f"Zona agregada: {zone_name} (BLE: {ble_adapter}, Motion: {motion_sensor_entity})")

    def register_user_device(
        self,
        user_id: str,
        ble_address: str,
        friendly_name: str = None,
        device_type: DeviceType = DeviceType.PHONE_ANDROID
    ):
        """
        Registrar dispositivo BLE de un usuario.

        Args:
            user_id: ID del usuario
            ble_address: MAC address del dispositivo BLE
            friendly_name: Nombre amigable
            device_type: Tipo de dispositivo
        """
        ble_address = ble_address.upper()

        # Registrar en mapeo
        self._device_to_user[ble_address] = user_id

        # Crear/actualizar presencia del usuario
        if user_id not in self._user_presence:
            self._user_presence[user_id] = UserPresence(user_id=user_id)

        if ble_address not in self._user_presence[user_id].ble_devices:
            self._user_presence[user_id].ble_devices.append(ble_address)

        # Registrar en todos los scanners
        for scanner in self._scanners.values():
            scanner.register_device(
                address=ble_address,
                user_id=user_id,
                friendly_name=friendly_name,
                device_type=device_type
            )

        logger.info(f"Dispositivo registrado: {friendly_name or ble_address} -> Usuario {user_id}")

    def unregister_user_device(self, ble_address: str):
        """Eliminar registro de dispositivo"""
        ble_address = ble_address.upper()

        if ble_address in self._device_to_user:
            user_id = self._device_to_user[ble_address]
            del self._device_to_user[ble_address]

            if user_id in self._user_presence:
                devices = self._user_presence[user_id].ble_devices
                if ble_address in devices:
                    devices.remove(ble_address)

        for scanner in self._scanners.values():
            scanner.unregister_device(ble_address)

    async def start(self):
        """Iniciar detección de presencia"""
        self._running = True

        # Iniciar scanners BLE
        for zone_id, scanner in self._scanners.items():
            await scanner.start()
            task = asyncio.create_task(self._run_zone_scanner(zone_id, scanner))
            self._tasks.append(task)

        # Iniciar polling de sensores HA
        if self.ha_client:
            task = asyncio.create_task(self._poll_motion_sensors())
            self._tasks.append(task)

        # Iniciar actualizador de estados
        task = asyncio.create_task(self._update_presence_states())
        self._tasks.append(task)

        logger.info(f"Presence Detector iniciado ({len(self._zones)} zonas)")

    async def stop(self):
        """Detener detección"""
        self._running = False

        # Cancelar tareas
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Detener scanners
        for scanner in self._scanners.values():
            await scanner.stop()

        logger.info("Presence Detector detenido")

    async def _run_zone_scanner(self, zone_id: str, scanner: BLEScanner):
        """Ejecutar scanner de una zona"""
        try:
            async for device in scanner.scan_continuous():
                if not self._running:
                    break

                # Actualizar ocupación de zona
                self._update_zone_occupancy(zone_id)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error en scanner zona {zone_id}: {e}")

    async def _poll_motion_sensors(self):
        """Polling de sensores de movimiento de HA"""
        while self._running:
            try:
                for zone_id, config in self._zones.items():
                    sensor_entity = config.get("motion_sensor")
                    if sensor_entity and self.ha_client:
                        state = await self.ha_client.get_state(sensor_entity)
                        if state and state.get("state") == "on":
                            self._zone_occupancy[zone_id].last_motion = time.time()

                await asyncio.sleep(5)  # Poll cada 5 segundos

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling motion sensors: {e}")
                await asyncio.sleep(10)

    async def _update_presence_states(self):
        """Actualizar estados de presencia periódicamente"""
        while self._running:
            try:
                now = time.time()

                for user_id, presence in self._user_presence.items():
                    old_state = presence.state
                    new_state = self._calculate_user_state(presence, now)

                    if new_state != old_state:
                        presence.state = new_state
                        presence.last_state_change = now
                        self._handle_state_change(user_id, old_state, new_state)

                await asyncio.sleep(10)  # Actualizar cada 10 segundos

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error actualizando estados: {e}")
                await asyncio.sleep(10)

    def _calculate_user_state(self, presence: UserPresence, now: float) -> PresenceState:
        """Calcular estado de presencia de un usuario"""
        time_since_seen = now - presence.last_seen if presence.last_seen > 0 else float('inf')

        # Si nunca lo hemos visto
        if presence.last_seen == 0:
            return PresenceState.UNKNOWN

        # Si lo vimos recientemente
        if time_since_seen < self.away_timeout:
            # ¿Recién llegó?
            if presence.arrival_time and (now - presence.arrival_time) < self.just_arrived_duration:
                return PresenceState.JUST_ARRIVED
            return PresenceState.HOME

        # No lo hemos visto en un rato
        # ¿Recién se fue?
        if presence.departure_time and (now - presence.departure_time) < self.just_arrived_duration:
            return PresenceState.JUST_LEFT

        return PresenceState.AWAY

    def _handle_state_change(self, user_id: str, old_state: PresenceState, new_state: PresenceState):
        """Manejar cambio de estado de usuario"""
        presence = self._user_presence[user_id]
        now = time.time()

        # Usuario llegó a casa
        if new_state in [PresenceState.HOME, PresenceState.JUST_ARRIVED]:
            if old_state in [PresenceState.AWAY, PresenceState.UNKNOWN, PresenceState.JUST_LEFT]:
                presence.arrival_time = now
                presence.departure_time = None
                logger.info(f"Usuario {user_id} llegó a casa")
                if self._on_user_arrived:
                    self._on_user_arrived(user_id, presence)

        # Usuario se fue
        elif new_state in [PresenceState.AWAY, PresenceState.JUST_LEFT]:
            if old_state in [PresenceState.HOME, PresenceState.JUST_ARRIVED]:
                presence.departure_time = now
                presence.arrival_time = None
                logger.info(f"Usuario {user_id} se fue de casa")
                if self._on_user_left:
                    self._on_user_left(user_id, presence)

    def _on_ble_device_detected(self, device: BLEDevice):
        """Callback cuando se detecta dispositivo BLE registrado"""
        if not device.user_id:
            return

        user_id = device.user_id
        now = time.time()

        # Actualizar presencia del usuario
        if user_id not in self._user_presence:
            self._user_presence[user_id] = UserPresence(user_id=user_id)

        presence = self._user_presence[user_id]
        presence.last_seen = now
        presence.current_zone = device.zone_id
        presence.confidence = device.presence_confidence

        # Actualizar zona
        if device.zone_id and presence.current_zone != presence.last_zone:
            presence.last_zone = presence.current_zone
            logger.debug(f"Usuario {user_id} detectado en zona {device.zone_id}")

    def _on_ble_device_lost(self, device: BLEDevice):
        """Callback cuando se pierde dispositivo BLE"""
        if device.user_id:
            logger.debug(f"Dispositivo de {device.user_id} perdido en zona {device.zone_id}")

    def _update_zone_occupancy(self, zone_id: str):
        """Actualizar ocupación de una zona"""
        if zone_id not in self._zone_occupancy:
            return

        scanner = self._scanners.get(zone_id)
        if not scanner:
            return

        occupancy = self._zone_occupancy[zone_id]
        active_devices = scanner.get_active_devices()

        # Contar usuarios conocidos
        known_users = set()
        for device in active_devices:
            if device.user_id and device.is_nearby:
                known_users.add(device.user_id)

        # Contar dispositivos desconocidos (probablemente personas)
        unknown_phones = sum(
            1 for d in active_devices
            if not d.user_id and d.is_nearby
            and d.device_type in [DeviceType.PHONE_IOS, DeviceType.PHONE_ANDROID]
        )

        old_people = occupancy.estimated_people
        occupancy.known_users = list(known_users)
        occupancy.unknown_devices = unknown_phones
        occupancy.estimated_people = len(known_users) + unknown_phones
        occupancy.devices = active_devices
        occupancy.last_presence = time.time()

        # Callbacks
        if old_people == 0 and occupancy.estimated_people > 0:
            if self._on_zone_occupied:
                self._on_zone_occupied(zone_id, occupancy)

        elif old_people > 0 and occupancy.estimated_people == 0:
            if self._on_zone_empty:
                self._on_zone_empty(zone_id, occupancy)

    # =========================================================================
    # Public API
    # =========================================================================

    def get_user_presence(self, user_id: str) -> Optional[UserPresence]:
        """Obtener estado de presencia de un usuario"""
        return self._user_presence.get(user_id)

    def get_zone_occupancy(self, zone_id: str) -> Optional[RoomOccupancy]:
        """Obtener ocupación de una zona"""
        return self._zone_occupancy.get(zone_id)

    def who_is_home(self) -> list[str]:
        """Obtener lista de usuarios en casa"""
        return [
            user_id for user_id, presence in self._user_presence.items()
            if presence.is_home
        ]

    def who_is_in_zone(self, zone_id: str) -> list[str]:
        """Obtener usuarios en una zona específica"""
        occupancy = self._zone_occupancy.get(zone_id)
        if occupancy:
            return occupancy.known_users
        return []

    def get_user_zone(self, user_id: str) -> Optional[str]:
        """Obtener zona actual de un usuario"""
        presence = self._user_presence.get(user_id)
        if presence:
            return presence.current_zone
        return None

    def get_total_people_home(self) -> int:
        """Total de personas detectadas en casa"""
        total = 0
        counted_users = set()

        for occupancy in self._zone_occupancy.values():
            # Sumar usuarios únicos
            for user_id in occupancy.known_users:
                if user_id not in counted_users:
                    counted_users.add(user_id)
                    total += 1

            # Sumar desconocidos
            total += occupancy.unknown_devices

        return total

    def is_anyone_home(self) -> bool:
        """¿Hay alguien en casa?"""
        return self.get_total_people_home() > 0

    def is_user_home(self, user_id: str) -> bool:
        """¿Está un usuario específico en casa?"""
        presence = self._user_presence.get(user_id)
        return presence.is_home if presence else False

    def get_unknown_devices(self) -> list[dict]:
        """
        Obtener dispositivos BLE desconocidos (no registrados).
        Útil para detectar invitados o nuevos dispositivos.

        Returns:
            Lista de dispositivos desconocidos con info relevante
        """
        unknown = []
        seen_macs = set()

        for zone_id, scanner in self._scanners.items():
            active = scanner.get_active_devices()
            for device in active:
                if device.user_id:  # Ya está registrado
                    continue
                if device.mac_address in seen_macs:
                    continue
                if not device.is_nearby:  # Muy lejos
                    continue

                seen_macs.add(device.mac_address)

                # Marcar como "nuevo" si es la primera vez
                is_new = device.mac_address not in self._seen_unknown_devices
                if is_new:
                    self._seen_unknown_devices.add(device.mac_address)

                unknown.append({
                    "mac_address": device.mac_address,
                    "device_type": device.device_type.value if device.device_type else "unknown",
                    "rssi": device.rssi,
                    "zone_id": zone_id,
                    "is_new": is_new,
                    "first_seen": device.first_seen,
                    "friendly_name": device.friendly_name or f"Dispositivo {device.mac_address[-5:]}"
                })

        return unknown

    def get_tracked_users(self) -> list[str]:
        """Obtener lista de user_ids siendo rastreados"""
        return list(self._user_presence.keys())

    def get_summary(self) -> dict:
        """Obtener resumen completo de presencia"""
        return {
            "total_people": self.get_total_people_home(),
            "anyone_home": self.is_anyone_home(),
            "users_home": self.who_is_home(),
            "zones": {
                zone_id: {
                    "name": occ.zone_name,
                    "people": occ.estimated_people,
                    "users": occ.known_users,
                    "unknown": occ.unknown_devices,
                    "occupied": occ.is_occupied
                }
                for zone_id, occ in self._zone_occupancy.items()
            },
            "user_states": {
                user_id: {
                    "state": presence.state.value,
                    "zone": presence.current_zone,
                    "confidence": presence.confidence,
                    "time_home": presence.time_home_str if presence.is_home else None
                }
                for user_id, presence in self._user_presence.items()
            }
        }

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_user_arrived(self, callback: Callable[[str, UserPresence], None]):
        """Callback cuando un usuario llega a casa"""
        self._on_user_arrived = callback

    def on_user_left(self, callback: Callable[[str, UserPresence], None]):
        """Callback cuando un usuario se va"""
        self._on_user_left = callback

    def on_zone_occupied(self, callback: Callable[[str, RoomOccupancy], None]):
        """Callback cuando una zona se ocupa"""
        self._on_zone_occupied = callback

    def on_zone_empty(self, callback: Callable[[str, RoomOccupancy], None]):
        """Callback cuando una zona se vacía"""
        self._on_zone_empty = callback
