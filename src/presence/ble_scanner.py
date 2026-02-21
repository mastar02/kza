"""
BLE Scanner Module
Escanea dispositivos Bluetooth Low Energy para detección de presencia.

Soporta:
- Escaneo pasivo de advertisements BLE
- Detección de dispositivos conocidos (teléfonos, relojes, etc.)
- Estimación de distancia por RSSI
- Tracking de dispositivos por zona (múltiples adaptadores)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Tipo de dispositivo BLE detectado"""
    PHONE_IOS = "phone_ios"
    PHONE_ANDROID = "phone_android"
    WATCH = "watch"
    TABLET = "tablet"
    LAPTOP = "laptop"
    BEACON = "beacon"
    TRACKER = "tracker"  # AirTag, Tile, etc.
    HEADPHONES = "headphones"
    SPEAKER = "speaker"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class BLEDevice:
    """Dispositivo BLE detectado"""
    address: str                    # MAC address (puede ser random en iOS)
    name: Optional[str] = None      # Nombre del dispositivo
    rssi: int = -100                # Señal (dBm), -30=muy cerca, -90=lejos
    device_type: DeviceType = DeviceType.UNKNOWN
    manufacturer_data: dict = field(default_factory=dict)
    service_uuids: list = field(default_factory=list)

    # Tracking temporal
    first_seen: float = 0
    last_seen: float = 0
    seen_count: int = 0

    # Asociación con usuario (si está registrado)
    user_id: Optional[str] = None
    friendly_name: Optional[str] = None

    # Estimación de distancia
    estimated_distance_m: Optional[float] = None
    zone_id: Optional[str] = None

    # FIX: Para iOS MAC randomization
    is_random_mac: bool = False           # ¿MAC es aleatoria?
    fingerprint: Optional[str] = None     # Fingerprint para identificar dispositivo
    previous_macs: list = field(default_factory=list)  # MACs anteriores del mismo dispositivo

    def update(self, rssi: int, name: str = None):
        """Actualizar con nueva detección"""
        self.rssi = rssi
        self.last_seen = time.time()
        self.seen_count += 1
        if name and name != self.name:
            self.name = name
        self._estimate_distance()

    def generate_fingerprint(self) -> str:
        """
        Generar fingerprint para identificar dispositivo iOS a pesar de MAC randomization.

        iOS cambia la MAC cada ~15 minutos, pero mantiene ciertos patrones:
        - Service UUIDs anunciados
        - Manufacturer data structure (no el contenido)
        - Nombre del dispositivo (si está disponible)
        - Características del advertisement
        """
        import hashlib

        components = []

        # 1. Service UUIDs (ordenados para consistencia)
        if self.service_uuids:
            sorted_uuids = sorted(str(u).lower() for u in self.service_uuids)
            components.append(f"uuids:{','.join(sorted_uuids)}")

        # 2. Manufacturer data keys (IDs, no el contenido que cambia)
        if self.manufacturer_data:
            mfr_ids = sorted(self.manufacturer_data.keys())
            components.append(f"mfr:{','.join(str(m) for m in mfr_ids)}")

            # Para Apple (0x004C), el primer byte indica tipo de dispositivo
            if 0x004C in self.manufacturer_data:
                apple_data = self.manufacturer_data[0x004C]
                if len(apple_data) >= 2:
                    # Tipo de dispositivo Apple (iPhone, iPad, Watch, etc.)
                    device_type_byte = apple_data[0] if isinstance(apple_data, (bytes, list)) else 0
                    components.append(f"apple_type:{device_type_byte}")

        # 3. Nombre del dispositivo (muy útil si está disponible)
        if self.name:
            # Normalizar: quitar números que podrían cambiar
            import re
            normalized_name = re.sub(r'\d+', '#', self.name.lower())
            components.append(f"name:{normalized_name}")

        # 4. Tipo de dispositivo detectado
        if self.device_type != DeviceType.UNKNOWN:
            components.append(f"type:{self.device_type.value}")

        if not components:
            return None

        # Generar hash del fingerprint
        fingerprint_str = "|".join(components)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    def matches_fingerprint(self, other_fingerprint: str) -> bool:
        """Verificar si este dispositivo coincide con un fingerprint"""
        if not other_fingerprint:
            return False
        my_fp = self.fingerprint or self.generate_fingerprint()
        return my_fp == other_fingerprint

    def _estimate_distance(self):
        """
        Estimar distancia basada en RSSI.

        Fórmula: distance = 10 ^ ((TxPower - RSSI) / (10 * n))
        - TxPower: potencia a 1 metro (~-59 dBm típico)
        - n: factor de propagación (2.0 en espacio libre, 2.5-4 en interiores)
        """
        TX_POWER = -59  # dBm a 1 metro (valor típico)
        N = 2.5  # Factor de propagación en interiores

        if self.rssi >= 0:
            self.estimated_distance_m = None
            return

        self.estimated_distance_m = 10 ** ((TX_POWER - self.rssi) / (10 * N))

    @property
    def is_nearby(self) -> bool:
        """¿Está cerca? (< 5 metros)"""
        return self.estimated_distance_m is not None and self.estimated_distance_m < 5

    @property
    def is_very_close(self) -> bool:
        """¿Está muy cerca? (< 2 metros)"""
        return self.estimated_distance_m is not None and self.estimated_distance_m < 2

    @property
    def is_stale(self) -> bool:
        """¿No se ha visto recientemente? (> 60 segundos)"""
        return (time.time() - self.last_seen) > 60

    @property
    def presence_confidence(self) -> float:
        """Confianza de presencia (0-1) basada en RSSI y frecuencia de detección"""
        if self.is_stale:
            return 0.0

        # Factor por RSSI
        rssi_factor = max(0, min(1, (self.rssi + 90) / 60))  # -90 a -30 -> 0 a 1

        # Factor por frecuencia de detección
        time_since_first = time.time() - self.first_seen
        if time_since_first > 0:
            detection_rate = self.seen_count / (time_since_first / 10)  # detecciones por 10s
            freq_factor = min(1, detection_rate / 3)  # 3+ por 10s = 100%
        else:
            freq_factor = 0.5

        return (rssi_factor * 0.6 + freq_factor * 0.4)


# Manufacturer IDs conocidos
MANUFACTURER_IDS = {
    0x004C: ("Apple", DeviceType.PHONE_IOS),
    0x0075: ("Samsung", DeviceType.PHONE_ANDROID),
    0x00E0: ("Google", DeviceType.PHONE_ANDROID),
    0x0006: ("Microsoft", DeviceType.LAPTOP),
    0x001D: ("Qualcomm", DeviceType.PHONE_ANDROID),
    0x0310: ("Xiaomi", DeviceType.PHONE_ANDROID),
    0x0157: ("Huawei", DeviceType.PHONE_ANDROID),
    0x038F: ("Garmin", DeviceType.WATCH),
    0x0499: ("Tile", DeviceType.TRACKER),
}

# Service UUIDs conocidos
SERVICE_UUIDS = {
    "0000180f-0000-1000-8000-00805f9b34fb": DeviceType.PHONE_ANDROID,  # Battery
    "0000180a-0000-1000-8000-00805f9b34fb": DeviceType.UNKNOWN,  # Device Info
    "0000fe9f-0000-1000-8000-00805f9b34fb": DeviceType.PHONE_ANDROID,  # Google
    "7905f431-b5ce-4e99-a40f-4b1e122d00d0": DeviceType.WATCH,  # Apple Watch
}


class BLEScanner:
    """
    Scanner BLE para detección de presencia.

    Uso:
        scanner = BLEScanner(adapter="hci0")
        scanner.register_device("AA:BB:CC:DD:EE:FF", user_id="user123", name="iPhone de Juan")

        async with scanner:
            async for device in scanner.scan_continuous():
                logger.debug(f"Detectado: {device.friendly_name or device.address}")

    FIX: Soporta iOS MAC randomization mediante fingerprinting de dispositivos.
    """

    def __init__(
        self,
        adapter: str = "hci0",
        scan_interval: float = 5.0,
        device_timeout: float = 120.0,
        zone_id: str = "default"
    ):
        """
        Args:
            adapter: Adaptador Bluetooth (hci0, hci1, etc.)
            scan_interval: Intervalo entre escaneos (segundos)
            device_timeout: Tiempo sin ver un dispositivo para considerarlo ausente
            zone_id: ID de la zona que cubre este scanner
        """
        self.adapter = adapter
        self.scan_interval = scan_interval
        self.device_timeout = device_timeout
        self.zone_id = zone_id

        # Dispositivos detectados
        self._devices: dict[str, BLEDevice] = {}

        # Dispositivos registrados (conocidos)
        self._registered_devices: dict[str, dict] = {}  # address -> {user_id, name}

        # FIX: Registro por fingerprint para iOS MAC randomization
        self._registered_fingerprints: dict[str, dict] = {}  # fingerprint -> {user_id, name, device_type}
        self._fingerprint_to_current_mac: dict[str, str] = {}  # fingerprint -> MAC actual

        # Callbacks
        self._on_device_detected: Optional[Callable] = None
        self._on_device_lost: Optional[Callable] = None
        self._on_registered_device_detected: Optional[Callable] = None
        self._on_mac_rotation: Optional[Callable] = None  # FIX: Callback para rotación de MAC

        # Estado
        self._running = False
        self._scanner = None
        self._bleak_available = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(self):
        """Iniciar scanner"""
        try:
            from bleak import BleakScanner
            self._bleak_available = True
            logger.info(f"BLE Scanner iniciado en {self.adapter}, zona: {self.zone_id}")
        except ImportError:
            logger.error("bleak no instalado: pip install bleak")
            self._bleak_available = False
            raise

        self._running = True

    async def stop(self):
        """Detener scanner"""
        self._running = False
        if self._scanner:
            await self._scanner.stop()
            self._scanner = None
        logger.info("BLE Scanner detenido")

    def register_device(
        self,
        address: str,
        user_id: str = None,
        friendly_name: str = None,
        device_type: DeviceType = DeviceType.UNKNOWN
    ):
        """
        Registrar dispositivo conocido para tracking.

        Args:
            address: MAC address del dispositivo
            user_id: ID del usuario asociado
            friendly_name: Nombre amigable ("iPhone de Juan")
            device_type: Tipo de dispositivo
        """
        address = address.upper()
        self._registered_devices[address] = {
            "user_id": user_id,
            "friendly_name": friendly_name,
            "device_type": device_type
        }
        logger.info(f"Dispositivo registrado: {address} -> {friendly_name or user_id}")

    def register_device_by_fingerprint(
        self,
        fingerprint: str,
        user_id: str = None,
        friendly_name: str = None,
        device_type: DeviceType = DeviceType.PHONE_IOS
    ):
        """
        FIX: Registrar dispositivo iOS por fingerprint (para MAC randomization).

        Args:
            fingerprint: Fingerprint del dispositivo (generado con BLEDevice.generate_fingerprint())
            user_id: ID del usuario asociado
            friendly_name: Nombre amigable ("iPhone de Juan")
            device_type: Tipo de dispositivo
        """
        self._registered_fingerprints[fingerprint] = {
            "user_id": user_id,
            "friendly_name": friendly_name,
            "device_type": device_type
        }
        logger.info(f"Dispositivo iOS registrado por fingerprint: {fingerprint[:8]}... -> {friendly_name or user_id}")

    def learn_device_fingerprint(self, address: str) -> Optional[str]:
        """
        FIX: Aprender el fingerprint de un dispositivo actualmente visible.

        Útil para registrar dispositivos iOS sin conocer su MAC (que cambia).

        Args:
            address: MAC address actual del dispositivo

        Returns:
            Fingerprint del dispositivo o None si no se encuentra
        """
        address = address.upper()
        if address in self._devices:
            device = self._devices[address]
            fingerprint = device.generate_fingerprint()
            if fingerprint:
                device.fingerprint = fingerprint
                logger.info(f"Fingerprint aprendido para {address}: {fingerprint[:8]}...")
            return fingerprint
        return None

    def get_registered_fingerprints(self) -> dict:
        """Obtener lista de fingerprints registrados"""
        return self._registered_fingerprints.copy()

    def on_mac_rotation(self, callback: Callable[[BLEDevice, str, str], None]):
        """
        FIX: Callback cuando se detecta rotación de MAC en iOS.

        Args:
            callback: Función que recibe (device, old_mac, new_mac)
        """
        self._on_mac_rotation = callback

    def unregister_device(self, address: str):
        """Eliminar dispositivo registrado"""
        address = address.upper()
        if address in self._registered_devices:
            del self._registered_devices[address]
            logger.info(f"Dispositivo eliminado: {address}")

    def get_registered_devices(self) -> dict:
        """Obtener lista de dispositivos registrados"""
        return self._registered_devices.copy()

    def on_device_detected(self, callback: Callable[[BLEDevice], None]):
        """Callback cuando se detecta cualquier dispositivo"""
        self._on_device_detected = callback

    def on_device_lost(self, callback: Callable[[BLEDevice], None]):
        """Callback cuando un dispositivo desaparece"""
        self._on_device_lost = callback

    def on_registered_device(self, callback: Callable[[BLEDevice], None]):
        """Callback cuando se detecta un dispositivo registrado"""
        self._on_registered_device_detected = callback

    async def scan_once(self, timeout: float = 5.0) -> list[BLEDevice]:
        """
        Realizar un escaneo único.

        Returns:
            Lista de dispositivos detectados
        """
        if not self._bleak_available:
            return []

        from bleak import BleakScanner

        detected = []

        try:
            devices = await BleakScanner.discover(
                timeout=timeout,
                adapter=self.adapter
            )

            for d in devices:
                device = self._process_detection(
                    address=d.address,
                    name=d.name,
                    rssi=d.rssi,
                    manufacturer_data=d.metadata.get("manufacturer_data", {}),
                    service_uuids=d.metadata.get("uuids", [])
                )
                detected.append(device)

            # Limpiar dispositivos stale
            self._cleanup_stale_devices()

        except Exception as e:
            logger.error(f"Error en escaneo BLE: {e}")

        return detected

    async def scan_continuous(self):
        """
        Escaneo continuo (generador async).

        Yields:
            BLEDevice cuando se detecta
        """
        if not self._bleak_available:
            logger.error("bleak no disponible")
            return

        from bleak import BleakScanner

        def detection_callback(device, advertisement_data):
            """Callback interno para detecciones"""
            ble_device = self._process_detection(
                address=device.address,
                name=device.name or advertisement_data.local_name,
                rssi=advertisement_data.rssi,
                manufacturer_data=advertisement_data.manufacturer_data,
                service_uuids=[str(u) for u in advertisement_data.service_uuids]
            )

            # Callbacks
            if self._on_device_detected:
                self._on_device_detected(ble_device)

            if ble_device.user_id and self._on_registered_device_detected:
                self._on_registered_device_detected(ble_device)

        self._scanner = BleakScanner(
            detection_callback=detection_callback,
            adapter=self.adapter
        )

        await self._scanner.start()
        logger.info("Escaneo continuo BLE iniciado")

        try:
            while self._running:
                await asyncio.sleep(self.scan_interval)

                # Limpiar dispositivos stale y notificar
                lost_devices = self._cleanup_stale_devices()
                for device in lost_devices:
                    if self._on_device_lost:
                        self._on_device_lost(device)
                    yield device  # Yield con is_stale=True

                # Yield dispositivos activos
                for device in self.get_active_devices():
                    yield device

        finally:
            await self._scanner.stop()

    def _process_detection(
        self,
        address: str,
        name: str,
        rssi: int,
        manufacturer_data: dict,
        service_uuids: list
    ) -> BLEDevice:
        """Procesar una detección BLE con soporte para iOS MAC randomization"""
        address = address.upper()
        now = time.time()

        # Dispositivo existente o nuevo
        if address in self._devices:
            device = self._devices[address]
            device.update(rssi, name)
        else:
            device = BLEDevice(
                address=address,
                name=name,
                rssi=rssi,
                manufacturer_data=manufacturer_data,
                service_uuids=service_uuids,
                first_seen=now,
                last_seen=now,
                seen_count=1,
                zone_id=self.zone_id
            )
            device._estimate_distance()
            self._devices[address] = device

        # Detectar tipo de dispositivo
        device.device_type = self._detect_device_type(manufacturer_data, service_uuids, name)

        # FIX: Detectar si es MAC aleatoria (iOS usa bit específico)
        # En MAC random, el segundo nibble del primer byte tiene bit 1 set (locally administered)
        first_byte = int(address.split(":")[0], 16)
        device.is_random_mac = bool(first_byte & 0x02)

        # Generar/actualizar fingerprint
        device.fingerprint = device.generate_fingerprint()

        # Asociar con usuario registrado (por MAC o por fingerprint)
        matched = False

        # 1. Primero intentar por MAC directa
        if address in self._registered_devices:
            reg = self._registered_devices[address]
            device.user_id = reg.get("user_id")
            device.friendly_name = reg.get("friendly_name")
            if reg.get("device_type"):
                device.device_type = reg["device_type"]
            matched = True

        # 2. FIX: Si no coincide por MAC, intentar por fingerprint (iOS)
        if not matched and device.fingerprint and device.fingerprint in self._registered_fingerprints:
            reg = self._registered_fingerprints[device.fingerprint]
            device.user_id = reg.get("user_id")
            device.friendly_name = reg.get("friendly_name")
            if reg.get("device_type"):
                device.device_type = reg["device_type"]
            matched = True

            # Detectar rotación de MAC
            old_mac = self._fingerprint_to_current_mac.get(device.fingerprint)
            if old_mac and old_mac != address:
                # MAC rotó
                device.previous_macs.append(old_mac)
                if len(device.previous_macs) > 10:
                    device.previous_macs = device.previous_macs[-10:]

                logger.info(f"MAC rotation detectada para {device.friendly_name}: {old_mac} -> {address}")

                if self._on_mac_rotation:
                    self._on_mac_rotation(device, old_mac, address)

            # Actualizar MAC actual para este fingerprint
            self._fingerprint_to_current_mac[device.fingerprint] = address

        return device

    def _detect_device_type(
        self,
        manufacturer_data: dict,
        service_uuids: list,
        name: str
    ) -> DeviceType:
        """Detectar tipo de dispositivo basado en metadata"""
        # Por manufacturer ID
        for mfr_id in manufacturer_data.keys():
            if mfr_id in MANUFACTURER_IDS:
                return MANUFACTURER_IDS[mfr_id][1]

        # Por service UUID
        for uuid in service_uuids:
            uuid_lower = str(uuid).lower()
            if uuid_lower in SERVICE_UUIDS:
                return SERVICE_UUIDS[uuid_lower]

        # Por nombre
        if name:
            name_lower = name.lower()
            if any(x in name_lower for x in ["iphone", "ipad", "macbook", "apple"]):
                return DeviceType.PHONE_IOS
            if any(x in name_lower for x in ["galaxy", "pixel", "oneplus", "xiaomi"]):
                return DeviceType.PHONE_ANDROID
            if any(x in name_lower for x in ["watch", "band", "fit"]):
                return DeviceType.WATCH
            if any(x in name_lower for x in ["airpods", "buds", "headphone"]):
                return DeviceType.HEADPHONES

        return DeviceType.UNKNOWN

    def _cleanup_stale_devices(self) -> list[BLEDevice]:
        """Limpiar dispositivos que no se han visto recientemente"""
        now = time.time()
        lost = []

        for address, device in list(self._devices.items()):
            if (now - device.last_seen) > self.device_timeout:
                lost.append(device)
                del self._devices[address]
                logger.debug(f"Dispositivo perdido: {device.friendly_name or address}")

        return lost

    def get_active_devices(self) -> list[BLEDevice]:
        """Obtener dispositivos activos (no stale)"""
        return [d for d in self._devices.values() if not d.is_stale]

    def get_registered_active_devices(self) -> list[BLEDevice]:
        """Obtener solo dispositivos registrados que están activos"""
        return [d for d in self._devices.values() if d.user_id and not d.is_stale]

    def get_device_by_user(self, user_id: str) -> Optional[BLEDevice]:
        """Obtener dispositivo de un usuario específico"""
        for device in self._devices.values():
            if device.user_id == user_id and not device.is_stale:
                return device
        return None

    def get_nearby_count(self) -> int:
        """Contar dispositivos cercanos"""
        return sum(1 for d in self._devices.values() if d.is_nearby and not d.is_stale)

    def get_occupancy_estimate(self) -> int:
        """
        Estimar número de personas en la zona.

        Heurística:
        - Cada teléfono = 1 persona (probablemente)
        - Relojes pueden ser del mismo dueño del teléfono
        - Otros dispositivos no cuentan como personas
        """
        phones = set()

        for device in self.get_active_devices():
            if device.device_type in [DeviceType.PHONE_IOS, DeviceType.PHONE_ANDROID]:
                if device.is_nearby:
                    # Usar user_id si está registrado, sino address
                    identifier = device.user_id or device.address
                    phones.add(identifier)

        return len(phones)
