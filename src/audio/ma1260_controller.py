"""
Dayton Audio MA1260 Controller
Control del amplificador multizona 12-channel (6 zonas estéreo).

Protocolo RS-232:
- Baudrate: 9600
- Data bits: 8
- Stop bits: 1
- Parity: None

Comandos (hex):
- Estructura: <STX> <Zone> <Command> <Data> <ETX>
- STX = 0x02, ETX = 0x03
"""

import logging
import time
from typing import Optional, List, Tuple
from enum import IntEnum
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


class MA1260Command(IntEnum):
    """Comandos del MA1260"""
    POWER = 0x01
    VOLUME = 0x02
    MUTE = 0x03
    SOURCE = 0x04
    BASS = 0x05
    TREBLE = 0x06
    BALANCE = 0x07
    QUERY_STATUS = 0x10


class MA1260Source(IntEnum):
    """Fuentes de entrada del MA1260"""
    SOURCE_1 = 0x01  # Típicamente: línea local/stream
    SOURCE_2 = 0x02
    SOURCE_3 = 0x03
    SOURCE_4 = 0x04
    SOURCE_5 = 0x05
    SOURCE_6 = 0x06
    BUS_A = 0x07     # Bus compartido A
    BUS_B = 0x08     # Bus compartido B


@dataclass
class ZoneStatus:
    """Estado de una zona del MA1260"""
    zone: int
    power: bool
    volume: int      # 0-100
    muted: bool
    source: int
    bass: int        # -10 a +10
    treble: int      # -10 a +10
    balance: int     # -10 a +10


class MA1260Controller:
    """
    Controlador para Dayton Audio MA1260.
    
    Soporta:
    - Control RS-232 (serial)
    - Control IP (si tiene módulo de red)
    - Simulación (para desarrollo)
    
    Features:
    - Control individual por zona (1-6)
    - Control de todas las zonas simultáneamente
    - Selección de fuente de audio
    - Control de volumen, bass, treble, balance
    - Estado de mute por zona
    - Audio output routing
    """
    
    # Constantes del protocolo
    STX = 0x02
    ETX = 0x03
    ALL_ZONES = 0x00
    
    def __init__(
        self,
        connection_type: str = "serial",  # "serial", "ip", "simulation"
        serial_port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        ip_address: str = None,
        ip_port: int = 8080,
        audio_output_device: int = None,  # Dispositivo de audio para output
        default_source: MA1260Source = MA1260Source.SOURCE_1
    ):
        """
        Args:
            connection_type: Tipo de conexión ("serial", "ip", "simulation")
            serial_port: Puerto serial (para RS-232)
            baudrate: Velocidad del puerto serial
            ip_address: IP del MA1260 (si tiene módulo de red)
            ip_port: Puerto TCP
            audio_output_device: Índice del dispositivo de audio para output
            default_source: Fuente de entrada por defecto
        """
        self.connection_type = connection_type
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ip_address = ip_address
        self.ip_port = ip_port
        self.audio_output_device = audio_output_device
        self.default_source = default_source
        
        self._serial = None
        self._socket = None
        self._lock = threading.Lock()
        
        # Estado de zonas (cache)
        self._zone_status: dict[int, ZoneStatus] = {}
        
        # Zona actualmente seleccionada para audio
        self._selected_zones: List[int] = []
        
        # Inicializar conexión
        if connection_type != "simulation":
            self._connect()
        else:
            logger.info("MA1260 en modo simulación")
            self._init_simulation()
    
    def _connect(self):
        """Establecer conexión con el MA1260"""
        if self.connection_type == "serial":
            self._connect_serial()
        elif self.connection_type == "ip":
            self._connect_ip()
    
    def _connect_serial(self):
        """Conectar vía RS-232"""
        try:
            import serial
            self._serial = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            logger.info(f"Conectado a MA1260 vía serial: {self.serial_port}")
            
            # Inicializar estado
            self._query_all_zones()
            
        except Exception as e:
            logger.error(f"Error conectando a MA1260 serial: {e}")
            raise
    
    def _connect_ip(self):
        """Conectar vía TCP/IP"""
        try:
            import socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.ip_address, self.ip_port))
            logger.info(f"Conectado a MA1260 vía IP: {self.ip_address}:{self.ip_port}")
            
            self._query_all_zones()
            
        except Exception as e:
            logger.error(f"Error conectando a MA1260 IP: {e}")
            raise
    
    def _init_simulation(self):
        """Inicializar estado para simulación"""
        for zone in range(1, 7):
            self._zone_status[zone] = ZoneStatus(
                zone=zone,
                power=True,
                volume=50,
                muted=False,
                source=self.default_source,
                bass=0,
                treble=0,
                balance=0
            )
    
    def _send_command(self, zone: int, command: MA1260Command, data: int = 0) -> bool:
        """
        Enviar comando al MA1260.
        
        Args:
            zone: Zona (1-6) o 0 para todas
            command: Comando a ejecutar
            data: Datos del comando
        
        Returns:
            True si el comando fue exitoso
        """
        with self._lock:
            if self.connection_type == "simulation":
                return self._simulate_command(zone, command, data)
            
            # Construir paquete
            packet = bytes([self.STX, zone, command, data, self.ETX])
            
            try:
                if self._serial:
                    self._serial.write(packet)
                    self._serial.flush()
                    # Esperar ACK
                    response = self._serial.read(1)
                    return len(response) > 0 and response[0] == 0x06  # ACK
                    
                elif self._socket:
                    self._socket.send(packet)
                    response = self._socket.recv(1)
                    return len(response) > 0 and response[0] == 0x06
                    
            except Exception as e:
                logger.error(f"Error enviando comando a MA1260: {e}")
                return False
        
        return False
    
    def _simulate_command(self, zone: int, command: MA1260Command, data: int) -> bool:
        """Simular ejecución de comando"""
        zones = [zone] if zone > 0 else list(range(1, 7))
        
        for z in zones:
            if z not in self._zone_status:
                continue
            
            status = self._zone_status[z]
            
            if command == MA1260Command.POWER:
                status.power = data == 1
            elif command == MA1260Command.VOLUME:
                status.volume = data
            elif command == MA1260Command.MUTE:
                status.muted = data == 1
            elif command == MA1260Command.SOURCE:
                status.source = data
            elif command == MA1260Command.BASS:
                status.bass = data - 10  # 0-20 -> -10 to +10
            elif command == MA1260Command.TREBLE:
                status.treble = data - 10
            elif command == MA1260Command.BALANCE:
                status.balance = data - 10
        
        logger.debug(f"[SIM] Zona {zone}: {command.name} = {data}")
        return True
    
    def _query_all_zones(self):
        """Consultar estado de todas las zonas"""
        for zone in range(1, 7):
            self._query_zone(zone)
    
    def _query_zone(self, zone: int):
        """Consultar estado de una zona"""
        if self.connection_type == "simulation":
            return
        
        # Enviar query
        with self._lock:
            packet = bytes([self.STX, zone, MA1260Command.QUERY_STATUS, 0, self.ETX])
            
            try:
                if self._serial:
                    self._serial.write(packet)
                    # Leer respuesta (8 bytes típicamente)
                    response = self._serial.read(8)
                    self._parse_status_response(zone, response)
                elif self._socket:
                    self._socket.send(packet)
                    response = self._socket.recv(8)
                    self._parse_status_response(zone, response)
            except Exception as e:
                logger.error(f"Error consultando zona {zone}: {e}")
    
    def _parse_status_response(self, zone: int, response: bytes):
        """Parsear respuesta de estado"""
        if len(response) < 8:
            return
        
        # Formato típico: STX, Zone, Power, Volume, Mute, Source, Bass, Treble, ETX
        self._zone_status[zone] = ZoneStatus(
            zone=zone,
            power=response[2] == 1,
            volume=response[3],
            muted=response[4] == 1,
            source=response[5],
            bass=response[6] - 10,
            treble=response[7] - 10,
            balance=0  # No siempre reportado
        )
    
    # =========================================================================
    # Control de Zonas
    # =========================================================================
    
    def power_on(self, zone: int = ALL_ZONES):
        """Encender zona(s)"""
        return self._send_command(zone, MA1260Command.POWER, 1)
    
    def power_off(self, zone: int = ALL_ZONES):
        """Apagar zona(s)"""
        return self._send_command(zone, MA1260Command.POWER, 0)
    
    def set_volume(self, zone: int, volume: int):
        """
        Establecer volumen de una zona.
        
        Args:
            zone: Zona (1-6) o 0 para todas
            volume: Volumen (0-100)
        """
        volume = max(0, min(100, volume))
        # Convertir 0-100 a 0-38 (rango del MA1260)
        hw_volume = int(volume * 38 / 100)
        return self._send_command(zone, MA1260Command.VOLUME, hw_volume)
    
    def get_volume(self, zone: int) -> int:
        """Obtener volumen de una zona"""
        if zone in self._zone_status:
            return self._zone_status[zone].volume
        return 0
    
    def mute_zone(self, zone: int = ALL_ZONES):
        """Silenciar zona(s)"""
        return self._send_command(zone, MA1260Command.MUTE, 1)
    
    def unmute_zone(self, zone: int = ALL_ZONES):
        """Quitar silencio de zona(s)"""
        return self._send_command(zone, MA1260Command.MUTE, 0)
    
    def toggle_mute(self, zone: int):
        """Alternar silencio de una zona"""
        if zone in self._zone_status:
            if self._zone_status[zone].muted:
                return self.unmute_zone(zone)
            else:
                return self.mute_zone(zone)
        return False
    
    def set_source(self, zone: int, source: MA1260Source):
        """
        Establecer fuente de audio de una zona.
        
        Args:
            zone: Zona (1-6) o 0 para todas
            source: Fuente de entrada
        """
        return self._send_command(zone, MA1260Command.SOURCE, source)
    
    def set_bass(self, zone: int, bass: int):
        """Establecer bass (-10 a +10)"""
        bass = max(-10, min(10, bass))
        return self._send_command(zone, MA1260Command.BASS, bass + 10)
    
    def set_treble(self, zone: int, treble: int):
        """Establecer treble (-10 a +10)"""
        treble = max(-10, min(10, treble))
        return self._send_command(zone, MA1260Command.TREBLE, treble + 10)
    
    def set_balance(self, zone: int, balance: int):
        """Establecer balance (-10 a +10)"""
        balance = max(-10, min(10, balance))
        return self._send_command(zone, MA1260Command.BALANCE, balance + 10)
    
    # =========================================================================
    # Audio Output
    # =========================================================================
    
    def select_zone(self, zone: int):
        """Seleccionar una zona para el siguiente audio output"""
        self._selected_zones = [zone]
        # Asegurar que la zona use la fuente del servidor
        self.set_source(zone, self.default_source)
        logger.debug(f"Zona seleccionada para output: {zone}")
    
    def select_zones(self, zones: List[int]):
        """Seleccionar múltiples zonas para audio output"""
        self._selected_zones = zones
        for zone in zones:
            self.set_source(zone, self.default_source)
        logger.debug(f"Zonas seleccionadas para output: {zones}")
    
    def select_all_zones(self):
        """Seleccionar todas las zonas para audio output"""
        self._selected_zones = list(range(1, 7))
        self.set_source(self.ALL_ZONES, self.default_source)
        logger.debug("Todas las zonas seleccionadas para output")
    
    def play_audio(
        self,
        audio_data,  # numpy array
        sample_rate: int = 22050,
        block: bool = True
    ):
        """
        Reproducir audio en las zonas seleccionadas.
        
        El audio se envía al dispositivo de salida configurado,
        que debe estar conectado a la entrada del MA1260.
        
        Args:
            audio_data: Datos de audio (numpy array)
            sample_rate: Sample rate
            block: Esperar a que termine la reproducción
        """
        if not self._selected_zones:
            logger.warning("No hay zonas seleccionadas para reproducir audio")
            return
        
        try:
            import sounddevice as sd
            import numpy as np
            
            # Asegurar que el audio es float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalizar si es necesario
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Reproducir en el dispositivo de salida configurado
            sd.play(
                audio_data,
                samplerate=sample_rate,
                device=self.audio_output_device
            )
            
            if block:
                sd.wait()
                
            logger.debug(f"Audio reproducido en zonas: {self._selected_zones}")
            
        except Exception as e:
            logger.error(f"Error reproduciendo audio: {e}")
    
    # =========================================================================
    # Estado
    # =========================================================================
    
    def get_zone_status(self, zone: int) -> Optional[ZoneStatus]:
        """Obtener estado de una zona"""
        return self._zone_status.get(zone)
    
    def get_all_status(self) -> dict:
        """Obtener estado de todas las zonas"""
        return {
            "connection": self.connection_type,
            "selected_zones": self._selected_zones,
            "zones": {
                zone: {
                    "power": status.power,
                    "volume": status.volume,
                    "muted": status.muted,
                    "source": status.source,
                    "bass": status.bass,
                    "treble": status.treble
                }
                for zone, status in self._zone_status.items()
            }
        }
    
    def refresh_status(self):
        """Refrescar estado de todas las zonas"""
        self._query_all_zones()
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self):
        """Cerrar conexión"""
        if self._serial:
            self._serial.close()
            self._serial = None
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("Conexión MA1260 cerrada")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
