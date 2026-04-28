"""
Zone Manager - Multi-Zone Audio Routing
Detecta la zona de origen del comando y enruta la respuesta.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable
from enum import StrEnum
import numpy as np
import time

logger = logging.getLogger(__name__)


class ZoneState(StrEnum):
    """Estado de una zona"""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    MUTED = "muted"


@dataclass
class Zone:
    """Configuración de una zona de audio"""
    id: str                          # "zone_1", "zone_2", etc.
    name: str                        # "Living Room", "Kitchen", etc.
    mic_device_index: int            # Índice del dispositivo de micrófono
    ma1260_zone: int                 # Zona en el MA1260 (1-6)
    
    # Estado
    state: ZoneState = ZoneState.IDLE
    last_activity: float = 0.0
    volume: int = 50                 # 0-100
    
    # Detección
    noise_floor: float = 0.01       # Nivel de ruido base
    detection_threshold: float = 0.05  # Umbral para detectar voz
    
    # Usuarios asociados (opcional)
    default_users: list[str] = field(default_factory=list)

    # Integración con intercom/media — opcionales por zona.
    # Se llenan solo si la zona está configurada para anunciar o reproducir
    # media; si son None, IntercomSystem simplemente no rutea a esa zona.
    media_player_entity: str | None = None
    speaker_entity: str | None = None
    tts_target: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "mic_device_index": self.mic_device_index,
            "ma1260_zone": self.ma1260_zone,
            "state": self.state.value,
            "volume": self.volume,
            "media_player_entity": self.media_player_entity,
            "speaker_entity": self.speaker_entity,
            "tts_target": self.tts_target,
        }


class ZoneManager:
    """
    Gestiona múltiples zonas de audio.
    
    Funcionalidades:
    - Detecta qué zona originó un comando de voz
    - Enruta la respuesta TTS a la zona correcta
    - Maneja prioridades si múltiples zonas detectan voz
    - Integra con MA1260 para control de audio
    """
    
    def __init__(
        self,
        zones: list[Zone] = None,
        ma1260_controller = None,
        detection_window_ms: int = 500,
        priority_mode: str = "loudest"  # "loudest", "first", "last_active"
    ):
        """
        Args:
            zones: Lista de zonas configuradas
            ma1260_controller: Controlador del Dayton MA1260
            detection_window_ms: Ventana para detectar zona de origen
            priority_mode: Cómo resolver conflictos entre zonas
        """
        self.zones: dict[str, Zone] = {}
        self.ma1260 = ma1260_controller
        self.detection_window_ms = detection_window_ms
        self.priority_mode = priority_mode
        
        # Audio levels por zona (para detección)
        self._audio_levels: dict[str, float] = {}
        self._detection_timestamps: dict[str, float] = {}
        
        # Zona activa actual
        self._active_zone: str | None = None
        
        # Callbacks
        self._on_zone_detected: Callable | None = None
        self._on_zone_speaking: Callable | None = None
        
        # Registrar zonas
        if zones:
            for zone in zones:
                self.add_zone(zone)
        
        logger.info(f"ZoneManager inicializado con {len(self.zones)} zonas")
    
    def add_zone(self, zone: Zone):
        """Agregar una zona"""
        self.zones[zone.id] = zone
        self._audio_levels[zone.id] = 0.0
        self._detection_timestamps[zone.id] = 0.0
        logger.info(f"Zona agregada: {zone.name} (mic: {zone.mic_device_index}, ma1260: {zone.ma1260_zone})")
    
    def remove_zone(self, zone_id: str):
        """Eliminar una zona"""
        if zone_id in self.zones:
            del self.zones[zone_id]
            del self._audio_levels[zone_id]
            del self._detection_timestamps[zone_id]
    
    def get_zone(self, zone_id: str) -> Zone | None:
        """Obtener zona por ID (soporta prefijo 'zone_' por compatibilidad legacy)."""
        zone = self.zones.get(zone_id)
        if zone is None and zone_id.startswith("zone_"):
            zone = self.zones.get(zone_id[len("zone_"):])
        if zone is None and not zone_id.startswith("zone_"):
            zone = self.zones.get(f"zone_{zone_id}")
        return zone

    def get_all_zones(self) -> dict[str, "Zone"]:
        """Devuelve todas las zonas como dict {zone_id: Zone}."""
        return dict(self.zones)
    
    def get_zone_by_mic(self, mic_index: int) -> Zone | None:
        """Obtener zona por índice de micrófono"""
        for zone in self.zones.values():
            if zone.mic_device_index == mic_index:
                return zone
        return None
    
    def get_zone_by_ma1260(self, ma1260_zone: int) -> Zone | None:
        """Obtener zona por número de zona MA1260"""
        for zone in self.zones.values():
            if zone.ma1260_zone == ma1260_zone:
                return zone
        return None
    
    # =========================================================================
    # Detección de Zona
    # =========================================================================
    
    def update_audio_level(self, zone_id: str, audio_data: np.ndarray):
        """
        Actualizar nivel de audio para una zona.
        Llamar continuamente con datos del micrófono.
        """
        if zone_id not in self.zones:
            return
        
        zone = self.zones[zone_id]
        
        # Calcular RMS del audio
        rms = np.sqrt(np.mean(audio_data ** 2))
        self._audio_levels[zone_id] = rms
        
        # Detectar si hay voz (sobre el umbral)
        if rms > zone.detection_threshold:
            self._detection_timestamps[zone_id] = time.time()
            zone.last_activity = time.time()
            
            if zone.state == ZoneState.IDLE:
                zone.state = ZoneState.LISTENING
                logger.debug(f"Zona {zone.name}: detectada actividad (RMS: {rms:.4f})")
    
    def detect_source_zone(self) -> Zone | None:
        """
        Detectar qué zona originó el comando de voz.
        Usar después de detectar wake word.
        
        Returns:
            Zona que probablemente originó el comando, o None
        """
        now = time.time()
        window_sec = self.detection_window_ms / 1000.0
        
        # Filtrar zonas con actividad reciente
        active_zones = []
        for zone_id, timestamp in self._detection_timestamps.items():
            if now - timestamp < window_sec:
                zone = self.zones[zone_id]
                level = self._audio_levels[zone_id]
                active_zones.append((zone, level, timestamp))
        
        if not active_zones:
            # Usar última zona activa como fallback
            if self._active_zone:
                return self.zones.get(self._active_zone)
            return None
        
        # Resolver según modo de prioridad
        if self.priority_mode == "loudest":
            # Zona con mayor nivel de audio
            active_zones.sort(key=lambda x: x[1], reverse=True)
        elif self.priority_mode == "first":
            # Primera zona en detectar
            active_zones.sort(key=lambda x: x[2])
        elif self.priority_mode == "last_active":
            # Última zona que tuvo actividad previa
            active_zones.sort(key=lambda x: x[0].last_activity, reverse=True)
        
        winner = active_zones[0][0]
        self._active_zone = winner.id
        
        logger.info(f"Zona detectada: {winner.name} (nivel: {self._audio_levels[winner.id]:.4f})")
        
        if self._on_zone_detected:
            self._on_zone_detected(winner)
        
        return winner
    
    def get_active_zone(self) -> Zone | None:
        """Obtener la zona actualmente activa"""
        if self._active_zone:
            return self.zones.get(self._active_zone)
        return None
    
    def set_active_zone(self, zone_id: str):
        """Establecer zona activa manualmente"""
        if zone_id in self.zones:
            self._active_zone = zone_id
            logger.info(f"Zona activa establecida: {self.zones[zone_id].name}")
    
    # =========================================================================
    # Output de Audio
    # =========================================================================
    
    def play_to_zone(
        self,
        zone_id: str,
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        block: bool = True
    ):
        """
        Reproducir audio en una zona específica.
        
        Args:
            zone_id: ID de la zona
            audio_data: Datos de audio (numpy array)
            sample_rate: Sample rate del audio
            block: Esperar a que termine la reproducción
        """
        zone = self.get_zone(zone_id)
        if not zone:
            logger.error(f"Zona no encontrada: {zone_id}")
            return
        
        zone.state = ZoneState.SPEAKING
        
        if self._on_zone_speaking:
            self._on_zone_speaking(zone)
        
        try:
            # Usar MA1260 para enrutar audio
            if self.ma1260:
                self.ma1260.select_zone(zone.ma1260_zone)
                self.ma1260.set_volume(zone.ma1260_zone, zone.volume)
                self.ma1260.play_audio(audio_data, sample_rate, block=block)
            else:
                # Fallback: reproducir en dispositivo default
                import sounddevice as sd
                sd.play(audio_data, sample_rate)
                if block:
                    sd.wait()
        finally:
            zone.state = ZoneState.IDLE
    
    def play_to_active_zone(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        block: bool = True
    ):
        """Reproducir audio en la zona activa"""
        if self._active_zone:
            self.play_to_zone(self._active_zone, audio_data, sample_rate, block)
        else:
            logger.warning("No hay zona activa para reproducir audio")
    
    def play_to_all_zones(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 22050
    ):
        """Reproducir audio en todas las zonas (anuncio)"""
        if self.ma1260:
            self.ma1260.select_all_zones()
            self.ma1260.play_audio(audio_data, sample_rate)
        else:
            import sounddevice as sd
            sd.play(audio_data, sample_rate)
            sd.wait()

    # =========================================================================
    # Streaming Output (menor latencia)
    # =========================================================================

    def play_stream_to_zone(
        self,
        zone_id: str,
        audio_generator,
        sample_rate: int = 22050,
        buffer_ms: int = 150,
        prebuffer_ms: int = 30
    ):
        """
        Reproducir audio en streaming a una zona específica.
        El audio empieza a sonar mientras se sigue generando.

        Args:
            zone_id: ID de la zona
            audio_generator: Generador que yield chunks de audio (numpy arrays)
            sample_rate: Sample rate del audio
            buffer_ms: Tamaño del buffer circular (mayor = más suave)
            prebuffer_ms: Pre-buffer antes de iniciar (latencia inicial)
        """
        zone = self.get_zone(zone_id)
        if not zone:
            logger.error(f"Zona no encontrada: {zone_id}")
            return

        zone.state = ZoneState.SPEAKING

        if self._on_zone_speaking:
            self._on_zone_speaking(zone)

        try:
            # Configurar MA1260 para esta zona
            if self.ma1260:
                self.ma1260.select_zone(zone.ma1260_zone)
                self.ma1260.set_volume(zone.ma1260_zone, zone.volume)

            # Crear streaming player con buffer configurado
            from src.tts.piper_tts import StreamingAudioPlayer
            player = StreamingAudioPlayer(
                sample_rate=sample_rate,
                device=self.ma1260.audio_output_device if self.ma1260 else None,
                buffer_ms=buffer_ms,
                prebuffer_ms=prebuffer_ms
            )

            # Reproducir en streaming
            player.play_stream(audio_generator)

            stats = player.get_stats()
            if stats["underruns"] > 0:
                logger.warning(f"Streaming zona {zone.name}: {stats['underruns']} underruns (aumentar buffer)")
            else:
                logger.debug(f"Streaming completado en zona: {zone.name} (sin underruns)")

        except Exception as e:
            logger.error(f"Error en streaming a zona {zone_id}: {e}")
        finally:
            zone.state = ZoneState.IDLE

    def play_stream_to_active_zone(
        self,
        audio_generator,
        sample_rate: int = 22050,
        buffer_ms: int = 150,
        prebuffer_ms: int = 30
    ):
        """Reproducir audio en streaming a la zona activa"""
        if self._active_zone:
            self.play_stream_to_zone(
                self._active_zone,
                audio_generator,
                sample_rate,
                buffer_ms=buffer_ms,
                prebuffer_ms=prebuffer_ms
            )
        else:
            logger.warning("No hay zona activa para streaming")

    # =========================================================================
    # Control de Zonas
    # =========================================================================
    
    def set_zone_volume(self, zone_id: str, volume: int):
        """Establecer volumen de una zona (0-100)"""
        zone = self.get_zone(zone_id)
        if zone:
            zone.volume = max(0, min(100, volume))
            if self.ma1260:
                self.ma1260.set_volume(zone.ma1260_zone, zone.volume)

    def mute_zone(self, zone_id: str):
        """Silenciar una zona"""
        zone = self.get_zone(zone_id)
        if zone:
            zone.state = ZoneState.MUTED
            if self.ma1260:
                self.ma1260.mute_zone(zone.ma1260_zone)

    def unmute_zone(self, zone_id: str):
        """Quitar silencio de una zona"""
        zone = self.get_zone(zone_id)
        if zone:
            zone.state = ZoneState.IDLE
            if self.ma1260:
                self.ma1260.unmute_zone(zone.ma1260_zone)
    
    def mute_all(self):
        """Silenciar todas las zonas"""
        for zone_id in self.zones:
            self.mute_zone(zone_id)
    
    def unmute_all(self):
        """Quitar silencio de todas las zonas"""
        for zone_id in self.zones:
            self.unmute_zone(zone_id)
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def on_zone_detected(self, callback: Callable[[Zone], None]):
        """Registrar callback cuando se detecta una zona"""
        self._on_zone_detected = callback
    
    def on_zone_speaking(self, callback: Callable[[Zone], None]):
        """Registrar callback cuando una zona empieza a hablar"""
        self._on_zone_speaking = callback
    
    # =========================================================================
    # Estado
    # =========================================================================
    
    def get_status(self) -> dict:
        """Obtener estado de todas las zonas"""
        return {
            "active_zone": self._active_zone,
            "zones": {
                zone_id: {
                    **zone.to_dict(),
                    "audio_level": self._audio_levels.get(zone_id, 0),
                    "last_detection": self._detection_timestamps.get(zone_id, 0)
                }
                for zone_id, zone in self.zones.items()
            }
        }
