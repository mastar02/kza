"""
Multi-Microphone Capture
Captura audio de múltiples micrófonos para detección de zona.
"""

import logging
import threading
import queue
import time
from typing import Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MicrophoneConfig:
    """Configuración de un micrófono"""
    device_index: int
    zone_id: str
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100  # Duración de cada chunk


@dataclass
class AudioChunk:
    """Un chunk de audio capturado"""
    zone_id: str
    device_index: int
    audio_data: np.ndarray
    timestamp: float
    rms_level: float


class MultiMicCapture:
    """
    Captura audio de múltiples micrófonos simultáneamente.
    
    Cada micrófono está asociado a una zona. El sistema:
    1. Captura audio de todos los micrófonos en paralelo
    2. Calcula niveles de audio para detectar actividad
    3. Identifica qué micrófono/zona tiene mayor actividad
    4. Provee el audio para procesamiento STT
    
    Hardware típico:
    - USB microphones individuales
    - Multi-channel audio interface (ej: Behringer UMC1820)
    - Array de micrófonos con múltiples canales
    """
    
    def __init__(
        self,
        microphones: list[MicrophoneConfig],
        buffer_duration_sec: float = 5.0,
        vad_threshold: float = 0.02,
        on_voice_detected: Callable[[AudioChunk], None] | None = None
    ):
        """
        Args:
            microphones: Lista de configuraciones de micrófonos
            buffer_duration_sec: Duración del buffer circular por micrófono
            vad_threshold: Umbral para detectar voz
            on_voice_detected: Callback cuando se detecta voz
        """
        self.microphones = {m.zone_id: m for m in microphones}
        self.buffer_duration_sec = buffer_duration_sec
        self.vad_threshold = vad_threshold
        self.on_voice_detected = on_voice_detected
        
        # Streams de audio por micrófono
        self._streams: dict[str, any] = {}
        
        # Buffers circulares por zona
        self._buffers: dict[str, np.ndarray] = {}
        self._buffer_positions: dict[str, int] = {}
        
        # Niveles de audio actuales
        self._current_levels: dict[str, float] = {}
        
        # Queue para chunks procesados
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Control
        self._running = False
        self._lock = threading.Lock()
        
        # Zona con voz detectada
        self._active_zone: str | None = None
        self._voice_start_time: float = 0
        
        self._init_buffers()
    
    def _init_buffers(self):
        """Inicializar buffers circulares"""
        for zone_id, mic in self.microphones.items():
            buffer_samples = int(mic.sample_rate * self.buffer_duration_sec)
            self._buffers[zone_id] = np.zeros(buffer_samples, dtype=np.float32)
            self._buffer_positions[zone_id] = 0
            self._current_levels[zone_id] = 0.0
    
    def start(self):
        """Iniciar captura de todos los micrófonos"""
        if self._running:
            return
        
        import sounddevice as sd
        
        self._running = True
        
        for zone_id, mic in self.microphones.items():
            try:
                # Crear callback para este micrófono
                callback = self._create_callback(zone_id, mic)
                
                # Iniciar stream
                stream = sd.InputStream(
                    device=mic.device_index,
                    samplerate=mic.sample_rate,
                    channels=mic.channels,
                    dtype=np.float32,
                    blocksize=int(mic.sample_rate * mic.chunk_duration_ms / 1000),
                    callback=callback
                )
                stream.start()
                self._streams[zone_id] = stream
                
                logger.info(f"Micrófono iniciado: {zone_id} (device {mic.device_index})")
                
            except Exception as e:
                logger.error(f"Error iniciando micrófono {zone_id}: {e}")
    
    def _create_callback(self, zone_id: str, mic: MicrophoneConfig):
        """Crear callback para un stream de audio"""
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic {zone_id} status: {status}")
            
            if not self._running:
                return
            
            # Convertir a mono si es necesario
            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            
            # Calcular RMS
            rms = np.sqrt(np.mean(audio ** 2))
            
            with self._lock:
                # Actualizar nivel
                self._current_levels[zone_id] = rms
                
                # Agregar al buffer circular
                buffer = self._buffers[zone_id]
                pos = self._buffer_positions[zone_id]
                chunk_len = len(audio)
                
                if pos + chunk_len <= len(buffer):
                    buffer[pos:pos + chunk_len] = audio
                else:
                    # Wrap around
                    first_part = len(buffer) - pos
                    buffer[pos:] = audio[:first_part]
                    buffer[:chunk_len - first_part] = audio[first_part:]
                
                self._buffer_positions[zone_id] = (pos + chunk_len) % len(buffer)
            
            # Detectar voz
            if rms > self.vad_threshold:
                chunk = AudioChunk(
                    zone_id=zone_id,
                    device_index=mic.device_index,
                    audio_data=audio.copy(),
                    timestamp=time.time(),
                    rms_level=rms
                )
                
                # Agregar a queue si no está llena
                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass
                
                # Callback
                if self.on_voice_detected:
                    self.on_voice_detected(chunk)
        
        return callback
    
    def stop(self):
        """Detener captura"""
        self._running = False
        
        for zone_id, stream in self._streams.items():
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                logger.error(f"Error deteniendo stream {zone_id}: {e}")
        
        self._streams.clear()
        logger.info("Captura multi-micrófono detenida")
    
    def get_loudest_zone(self) -> tuple[str, float] | None:
        """
        Obtener la zona con mayor nivel de audio actual.
        
        Returns:
            Tupla (zone_id, rms_level) o None
        """
        with self._lock:
            if not self._current_levels:
                return None
            
            loudest = max(self._current_levels.items(), key=lambda x: x[1])
            return loudest if loudest[1] > self.vad_threshold else None
    
    def get_audio_levels(self) -> dict[str, float]:
        """Obtener niveles de audio de todas las zonas"""
        with self._lock:
            return self._current_levels.copy()
    
    def get_recent_audio(
        self,
        zone_id: str,
        duration_sec: float = 3.0
    ) -> np.ndarray | None:
        """
        Obtener audio reciente de una zona.
        
        Args:
            zone_id: ID de la zona
            duration_sec: Duración del audio a obtener
        
        Returns:
            Array de audio o None
        """
        if zone_id not in self.microphones:
            return None
        
        mic = self.microphones[zone_id]
        samples_needed = int(mic.sample_rate * duration_sec)
        
        with self._lock:
            buffer = self._buffers[zone_id]
            pos = self._buffer_positions[zone_id]
            
            if samples_needed >= len(buffer):
                # Retornar todo el buffer en orden correcto
                return np.concatenate([buffer[pos:], buffer[:pos]])
            else:
                # Retornar últimos N samples
                start = (pos - samples_needed) % len(buffer)
                if start < pos:
                    return buffer[start:pos].copy()
                else:
                    return np.concatenate([buffer[start:], buffer[:pos]])
    
    def get_next_chunk(self, timeout: float = 1.0) -> AudioChunk | None:
        """
        Obtener siguiente chunk de audio de la queue.
        
        Args:
            timeout: Tiempo máximo de espera
        
        Returns:
            AudioChunk o None si timeout
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queue(self):
        """Limpiar queue de audio"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
    
    @staticmethod
    def list_devices() -> list[dict]:
        """Listar dispositivos de audio disponibles"""
        import sounddevice as sd
        
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:  # Solo inputs
                devices.append({
                    "index": i,
                    "name": dev['name'],
                    "channels": dev['max_input_channels'],
                    "sample_rate": dev['default_samplerate']
                })
        return devices
    
    @staticmethod
    def test_microphone(device_index: int, duration_sec: float = 2.0) -> dict:
        """
        Probar un micrófono.
        
        Args:
            device_index: Índice del dispositivo
            duration_sec: Duración de la prueba
        
        Returns:
            Diccionario con resultados de la prueba
        """
        import sounddevice as sd
        
        try:
            # Grabar audio
            recording = sd.rec(
                int(16000 * duration_sec),
                samplerate=16000,
                channels=1,
                device=device_index,
                dtype=np.float32
            )
            sd.wait()
            
            # Analizar
            rms = np.sqrt(np.mean(recording ** 2))
            peak = np.abs(recording).max()
            
            return {
                "success": True,
                "device_index": device_index,
                "rms_level": float(rms),
                "peak_level": float(peak),
                "duration_sec": duration_sec,
                "has_signal": rms > 0.001
            }
            
        except Exception as e:
            return {
                "success": False,
                "device_index": device_index,
                "error": str(e)
            }
