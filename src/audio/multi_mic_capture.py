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

from src.audio.zone_capture import ZoneCaptureOwner, ZoneCaptureStopped

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
        on_voice_detected: Callable[[AudioChunk], None] | None = None,
        *,
        authority_factory: Callable | None = None,
    ):
        """Configure metadata; live admission requires a fresh authority."""
        self.microphones = {m.zone_id: m for m in microphones}
        self.buffer_duration_sec = buffer_duration_sec
        self.vad_threshold = vad_threshold
        self.on_voice_detected = on_voice_detected
        self._authority_factory = authority_factory
        self._capture_owner: ZoneCaptureOwner | None = None
        self._streams: dict[str, any] = {}
        self._buffers: dict[str, np.ndarray] = {}
        self._buffer_positions: dict[str, int] = {}
        self._current_levels: dict[str, float] = {}
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._lock = threading.Lock()
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
        """Start one fresh, explicitly authorized capture invocation."""
        with self._lock:
            previous = self._capture_owner
            if previous is not None and not previous.retired:
                if self._running and not previous._sealed:
                    return
                raise RuntimeError("zone_capture_retirement_unconfirmed")
            owner = ZoneCaptureOwner(self._authority_factory)
            self._capture_owner = owner
            self._streams = {}
            self._audio_queue = queue.Queue(maxsize=100)
            self._active_zone = None
            self._voice_start_time = 0
            self._init_buffers()
            self._running = True
        try:
            with owner.operation("startup", require_authority=False):
                try:
                    owner.initialize()
                    import sounddevice as sd

                    for zone_id, mic in self.microphones.items():
                        owner.require_current()
                        callback = self._create_callback(zone_id, mic)
                        owner.open_stream(
                            zone_id,
                            lambda: sd.InputStream(
                                device=mic.device_index,
                                samplerate=mic.sample_rate,
                                channels=mic.channels,
                                dtype=np.float32,
                                blocksize=int(mic.sample_rate * mic.chunk_duration_ms / 1000),
                                callback=callback,
                            ),
                            lambda stream: self._streams.__setitem__(zone_id, stream),
                        )
                        owner.publish(
                            logger.info,
                            "Micrófono iniciado: %s (device %s)", zone_id, mic.device_index,
                        )
                finally:
                    callback = None
        except BaseException as error:
            self._running = False
            try:
                owner.finish()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "zone_capture_start_failed", [error, cleanup]
                ) from None
            raise
    
    def _create_callback(self, zone_id: str, mic: MicrophoneConfig):
        """Bind callbacks to this invocation, including their actual return."""
        owner = self._capture_owner
        if owner is None:
            raise RuntimeError("zone_capture_authority_required")

        def callback(indata, frames, time_info, status):
            audio = chunk = None
            try:
                if status:
                    owner.publish(logger.warning, "Mic %s status: %s", zone_id, status)
                owner.require_current()
                audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                rms = np.sqrt(np.mean(audio ** 2))

                def commit():
                    with self._lock:
                        owner.require_current()
                        self._current_levels[zone_id] = rms
                        buffer = self._buffers[zone_id]
                        pos = self._buffer_positions[zone_id]
                        chunk_len = len(audio)
                        if pos + chunk_len <= len(buffer):
                            buffer[pos:pos + chunk_len] = audio
                        else:
                            first_part = len(buffer) - pos
                            buffer[pos:] = audio[:first_part]
                            buffer[:chunk_len - first_part] = audio[first_part:]
                        self._buffer_positions[zone_id] = (pos + chunk_len) % len(buffer)

                owner.publish(commit)
                if rms > self.vad_threshold:
                    chunk = AudioChunk(
                        zone_id=zone_id,
                        device_index=mic.device_index,
                        audio_data=audio.copy(),
                        timestamp=time.time(),
                        rms_level=rms,
                    )
                    try:
                        owner.publish(self._audio_queue.put_nowait, chunk)
                    except queue.Full:
                        pass
                    if self.on_voice_detected:
                        owner.publish(self.on_voice_detected, chunk)
            finally:
                indata = audio = chunk = status = None
                del frames, time_info

        return owner.callback(callback)
    
    def stop(self):
        """Seal and drain known resources; uncertain retirement is not retried."""
        owner = self._capture_owner
        if owner is None:
            return
        try:
            owner.finish()
        finally:
            with self._lock:
                if self._capture_owner is owner:
                    if owner._sealed:
                        self._running = False
                    if owner.retired:
                        self._streams.clear()

    def require_current(self) -> None:
        owner = self._capture_owner
        if owner is None:
            raise RuntimeError("zone_capture_authority_required")
        owner.require_current()

    def publish(self, function: Callable, *args, **kwargs):
        owner = self._capture_owner
        if owner is None:
            raise RuntimeError("zone_capture_authority_required")
        return owner.publish(function, *args, **kwargs)

    def pause(self, seconds: float) -> None:
        owner = self._capture_owner
        if owner is None:
            raise RuntimeError("zone_capture_authority_required")
        owner.pause(seconds)
    
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
    def test_microphone(
        device_index: int,
        duration_sec: float = 2.0,
        *,
        authority_factory: Callable | None = None,
        report: Callable | None = None,
    ) -> dict:
        """Measure one owned live input; optional reporting precedes retire."""
        owner = ZoneCaptureOwner(authority_factory)
        recording = stream = result = None
        try:
            owner.initialize()
            import sounddevice as sd

            stream = owner.open_stream(
                "test_microphone",
                lambda: sd.InputStream(
                    samplerate=16000, channels=1, device=device_index,
                    dtype=np.float32,
                ),
                lambda value: None,
            )
            with owner.operation("read"):
                try:
                    recording, _overflowed = stream.read(int(16000 * duration_sec))
                    owner.require_current()
                    rms = np.sqrt(np.mean(recording ** 2))
                    peak = np.abs(recording).max()
                    result = {
                        "success": True,
                        "device_index": device_index,
                        "rms_level": float(rms),
                        "peak_level": float(peak),
                        "duration_sec": duration_sec,
                        "has_signal": rms > 0.001,
                    }
                finally:
                    recording = stream = None
                    _overflowed = None
            if report is not None:
                owner.publish(report, result, owner.publish)
        except Exception as error:
            result = {
                "success": False,
                "device_index": device_index,
                "error": ("zone_capture_stopped" if isinstance(error, ZoneCaptureStopped)
                          else "zone_capture_failed"),
            }
        except BaseException as error:
            owner.record_failure(error)
        finally:
            recording = stream = report = None
        owner.finish()
        return result
