"""
Audio Manager Module
Gestiona la captura de audio, detección de wake word, VAD y detección de zonas.
"""

import asyncio
import logging
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

from src.wakeword.detector import WakeWordDetector

logger = logging.getLogger(__name__)


class AudioManager:
    """
    Gestiona la captura de audio y detección de wake word.

    Responsabilidades:
    - Cargar modelos de wake word
    - Detectar wake word en streaming
    - Capturar comandos de audio
    - Detectar zona de origen
    - Detección de VAD (Voice Activity Detection)
    """

    def __init__(
        self,
        zone_manager=None,
        wake_word_model: str = "hey_jarvis",
        wake_word_threshold: float = 0.5,
        sample_rate: int = 16000,
        command_duration: float = 2.0,
    ):
        """
        Inicializar AudioManager.

        Args:
            zone_manager: ZoneManager para detectar zona de origen
            wake_word_model: Nombre del modelo de wake word
            wake_word_threshold: Umbral de confianza para wake word
            sample_rate: Sample rate de audio (Hz)
            command_duration: Duración máxima de comando (segundos)
        """
        self.zone_manager = zone_manager
        self.wake_word_model = wake_word_model
        self.wake_word_threshold = wake_word_threshold
        self.sample_rate = sample_rate
        self.command_duration = command_duration

        self._wake_model: Optional[WakeWordDetector] = None

    def load_wake_word(self):
        """
        Cargar modelo de wake word (pre-entrenado o personalizado).

        Soporta múltiples modelos separados por coma.
        """
        logger.info(f"Cargando wake word: {self.wake_word_model}")

        # Soportar múltiples modelos separados por coma
        models = [m.strip() for m in self.wake_word_model.split(",")]

        self._wake_model = WakeWordDetector(
            models=models,
            threshold=self.wake_word_threshold,
            refractory_period=2.0
        )
        self._wake_model.load()

        logger.info(f"Wake word cargado: {self._wake_model.get_active_models()}")

    def detect_wake_word(self, audio_chunk: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Detectar wake word en un chunk de audio.

        Args:
            audio_chunk: Chunk de audio para analizar

        Returns:
            Tuple[model_name, confidence] si detecta, None si no detecta
        """
        if self._wake_model is None:
            return None

        detection = self._wake_model.detect(audio_chunk)
        if detection:
            model_name, confidence = detection
            logger.info(f"🎤 Wake word detectado: {model_name} ({confidence:.2f})")
            return detection

        return None

    def detect_source_zone(self, audio: np.ndarray = None) -> Optional[str]:
        """
        Detectar qué zona originó el comando de voz.

        Args:
            audio: Audio del comando (opcional, para análisis adicional)

        Returns:
            ID de la zona detectada o None
        """
        if not self.zone_manager:
            return None

        zone = self.zone_manager.detect_source_zone()
        if zone:
            logger.info(f"Zona detectada: {zone.name}")
            return zone.id

        return None

    def set_active_zone(self, zone_id: str):
        """Establecer zona activa para respuestas."""
        if self.zone_manager:
            self.zone_manager.set_active_zone(zone_id)
        logger.info(f"Zona activa: {zone_id}")

    def capture_command(
        self,
        audio_buffer: list,
        command_start_time: float,
        chunk_size: int = 1280
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Verificar si se completó la captura de comando.

        Args:
            audio_buffer: Buffer de audio capturado
            command_start_time: Tiempo de inicio del comando
            chunk_size: Tamaño del chunk de audio

        Returns:
            Tuple[is_complete, elapsed_ms, audio_data]
        """
        elapsed = time.time() - command_start_time
        elapsed_ms = elapsed * 1000
        is_complete = elapsed >= self.command_duration

        if is_complete and audio_buffer:
            audio_data = np.array(audio_buffer, dtype=np.float32)
            return True, elapsed_ms, audio_data

        return False, elapsed_ms, None

    def capture_command_with_vad(
        self,
        audio_buffer: list,
        command_start_time: float,
        silence_threshold: float = 0.015,
        silence_duration_ms: int = 300,
        min_speech_ms: int = 300
    ) -> Tuple[bool, float, np.ndarray, bool]:
        """
        Captura de comando con VAD temprano - termina ANTES si detecta silencio.

        Optimización: En lugar de esperar el timeout completo (2s),
        termina cuando detecta que el usuario dejó de hablar.
        Ahorro estimado: ~200-500ms en comandos cortos.

        Args:
            audio_buffer: Buffer de audio capturado
            command_start_time: Tiempo de inicio del comando
            silence_threshold: Umbral RMS para silencio
            silence_duration_ms: ms de silencio para considerar fin
            min_speech_ms: Mínimo de habla antes de considerar silencio

        Returns:
            Tuple[is_complete, elapsed_ms, audio_data, early_exit]
            - early_exit=True si terminó por VAD antes del timeout
        """
        elapsed = time.time() - command_start_time
        elapsed_ms = elapsed * 1000

        # Mínimo de audio antes de considerar early exit
        if elapsed_ms < min_speech_ms:
            return False, elapsed_ms, None, False

        if not audio_buffer:
            return False, elapsed_ms, None, False

        # Analizar últimos chunks para detectar silencio
        samples_per_ms = self.sample_rate // 1000
        silence_samples = int(silence_duration_ms * samples_per_ms)

        # Obtener últimos N samples
        recent_audio = audio_buffer[-silence_samples:] if len(audio_buffer) > silence_samples else audio_buffer

        if recent_audio:
            recent_array = np.array(recent_audio, dtype=np.float32)
            rms = np.sqrt(np.mean(recent_array ** 2))

            # Si hay silencio prolongado, terminar temprano
            if rms < silence_threshold:
                audio_data = np.array(audio_buffer, dtype=np.float32)
                logger.debug(f"VAD early exit: {elapsed_ms:.0f}ms (silencio detectado)")
                return True, elapsed_ms, audio_data, True

        # Timeout normal
        if elapsed >= self.command_duration:
            audio_data = np.array(audio_buffer, dtype=np.float32)
            return True, elapsed_ms, audio_data, False

        return False, elapsed_ms, None, False

    def get_audio_stream_callback(self):
        """
        Obtener un callback para usar con sounddevice InputStream.

        Returns:
            Función callback que procesa audio en streaming
        """
        def audio_callback(indata, frames, time_info, status):
            # Este callback será configurado por el caller
            pass

        return audio_callback

    @staticmethod
    def validate_audio(audio: np.ndarray) -> bool:
        """
        Validar que el audio tiene formato correcto.

        Args:
            audio: Array de audio

        Returns:
            True si el audio es válido
        """
        if audio is None or len(audio) == 0:
            return False

        if audio.dtype != np.float32:
            return False

        # Verificar que no sea todo silencio
        if np.abs(audio).max() < 0.01:
            return False

        return True
