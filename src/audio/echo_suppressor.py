"""
Echo Suppressor - Supresión de Eco para Asistente de Voz
Evita que la respuesta del TTS sea capturada por el micrófono.

Técnicas implementadas:
1. Ducking temporal: Silencia el mic mientras habla KZA
2. Detección de eco: Compara audio capturado con audio reproducido
3. VAD inteligente: Distingue entre voz humana y TTS
4. Spectral gating: Filtrado espectral del eco conocido
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
from enum import Enum
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerState(Enum):
    """Estado del sistema de audio"""
    IDLE = "idle"                    # Nada reproduciéndose
    SPEAKING = "speaking"            # TTS activo
    COOLDOWN = "cooldown"            # Esperando eco residual
    LISTENING = "listening"          # Escuchando al usuario


@dataclass
class EchoSuppressionConfig:
    """Configuración de supresión de eco"""
    # Ducking temporal
    ducking_enabled: bool = True
    pre_speech_buffer_ms: int = 50       # Buffer antes de hablar
    post_speech_buffer_ms: int = 300     # Buffer después de hablar (eco)

    # Detección de eco
    echo_detection_enabled: bool = True
    echo_correlation_threshold: float = 0.6  # Umbral de correlación
    echo_window_ms: int = 500            # Ventana de búsqueda de eco

    # VAD para distinguir humano vs TTS
    vad_enabled: bool = True
    human_voice_freq_range: tuple = (85, 300)   # Hz - voz humana fundamental
    tts_voice_freq_range: tuple = (100, 400)    # Hz - TTS típico

    # Spectral gating
    spectral_gating_enabled: bool = False  # Más costoso computacionalmente
    spectral_threshold_db: float = -20

    # Umbrales de energía
    silence_threshold: float = 0.01
    speech_threshold: float = 0.03


class EchoSuppressor:
    """
    Sistema de supresión de eco para evitar feedback del TTS.

    El problema:
    - KZA habla por el parlante
    - El micrófono captura esa voz
    - El sistema procesa la propia voz de KZA como comando

    Soluciones implementadas:
    1. Ducking: Ignorar mic mientras KZA habla + cooldown
    2. Correlación: Detectar eco comparando con audio reproducido
    3. Fingerprinting: Guardar "huella" del audio reproducido
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        config: EchoSuppressionConfig = None
    ):
        self.sample_rate = sample_rate
        self.config = config or EchoSuppressionConfig()

        # Estado
        self._state = SpeakerState.IDLE
        self._state_lock = threading.Lock()

        # Timestamps
        self._speech_start_time: Optional[float] = None
        self._speech_end_time: Optional[float] = None
        self._last_tts_end: float = 0

        # Buffer de audio reproducido (para correlación)
        self._playback_buffer: deque = deque(maxlen=int(sample_rate * 2))  # 2 segundos
        self._playback_fingerprints: list = []

        # Estadísticas
        self._stats = {
            "total_suppressed_ms": 0,
            "echo_detections": 0,
            "false_triggers_prevented": 0
        }

        # Callbacks
        self._on_state_change: Optional[Callable] = None

    @property
    def state(self) -> SpeakerState:
        """Estado actual del supresor"""
        with self._state_lock:
            return self._state

    @property
    def is_safe_to_listen(self) -> bool:
        """¿Es seguro procesar audio del micrófono?"""
        with self._state_lock:
            if self._state in [SpeakerState.SPEAKING, SpeakerState.COOLDOWN]:
                return False

            # Verificar cooldown temporal
            if self._last_tts_end > 0:
                elapsed_ms = (time.time() - self._last_tts_end) * 1000
                if elapsed_ms < self.config.post_speech_buffer_ms:
                    return False

            return True

    def notify_tts_start(self, audio_data: np.ndarray = None):
        """
        Notificar que el TTS va a empezar a reproducir.

        Args:
            audio_data: Audio que se va a reproducir (para correlación)
        """
        with self._state_lock:
            self._state = SpeakerState.SPEAKING
            self._speech_start_time = time.time()

        # Guardar audio para correlación
        if audio_data is not None and self.config.echo_detection_enabled:
            self._store_playback_audio(audio_data)

        logger.debug("🔇 Echo suppressor: TTS iniciando, mic silenciado")

        if self._on_state_change:
            self._on_state_change(SpeakerState.SPEAKING)

    def notify_tts_end(self):
        """Notificar que el TTS terminó de reproducir."""
        with self._state_lock:
            self._state = SpeakerState.COOLDOWN
            self._speech_end_time = time.time()
            self._last_tts_end = time.time()

        logger.debug(f"🔇 Echo suppressor: TTS terminó, cooldown {self.config.post_speech_buffer_ms}ms")

        # Programar fin del cooldown
        asyncio.get_event_loop().call_later(
            self.config.post_speech_buffer_ms / 1000,
            self._end_cooldown
        )

        if self._on_state_change:
            self._on_state_change(SpeakerState.COOLDOWN)

    def _end_cooldown(self):
        """Terminar período de cooldown"""
        with self._state_lock:
            if self._state == SpeakerState.COOLDOWN:
                self._state = SpeakerState.IDLE
                logger.debug("🎤 Echo suppressor: Escuchando de nuevo")

                if self._on_state_change:
                    self._on_state_change(SpeakerState.IDLE)

    def _store_playback_audio(self, audio: np.ndarray):
        """Guardar audio reproducido para detección de eco"""
        # Guardar en buffer circular
        self._playback_buffer.extend(audio.flatten())

        # Crear fingerprint del audio
        fingerprint = self._create_audio_fingerprint(audio)
        self._playback_fingerprints.append({
            "fingerprint": fingerprint,
            "timestamp": time.time(),
            "duration_ms": len(audio) / self.sample_rate * 1000
        })

        # Limpiar fingerprints viejos (>5 segundos)
        cutoff = time.time() - 5
        self._playback_fingerprints = [
            fp for fp in self._playback_fingerprints
            if fp["timestamp"] > cutoff
        ]

    def _create_audio_fingerprint(self, audio: np.ndarray) -> dict:
        """Crear fingerprint del audio para comparación rápida"""
        audio = audio.flatten()

        # Características básicas
        return {
            "energy": float(np.mean(audio ** 2)),
            "zcr": float(np.mean(np.abs(np.diff(np.signbit(audio))))),  # Zero crossing rate
            "peak": float(np.max(np.abs(audio))),
            "length": len(audio)
        }

    def should_process_audio(self, audio: np.ndarray) -> tuple[bool, str]:
        """
        Determinar si el audio del micrófono debe procesarse.

        Returns:
            (should_process, reason)
        """
        # 1. Verificar estado de ducking
        if not self.is_safe_to_listen:
            self._stats["total_suppressed_ms"] += len(audio) / self.sample_rate * 1000
            return False, "tts_active"

        # 2. Verificar si es eco del TTS (correlación)
        if self.config.echo_detection_enabled and self._is_echo(audio):
            self._stats["echo_detections"] += 1
            self._stats["false_triggers_prevented"] += 1
            return False, "echo_detected"

        # 3. Verificar energía mínima (silencio)
        energy = np.sqrt(np.mean(audio ** 2))
        if energy < self.config.silence_threshold:
            return False, "silence"

        return True, "ok"

    def _is_echo(self, captured_audio: np.ndarray) -> bool:
        """
        Detectar si el audio capturado es eco del TTS.

        Usa correlación cruzada con el audio reproducido recientemente.
        """
        if len(self._playback_buffer) == 0:
            return False

        captured = captured_audio.flatten()
        playback = np.array(self._playback_buffer)

        # Normalizar
        if np.std(captured) == 0 or np.std(playback) == 0:
            return False

        captured_norm = (captured - np.mean(captured)) / np.std(captured)
        playback_norm = (playback - np.mean(playback)) / np.std(playback)

        # Correlación cruzada
        # Usamos una versión simplificada para velocidad
        min_len = min(len(captured_norm), len(playback_norm))
        if min_len < 100:
            return False

        # Correlación en ventana
        correlation = np.abs(np.corrcoef(
            captured_norm[:min_len],
            playback_norm[-min_len:]
        )[0, 1])

        if np.isnan(correlation):
            return False

        is_echo = correlation > self.config.echo_correlation_threshold

        if is_echo:
            logger.debug(f"🔇 Echo detectado: correlación={correlation:.2f}")

        return is_echo

    def process_audio(self, audio: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Procesar audio del micrófono aplicando supresión de eco.

        Returns:
            (audio_procesado, fue_suprimido)
        """
        should_process, reason = self.should_process_audio(audio)

        if not should_process:
            # Retornar silencio
            return np.zeros_like(audio), True

        return audio, False

    def filter_echo_spectral(self, audio: np.ndarray) -> np.ndarray:
        """
        Filtrado espectral del eco (más costoso pero más preciso).

        Sustrae el espectro del audio reproducido del audio capturado.
        """
        if not self.config.spectral_gating_enabled:
            return audio

        if len(self._playback_buffer) == 0:
            return audio

        try:
            # FFT del audio capturado
            captured_fft = np.fft.rfft(audio.flatten())

            # FFT del audio reproducido (últimos N samples)
            playback = np.array(list(self._playback_buffer)[-len(audio):])
            if len(playback) < len(audio):
                playback = np.pad(playback, (0, len(audio) - len(playback)))
            playback_fft = np.fft.rfft(playback)

            # Calcular máscara de sustracción
            playback_mag = np.abs(playback_fft)
            captured_mag = np.abs(captured_fft)

            # Sustracción espectral con floor
            alpha = 2.0  # Factor de sobresustracción
            beta = 0.01  # Floor para evitar artefactos

            mask = np.maximum(
                captured_mag - alpha * playback_mag,
                beta * captured_mag
            ) / (captured_mag + 1e-10)

            # Aplicar máscara
            filtered_fft = captured_fft * mask

            # Reconstruir
            filtered = np.fft.irfft(filtered_fft)

            return filtered.astype(audio.dtype)

        except Exception as e:
            logger.error(f"Error en filtrado espectral: {e}")
            return audio

    # ==================== VAD para distinguir humano vs TTS ====================

    def is_human_voice(self, audio: np.ndarray) -> bool:
        """
        Intentar distinguir si el audio es voz humana o TTS.

        Características que difieren:
        - Frecuencia fundamental (pitch)
        - Variabilidad en el pitch
        - Características espectrales
        """
        if not self.config.vad_enabled:
            return True

        try:
            # Calcular espectro
            fft = np.abs(np.fft.rfft(audio.flatten()))
            freqs = np.fft.rfftfreq(len(audio.flatten()), 1/self.sample_rate)

            # Buscar pico en rango de voz humana
            human_range = self.config.human_voice_freq_range
            mask = (freqs >= human_range[0]) & (freqs <= human_range[1])

            if not np.any(mask):
                return True

            human_energy = np.sum(fft[mask] ** 2)
            total_energy = np.sum(fft ** 2) + 1e-10

            # Ratio de energía en frecuencias de voz
            voice_ratio = human_energy / total_energy

            # Si hay mucha energía en frecuencias altas, probablemente es TTS
            high_freq_mask = freqs > 4000
            high_freq_energy = np.sum(fft[high_freq_mask] ** 2)
            high_freq_ratio = high_freq_energy / total_energy

            # TTS suele tener espectro más "limpio" y consistente
            # Voz humana tiene más variabilidad

            # Heurística simple: si ratio de voz bajo + alta frecuencia alta = TTS
            is_likely_tts = voice_ratio < 0.3 and high_freq_ratio > 0.2

            return not is_likely_tts

        except Exception as e:
            logger.debug(f"Error en VAD: {e}")
            return True

    # ==================== Callbacks ====================

    def on_state_change(self, callback: Callable[[SpeakerState], None]):
        """Registrar callback para cambios de estado"""
        self._on_state_change = callback

    # ==================== Estado y Stats ====================

    def get_stats(self) -> dict:
        """Obtener estadísticas de supresión"""
        return {
            **self._stats,
            "state": self.state.value,
            "playback_buffer_size": len(self._playback_buffer),
            "fingerprints_stored": len(self._playback_fingerprints)
        }

    def reset(self):
        """Resetear estado del supresor"""
        with self._state_lock:
            self._state = SpeakerState.IDLE
            self._speech_start_time = None
            self._speech_end_time = None
            self._last_tts_end = 0

        self._playback_buffer.clear()
        self._playback_fingerprints.clear()


class EchoSuppressorContext:
    """
    Context manager para supresión de eco durante TTS.

    Uso:
        async with echo_suppressor.speaking(audio_to_play):
            await play_audio(audio_to_play)
    """

    def __init__(self, suppressor: EchoSuppressor, audio: np.ndarray = None):
        self.suppressor = suppressor
        self.audio = audio

    async def __aenter__(self):
        self.suppressor.notify_tts_start(self.audio)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.suppressor.notify_tts_end()
        return False
