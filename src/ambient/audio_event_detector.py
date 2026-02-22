"""
Audio Event Detector - Detección Ambiental Sin Wake Word
Detecta eventos de audio importantes: timbre, bebé, alarmas, vidrio, etc.

Utiliza modelos de clasificación de audio (YAMNet o AudioSet) para detectar
eventos en tiempo real sin necesidad de wake word.
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Any
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioEventType(StrEnum):
    """Tipos de eventos de audio detectables"""
    # Alertas de seguridad (prioridad alta)
    SMOKE_ALARM = "smoke_alarm"
    GLASS_BREAKING = "glass_breaking"
    GUNSHOT = "gunshot"
    SCREAM = "scream"
    DOG_BARKING = "dog_barking"

    # Hogar
    DOORBELL = "doorbell"
    KNOCK = "knock"
    PHONE_RINGING = "phone_ringing"
    BABY_CRYING = "baby_crying"
    APPLIANCE_BEEP = "appliance_beep"  # Microondas, lavadora, etc.

    # Actividad
    COUGHING = "coughing"
    SNORING = "snoring"
    WATER_RUNNING = "water_running"

    # Otros
    MUSIC = "music"
    SPEECH = "speech"
    SILENCE = "silence"
    UNKNOWN = "unknown"


@dataclass
class AudioEvent:
    """Evento de audio detectado"""
    event_type: AudioEventType
    confidence: float
    timestamp: float
    duration_ms: float = 0
    zone_id: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class EventConfig:
    """Configuración por tipo de evento"""
    enabled: bool = True
    min_confidence: float = 0.7
    cooldown_seconds: float = 5.0  # Evitar detecciones repetidas
    priority: int = 5  # 1-10, mayor = más prioritario
    notify: bool = True
    action: str | None = None  # Acción automática a ejecutar


class AudioEventDetector:
    """
    Detector de eventos de audio ambientales.

    Características:
    - Detección en tiempo real usando clasificador de audio
    - Configuración por tipo de evento
    - Cooldown para evitar spam de detecciones
    - Integración con sistema de alertas
    - Priorización de eventos de seguridad
    """

    # Configuración por defecto para cada tipo de evento
    DEFAULT_CONFIGS = {
        AudioEventType.SMOKE_ALARM: EventConfig(
            enabled=True, min_confidence=0.6, cooldown_seconds=10,
            priority=10, notify=True, action="alert_emergency"
        ),
        AudioEventType.GLASS_BREAKING: EventConfig(
            enabled=True, min_confidence=0.7, cooldown_seconds=5,
            priority=9, notify=True, action="alert_security"
        ),
        AudioEventType.BABY_CRYING: EventConfig(
            enabled=True, min_confidence=0.7, cooldown_seconds=30,
            priority=8, notify=True, action=None
        ),
        AudioEventType.DOORBELL: EventConfig(
            enabled=True, min_confidence=0.8, cooldown_seconds=3,
            priority=6, notify=True, action="show_camera"
        ),
        AudioEventType.KNOCK: EventConfig(
            enabled=True, min_confidence=0.75, cooldown_seconds=5,
            priority=5, notify=True, action=None
        ),
        AudioEventType.DOG_BARKING: EventConfig(
            enabled=True, min_confidence=0.7, cooldown_seconds=60,
            priority=4, notify=True, action=None
        ),
        AudioEventType.COUGHING: EventConfig(
            enabled=False, min_confidence=0.8, cooldown_seconds=300,
            priority=2, notify=False, action=None
        ),
        AudioEventType.APPLIANCE_BEEP: EventConfig(
            enabled=True, min_confidence=0.75, cooldown_seconds=5,
            priority=3, notify=True, action=None
        ),
    }

    # Mapeo de clases YAMNet a nuestros tipos
    YAMNET_MAPPING = {
        # Alarmas
        "Smoke detector": AudioEventType.SMOKE_ALARM,
        "Fire alarm": AudioEventType.SMOKE_ALARM,
        "Alarm": AudioEventType.SMOKE_ALARM,

        # Seguridad
        "Glass": AudioEventType.GLASS_BREAKING,
        "Shatter": AudioEventType.GLASS_BREAKING,
        "Gunshot": AudioEventType.GUNSHOT,
        "Scream": AudioEventType.SCREAM,

        # Hogar
        "Doorbell": AudioEventType.DOORBELL,
        "Ding-dong": AudioEventType.DOORBELL,
        "Knock": AudioEventType.KNOCK,
        "Telephone": AudioEventType.PHONE_RINGING,
        "Ringtone": AudioEventType.PHONE_RINGING,

        # Bebé
        "Baby cry": AudioEventType.BABY_CRYING,
        "Infant cry": AudioEventType.BABY_CRYING,
        "Crying": AudioEventType.BABY_CRYING,

        # Mascotas
        "Dog": AudioEventType.DOG_BARKING,
        "Bark": AudioEventType.DOG_BARKING,
        "Bow-wow": AudioEventType.DOG_BARKING,

        # Salud
        "Cough": AudioEventType.COUGHING,
        "Snoring": AudioEventType.SNORING,

        # Otros
        "Beep": AudioEventType.APPLIANCE_BEEP,
        "Water": AudioEventType.WATER_RUNNING,
        "Music": AudioEventType.MUSIC,
        "Speech": AudioEventType.SPEECH,
        "Silence": AudioEventType.SILENCE,
    }

    def __init__(
        self,
        model_path: str = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 1000,
        use_gpu: bool = False
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.use_gpu = use_gpu

        # Configuraciones de eventos
        self._event_configs: dict[AudioEventType, EventConfig] = {
            **self.DEFAULT_CONFIGS
        }

        # Estado
        self._running = False
        self._model = None
        self._class_names: list[str] = []

        # Historial de detecciones (para cooldown)
        self._last_detections: dict[AudioEventType, float] = {}

        # Buffer de audio para análisis
        self._audio_buffer = deque(maxlen=int(sample_rate * 5))  # 5 segundos

        # Callbacks
        self._on_event: Callable[[AudioEvent], None] | None = None
        self._on_security_event: Callable[[AudioEvent], None] | None = None

        # Estadísticas
        self._stats = {
            "total_chunks_analyzed": 0,
            "events_detected": 0,
            "events_by_type": {}
        }

    async def initialize(self):
        """Inicializar el detector y cargar modelo"""
        logger.info("Inicializando detector de eventos de audio...")

        try:
            # Intentar cargar YAMNet o modelo similar
            await self._load_model()
            logger.info(f"Modelo cargado con {len(self._class_names)} clases")

        except Exception as e:
            logger.warning(f"No se pudo cargar modelo ML: {e}")
            logger.info("Usando detección basada en reglas como fallback")
            self._setup_rule_based_detection()

    async def _load_model(self):
        """Cargar modelo de clasificación de audio"""
        # Implementación con TensorFlow Lite / ONNX
        # Por ahora, setup básico

        try:
            # Intentar cargar YAMNet
            import tensorflow_hub as hub
            self._model = hub.load('https://tfhub.dev/google/yamnet/1')

            # Cargar nombres de clases
            class_map_path = Path(__file__).parent / "yamnet_class_map.csv"
            if class_map_path.exists():
                import csv
                with open(class_map_path) as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    self._class_names = [row[2] for row in reader]
            else:
                # Usar nombres básicos
                self._class_names = list(self.YAMNET_MAPPING.keys())

            logger.info("YAMNet cargado correctamente")

        except ImportError:
            logger.info("TensorFlow Hub no disponible, usando modo lightweight")
            self._setup_lightweight_detection()

    def _setup_lightweight_detection(self):
        """Configurar detección lightweight sin ML pesado"""
        # Usar librosa o detección basada en características
        self._model = None
        self._class_names = list(self.YAMNET_MAPPING.keys())
        logger.info("Modo lightweight activado")

    def _setup_rule_based_detection(self):
        """Configurar detección basada en reglas simples"""
        self._model = None
        self._class_names = []
        logger.info("Detección basada en reglas activada")

    async def start(self):
        """Iniciar detección continua"""
        if self._running:
            return

        self._running = True
        logger.info("Detector de eventos de audio iniciado")

    async def stop(self):
        """Detener detección"""
        self._running = False
        logger.info("Detector de eventos de audio detenido")

    async def analyze_chunk(self, audio_data: np.ndarray, zone_id: str = "default") -> list[AudioEvent]:
        """
        Analizar chunk de audio y detectar eventos.

        Args:
            audio_data: Array de audio (float32, normalizado -1 a 1)
            zone_id: ID de la zona donde se capturó

        Returns:
            Lista de eventos detectados
        """
        if not self._running:
            return []

        self._stats["total_chunks_analyzed"] += 1
        events = []

        try:
            if self._model is not None:
                # Clasificación con modelo ML
                events = await self._classify_with_model(audio_data, zone_id)
            else:
                # Detección basada en características
                events = await self._classify_with_features(audio_data, zone_id)

            # Filtrar por configuración y cooldown
            events = self._filter_events(events)

            # Notificar eventos
            for event in events:
                await self._notify_event(event)

        except Exception as e:
            logger.error(f"Error analizando audio: {e}")

        return events

    async def _classify_with_model(self, audio: np.ndarray, zone_id: str) -> list[AudioEvent]:
        """Clasificar con modelo YAMNet"""
        events = []

        try:
            # Asegurar formato correcto
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalizar si es necesario
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            # Ejecutar modelo
            scores, embeddings, spectrogram = self._model(audio)
            scores = scores.numpy()

            # Obtener predicciones top
            mean_scores = scores.mean(axis=0)
            top_indices = np.argsort(mean_scores)[-5:][::-1]

            for idx in top_indices:
                confidence = float(mean_scores[idx])

                if confidence < 0.3:
                    continue

                class_name = self._class_names[idx] if idx < len(self._class_names) else "Unknown"

                # Mapear a nuestro tipo de evento
                event_type = self._map_class_to_event(class_name)

                if event_type != AudioEventType.UNKNOWN:
                    events.append(AudioEvent(
                        event_type=event_type,
                        confidence=confidence,
                        timestamp=time.time(),
                        zone_id=zone_id,
                        metadata={"raw_class": class_name}
                    ))

        except Exception as e:
            logger.error(f"Error en clasificación ML: {e}")

        return events

    async def _classify_with_features(self, audio: np.ndarray, zone_id: str) -> list[AudioEvent]:
        """Clasificación basada en características de audio (fallback)"""
        events = []

        try:
            # Características básicas
            rms = np.sqrt(np.mean(audio ** 2))
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2 / len(audio)

            # Detectar silencio
            if rms < 0.01:
                return events

            # Detectar posibles eventos por características
            # (Implementación simplificada - en producción usar librosa)

            # Alto nivel + muchos zero crossings = posible alarma
            if rms > 0.3 and zero_crossings > 0.1:
                # Verificar periodicidad (alarmas son periódicas)
                if self._is_periodic(audio):
                    events.append(AudioEvent(
                        event_type=AudioEventType.SMOKE_ALARM,
                        confidence=0.6,
                        timestamp=time.time(),
                        zone_id=zone_id,
                        metadata={"detection": "rule_based"}
                    ))

            # Pico súbito = posible vidrio o golpe
            peak = np.max(np.abs(audio))
            if peak > 0.8 and rms < 0.3:  # Pico alto pero energía baja = impulsivo
                events.append(AudioEvent(
                    event_type=AudioEventType.GLASS_BREAKING,
                    confidence=0.5,
                    timestamp=time.time(),
                    zone_id=zone_id,
                    metadata={"detection": "rule_based", "peak": float(peak)}
                ))

        except Exception as e:
            logger.error(f"Error en clasificación por características: {e}")

        return events

    def _is_periodic(self, audio: np.ndarray, threshold: float = 0.5) -> bool:
        """Detectar si el audio tiene patrón periódico"""
        try:
            # Autocorrelación simplificada
            corr = np.correlate(audio, audio, mode='full')
            corr = corr[len(corr)//2:]
            corr = corr / corr[0]

            # Buscar picos en la autocorrelación
            peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
            if len(peaks) > 2:
                # Verificar regularidad de picos
                peak_diffs = np.diff(peaks)
                if np.std(peak_diffs) / np.mean(peak_diffs) < 0.3:  # Variación < 30%
                    return True
        except Exception as e:
            logger.debug(f"Beep pattern detection failed: {e}")
        return False

    def _map_class_to_event(self, class_name: str) -> AudioEventType:
        """Mapear nombre de clase a tipo de evento"""
        class_lower = class_name.lower()

        for key, event_type in self.YAMNET_MAPPING.items():
            if key.lower() in class_lower:
                return event_type

        return AudioEventType.UNKNOWN

    def _filter_events(self, events: list[AudioEvent]) -> list[AudioEvent]:
        """Filtrar eventos por configuración y cooldown"""
        filtered = []
        now = time.time()

        for event in events:
            config = self._event_configs.get(event.event_type)

            if not config or not config.enabled:
                continue

            if event.confidence < config.min_confidence:
                continue

            # Verificar cooldown
            last_time = self._last_detections.get(event.event_type, 0)
            if now - last_time < config.cooldown_seconds:
                continue

            # Pasó todos los filtros
            self._last_detections[event.event_type] = now
            filtered.append(event)

        return filtered

    async def _notify_event(self, event: AudioEvent):
        """Notificar evento detectado"""
        self._stats["events_detected"] += 1
        self._stats["events_by_type"][event.event_type.value] = \
            self._stats["events_by_type"].get(event.event_type.value, 0) + 1

        config = self._event_configs.get(event.event_type)

        logger.info(
            f"Evento detectado: {event.event_type.value} "
            f"(confianza: {event.confidence:.2f}, zona: {event.zone_id})"
        )

        # Callback general
        if self._on_event:
            self._on_event(event)

        # Callback de seguridad para eventos prioritarios
        if config and config.priority >= 8:
            if self._on_security_event:
                self._on_security_event(event)

    # ==================== Configuración ====================

    def configure_event(self, event_type: AudioEventType, **kwargs):
        """Configurar un tipo de evento"""
        if event_type not in self._event_configs:
            self._event_configs[event_type] = EventConfig()

        config = self._event_configs[event_type]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def enable_event(self, event_type: AudioEventType):
        """Habilitar detección de evento"""
        self.configure_event(event_type, enabled=True)

    def disable_event(self, event_type: AudioEventType):
        """Deshabilitar detección de evento"""
        self.configure_event(event_type, enabled=False)

    def set_sensitivity(self, event_type: AudioEventType, sensitivity: float):
        """Ajustar sensibilidad (0-1, mayor = más sensible)"""
        min_confidence = 1.0 - (sensitivity * 0.5)  # 0.5 a 1.0
        self.configure_event(event_type, min_confidence=min_confidence)

    # ==================== Callbacks ====================

    def on_event(self, callback: Callable[[AudioEvent], None]):
        """Registrar callback para todos los eventos"""
        self._on_event = callback

    def on_security_event(self, callback: Callable[[AudioEvent], None]):
        """Registrar callback para eventos de seguridad"""
        self._on_security_event = callback

    # ==================== Estado ====================

    def get_status(self) -> dict:
        """Obtener estado del detector"""
        return {
            "running": self._running,
            "model_loaded": self._model is not None,
            "enabled_events": [
                et.value for et, cfg in self._event_configs.items()
                if cfg.enabled
            ],
            "stats": self._stats
        }

    def get_event_configs(self) -> dict:
        """Obtener configuraciones de eventos"""
        return {
            et.value: {
                "enabled": cfg.enabled,
                "min_confidence": cfg.min_confidence,
                "cooldown_seconds": cfg.cooldown_seconds,
                "priority": cfg.priority,
                "notify": cfg.notify
            }
            for et, cfg in self._event_configs.items()
        }
