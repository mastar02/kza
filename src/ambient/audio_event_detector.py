"""
Audio Event Detector - Detección Ambiental Sin Wake Word
Detecta eventos de audio importantes: timbre, bebé, alarmas, vidrio, etc.

Utiliza modelos de clasificación de audio (YAMNet o AudioSet) para detectar
eventos en tiempo real sin necesidad de wake word.
"""

import asyncio
from contextvars import ContextVar
import inspect
import logging
import threading
import time
import numpy as np
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Callable
from collections import deque
from pathlib import Path

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.fact_ledger import MemorySource
from src.memory.producer_context import PRODUCER_CONTEXT, model_admission
from src.memory.prompt_context import SourcesInvalidated, run_private

logger = logging.getLogger(__name__)
_EVENT_OWNER = ContextVar('audio_event_owner', default=None)


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
    sources: tuple[MemorySource, ...] = ()


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
        use_gpu: bool = False,
        *, privacy=None, close_timeout_s: float = 1,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.use_gpu = use_gpu
        self._privacy = None
        self._work = OwnedBlockingWork(capacity=4)
        self._native_lock = threading.Lock()
        self._analyses = set()
        self._closed = False
        self._closing = None
        self._initialized = False
        self._close_timeout_s = close_timeout_s
        if privacy is not None:
            self.attach_privacy(privacy)

        # Configuraciones de eventos
        self._event_configs = {kind: replace(config) for kind, config in self.DEFAULT_CONFIGS.items()}

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

    def attach_privacy(self, privacy) -> None:
        """Bind a shared authority before initialization, independently of memory."""
        if self._initialized or self._work.status()['operations'] or self._closed or self._privacy not in (None, privacy):
            raise RuntimeError('audio_event_privacy_binding_unavailable')
        self._privacy = privacy
        privacy.add_borrower(self.begin_close)

    def _live(self, sources) -> bool:
        return not self._closed and (self._privacy is None or self._privacy.sources_available(sources))

    def _invoke(self, function, args, private, cancelled):
        result = None
        invalidated = False
        def check():
            if self._closed or cancelled.is_set() or private and not self._live(PRODUCER_CONTEXT.get().sources):
                raise SourcesInvalidated('audio_event_sources_invalidated')
        try:
            with model_admission(self._native_lock, check):
                result = function(*args)
                check()
                return result
        except SourcesInvalidated:
            invalidated = True
        except Exception as error:
            logger.warning('Audio event native operation failed: %s', type(error).__name__)
        finally:
            function = args = result = cancelled = check = None
        if invalidated:
            raise SourcesInvalidated('audio_event_sources_invalidated')
        raise RuntimeError('audio_event_native_failed')

    async def _native(self, function, *args, private=True):
        cancelled = threading.Event()
        try:
            return await self._work.run(self._invoke, function, args, private, cancelled)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            function = args = cancelled = None

    async def initialize(self) -> None:
        """Load the optional classifier outside the loop and retain native exit."""
        if self._closed:
            raise RuntimeError('audio_event_closed')
        if self._initialized:
            return
        try:
            await self._native(self._load_model_native, private=False)
        except SourcesInvalidated:
            raise
        except Exception:
            self._setup_rule_based_detection()
        self._initialized = True

    def _load_model_native(self):
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

        if self._closed:
            raise RuntimeError('audio_event_closed')
        self._running = True
        logger.info("Detector de eventos de audio iniciado")

    def _track(self, task):
        self._analyses.add(task)
        def finished(done):
            self._analyses.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task

    async def analyze_chunk(self, audio_data: np.ndarray, zone_id: str = 'default') -> list[AudioEvent]:
        """Retain actual sourced analysis and callbacks through cancellation."""
        task = None
        try:
            if not self._running or self._closed:
                return []
            if len(self._analyses) >= 4:
                return []
            task = self._track(asyncio.create_task(self._analyze_owned(audio_data, zone_id)))
            audio_data = None
            return await task
        finally:
            audio_data = task = None

    async def _analyze_owned(self, audio, zone):
        token = _EVENT_OWNER.set(asyncio.current_task())
        sources = PRODUCER_CONTEXT.get().sources
        try:
            operation = self._analyze(audio, zone, sources)
            audio = None
            return await run_private(operation, lambda: self._live(sources))
        except SourcesInvalidated:
            return []
        finally:
            _EVENT_OWNER.reset(token)
            audio = None

    async def _analyze(self, audio, zone, sources):
        events = event = None
        try:
            self._stats['total_chunks_analyzed'] += 1
            function = self._classify_model if self._model is not None else self._classify_features
            events = await self._native(function, audio, zone)
            audio = None  # Notification callbacks never need the original waveform.
            if not self._live(sources):
                raise SourcesInvalidated('audio_event_sources_invalidated')
            events = self._filter_events(events)
            for event in events:
                event.sources = sources
                await self._notify_event(event)
            return events
        finally:
            audio = events = event = None

    def begin_close(self) -> asyncio.Task:
        """Retain actual model/callback exit before dropping loaded state."""
        self._closed = True
        self._running = False
        if self._closing is None:
            tasks = tuple(self._analyses)
            exempt = _EVENT_OWNER.get()
            for task in tasks:
                if task is not exempt:
                    task.cancel()
            native = self._work.begin_close()
            async def finish():
                await asyncio.gather(*tasks, native, return_exceptions=True)
                self._audio_buffer.clear()
                self._last_detections.clear()
                self._model = None
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def stop(self) -> None:
        """Bound close observation while retaining real native cleanup."""
        closing = self.begin_close()
        if _EVENT_OWNER.get() in self._analyses:
            return
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(closing)
        except TimeoutError:
            raise WorkCloseTimeout('audio_event_still_running') from None

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose actual finite analyses and native work."""
        return tuple(self._analyses | self._work._tasks)

    def _classify_model(self, audio: np.ndarray, zone_id: str) -> list[AudioEvent]:
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
            logger.warning("Audio event model classification failed: %s", type(e).__name__)

        return events

    def _classify_features(self, audio: np.ndarray, zone_id: str) -> list[AudioEvent]:
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
            logger.warning("Audio event feature classification failed: %s", type(e).__name__)

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
            logger.debug('Audio periodicity check failed: %s', type(e).__name__)
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

    def _sync_callback(self, callback, event):
        try:
            result = callback(event)
            if inspect.iscoroutine(result):
                result.close()
                raise TypeError('audio_event_callback_must_declare_async')
        finally:
            callback = event = None

    async def _notify_event(self, event: AudioEvent):
        try:
            if not self._live(event.sources):
                raise SourcesInvalidated('audio_event_sources_invalidated')
            self._stats['events_detected'] += 1
            counts = self._stats['events_by_type']
            counts[event.event_type.value] = counts.get(event.event_type.value, 0) + 1
            config = self._event_configs.get(event.event_type)
            callbacks = [self._on_event]
            if config and config.priority >= 8:
                callbacks.append(self._on_security_event)
            for callback in callbacks:
                if callback is None:
                    continue
                if not self._live(event.sources):
                    raise SourcesInvalidated('audio_event_sources_invalidated')
                if inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(getattr(type(callback), '__call__', None)):
                    await callback(event)
                else:
                    await self._native(self._sync_callback, callback, event)
        finally:
            event = callback = callbacks = None

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
