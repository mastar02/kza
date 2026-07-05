# KZA - Análisis Completo y Roadmap de Mejoras

> ⚠️ **Documento histórico.** Las propuestas de infraestructura de este doc (p. ej.
> docker-compose para ChromaDB) NO reflejan producción. La fuente de verdad de deploy
> es [`docs/architecture/DEPLOYMENT.md`](../architecture/DEPLOYMENT.md).

## Resumen Ejecutivo

**Estado actual**: Tienes un proyecto muy sólido y bien arquitecturado que ya supera las capacidades de Alexa en varios aspectos. La base está lista para sustituir asistentes comerciales.

**Hardware disponible**:
- CPU: Threadripper PRO 9965WX (24 cores)
- RAM: 128GB DDR5 ECC (expandible a 512GB)
- GPU: 4x RTX 3070 (32GB VRAM total)
- Red: 10GbE dual

**Filosofía del proyecto**: "Toda la inteligencia, la red y la automatización bajo tu control" ✓

---

## 1. Lo Que YA Tienes Funcionando

### 1.1 Pipeline de Voz Completo ✅
| Componente | Modelo | Latencia | Estado |
|------------|--------|----------|--------|
| Wake Word | OpenWakeWord | ~10ms | ✅ Listo |
| STT | Faster-Whisper | ~150ms | ✅ Listo |
| Router | Qwen2.5-7B | ~200ms | ✅ Listo |
| TTS | Piper/XTTS | ~80ms | ✅ Listo |
| **Total Fast Path** | | **<300ms** | ✅ Listo |

### 1.2 Identificación de Personas ✅
```
src/users/speaker_identifier.py
├── ECAPA-TDNN (speechbrain)
├── Embeddings de 192/256 dimensiones
├── Verificación por similitud coseno
└── Enrollment de múltiples muestras
```
**Funcionalidad**: Identifica quién habla con ~75% umbral de confianza.

### 1.3 Sistema de Memoria ✅
```
src/memory/memory_manager.py
├── ShortTermMemory (últimos 10 turnos)
├── LongTermMemory (ChromaDB)
└── PreferencesStore (JSON)
```
**Diferenciador vs Alexa**: Recuerda preferencias, patrones y conversaciones.

### 1.4 Detección de Patrones ✅
```
src/analytics/pattern_analyzer.py
├── Patrones temporales (diarios, entre semana, fin de semana)
├── Patrones de secuencia (A → B)
├── Patrones por usuario
└── Sugerencias automáticas de automatización
```

### 1.5 Personalidad Configurable ✅
```
src/training/personality.py
├── 5 tonos: formal, friendly, casual, technical, butler
├── Frases personalizables
├── Reglas de comportamiento
└── System prompts dinámicos
```

### 1.6 Integraciones ✅
- **Home Assistant**: Control completo de domótica
- **Spotify**: Búsqueda por mood, artista, contexto ("música para cocinar")
- **Multi-zona**: Dayton Audio MA1260 (6 zonas)

---

## 2. Lo Que FALTA para Tu Visión

### 2.1 Detección de Emociones/Tonos de Voz ❌
**Estado**: No implementado
**Importancia**: Alta - Permite respuestas empáticas y contextuales

**Solución propuesta**:
```python
# Nuevo módulo: src/users/emotion_detector.py
# Modelos recomendados:
# - Wav2Vec2-Emotion (GPU 1, compartido con embeddings)
# - SpeechBrain emotion recognition
# - Hugging Face: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim

class EmotionDetector:
    """
    Detecta emociones en audio:
    - happy, sad, angry, fearful, neutral, surprised
    - Arousal (energía) y Valence (positivo/negativo)
    """
```

### 2.2 Sistema de Alertas Proactivo ❌
**Estado**: No implementado
**Importancia**: Alta - Diferenciador clave vs Alexa

**Tipos de alertas a implementar**:
```
1. Alertas de seguridad
   - Detección de vidrio roto (audio)
   - Movimiento inusual a horas inusuales
   - Puerta abierta por mucho tiempo

2. Alertas de patrones
   - "No has apagado las luces hoy a tu hora habitual"
   - "Detecté que llegaste tarde, ¿ajusto las rutinas?"

3. Alertas de bienestar
   - "No he detectado actividad en 8 horas"
   - "Tu tono de voz parece diferente hoy"

4. Alertas de dispositivos
   - Batería baja
   - Dispositivo offline
   - Consumo eléctrico anormal
```

### 2.3 Aprendizaje Continuo de Conductas ⚠️
**Estado**: Parcialmente implementado (PatternAnalyzer existe)
**Falta**:
- Aprendizaje de nuevos comandos por voz
- Fine-tuning del modelo con tus interacciones
- Adaptación de tono por usuario

### 2.4 Wake Word Personalizado ⚠️
**Estado**: Estructura existe pero no entrenado
**Falta**: Entrenar wake word con tu voz

---

## 3. Roadmap de Mejoras (Priorizado)

### Fase 1: Estabilización (1-2 semanas)
> **Objetivo**: Consolidar lo existente antes de expandir

1. **Tests Unitarios** 🔴 CRÍTICO
   ```bash
   # Cobertura actual: ~0%
   # Objetivo: >70% en módulos críticos
   pytest tests/ --cov=src --cov-report=html
   ```

2. **Refactoring de VoicePipeline**
   - Dividir el "God Object" de 1400 líneas
   - Crear: AudioManager, CommandProcessor, ResponseHandler

3. **Documentación de APIs internas**

### Fase 2: Funcionalidades Core (2-4 semanas)
> **Objetivo**: Completar funcionalidades para sustituir Alexa

4. **Implementar Detección de Emociones**
   ```python
   # src/users/emotion_detector.py
   # Integrar con GPU 1 (compartido con speaker_id)
   # Usar embeddings de wav2vec2-emotion
   ```

5. **Sistema de Alertas**
   ```python
   # src/alerts/
   # ├── alert_manager.py       # Gestor central
   # ├── security_alerts.py     # Seguridad
   # ├── pattern_alerts.py      # Patrones anómalos
   # └── device_alerts.py       # Estado de dispositivos
   ```

6. **Entrenamiento de Wake Word Personalizado**
   ```bash
   # Ya tienes scripts/train_wakeword.py
   # Grabar 50+ muestras de "Hey KZA" (o tu wake word)
   python scripts/train_wakeword.py --samples 50 --word "hey_kza"
   ```

### Fase 3: Personalización Avanzada (1-2 meses)
> **Objetivo**: IA que se adapta a ti

7. **Fine-tuning con LoRA**
   ```python
   # Ya tienes config en settings.yaml
   # training.lora.*
   # Recopilar conversaciones → Entrenar adapter
   python scripts/train_assistant.py --epochs 3
   ```

8. **Perfiles de Usuario Avanzados**
   ```python
   # Cada usuario tiene:
   # - Embeddings de voz
   # - Preferencias de tono
   # - Horarios típicos
   # - Dispositivos permitidos (por nivel de permisos)
   ```

9. **Comandos Personalizados por Voz**
   ```
   Usuario: "Cuando diga 'modo cine' quiero que apagues todas
            las luces y bajes las persianas"
   KZA: "Entendido, he creado el comando 'modo cine'"
   ```

### Fase 4: Escalabilidad (2-3 meses)
> **Objetivo**: Arquitectura preparada para crecer

10. **ChromaDB como Servicio Separado**
    ```yaml
    # docker-compose.yml
    services:
      chromadb:
        image: ghcr.io/chroma-core/chroma:latest
        ports:
          - "8000:8000"
    ```

11. **Pool de Workers LLM**
    - Worker 1: Llama 70B Q4 (~35GB RAM)
    - Worker 2: Qwen 32B Q8 (~35GB RAM)
    - Disponible: ~58GB para OS/buffers

12. **API REST/WebSocket**
    ```python
    # src/api/
    # ├── rest_api.py      # FastAPI endpoints
    # └── websocket.py     # Streaming de audio
    ```

### Fase 5: Integración Avanzada (Continuo)
> **Objetivo**: Ecosistema completo

13. **Más Integraciones**
    - Calendarios (Google/Outlook)
    - WhatsApp/Telegram (notificaciones)
    - Cámaras (frigate/deepstack)
    - Sensores ambientales

14. **Dashboard de Monitoreo**
    ```python
    # scripts/latency_dashboard.py ya existe
    # Expandir con Grafana + InfluxDB
    ```

15. **Apps Móviles (Futuro)**
    - Control remoto
    - Enrollment de voz
    - Configuración

---

## 4. Arquitectura Propuesta (Mejorada)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AUDIO INPUT                                     │
│                    (Multi-Zone Microphones)                                  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────────┐
        │   Wake Word       │   │   Audio Classifier    │  ← NUEVO
        │   (OpenWakeWord)  │   │   (Sonidos anómalos)  │
        └─────────┬─────────┘   └───────────┬───────────┘
                  │                         │
                  ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PARALLEL PROCESSING                                  │
│                                                                             │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                │
│   │   STT         │   │   Speaker ID  │   │   Emotion     │  ← NUEVO       │
│   │   (GPU 0)     │   │   (GPU 1)     │   │   Detection   │                │
│   │   Whisper     │   │   ECAPA-TDNN  │   │   (GPU 1)     │                │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘                │
│           │                   │                   │                         │
│           └───────────────────┴───────────────────┘                         │
│                               │                                             │
│                               ▼                                             │
│           ┌─────────────────────────────────────────────┐                   │
│           │           CONTEXT AGGREGATOR                │                   │
│           │   - Texto transcrito                        │                   │
│           │   - ID de usuario + permisos                │                   │
│           │   - Estado emocional                        │  ← NUEVO          │
│           │   - Contexto de memoria                     │                   │
│           │   - Hora, zona, actividad reciente          │                   │
│           └───────────────────────────────────────────┘                   │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DISPATCHER                                          │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │ DOMOTICS    │   │ MUSIC       │   │ ALERTS      │   │ REASONING   │    │
│   │ Fast Path   │   │ Path        │   │ Path        │   │ Slow Path   │    │
│   │ <100ms      │   │ ~500ms      │   │ NUEVO       │   │ 5-30s       │    │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │
│          │                 │                 │                 │            │
│          ▼                 ▼                 ▼                 ▼            │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │ VectorDB    │   │ Spotify     │   │ Alert       │   │ LLM 70B     │    │
│   │ + HA Client │   │ + Mood Map  │   │ Manager     │   │ + Memory    │    │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESPONSE HANDLER                                    │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    ADAPTIVE TTS                                    │    │
│   │   - Tono ajustado a emoción detectada          ← NUEVO            │    │
│   │   - Velocidad según urgencia                                       │    │
│   │   - Voz personalizada por usuario                                  │    │
│   └───────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ZONE MANAGER                                        │
│                   (MA1260 - 6 Zones)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Código de Referencia para Nuevos Módulos

### 5.1 Detector de Emociones (Prioridad Alta)

```python
# src/users/emotion_detector.py
"""
Emotion Detector Module
Detecta emociones en el audio para respuestas empáticas.
"""

import logging
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from transformers import pipeline

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Resultado de detección de emoción"""
    primary_emotion: str      # happy, sad, angry, neutral, fearful, surprised
    confidence: float         # 0-1
    arousal: float           # energía (0=calmado, 1=excitado)
    valence: float           # sentimiento (-1=negativo, 1=positivo)
    all_emotions: dict       # {emotion: probability}


class EmotionDetector:
    """
    Detecta emociones usando modelos de audio.

    Modelos disponibles:
    - audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (recomendado)
    - speechbrain/emotion-recognition-wav2vec2-IEMOCAP
    - ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    """

    EMOTION_VALENCE = {
        "happy": 0.8,
        "excited": 0.6,
        "neutral": 0.0,
        "sad": -0.6,
        "angry": -0.5,
        "fearful": -0.7,
        "surprised": 0.2,
        "disgusted": -0.8
    }

    EMOTION_AROUSAL = {
        "happy": 0.6,
        "excited": 0.9,
        "neutral": 0.3,
        "sad": 0.2,
        "angry": 0.9,
        "fearful": 0.8,
        "surprised": 0.7,
        "disgusted": 0.5
    }

    def __init__(
        self,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device: str = "cuda:1",
        min_audio_length: float = 1.0  # segundos mínimos
    ):
        self.model_name = model_name
        self.device = device
        self.min_audio_length = min_audio_length
        self._pipeline = None

    def load(self):
        """Cargar modelo de emociones"""
        logger.info(f"Cargando emotion detector: {self.model_name}")

        self._pipeline = pipeline(
            "audio-classification",
            model=self.model_name,
            device=0 if "cuda" in self.device else -1
        )

        logger.info("Emotion detector cargado")

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> EmotionResult:
        """
        Detectar emoción en audio.

        Args:
            audio: Audio como numpy array (float32, mono)
            sample_rate: Sample rate del audio

        Returns:
            EmotionResult con emoción detectada
        """
        if self._pipeline is None:
            self.load()

        # Verificar longitud mínima
        duration = len(audio) / sample_rate
        if duration < self.min_audio_length:
            return EmotionResult(
                primary_emotion="neutral",
                confidence=0.5,
                arousal=0.3,
                valence=0.0,
                all_emotions={"neutral": 1.0}
            )

        # Ejecutar modelo
        results = self._pipeline(audio)

        # Procesar resultados
        emotions_dict = {r["label"].lower(): r["score"] for r in results}

        # Encontrar emoción principal
        primary = max(emotions_dict, key=emotions_dict.get)
        confidence = emotions_dict[primary]

        # Calcular arousal y valence
        arousal = self.EMOTION_AROUSAL.get(primary, 0.5)
        valence = self.EMOTION_VALENCE.get(primary, 0.0)

        return EmotionResult(
            primary_emotion=primary,
            confidence=confidence,
            arousal=arousal,
            valence=valence,
            all_emotions=emotions_dict
        )

    def get_response_adjustment(self, emotion: EmotionResult) -> dict:
        """
        Obtener ajustes para la respuesta según emoción.

        Returns:
            {
                "tone": str,           # Tono sugerido para respuesta
                "speed": float,        # Velocidad de TTS (0.8-1.2)
                "empathy_prefix": str  # Frase empática opcional
            }
        """
        adjustments = {
            "happy": {
                "tone": "friendly",
                "speed": 1.1,
                "empathy_prefix": ""
            },
            "sad": {
                "tone": "gentle",
                "speed": 0.9,
                "empathy_prefix": "Entiendo... "
            },
            "angry": {
                "tone": "calm",
                "speed": 0.95,
                "empathy_prefix": "Comprendo tu frustración. "
            },
            "fearful": {
                "tone": "reassuring",
                "speed": 0.9,
                "empathy_prefix": "No te preocupes. "
            },
            "neutral": {
                "tone": "normal",
                "speed": 1.0,
                "empathy_prefix": ""
            }
        }

        return adjustments.get(emotion.primary_emotion, adjustments["neutral"])
```

### 5.2 Sistema de Alertas (Prioridad Alta)

```python
# src/alerts/alert_manager.py
"""
Alert Manager Module
Sistema de alertas proactivas.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    CRITICAL = 0    # Seguridad, emergencias
    HIGH = 1        # Requiere atención inmediata
    MEDIUM = 2      # Informativo importante
    LOW = 3         # Informativo


class AlertType(Enum):
    SECURITY = "security"
    PATTERN = "pattern"
    DEVICE = "device"
    REMINDER = "reminder"
    WELLNESS = "wellness"


@dataclass
class Alert:
    """Una alerta del sistema"""
    id: str
    type: AlertType
    priority: AlertPriority
    title: str
    message: str
    created_at: datetime
    entity_id: Optional[str] = None
    user_id: Optional[str] = None
    acknowledged: bool = False
    data: dict = None


class AlertManager:
    """
    Gestor central de alertas.

    Responsabilidades:
    - Recibir alertas de diferentes fuentes
    - Priorizar y deduplicar
    - Notificar por voz o notificación
    - Registrar historial
    """

    def __init__(
        self,
        tts_callback: Callable,
        home_assistant_client=None,
        cooldown_seconds: int = 300  # 5 min entre alertas similares
    ):
        self.tts_callback = tts_callback
        self.ha_client = home_assistant_client
        self.cooldown_seconds = cooldown_seconds

        self._alerts: list[Alert] = []
        self._recent_alerts: dict[str, datetime] = {}
        self._alert_handlers: dict[AlertType, list[Callable]] = {}

    def register_handler(self, alert_type: AlertType, handler: Callable):
        """Registrar un handler para un tipo de alerta"""
        if alert_type not in self._alert_handlers:
            self._alert_handlers[alert_type] = []
        self._alert_handlers[alert_type].append(handler)

    async def create_alert(
        self,
        type: AlertType,
        priority: AlertPriority,
        title: str,
        message: str,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        speak: bool = True,
        data: dict = None
    ) -> Optional[Alert]:
        """
        Crear y procesar una nueva alerta.

        Args:
            type: Tipo de alerta
            priority: Prioridad
            title: Título corto
            message: Mensaje completo
            entity_id: Entidad relacionada
            user_id: Usuario específico (None = todos)
            speak: Si debe anunciarse por voz
            data: Datos adicionales

        Returns:
            Alert creada o None si fue deduplicada
        """
        # Generar ID para deduplicación
        dedup_key = f"{type.value}:{entity_id}:{title}"

        # Verificar cooldown
        if dedup_key in self._recent_alerts:
            last_time = self._recent_alerts[dedup_key]
            if (datetime.now() - last_time).seconds < self.cooldown_seconds:
                logger.debug(f"Alerta deduplicada: {dedup_key}")
                return None

        # Crear alerta
        alert = Alert(
            id=f"alert_{int(datetime.now().timestamp() * 1000)}",
            type=type,
            priority=priority,
            title=title,
            message=message,
            created_at=datetime.now(),
            entity_id=entity_id,
            user_id=user_id,
            data=data or {}
        )

        # Registrar
        self._alerts.append(alert)
        self._recent_alerts[dedup_key] = datetime.now()

        # Anunciar por voz si corresponde
        if speak and priority.value <= AlertPriority.HIGH.value:
            await self._speak_alert(alert)

        # Ejecutar handlers
        if type in self._alert_handlers:
            for handler in self._alert_handlers[type]:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error en handler de alerta: {e}")

        logger.info(f"Alerta creada: [{priority.name}] {title}")
        return alert

    async def _speak_alert(self, alert: Alert):
        """Anunciar alerta por voz"""
        # Prefijo según prioridad
        prefixes = {
            AlertPriority.CRITICAL: "¡Atención urgente! ",
            AlertPriority.HIGH: "Atención. ",
            AlertPriority.MEDIUM: "",
            AlertPriority.LOW: ""
        }

        text = prefixes[alert.priority] + alert.message

        if self.tts_callback:
            await self.tts_callback(text, priority=alert.priority.value)

    def acknowledge(self, alert_id: str) -> bool:
        """Marcar alerta como reconocida"""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_pending(self, user_id: Optional[str] = None) -> list[Alert]:
        """Obtener alertas pendientes"""
        pending = [a for a in self._alerts if not a.acknowledged]
        if user_id:
            pending = [a for a in pending if a.user_id is None or a.user_id == user_id]
        return sorted(pending, key=lambda a: a.priority.value)

    def get_summary(self) -> str:
        """Obtener resumen de alertas pendientes"""
        pending = self.get_pending()
        if not pending:
            return "No hay alertas pendientes."

        critical = len([a for a in pending if a.priority == AlertPriority.CRITICAL])
        high = len([a for a in pending if a.priority == AlertPriority.HIGH])

        parts = []
        if critical:
            parts.append(f"{critical} críticas")
        if high:
            parts.append(f"{high} importantes")
        if len(pending) - critical - high:
            parts.append(f"{len(pending) - critical - high} informativas")

        return f"Tienes {len(pending)} alertas: {', '.join(parts)}."


# Ejemplos de alertas específicas

async def check_security_alerts(
    alert_manager: AlertManager,
    ha_client
):
    """Verificar condiciones de seguridad"""
    # Ejemplo: Puerta abierta por mucho tiempo
    doors = await ha_client.get_entities("binary_sensor.door*")
    for door in doors:
        if door.state == "on":  # Abierta
            duration = door.last_changed_minutes
            if duration > 10:
                await alert_manager.create_alert(
                    type=AlertType.SECURITY,
                    priority=AlertPriority.HIGH,
                    title="Puerta abierta",
                    message=f"La {door.friendly_name} lleva abierta {duration} minutos.",
                    entity_id=door.entity_id
                )


async def check_pattern_alerts(
    alert_manager: AlertManager,
    pattern_analyzer,
    user_id: str
):
    """Verificar desviaciones de patrones"""
    # Ejemplo: Usuario no ha hecho su rutina habitual
    patterns = pattern_analyzer.get_patterns_for_user(user_id)
    for pattern in patterns:
        if pattern.should_have_triggered() and not pattern.triggered_today():
            await alert_manager.create_alert(
                type=AlertType.PATTERN,
                priority=AlertPriority.LOW,
                title="Rutina no ejecutada",
                message=f"Normalmente a esta hora haces: {pattern.description}. ¿Quieres que lo haga?",
                user_id=user_id
            )
```

---

## 6. Comparativa: KZA vs Alexa

| Característica | Alexa | KZA (Actual) | KZA (Roadmap) |
|----------------|-------|--------------|---------------|
| **Privacidad** | ❌ Cloud | ✅ 100% Local | ✅ 100% Local |
| **Latencia domótica** | ~500ms | ✅ <300ms | <300ms |
| **Identificación de voz** | ⚠️ Básica | ✅ ECAPA-TDNN | ✅ Mejorada |
| **Detección de emociones** | ❌ No | ❌ No | ✅ wav2vec2 |
| **Memoria contextual** | ❌ Limitada | ✅ ChromaDB | ✅ Expandida |
| **Aprendizaje de patrones** | ⚠️ Básico | ✅ Avanzado | ✅ ML continuo |
| **Personalización** | ❌ Limitada | ✅ Total | ✅ Fine-tuning |
| **Alertas proactivas** | ⚠️ Básicas | ❌ No | ✅ Multi-tipo |
| **Multi-zona** | ⚠️ Caro | ✅ MA1260 | ✅ Expandible |
| **Razonamiento** | ❌ No | ✅ LLM 70B | ✅ Mejorado |
| **Costo mensual** | $$ | $0 | $0 |
| **Dependencia internet** | ❌ Sí | ✅ No | ✅ No |

---

## 7. Estimación de Recursos por Fase

### Fase 1 (Estabilización)
- **Tiempo**: 1-2 semanas
- **Recursos**: Solo tiempo de desarrollo
- **Riesgo**: Bajo

### Fase 2 (Funcionalidades Core)
- **Tiempo**: 2-4 semanas
- **GPU adicional**: No (usa GPU 1 existente)
- **RAM adicional**: No
- **Riesgo**: Medio

### Fase 3 (Personalización)
- **Tiempo**: 1-2 meses
- **Storage**: ~50GB para datos de entrenamiento
- **GPU para training**: Usar las 4 RTX 3070 (temporal)
- **Riesgo**: Medio-Alto (fine-tuning puede fallar)

### Fase 4 (Escalabilidad)
- **Tiempo**: 2-3 meses
- **RAM**: Expandir a 256GB si necesitas 2 LLMs
- **Riesgo**: Medio

---

## 8. Próximos Pasos Inmediatos

1. **Esta semana**:
   - [ ] Crear estructura de tests
   - [ ] Configurar pytest y coverage
   - [ ] Escribir tests para módulos críticos

2. **Próxima semana**:
   - [ ] Implementar `EmotionDetector` básico
   - [ ] Integrar con pipeline existente
   - [ ] Probar con diferentes tonos de voz

3. **Siguiente**:
   - [ ] Implementar `AlertManager`
   - [ ] Crear primeras alertas de seguridad
   - [ ] Entrenar wake word personalizado

---

## 9. Conclusión

Tu proyecto KZA ya es **significativamente superior a Alexa** en varios aspectos:
- Privacidad total
- Latencia menor
- Capacidad de razonamiento
- Personalización completa

Lo que falta para tu visión completa:
1. **Detección de emociones** - 2 semanas de trabajo
2. **Sistema de alertas** - 2-3 semanas de trabajo
3. **Fine-tuning personalizado** - 1 mes de trabajo

Con el hardware que tienes, puedes implementar TODO lo que describes sin comprar nada adicional.

**Recomendación**: Empieza por estabilizar con tests, luego agrega detección de emociones (alto impacto, poco esfuerzo), y después el sistema de alertas.

---

*Documento generado: Febrero 2026*
*Proyecto: KZA - Ecosistema de IA Local*
