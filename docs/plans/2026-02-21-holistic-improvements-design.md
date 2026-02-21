# KZA Holistic Improvement Plan — Foundation First

**Date:** 2026-02-21
**Approach:** Foundation First (A) — Architecture → Robustness → Tests → Production
**Timeline:** 4 trimestres (~48 semanas)
**Current state:** Desarrollo activo, no en producción

---

## Hallazgos del Análisis

### Fortalezas
- Spotify module (4,568 líneas, 169 tests) — rated 5/5, patrón ejemplar
- Multi-user orchestration (priority queue, cancellation, context) — bien diseñado
- BufferedLLMStreamer — streaming inteligente con 4 estrategias
- Latencia fast path: ~150-280ms (cumple <300ms)
- 617+ tests existentes con buena estructura de mocks

### Problemas Críticos
1. **VoicePipeline god object**: 1,870 líneas, 39 params en `__init__`, 10 responsabilidades
2. **Blocking I/O en async**: `ha_client.py` usa `requests` (sync) desde contexto async
3. **7 bare `except:` clauses**: Swallow KeyboardInterrupt, SystemExit, MemoryError
4. **15+ módulos con 0% test coverage**: pipeline core, presence, TTS, timers, analytics
5. **DI violation**: VoicePipeline crea MultiUserOrchestrator internamente
6. **Dead code**: 3 example files en src/, TODOs críticos sin implementar, método muerto
7. **Type safety**: Constructor params sin tipos, `any` vs `Any`, clase anónima
8. **Fire-and-forget tasks**: `asyncio.create_task()` sin error handlers
9. **Cache sin TTL**: _query_cache puede servir datos stale indefinidamente
10. **requirements.txt incompleto**: faltan bleak, torch, speechbrain, fastapi, uvicorn

---

## Q1: Refactoring Arquitectural (Semanas 1-12)

### 1.1 Descomponer VoicePipeline

**De:**
```
VoicePipeline (1,870 líneas, 39 params)
├── Wake word + VAD + echo suppression
├── STT + SpeakerID + Emotion
├── Command routing (fast/slow/music)
├── TTS + streaming
├── Timers, intercom, notifications, alerts, briefing
├── Presence tracking
├── Enrollment flow
├── Feedback detection
└── Suggestion management
```

**A:**
```
AudioLoop (~300 líneas)
├── Wake word detection
├── VAD + echo suppression
└── Audio chunk management

CommandProcessor (~313 líneas) — ya existe, se mantiene
├── asyncio.gather(STT, SpeakerID, Emotion)
└── Parallel GPU execution

RequestRouter (~400 líneas) — extraído del pipeline
├── Request classification
├── FastPathHandler
├── MusicPathHandler
└── SlowPathHandler

ResponseHandler (~300 líneas) — ya existe, se mantiene
├── TTS routing
├── Streaming
└── Zone-aware playback

FeatureManager (~250 líneas)
├── Registry pattern para features
├── Timers, intercom, notifications
├── Proactive alerts, morning briefing
└── Cada feature se registra con callbacks

VoicePipeline (~200 líneas) — orquestador delgado
├── Conecta AudioLoop → CommandProcessor → RequestRouter → ResponseHandler
└── __init__ recibe 5-6 componentes (no 39)
```

### 1.2 Arreglar DI en main.py

```python
# Después del refactor, main.py construye todo explícitamente:
context_manager = ContextManager(config)
queue = PriorityRequestQueue(config)
dispatcher = RequestDispatcher(ha_client, chroma_sync, llm, router, music_dispatcher)
orchestrator = MultiUserOrchestrator(context_manager, queue, dispatcher)

audio_loop = AudioLoop(wake_word_detector, echo_suppressor, vad_config)
request_router = RequestRouter(orchestrator)
feature_manager = FeatureManager(timers, intercom, notifications, alerts)

pipeline = VoicePipeline(audio_loop, command_processor, request_router, response_handler, feature_manager)
```

### 1.3 Convertir ha_client a fully async

- Reemplazar `requests` por `aiohttp` en todas las llamadas
- `aiohttp.ClientSession` compartida (creada una vez en startup)
- Timeout configurable vía `aiohttp.ClientTimeout`
- Mantener fallback REST como async (no sync como ahora)
- Actualizar los 11 tests existentes con `AsyncMock`

### 1.4 Limpiar imports en main.py

- Eliminar `sys.path.insert(0, ...)` hack
- Migrar a `from src.module import Class` canónico
- Verificar que pytest y docker también usen el import style correcto

---

## Q2: Robustez y Calidad de Código (Semanas 13-24)

### 2.1 Eliminar bare excepts (7 ubicaciones)

| Archivo | Línea | Cambio |
|---------|-------|--------|
| `ha_client.py` | 183 | `except Exception as e: logger.error(f"Failed to reload automations: {e}")` |
| `ha_client.py` | 347 | `except Exception as e: logger.error(f"Connection test failed: {e}"); return False` |
| `context_persistence.py` | 271, 383 | `except Exception as e: logger.error(...)` |
| `nightly_trainer.py` | 717, 1074 | `except Exception as e: logger.error(...)` |
| `audio_event_detector.py` | 411 | `except Exception as e: logger.error(...)` |

### 2.2 Type hints con Protocol classes

Nuevo archivo `src/protocols.py`:

```python
from typing import Protocol, AsyncIterator, Optional
import numpy as np

class STTProvider(Protocol):
    async def transcribe(self, audio: np.ndarray) -> str: ...

class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> np.ndarray: ...
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...

class HAProvider(Protocol):
    async def call_service(self, domain: str, service: str, entity_id: str) -> bool: ...
    async def get_entity_state(self, entity_id: str) -> dict: ...

class VectorSearchProvider(Protocol):
    async def search_command(self, text: str) -> Optional[dict]: ...

class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...
    async def generate_stream(self, prompt: str) -> AsyncIterator[str]: ...
```

Tipar todos los constructores de componentes post-refactor.

### 2.3 Eliminar dead code

- `src/alerts/example_usage.py` → `docs/examples/alerts/`
- `src/alerts/integration_example.py` → `docs/examples/alerts/`
- `src/alerts/complete_integration_demo.py` → `docs/examples/alerts/`
- `command_processor.py:287` `classify_intent` → eliminar
- `voice_pipeline.py:832` anonymous `PermResult` → `@dataclass` en módulo
- `dispatcher.py:842` `audio: any` → `audio: Optional[np.ndarray]`

### 2.4 Seguridad

- `settings.yaml:5` — `url` → `"${HOME_ASSISTANT_URL}"`
- Input sanitization para texto STT antes de prompt injection al LLM
- Verificar que dashboard y health endpoints no expongan tokens
- Agregar scrubbing layer para serialización de objetos con tokens

### 2.5 Error handling en fire-and-forget tasks

```python
def _handle_task_error(self, task: asyncio.Task):
    if not task.cancelled() and task.exception():
        logger.error(f"Background task failed: {task.exception()}")

# Usage:
task = asyncio.create_task(self.analyze_ambient_audio(chunk))
task.add_done_callback(self._handle_task_error)
```

### 2.6 Cache con TTL

```python
@dataclass
class CacheEntry:
    value: dict
    created_at: float

class TTLCache:
    def __init__(self, max_size: int = 100, ttl_seconds: float = 300):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[dict]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry.created_at) < self._ttl:
            return entry.value
        if entry:
            del self._cache[key]
        return None

    def set(self, key: str, value: dict):
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = CacheEntry(value=value, created_at=time.time())
```

---

## Q3: Test Coverage (Semanas 25-36)

### Priorización por criticidad

| Prioridad | Módulo | Líneas | Meta Coverage | Tests Estimados |
|-----------|--------|--------|---------------|-----------------|
| **P0** | pipeline/ (5 componentes post-refactor) | ~1450 | 80% | ~60 tests |
| **P0** | orchestrator/cancellation.py | 331 | 80% | ~15 tests |
| **P0** | orchestrator/context_persistence.py | 430 | 80% | ~20 tests |
| **P1** | tts/piper_tts.py | 667 | 70% | ~25 tests |
| **P1** | presence/ (3 archivos) | 1858 | 70% | ~40 tests |
| **P1** | llm/buffered_streamer.py | 496 | 70% | ~20 tests |
| **P2** | timers/named_timers.py | 640 | 60% | ~20 tests |
| **P2** | notifications/smart_notifications.py | 635 | 60% | ~15 tests |
| **P2** | audio/ (4 archivos) | 1271 | 60% | ~25 tests |
| **P3** | analytics/ (3 archivos) | 1453 | 50% | ~20 tests |
| **P3** | routines/ (executor, scheduler) | ~1130 | 50% | ~20 tests |
| **P3** | dashboard/api.py | 594 | 50% | ~15 tests |
| **P3** | Resto (ambient, conversation, intercom, etc.) | ~3000 | 50% | ~30 tests |

**Total estimado: ~325 tests nuevos → meta final: ~940+ tests**

### Mocks para hardware

```python
# tests/mocks/mock_ble.py — BLE scanner
class MockBLEScanner:
    def __init__(self, devices: list[MockBLEDevice]):
        self.devices = devices
    async def discover(self, timeout: float) -> list:
        return self.devices

# tests/mocks/mock_audio.py — Audio devices
class MockAudioStream:
    def __init__(self, audio_data: np.ndarray):
        self._data = audio_data
        self._pos = 0
    def read(self, frames: int) -> np.ndarray:
        chunk = self._data[self._pos:self._pos + frames]
        self._pos += frames
        return chunk

# tests/mocks/mock_ma1260.py — Amplifier
class MockMA1260:
    def __init__(self):
        self.zones = {i: {"volume": 20, "power": False} for i in range(1, 7)}
    async def set_volume(self, zone: int, level: int):
        self.zones[zone]["volume"] = level
```

### Consolidar fixtures duplicados

Eliminar definiciones locales de `mock_chroma` y `mock_routine_manager` en `test_dispatcher.py`. Usar exclusivamente las de `tests/conftest.py`.

### Test de integración end-to-end

```python
# tests/integration/test_full_pipeline.py
async def test_fast_path_end_to_end(mock_audio, mock_stt, mock_ha, mock_tts):
    """Simula: audio → wake word → STT → router → HA → TTS"""
    pipeline = build_test_pipeline(mock_audio, mock_stt, mock_ha, mock_tts)
    result = await pipeline.process_command(mock_audio.wake_word_audio)
    assert result["success"]
    assert result["latency_ms"] < 300
    mock_ha.call_service.assert_called_once()
    mock_tts.synthesize.assert_called_once()
```

---

## Q4: Features Pendientes + Production-Ready (Semanas 37-48)

### 4.1 Completar TODOs críticos

| TODO | Archivo:Línea | Implementación |
|------|---------------|----------------|
| Notificación multi-usuario | `dispatcher.py:715` | Conectar `notify_waiting_users` a TTS real |
| Sugerencias → automatizaciones | `suggestion_engine.py:459` | Llamar `ha_client.create_automation()` |
| Security event handler | `voice_pipeline.py:1524` | Push notification + activar cámaras Frigate |
| Feedback TTS | `voice_pipeline.py:780` | Detectar "no entendí" y re-promptear |

### 4.2 Dashboard web

- Frontend: HTML+JS simple (o React si se justifica)
- Endpoints ya existentes en `dashboard/api.py`
- Agregar: latency real-time, user management, zone control, health status
- WebSocket para updates en tiempo real

### 4.3 Wake word personalizado

- Completar flujo `wakeword/trainer.py` + `recorder.py`
- Entrenar con 50+ samples del usuario
- Evaluar accuracy vs `hey_jarvis` default
- Documentar proceso de re-entrenamiento

### 4.4 Docker production

- Health checks reales en docker-compose (no solo TCP)
- CI/CD: build → test → deploy (GitHub Actions ya existe parcialmente)
- Monitorización con LatencyMonitor → alertas
- Validar y documentar systemd service

### 4.5 requirements.txt completo

Agregar dependencias faltantes:
- `bleak>=0.21.0` (BLE scanning)
- `torch>=2.1.0` (ML models)
- `torchaudio>=2.1.0` (audio processing)
- `speechbrain>=0.5.16` (speaker ID)
- `fastapi>=0.109.0` (dashboard)
- `uvicorn>=0.27.0` (ASGI server)
- `spotipy>=2.23.0` (Spotify client)

### 4.6 Logging unificado

Migrar todos los módulos a `src/core/logging.get_logger()`. Actualmente ~50% usa el custom logger y ~50% el estándar. Unificar a structured logging con `LogContext` y `generate_request_id()`.

---

## Métricas de Éxito

| Métrica | Actual | Post-Q1 | Post-Q2 | Post-Q3 | Post-Q4 |
|---------|--------|---------|---------|---------|---------|
| VoicePipeline LOC | 1,870 | ~200 | ~200 | ~200 | ~200 |
| Bare excepts | 7 | 7 | 0 | 0 | 0 |
| Typed constructors | ~10% | ~40% | ~90% | ~90% | ~95% |
| Test count | 617 | ~650 | ~680 | ~940 | ~960 |
| Modules with 0% tests | 15+ | ~12 | ~10 | 0 | 0 |
| Blocking I/O in async | 6+ calls | 0 | 0 | 0 | 0 |
| Dead code files | 3 | 3 | 0 | 0 | 0 |
| requirements.txt gaps | ~7 | ~7 | ~7 | ~7 | 0 |

---

## Riesgos y Mitigación

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Refactor Q1 rompe funcionalidad | Alta | Tests existentes (617) como safety net. Refactor incremental, no big-bang |
| ha_client async migration rompe WebSocket | Media | Los 11 tests existentes cubren REST+WS. Migrar método por método |
| Nuevos tests descubren bugs latentes | Alta | Esto es bueno. Documentar y corregir |
| Q4 features scope creep | Media | Mantener scope mínimo viable. Dashboard simple primero |
