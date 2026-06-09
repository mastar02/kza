# KZA - Reporte de Estado del Proyecto
**Fecha:** 9 de Marzo, 2026
**Versión:** 2.0

---

## Resumen Ejecutivo

KZA es un ecosistema de control por voz para domótica con latencia ultra-baja (<300ms), diseñado para reemplazar asistentes comerciales como Alexa con control total local y soberano. Python 3.13.

| Métrica | Valor |
|---------|-------|
| Líneas de código (src) | 40,163 |
| Líneas de tests | 16,074 |
| Tests totales | 975 (pasando) |
| Módulos principales | 32 |
| Archivos de documentación | 13 |
| Runtime | Python 3.13.9, `src/main.py` es el entry point canónico |

---

## Hardware Objetivo

```
CPU:  AMD Threadripper PRO 9965WX (24 cores/48 threads)
RAM:  128GB DDR5-5600 ECC
GPUs: 4x NVIDIA RTX 3070 (8GB cada una)
      ├── GPU 0: Speech-to-Text (Faster-Whisper)
      ├── GPU 1: Embeddings/RAG
      ├── GPU 2: Router/Clasificador
      └── GPU 3: Text-to-Speech
```

---

## Arquitectura de Módulos

### Core (Pipeline de Voz)
```
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE DE VOZ (~200-300ms total)                        │
├─────────────────────────────────────────────────────────────┤
│  Wake Word → STT → Router → Action/LLM → TTS → Respuesta  │
│     10ms    150ms   20ms      50ms       80ms              │
└─────────────────────────────────────────────────────────────┘
```

### Módulos Implementados

| Módulo | Archivos | Líneas | Función |
|--------|----------|--------|---------|
| **spotify** | 8 | 4,506 | Integración Spotify con mood mapping |
| **training** | 7 | 3,882 | Entrenamiento LoRA y personalidad |
| **pipeline** | 11 | 3,459 | Voice pipeline, CommandProcessor, ProcessedCommand |
| **orchestrator** | 6 | 3,265 | Multi-usuario, routing, FAST_LIST/FAST_REMINDER |
| **routines** | 5 | 2,526 | Rutinas de automatización |
| **presence** | 4 | 1,727 | BLE scanning, tracking por zona |
| **audio** | 5 | 1,723 | Captura y zonas, MA1260 |
| **users** | 5 | 1,629 | Identificación y emociones |
| **alerts** | 6 | 1,609 | Alertas de seguridad/patrones |
| **analytics** | 4 | 1,465 | Análisis y sugerencias |
| **llm** | 3 | 1,112 | Razonamiento LLM 72B + Router 7B |
| **learning** | 3 | 1,108 | Aprendizaje continuo |
| **wakeword** | 4 | 862 | Detección palabra clave |
| **tts** | 2 | 857 | Dual TTS: Kokoro + Qwen3-TTS |
| **rooms** | 2 | 766 | Contexto por habitación |
| **memory** | 3 | 724 | Memoria contextual |
| **dashboard** | 2 | 680 | API del dashboard |
| **notifications** | 2 | 653 | Sistema de notificaciones |
| **timers** | 2 | 648 | Timers nombrados |
| **integrations** | 2 | 634 | Integraciones HA |
| **home_assistant** | 3 | 616 | Cliente HA + circuit breaker |
| **reminders** | 5 | 586 | Recordatorios con recurrencia |
| **lists** | 4 | 564 | Listas compartidas (compras, etc.) |
| **ambient** | 2 | 540 | Detección de eventos ambientales |
| **intercom** | 2 | 530 | Sistema de intercomunicación |
| **proactive** | 2 | 505 | Morning briefing proactivo |
| **monitoring** | 2 | 413 | Latencia |
| **core** | 2 | 410 | Logging centralizado |
| **vectordb** | 2 | 399 | Base de datos vectorial |
| **health** | 2 | 381 | Chequeos de salud |
| **conversation** | 2 | 363 | Follow-up mode |
| **stt** | 2 | 332 | Speech-to-Text |

---

## Sistema de Spotify Multi-Zona (NUEVO)

### Componentes Implementados

```
┌─────────────────────────────────────────────────────────┐
│                    SPOTIFY MULTI-ROOM                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐    ┌──────────────────┐           │
│  │ SpeakerGroupMgr  │───▶│ SpotifyZoneCtrl  │           │
│  │  - Speakers      │    │  - Play/Transfer │           │
│  │  - Groups        │    │  - Volume        │           │
│  │  - Aliases       │    │  - Sync          │           │
│  └──────────────────┘    └──────────────────┘           │
│           │                       │                      │
│           ▼                       ▼                      │
│  ┌──────────────────┐    ┌──────────────────┐           │
│  │ SpeakerEnrollment│    │ MusicDispatcher  │           │
│  │  - Voice learning│    │  - Intent detect │           │
│  │  - Auto-discover │    │  - Zone routing  │           │
│  └──────────────────┘    └──────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Comandos de Voz Soportados

**Reproducción por zona:**
- "Pon música en la cocina"
- "Reproduce jazz en la sala"
- "Pon música en toda la casa"

**Transferencia:**
- "Mueve la música al dormitorio"
- "Pasa la música a la planta alta"

**Control de volumen:**
- "Sube el volumen en la cocina"
- "Volumen al 50 en la sala"

**Enrollment por voz (NUEVO):**
- "Esta bocina se llama cocina"
- "La cocina está en la planta baja"
- "Crea un grupo área social con cocina y sala"
- "Busca bocinas nuevas"
- "Qué bocinas tengo?"

---

## Cobertura de Tests

**Total: 975 tests pasando** (pytest, ~20.35s runtime, Python 3.13.9)

Los tests cubren todos los módulos principales. Se ejecutan con:
```bash
pytest tests/                              # 975 passed
pytest tests/ --cov=src --cov-report=html  # Con coverage
```

---

## Funcionalidades Completadas

### Voice Pipeline
- [x] Detección de wake word personalizable
- [x] STT con Faster-Whisper (GPU 0)
- [x] Router rápido Qwen2.5 7B (GPU 2)
- [x] Dual TTS: Kokoro-82M (fast) + Qwen3-TTS 0.6B (conversacional) (GPU 3)
- [x] Latencia <300ms objetivo
- [x] Pipeline refactorizado: VoicePipeline (169 LOC) orquesta AudioManager, CommandProcessor, ResponseHandler, FeatureManager
- [x] ProcessedCommand dataclass como contrato tipado entre CommandProcessor y RequestRouter

### Multi-Usuario
- [x] Identificación por voz (ECAPA-VOXCELEB)
- [x] Contextos persistentes por usuario
- [x] Cola de prioridades para requests
- [x] Tokens de cancelación

### Spotify Avanzado
- [x] Mood mapping contextual
- [x] Multi-room audio
- [x] Grupos de bocinas
- [x] Enrollment por voz
- [x] Auto-discovery de dispositivos
- [x] Transferencia entre zonas

### Sistema de Alertas
- [x] Alertas de seguridad (puertas/ventanas)
- [x] Alertas de patrones (energía/agua)
- [x] Alertas de dispositivos (batería)
- [x] Notificaciones por voz

### Listas y Recordatorios (NUEVO — marzo 2026)
- [x] Listas compartidas y por usuario (src/lists/)
- [x] Recordatorios con recurrencia (src/reminders/)
- [x] Dispatcher paths: FAST_LIST y FAST_REMINDER en RequestDispatcher
- [x] Sincronización con Home Assistant
- [x] Tests de integración para ambos módulos

### Aprendizaje
- [x] Entrenamiento nocturno LoRA
- [x] Detección de emociones
- [x] Sugerencias de automatización
- [x] Personalidad configurable

### Cambios de Infraestructura (BL-001 a BL-004)
- [x] `src/kza_server.py` eliminado — `src/main.py` es el runtime canónico
- [x] Docker services marcados como EXPERIMENTAL (scoped bajo BL-005)
- [x] ProcessedCommand dataclass como contrato tipado en el pipeline (BL-002)
- [x] Python 3.13 como baseline (BL-003)
- [x] 975/975 tests pasando como baseline (BL-004)

---

## Estructura de Archivos Clave

```
kza/
├── src/
│   ├── main.py                    # Entry point canónico (kza_server.py fue eliminado)
│   ├── pipeline/
│   │   ├── voice_pipeline.py      # Orquestador (169 LOC)
│   │   ├── command_processor.py   # ProcessedCommand dataclass + STT + Speaker ID
│   │   ├── request_router.py      # Consume ProcessedCommand, despacha paths
│   │   ├── response_handler.py    # TTS + streaming + zone routing
│   │   ├── audio_manager.py       # Captura de audio
│   │   └── feature_manager.py     # Analytics, memory, training
│   │
│   ├── orchestrator/
│   │   ├── dispatcher.py          # PathType enum (incluye FAST_LIST, FAST_REMINDER)
│   │   ├── context_manager.py     # Contextos de usuario
│   │   └── priority_queue.py      # Cola de prioridades
│   │
│   ├── lists/                     # NUEVO: Listas compartidas
│   │   ├── list_manager.py
│   │   ├── list_store.py
│   │   └── ha_sync.py
│   │
│   ├── reminders/                 # NUEVO: Recordatorios con recurrencia
│   │   ├── reminder_manager.py
│   │   ├── reminder_store.py
│   │   ├── reminder_scheduler.py
│   │   └── recurrence.py
│   │
│   └── [28 módulos más...]
│
├── config/
│   └── settings.yaml              # Configuración completa
│
├── tests/                         # 975 tests, 75 archivos
│
└── docs/
    └── 13 archivos .md
```

---

## Próximos Pasos Sugeridos

1. **Integración física** - Conectar hardware real (bocinas, sensores)
2. **Training inicial** - Entrenar wake word personalizado
3. **Calibración de zonas** - Configurar MA1260 con zonas reales
4. **Testing end-to-end** - Pruebas con voz real
5. **Dashboard de monitoreo** - UI para ver latencias/estado

---

## Notas de Desarrollo

- Todos los tests pasan: `pytest tests/ → 975 passed` (~20.35s, Python 3.13.9)
- Runtime canónico: `python -m src.main` (kza_server.py fue eliminado)
- Docker services son EXPERIMENTAL (bajo BL-005)
- Código en español para comandos de voz, código/logs en inglés
- Arquitectura modular y extensible (32 módulos)
- Configuración centralizada en `settings.yaml`
- Documentación en `/docs` (13 archivos)

---

*Última actualización: 9 de Marzo, 2026 — BL-006*
