# KZA - Reporte de Estado del Proyecto
**Fecha:** 4 de Febrero, 2026
**Versión:** 1.0

---

## Resumen Ejecutivo

KZA es un ecosistema de control por voz para domótica con latencia ultra-baja (<300ms), diseñado para reemplazar asistentes comerciales como Alexa con control total local y soberano.

| Métrica | Valor |
|---------|-------|
| Líneas de código (src) | 25,599 |
| Líneas de tests | 10,111 |
| Tests totales | 612 |
| Módulos principales | 18 |
| Archivos de documentación | 11 |

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
| **spotify** | 8 | 4,568 | Integración Spotify con mood mapping |
| **orchestrator** | 6 | 3,075 | Multi-usuario y routing |
| **alerts** | 9 | 3,159 | Alertas de seguridad/patrones |
| **training** | 5 | 2,600 | Entrenamiento y personalidad |
| **pipeline** | 6 | 2,492 | Coordinación de voz |
| **users** | 5 | 1,511 | Identificación y emociones |
| **analytics** | 3 | 1,453 | Análisis y sugerencias |
| **audio** | 4 | 1,271 | Captura y zonas |
| **llm** | 3 | 948 | Razonamiento LLM |
| **wakeword** | 4 | 857 | Detección palabra clave |
| **memory** | 3 | 721 | Memoria contextual |
| **tts** | 2 | 640 | Síntesis de voz |
| **routines** | 2 | 439 | Rutinas de automatización |
| **monitoring** | 2 | 409 | Latencia |
| **vectordb** | 2 | 400 | Base de datos vectorial |
| **health** | 2 | 378 | Chequeos de salud |
| **home_assistant** | 2 | 289 | Cliente HA |
| **stt** | 2 | 188 | Speech-to-Text |

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

### Distribución por Módulo

| Módulo | Tests | Estado |
|--------|-------|--------|
| Spotify | 169 | ✅ Pasando |
| Training | 65 | ✅ Pasando |
| Orchestrator | 79 | ✅ Pasando |
| Alerts | 74 | ✅ Pasando |
| Users | 58 | ✅ Pasando |
| Safety | 43 | ✅ Pasando |
| Memory | 38 | ✅ Pasando |
| Otros | 86 | ✅ Pasando |
| **TOTAL** | **612** | ✅ |

### Tests de Spotify Detallados

```
test_mood_mapper.py        → 32 tests ✅
test_music_dispatcher.py   → 47 tests ✅
test_speaker_enrollment.py → 53 tests ✅
test_speaker_groups.py     → 37 tests ✅
─────────────────────────────────────
                           169 tests
```

---

## Funcionalidades Completadas

### Voice Pipeline
- [x] Detección de wake word personalizable
- [x] STT con Faster-Whisper (GPU 0)
- [x] Router rápido Qwen2.5 7B (GPU 2)
- [x] TTS con Piper/XTTS-v2 (GPU 3)
- [x] Latencia <300ms objetivo

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

### Aprendizaje
- [x] Entrenamiento nocturno LoRA
- [x] Detección de emociones
- [x] Sugerencias de automatización
- [x] Personalidad configurable

---

## Estructura de Archivos Clave

```
kza/
├── src/
│   ├── spotify/
│   │   ├── client.py              # Cliente Spotify API
│   │   ├── mood_mapper.py         # Mapeo de mood → música
│   │   ├── music_dispatcher.py    # Routing de comandos
│   │   ├── speaker_groups.py      # Gestión de bocinas/grupos
│   │   ├── speaker_enrollment.py  # Enrollment por voz ⭐
│   │   └── zone_controller.py     # Control multi-zona ⭐
│   │
│   ├── orchestrator/
│   │   ├── context_manager.py     # Contextos de usuario
│   │   ├── request_dispatcher.py  # Fast/slow path routing
│   │   └── priority_queue.py      # Cola de prioridades
│   │
│   └── [otros módulos...]
│
├── config/
│   └── settings.yaml              # Configuración completa
│
├── tests/
│   └── unit/spotify/
│       ├── test_speaker_enrollment.py  # 53 tests ⭐
│       └── test_speaker_groups.py      # 37 tests ⭐
│
└── docs/
    ├── SPOTIFY.md
    ├── ORCHESTRATOR.md
    └── ESTADO_PROYECTO_KZA.md     # Este archivo
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

- Todos los tests pasan: `pytest tests/ → 612 passed`
- Código en español para comandos de voz
- Arquitectura modular y extensible
- Configuración centralizada en `settings.yaml`
- Documentación completa en `/docs`

---

*Generado automáticamente - KZA Project*
