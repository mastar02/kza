# 🏠 KZA - Asistente de Hogar Inteligente con IA Local

> **Reemplazo completo de Alexa con procesamiento 100% local**

---

## 📋 Resumen Ejecutivo

**KZA** es un sistema de domótica por voz que corre completamente en hardware local, sin depender de servicios en la nube. Diseñado para máxima privacidad, baja latencia y personalización total.

| Métrica | Valor |
|---------|-------|
| **Latencia objetivo** | < 300ms end-to-end |
| **Latencia actual** | ~150-280ms ✅ |
| **Líneas de código** | ~26,000+ |
| **Tests** | 617+ |
| **Módulos** | 20+ |

---

## 🖥️ Hardware

```
┌─────────────────────────────────────────────────────────┐
│  AMD Threadripper PRO 5965WX (24 cores / 48 threads)    │
│  128GB RAM DDR4 (8x16GB, 8 canales, ~358 GB/s)          │
├─────────────────────────────────────────────────────────┤
│  GPU 0: RTX 3070 - STT (Whisper)                        │
│  GPU 1: RTX 3070 - Embeddings + Speaker ID + Emotion    │
│  GPU 2: RTX 3070 - Router/Clasificador (Qwen 7B)        │
│  GPU 3: RTX 3070 - TTS (Piper/XTTS)                     │
├─────────────────────────────────────────────────────────┤
│  Modelo Razonamiento: Llama 3.3 70B Q4 (CPU, 128GB)     │
└─────────────────────────────────────────────────────────┘
```

### Periféricos de Audio
- **Micrófono**: ReSpeaker Mic Array v2.0 (XMOS XVF-3000, 4 mics, beamforming)
- **Extensión USB**: Cat5e/6 60m para distribución en toda la casa
- **Bluetooth**: UGREEN BT 5.3 para detección de presencia
- **Amplificador**: Monoprice MA1260 (12 canales, 6 zonas)

---

## 🧠 Arquitectura del Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Wake     │ -> │ STT      │ -> │ Router   │ -> │ TTS      │
│ Word     │    │ Whisper  │    │ Qwen 7B  │    │ Piper    │
│ (CPU)    │    │ (GPU 0)  │    │ (GPU 2)  │    │ (GPU 3)  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │
     │               ▼               │               │
     │         ┌──────────┐         │               │
     │         │ Parallel │         │               │
     │         │ Speaker  │         │               │
     │         │ Emotion  │         │               │
     │         │ (GPU 1)  │         │               │
     │         └──────────┘         │               │
     │                              ▼               │
     │                        ┌──────────┐         │
     │                        │ LLM 70B  │ (si es  │
     │                        │ (CPU)    │ complejo│
     │                        └──────────┘         │
     ▼                              ▼               ▼
┌─────────────────────────────────────────────────────────┐
│                   Home Assistant                         │
│              (control de dispositivos)                   │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Optimizaciones de Latencia Implementadas

### Fase 1: Optimizaciones Core (~120ms ahorrados)

| Optimización | Ahorro | Descripción |
|--------------|--------|-------------|
| **STT 100% en RAM** | ~25-30ms | Audio procesado como numpy array directo, sin I/O de disco |
| **Router 1 inferencia** | ~15-25ms | `classify_and_respond()` combina clasificación + respuesta |
| **TTS prebuffer 30ms** | ~20-30ms | Reducido de 80ms → 30ms para menor latencia perceptible |
| **Embeddings cache** | ~10-20ms | Cache con TTL 60s + invalidación por versión |
| **WebSocket HA** | ~40-60ms | Conexión persistente vs REST (15ms vs 60ms) |

### Fase 2: Optimizaciones Avanzadas (~135ms adicionales)

| Optimización | Ahorro | Descripción |
|--------------|--------|-------------|
| **asyncio.gather()** | ~15-25ms | Procesamiento paralelo real de STT + Speaker + Emotion |
| **VAD Streaming** | ~50-100ms | Comienza transcripción antes del silencio final |
| **TTS Pre-warming** | ~20-35ms | Modelo ONNX pre-calentado al iniciar |
| **KV-Cache Prefix** | ~40-80ms | System prompt cacheado en GPU (vLLM) |
| **Emotion Batch GPU** | ~10-20ms | Batch real con vectorización en GPU |

### Resultado Final

```
ANTES:       ~450-650ms
DESPUÉS:     ~150-280ms
─────────────────────────
AHORRO:      ~250-370ms (55-60%)
```

---

## 🎵 Módulo Spotify Multi-Room

Control completo de Spotify con interpretación de contexto por LLM:

### Comandos Soportados
```
"Pon música para cocinar en la cocina"
"Reproduce algo relajante en toda la casa"
"Pon Bad Bunny en el living"
"Pausa la música"
"Siguiente canción"
"Baja el volumen en el dormitorio"
```

### Características
- **Speaker Enrollment por voz**: "Registra este parlante como cocina"
- **Grupos de zonas**: "Crea un grupo con living y comedor"
- **Mood interpretation**: LLM interpreta "música para estudiar" → parámetros de Spotify
- **Multi-room sync**: Reproducción sincronizada en múltiples dispositivos

---

## 👤 Identificación de Usuario por Voz

```python
# Flujo de identificación
Audio → Speaker Embeddings → Comparación con usuarios registrados → Usuario identificado

# Enrollment
"Kza, registra mi voz como Juan"
[Captura 3 muestras de voz]
"Listo Juan, ya te reconozco"
```

### Capacidades
- Identificación en < 60ms
- Threshold de confianza configurable (default: 0.75)
- Soporte para múltiples usuarios
- Cache de embeddings para acceso rápido

---

## 📍 Sistema de Presencia BLE (Nuevo)

Detección de presencia usando Bluetooth Low Energy:

```
┌─────────────────────────────────────────────────┐
│              UGREEN BT 5.3 Adapter              │
│                      │                          │
│    ┌─────────────────┼─────────────────┐        │
│    ▼                 ▼                 ▼        │
│ ┌──────┐        ┌──────┐         ┌──────┐      │
│ │iPhone│        │Watch │         │Pixel │      │
│ │ -45dB│        │ -60dB│         │ -70dB│      │
│ │ ~2m  │        │ ~4m  │         │ ~6m  │      │
│ └──────┘        └──────┘         └──────┘      │
└─────────────────────────────────────────────────┘
```

### Comandos de Presencia
```
"¿Quién está en casa?"
"¿Está Juan en casa?"
"¿Hay alguien en el living?"
"¿Dónde está María?"
```

### Características
- Escaneo pasivo de BLE advertisements
- Estimación de distancia por RSSI
- Tracking por zona (múltiples adaptadores)
- Asociación dispositivo ↔ usuario
- Eventos de llegada/salida

---

## 😊 Detección de Emociones

Análisis de emoción en tiempo real usando wav2vec2:

| Emoción | Arousal | Valence | Ajuste de Respuesta |
|---------|---------|---------|---------------------|
| Happy | 0.8 | 0.9 | Tono alegre, +10% velocidad |
| Sad | 0.3 | 0.1 | Tono empático, -10% velocidad |
| Angry | 0.9 | 0.2 | Tono firme, +20% velocidad |
| Neutral | 0.5 | 0.5 | Sin cambios |

---

## 🔧 Estructura del Proyecto

```
kza/
├── src/
│   ├── stt/              # Speech-to-Text (Whisper, Moonshine)
│   ├── tts/              # Text-to-Speech (Piper, XTTS, Kokoro)
│   ├── llm/              # LLM Reasoner + Fast Router
│   ├── pipeline/         # Voice Pipeline + Command Processor
│   ├── home_assistant/   # Integración con Home Assistant
│   ├── users/            # User Manager + Emotion Detector
│   ├── presence/         # Detección de presencia BLE (NUEVO)
│   ├── spotify/          # Control de Spotify multi-room
│   ├── orchestrator/     # Multi-usuario + Context Manager
│   ├── routines/         # Gestión de rutinas
│   ├── audio/            # Zone Manager + Multi-mic capture
│   └── wakeword/         # Detección de wake word
├── config/
│   └── settings.yaml     # Configuración centralizada
├── tests/                # 617+ tests
├── tools/
│   └── benchmark_latency.py  # Benchmark de latencia
└── docs/                 # Documentación
```

---

## 🧪 Benchmark de Latencia

```bash
$ python tools/benchmark_latency.py --iterations 20

============================================================
📊 RESUMEN DE LATENCIAS
============================================================
Componente           Min      Avg      Med      P95      Max  Success
------------------------------------------------------------
STT                 45.2ms   52.3ms   51.1ms   58.4ms   65.2ms   100%
STT+VAD             38.1ms   44.6ms   43.2ms   52.1ms   58.3ms   100%
Router              35.4ms   42.1ms   41.3ms   48.2ms   55.1ms   100%
TTS                 28.3ms   35.2ms   34.1ms   42.1ms   48.3ms   100%
Emotion             22.1ms   28.4ms   27.2ms   35.1ms   42.2ms   100%
Home Assistant      12.3ms   18.2ms   17.1ms   25.3ms   32.1ms   100%
------------------------------------------------------------

Pipeline estimado:   129.6ms (STT + Router + TTS)
✅ Dentro del objetivo de 300ms
```

---

## 📈 Próximos Pasos

1. **Configuración multi-room BLE**: Múltiples adaptadores por zona
2. **Integración presencia con rutinas**: "Cuando llegue Juan, prende luces"
3. **Wake word personalizado**: Entrenar modelo con "Hey Kza"
4. **Dashboard web**: Monitor de estado en tiempo real
5. **Fine-tuning LoRA**: Personalizar respuestas del LLM

---

## 🔐 Privacidad

- ✅ **100% local**: Ningún dato sale de la red local
- ✅ **Sin dependencias cloud**: No requiere conexión a internet para funcionar
- ✅ **Control total**: Todo el código es auditable y modificable
- ✅ **Datos en tu hardware**: Embeddings, preferencias, historial - todo local

---

## 📦 Dependencias Principales

```
faster-whisper    # STT
piper-tts         # TTS rápido
vllm              # Router GPU
llama-cpp-python  # LLM 70B en CPU
chromadb          # Vector DB
bleak             # BLE scanning
spotipy           # Spotify API
homeassistant     # Integración HA
```

---

## 🚀 Quick Start

```bash
# Clonar e instalar
git clone https://github.com/tu-usuario/kza.git
cd kza
pip install -r requirements.txt

# Configurar
cp config/settings.example.yaml config/settings.yaml
# Editar con tus tokens y configuración

# Ejecutar
python -m src.main
```

---

**Desarrollado con ❤️ para domótica local y privada**

*Última actualización: Febrero 2026*
