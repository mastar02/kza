# 🏠 Home Assistant Voice Control

Sistema de control por voz para Home Assistant con latencia ultra-baja (<300ms) para domótica y capacidad de razonamiento profundo para consultas complejas.

## ✨ Características

- **Latencia <300ms** para comandos de domótica
- **Wake word** personalizable ("Hey Jarvis")
- **Base vectorial** sincronizada con Home Assistant
- **Creación de rutinas por voz**
- **Razonamiento profundo** con LLM 70B para consultas complejas
- **TTS natural** con Piper o XTTS
- **Multi-usuario** con contexto separado y cola priorizada
- **Spotify avanzado** con interpretación de contexto ("música para cocinar")

## 🖥️ Hardware del Sistema

### Especificaciones Completas

| Componente | Modelo | Especificaciones |
|------------|--------|------------------|
| **CPU** | AMD Threadripper PRO 9965WX | 24 cores/48 threads, Zen 5, 4.2-5.4 GHz, 128MB L3 |
| **Motherboard** | ASUS Pro WS WRX90E-SAGE SE | WRX90, 8 canales DDR5, 128 PCIe 5.0 lanes |
| **RAM** | A-Tech 128GB (2x64GB) | DDR5-5600 RDIMM ECC, expandible a 512GB |
| **GPU 0** | NVIDIA RTX 3070 | 8GB VRAM - Speech-to-Text |
| **GPU 1** | NVIDIA RTX 3070 | 8GB VRAM - Embeddings/RAG |
| **GPU 2** | NVIDIA RTX 3070 | 8GB VRAM - Router/Clasificador |
| **GPU 3** | NVIDIA RTX 3070 | 8GB VRAM - Text-to-Speech |
| **PSU** | 1600W+ recomendado | 80+ Platinum (CPU 350W + 4×GPU 220W) |

### Distribución de Recursos

```
┌─────────────────────────────────────────────────────────────────┐
│  CPU: Threadripper PRO 9965WX (24 cores)                        │
│  └─ LLM 70B Inference (llama.cpp, 24 threads)                   │
│  └─ ~45GB RAM para modelo Q4_K_M                                │
├─────────────────────────────────────────────────────────────────┤
│  RAM: 128GB DDR5-5600 RDIMM ECC (2 canales activos)             │
│  └─ LLM: ~45GB                                                  │
│  └─ ChromaDB: ~2-5GB                                            │
│  └─ OS + Buffers: ~10GB                                         │
│  └─ Disponible: ~60GB                                           │
├─────────────────────────────────────────────────────────────────┤
│  GPU 0 (cuda:0): STT                                            │
│  └─ Faster-Whisper Large-v3: ~4GB VRAM                          │
├─────────────────────────────────────────────────────────────────┤
│  GPU 1 (cuda:1): Embeddings                                     │
│  └─ BGE-M3 o BGE-small: ~2GB VRAM                               │
├─────────────────────────────────────────────────────────────────┤
│  GPU 2 (cuda:2): Router                                         │
│  └─ Qwen2.5-7B-Instruct: ~6GB VRAM                              │
├─────────────────────────────────────────────────────────────────┤
│  GPU 3 (cuda:3): TTS                                            │
│  └─ Piper: ~500MB VRAM                                          │
│  └─ XTTS-v2 (opcional): ~4GB VRAM                               │
└─────────────────────────────────────────────────────────────────┘
```

### Expansión Futura de RAM

| Configuración | Capacidad | Canales | Ancho de Banda |
|---------------|-----------|---------|----------------|
| Actual (2x64GB) | 128GB | 2 de 8 | ~102 GB/s |
| Fase 2 (4x64GB) | 256GB | 4 de 8 | ~205 GB/s |
| Máximo (8x64GB) | 512GB | 8 de 8 | ~358 GB/s |

> **Nota**: Más RAM permite correr modelos más grandes (Llama 405B) o múltiples modelos simultáneos.

## 🚀 Instalación

### 1. Clonar y configurar

```bash
cd home-assistant-voice
cp .env.example .env
# Editar .env con tu token de Home Assistant
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar modelos

```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

### 4. Ejecutar

```bash
# Recommended (includes environment checks, GPU verification, health check):
./scripts/start.sh

# Or directly:
python -m src.main
```

## 🎤 Comandos de Voz

### Domótica (< 300ms)

- "Prende la luz del living"
- "Apaga el aire del dormitorio"
- "Sube las persianas de la cocina"
- "Pon el aire a 22 grados"

### Rutinas

- "Crea una rutina que cuando llegue a casa prenda las luces"
- "Nueva automatización: a las 7am abre las persianas"
- "¿Qué rutinas tengo?"
- "Elimina la rutina de llegada"

### Sincronización

- "Sincroniza los comandos"
- "Actualiza la base de datos"

### Música / Spotify

- "Pon música de Bad Bunny"
- "Pon algo tranquilo"
- "Música para una cena romántica"
- "Pausa" / "Siguiente canción"
- "¿Qué está sonando?"

> Ver [docs/SPOTIFY.md](docs/SPOTIFY.md) para configuración completa.

## 📁 Estructura

```
home-assistant-voice/
├── CLAUDE.md              # Contexto para Claude Code
├── README.md              # Este archivo
├── requirements.txt       # Dependencias
├── .env.example          # Variables de entorno
├── config/
│   └── settings.yaml     # Configuración
├── docs/
│   ├── ORCHESTRATOR.md   # Sistema multi-usuario
│   └── SPOTIFY.md        # Integración Spotify
├── src/
│   ├── main.py           # Entry point
│   ├── stt/              # Speech-to-Text
│   ├── tts/              # Text-to-Speech
│   ├── vectordb/         # ChromaDB sync
│   ├── routines/         # Gestión de rutinas
│   ├── home_assistant/   # Cliente HA
│   ├── llm/              # LLM reasoner + buffering
│   ├── orchestrator/     # Multi-usuario, cola, contexto
│   ├── spotify/          # Integración Spotify
│   └── pipeline/         # Pipeline principal
├── scripts/
│   ├── download_models.sh
│   └── setup_spotify.py  # Configurar Spotify
├── models/               # Modelos descargados
└── data/
    ├── chroma_db/        # Base vectorial
    └── spotify_tokens.json  # Tokens OAuth
```

## ⚙️ Configuración

Editar `config/settings.yaml`:

```yaml
home_assistant:
  url: "http://192.168.1.100:8123"
  token: "${HOME_ASSISTANT_TOKEN}"

stt:
  model: "distil-whisper/distil-small.en"
  device: "cuda:0"

tts:
  engine: "piper"  # o "xtts" para mejor calidad

latency_targets:
  total: 300  # ms
```

## 📊 Latencia Esperada

| Componente | Tiempo |
|------------|--------|
| Wake word | ~10ms |
| STT | ~100-150ms |
| Vector search | ~15-20ms |
| Home Assistant | ~30-50ms |
| TTS | ~50-80ms |
| **TOTAL** | **~200-300ms** |

## 🔧 Troubleshooting

### "No se puede conectar a Home Assistant"

1. Verifica que la URL sea correcta
2. Verifica que el token sea válido
3. Prueba: `curl -H "Authorization: Bearer TU_TOKEN" http://TU_HA:8123/api/`

### "CUDA out of memory"

Reduce `gpu_memory_utilization` en settings.yaml o usa modelos más pequeños.

### Latencia > 300ms

1. Usa Whisper Tiny en vez de Small
2. Usa Piper en vez de XTTS
3. Verifica que Home Assistant esté en la red local

## 📚 Documentación Adicional

| Documento | Descripción |
|-----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Contexto del proyecto para Claude Code |
| [docs/ORCHESTRATOR.md](docs/ORCHESTRATOR.md) | Sistema multi-usuario, cola priorizada, cancelación |
| [docs/SPOTIFY.md](docs/SPOTIFY.md) | Integración Spotify con interpretación de contexto |

## 📝 Licencia

MIT

## 🤝 Contribuir

¡PRs bienvenidos! Revisa CLAUDE.md para contexto del proyecto.
