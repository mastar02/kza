# Review de Arquitectura - Home Assistant Voice

## Resumen Ejecutivo

El sistema tiene una arquitectura sólida para uso doméstico (1-4 usuarios) con excelente separación de paths rápidos y lentos. Sin embargo, tiene limitaciones de escalabilidad que deberían abordarse si se planea expandir.

## Diagrama de Arquitectura Actual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AUDIO INPUT                                     │
│                    (sounddevice, 16kHz, multi-zone)                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WAKE WORD DETECTION                                  │
│                      (OpenWakeWord, ~10ms)                                   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │   STT (GPU 0)         │       │   Speaker ID (GPU 1)  │
        │   Whisper ~150ms      │       │   ECAPA ~100ms        │
        └───────────┬───────────┘       └───────────┬───────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VOICE PIPELINE                                      │
│                     (Orquestación Central)                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    REQUEST DISPATCHER                                │    │
│  │                                                                      │    │
│  │   Keywords → Intent Classification → Path Routing                   │    │
│  │                                                                      │    │
│  │   DOMOTICS_KEYWORDS: prende, apaga, sube, baja...                   │    │
│  │   MUSIC_KEYWORDS: música de, pon algo, pausa...                     │    │
│  │   ROUTINE_KEYWORDS: rutina, automatiza, cuando...                   │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│         ┌───────────────────────┼───────────────────────┐                   │
│         │                       │                       │                   │
│         ▼                       ▼                       ▼                   │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐             │
│  │ FAST PATH   │        │ MUSIC PATH  │        │ SLOW PATH   │             │
│  │ (Paralelo)  │        │             │        │ (Cola)      │             │
│  ├─────────────┤        ├─────────────┤        ├─────────────┤             │
│  │             │        │             │        │             │             │
│  │ • Domotics  │        │ • Search    │        │ • Context   │             │
│  │   Vector DB │        │   (~500ms)  │        │   Manager   │             │
│  │   + HA      │        │             │        │             │             │
│  │   ~100ms    │        │ • Mood+LLM  │        │ • Priority  │             │
│  │             │        │   (~3-5s)   │        │   Queue     │             │
│  │ • Router 7B │        │             │        │   P0>P1>P2  │             │
│  │   ~200ms    │        │ • Playback  │        │             │             │
│  │             │        │   Control   │        │ • LLM 70B   │             │
│  │ • Rutinas   │        │   ~200ms    │        │   (CPU)     │             │
│  │   ~100ms    │        │             │        │   ~5-30s    │             │
│  │             │        │             │        │             │             │
│  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘             │
│         │                      │                      │                     │
│         └──────────────────────┴──────────────────────┘                     │
│                                │                                            │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TTS (GPU 3)                                       │
│                    Piper (~80ms) / XTTS (~1.5s)                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              BUFFERED STREAMING                                      │    │
│  │                                                                      │    │
│  │   LLM tokens ──► Sentence Buffer ──► TTS Chunks ──► Audio Stream    │    │
│  │                                                                      │    │
│  │   Filler phrases mientras LLM piensa: "Déjame ver..."               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ZONE MANAGER                                        │
│                   (Multi-room audio routing)                                 │
│                                                                             │
│   Zone 1 ────► Amplifier Ch 1 ────► Living Room Speakers                    │
│   Zone 2 ────► Amplifier Ch 2 ────► Kitchen Speakers                        │
│   Zone N ────► Amplifier Ch N ────► Room N Speakers                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evaluación por Componente

### 1. VoicePipeline (src/pipeline/voice_pipeline.py)

**Líneas de código:** ~1400
**Responsabilidades:** Demasiadas

| Aspecto | Puntuación | Comentario |
|---------|------------|------------|
| Cohesión | ⚠️ 2/5 | Mezcla audio, orquestación, analytics |
| Acoplamiento | ⚠️ 2/5 | Depende de 15+ componentes |
| Testabilidad | ⚠️ 2/5 | Difícil de testear en aislamiento |
| Mantenibilidad | ⚠️ 3/5 | Funciona pero complejo de modificar |

**Problema:** El pipeline es un "God Object" que conoce demasiado del sistema.

**Solución propuesta:**
```
VoicePipeline (actual)
    │
    ├── AudioManager (nuevo)
    │   └── Captura, wake word, VAD
    │
    ├── CommandProcessor (nuevo)
    │   └── STT, Speaker ID, dispatching
    │
    ├── ResponseHandler (nuevo)
    │   └── TTS, streaming, zone routing
    │
    └── FeatureManager (nuevo)
        └── Analytics, memory, training
```

### 2. Orchestrator (src/orchestrator/)

**Evaluación:** ★★★★☆ (4/5)

| Componente | Puntuación | Comentario |
|------------|------------|------------|
| dispatcher.py | ★★★★☆ | Bien diseñado, fácil de extender |
| priority_queue.py | ★★★★★ | Excelente implementación |
| context_manager.py | ★★★★☆ | Funcional, podría ser más robusto |
| cancellation.py | ★★★★☆ | Buen patrón cooperativo |

**Fortaleza:** Separación clara de responsabilidades dentro del módulo.

**Debilidad:** Solo soporta un worker de slow path.

### 3. Spotify Integration (src/spotify/)

**Evaluación:** ★★★★★ (5/5)

| Componente | Puntuación | Comentario |
|------------|------------|------------|
| auth.py | ★★★★★ | OAuth2 + PKCE bien implementado |
| client.py | ★★★★★ | API completa y async |
| mood_mapper.py | ★★★★☆ | Extensible, buenos defaults |
| music_dispatcher.py | ★★★★★ | Intent detection robusto |

**Fortaleza:** Módulo más limpio del proyecto, bien aislado.

### 4. LLM Subsystem (src/llm/)

**Evaluación:** ★★★★☆ (4/5)

| Componente | Puntuación | Comentario |
|------------|------------|------------|
| reasoner.py | ★★★★☆ | Funcional, soporta LoRA |
| buffered_streamer.py | ★★★★★ | Excelente para UX |

**Fortaleza:** El buffering hace que LLM lento se sienta fluido.

**Debilidad:** Sin batching, cada request es independiente.

## Análisis de Escalabilidad

### Límites Actuales

| Recurso | Límite | Bottleneck |
|---------|--------|------------|
| Usuarios simultáneos (fast path) | ~10 | GPU memory |
| Usuarios simultáneos (slow path) | 1 | LLM secuencial |
| Requests/segundo (domotics) | ~20 | HA API rate limit |
| Requests/segundo (LLM) | 0.1-0.2 | CPU throughput |
| Contextos activos | ~50 | RAM (~50MB/contexto) |

### Escenarios de Carga

```
┌─────────────────────────────────────────────────────────────────┐
│  ESCENARIO 1: Casa típica (2-4 personas)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Carga típica:                                                  │
│  • 10-20 comandos de domotics/hora                              │
│  • 2-5 consultas conversacionales/hora                          │
│  • 5-10 comandos de música/hora                                 │
│                                                                 │
│  Resultado: ✅ EXCELENTE                                        │
│  • Fast path siempre disponible                                 │
│  • Slow path raramente tiene cola                               │
│  • Latencia consistente                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ESCENARIO 2: Oficina pequeña (10-15 personas)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Carga típica:                                                  │
│  • 50-100 comandos de domotics/hora                             │
│  • 20-30 consultas conversacionales/hora                        │
│  • 10-20 comandos de música/hora                                │
│                                                                 │
│  Resultado: ⚠️ DEGRADADO                                        │
│  • Fast path OK                                                 │
│  • Slow path: cola de 3-5 requests frecuente                    │
│  • Tiempo de espera: 15-25 segundos                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ESCENARIO 3: Edificio (50+ personas)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Resultado: ❌ NO VIABLE                                        │
│  • Cola de slow path crece indefinidamente                      │
│  • Timeouts frecuentes                                          │
│  • ChromaDB se vuelve bottleneck                                │
│  • Necesita arquitectura distribuida                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Recomendaciones

### Prioridad Alta (Refactoring)

#### 1. Dividir VoicePipeline

**Antes:**
```python
class VoicePipeline:
    # 1400 líneas, 15+ dependencias
    def __init__(self, stt, tts, chroma, ha, llm, router, ...):
        ...
```

**Después:**
```python
class AudioCapture:
    """Solo captura de audio y wake word"""

class CommandProcessor:
    """STT + Speaker ID + Dispatching"""

class ResponseHandler:
    """TTS + Streaming + Zone routing"""

class VoicePipeline:
    """Coordinación de alto nivel"""
    def __init__(self, audio: AudioCapture, processor: CommandProcessor, ...):
        ...
```

#### 2. Inyección de Dependencias

**Antes:**
```python
# En VoicePipeline.__init__
if orchestrator_enabled:
    self._orchestrator = MultiUserOrchestrator(
        chroma_sync=self.chroma,
        ha_client=self.ha,
        ...  # Crea dependencias internamente
    )
```

**Después:**
```python
# En main.py o factory
orchestrator = create_orchestrator(config)
pipeline = VoicePipeline(orchestrator=orchestrator)  # Inyectado
```

### Prioridad Media (Escalabilidad)

#### 3. ChromaDB como Servicio

**Actual:** In-process, bloquea thread principal

**Propuesto:**
```
┌─────────────┐     HTTP/gRPC     ┌─────────────┐
│  Pipeline   │ ◄───────────────► │  ChromaDB   │
│  (async)    │                   │  Server     │
└─────────────┘                   └─────────────┘
```

Beneficios:
- No bloquea el pipeline
- Puede escalar independientemente
- Permite caching distribuido

#### 4. Pool de Workers LLM

**Actual:** 1 worker secuencial

**Propuesto:**
```python
class LLMWorkerPool:
    def __init__(self, num_workers: int = 2):
        self.workers = [LLMWorker() for _ in range(num_workers)]
        self.queue = asyncio.Queue()

    async def process(self, request):
        worker = await self.get_available_worker()
        return await worker.generate(request)
```

Con 128GB RAM:
- Worker 1: Llama 70B Q4 (~35GB)
- Worker 2: Qwen 32B Q8 (~35GB)
- Disponible: ~58GB para OS + buffers

### Prioridad Baja (Futuro)

#### 5. Arquitectura de Microservicios

Para escalar a edificio/enterprise:

```
┌─────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                              │
│                    (Load Balancer + Auth)                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   STT Service │       │  LLM Service  │       │  TTS Service  │
│   (GPU Pool)  │       │  (CPU Pool)   │       │  (GPU Pool)   │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │   Message Queue   │
                    │   (Redis/RabbitMQ)│
                    └───────────────────┘
```

## Métricas de Calidad

### Complejidad Ciclomática

| Módulo | Complejidad | Evaluación |
|--------|-------------|------------|
| voice_pipeline.py | Alta (~25) | ⚠️ Necesita refactoring |
| dispatcher.py | Media (~12) | ✅ Aceptable |
| spotify/*.py | Baja (~5) | ✅ Excelente |
| llm/*.py | Baja (~6) | ✅ Excelente |

### Cobertura de Tests (Estimada)

| Módulo | Cobertura | Prioridad |
|--------|-----------|-----------|
| orchestrator/ | ~0% | 🔴 Alta |
| spotify/ | ~0% | 🟡 Media |
| llm/ | ~0% | 🟡 Media |
| pipeline/ | ~0% | 🔴 Alta |

### Deuda Técnica

| Área | Severidad | Impacto |
|------|-----------|---------|
| Sin tests | 🔴 Alta | Regresiones frecuentes |
| VoicePipeline monolítico | 🟡 Media | Mantenimiento difícil |
| ChromaDB in-process | 🟡 Media | Escalabilidad limitada |
| Sin circuit breakers | 🟡 Media | Fallas en cascada |
| Logs inconsistentes | 🟢 Baja | Debug más difícil |

## Conclusión

### Lo que está bien

1. **Arquitectura dual-path** es la decisión correcta
2. **Distribución de GPUs** está bien pensada
3. **Spotify module** es ejemplar en diseño
4. **Buffered streaming** resuelve el problema de LLM lento
5. **Priority queue** funciona correctamente

### Lo que necesita trabajo

1. **VoicePipeline** necesita ser dividido
2. **Tests** son críticos antes de más cambios
3. **Inyección de dependencias** mejoraría testabilidad
4. **ChromaDB** debería ser servicio separado para escalar

### Recomendación Final

Para uso doméstico (1-4 usuarios): **El sistema está listo para producción.**

Para escalar más allá: Implementar las recomendaciones de prioridad alta antes de agregar más features.

---

*Documento generado: 2024*
*Última revisión de arquitectura*
