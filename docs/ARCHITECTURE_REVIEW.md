# Review de Arquitectura - Home Assistant Voice

> **Actualizado:** 9 de Marzo, 2026 (BL-006). Codebase: 40K LOC src, 16K LOC tests, 975 tests pasando, 32 módulos, Python 3.13.

## Resumen Ejecutivo

El sistema tiene una arquitectura sólida para uso doméstico (1-4 usuarios) con excelente separación de paths rápidos y lentos. El pipeline fue refactorizado (VoicePipeline dividido en componentes), se agregó un contrato tipado (ProcessedCommand), y la suite de tests cubre todos los módulos principales.

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
│  │             │        │   Control   │        │ • LLM 72B   │             │
│  │ • Rutinas   │        │   ~200ms    │        │   (CPU)     │             │
│  │   ~100ms    │        │             │        │   ~5-30s    │             │
│  │             │        │             │        │             │             │
│  │ • Lists     │        │             │        │             │             │
│  │ • Reminders │        │             │        │             │             │
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
│              Kokoro-82M (~30ms) / Qwen3-TTS 0.6B (conversacional)            │
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

### 1. Pipeline (src/pipeline/) -- REFACTORIZADO

**Estado:** El "God Object" fue dividido. VoicePipeline ahora es un orquestador ligero de 169 LOC. El módulo completo tiene 11 archivos y 3,459 LOC.

| Componente | LOC | Función |
|------------|-----|---------|
| voice_pipeline.py | 169 | Orquestador de alto nivel |
| command_processor.py | 284 | STT + Speaker ID + ProcessedCommand dataclass |
| request_router.py | 844 | Consume ProcessedCommand, despacha a paths |
| response_handler.py | 311 | TTS + streaming + zone routing |
| audio_manager.py | 244 | Captura de audio |
| audio_loop.py | 258 | Loop de audio principal |
| feature_manager.py | 409 | Analytics, memory, training |
| model_manager.py | 603 | Gestión de modelos GPU |
| multi_room_audio_loop.py | 269 | Loop multi-room |

**Contrato tipado:** `ProcessedCommand` dataclass (BL-002) es el contrato entre CommandProcessor y RequestRouter. Campos: `text`, `user`, `emotion`, `speaker_confidence`, `timings`, `success`.

| Aspecto | Puntuación | Comentario |
|---------|------------|------------|
| Cohesion | 4/5 | Cada componente tiene responsabilidad clara |
| Acoplamiento | 3/5 | Bien separado, ProcessedCommand como contrato |
| Testabilidad | 4/5 | Componentes testeables individualmente |
| Mantenibilidad | 4/5 | Mucho mas facil de modificar |

### 2. Orchestrator (src/orchestrator/) — 3,265 LOC

**Evaluación:** ★★★★☆ (4/5)

| Componente | Puntuacion | Comentario |
|------------|------------|------------|
| dispatcher.py | ★★★★☆ | PathType enum con 12 paths incluyendo FAST_LIST, FAST_REMINDER |
| priority_queue.py | ★★★★★ | Excelente implementacion |
| context_manager.py | ★★★★☆ | Funcional, confirmation ordering fix aplicado |
| cancellation.py | ★★★★☆ | Buen patron cooperativo |

**PathType enum (actual):** FAST_DOMOTICS, FAST_ROUTINE, FAST_ROUTER, FAST_MUSIC, SLOW_MUSIC, SLOW_LLM, SYNC, ENROLLMENT, FEEDBACK, FAST_LIST, FAST_REMINDER.

**Fortaleza:** Separación clara de responsabilidades. Facil de extender (lists y reminders se agregaron con paths nuevos sin tocar paths existentes).

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

### 5. Lists (src/lists/) y Reminders (src/reminders/) -- NUEVO

**Lists:** 4 archivos, 564 LOC. Listas compartidas y por usuario con sync a HA.
**Reminders:** 5 archivos, 586 LOC. Recordatorios con recurrencia y scheduler.

Ambos se integran con el dispatcher via FAST_LIST y FAST_REMINDER paths (fast path, Priority.HIGH). Tests de integracion incluidos.

## Analisis de Escalabilidad

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

### Completado (desde la revision original)

#### 1. Dividir VoicePipeline -- HECHO

VoicePipeline fue dividido en AudioManager, CommandProcessor, ResponseHandler, FeatureManager, ModelManager, RequestRouter. El orquestador central es ahora 169 LOC. DI por constructor en `src/main.py`.

#### 2. Inyeccion de Dependencias -- HECHO

`src/main.py` es el entry point canonico y crea todos los servicios con DI por constructor. `src/kza_server.py` fue eliminado. Docker services estan marcados como EXPERIMENTAL.

#### 3. Tests -- HECHO

975 tests pasando (~20.35s). Cubren todos los modulos principales incluyendo orchestrator, pipeline, spotify, alerts, lists, reminders.

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

## Metricas de Calidad

### Complejidad Ciclomatica

| Modulo | Complejidad | Evaluacion |
|--------|-------------|------------|
| voice_pipeline.py | Baja (~5) | Orquestador ligero post-refactor |
| request_router.py | Media-Alta (~15) | 844 LOC, muchos paths (unverified) |
| dispatcher.py | Media (~12) | Aceptable (unverified) |
| spotify/*.py | Baja (~5) | Excelente (unverified) |

### Tests

**975 tests pasando** (~20.35s, Python 3.13.9). Todos los modulos principales tienen tests unitarios. Test files: 75 archivos Python en tests/.

### Deuda Tecnica

| Area | Severidad | Impacto |
|------|-----------|---------|
| ChromaDB in-process | Media | Escalabilidad limitada |
| request_router.py 844 LOC | Media | Archivo mas grande del pipeline |
| Coverage % desconocido | Media | No se ejecuta coverage regularmente |

## Conclusion

### Lo que esta bien

1. **Arquitectura multi-path** con 12 PathTypes, bien separados
2. **Pipeline refactorizado** con componentes claros y contrato tipado (ProcessedCommand)
3. **975 tests** como baseline de calidad
4. **Distribucion de GPUs** bien pensada
5. **Spotify module** es ejemplar en diseno
6. **Buffered streaming** resuelve el problema de LLM lento
7. **Priority queue** funciona correctamente
8. **Lists y Reminders** se agregaron limpiamente con paths nuevos

### Lo que necesita trabajo

1. **request_router.py** (844 LOC) es el archivo mas grande del pipeline y podria beneficiarse de mas division
2. **ChromaDB** deberia ser servicio separado para escalar
3. **Coverage %** no se mide regularmente — agregar al CI
4. **Docker services** estan marcados EXPERIMENTAL (BL-005)

### Recomendacion Final

Para uso domestico (1-4 usuarios): **El sistema esta listo para produccion.**

Para escalar mas alla: Las recomendaciones de prioridad alta (pipeline, DI, tests) ya fueron implementadas. Las pendientes son de prioridad media (ChromaDB como servicio, pool de workers LLM).

---

*Documento original: 2024*
*Ultima actualizacion: 9 de Marzo, 2026 — BL-006*
