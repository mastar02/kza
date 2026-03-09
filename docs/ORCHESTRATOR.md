# Sistema de Orquestacion Multi-Usuario

> **Actualizado:** 9 de Marzo, 2026 (BL-006).

Sistema para manejar multiples usuarios concurrentes con contexto separado por usuario, cola priorizada y cancelacion inteligente.

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CAPA DE ENTRADA (PARALELA)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Zona 1] ──┐                                                               │
│  [Zona 2] ──┼──► [STT] ──► [Speaker ID] ──► [REQUEST DISPATCHER]            │
│  [Zona N] ──┘                                    │                          │
└──────────────────────────────────────────────────┼──────────────────────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    ▼                              ▼                              ▼
       ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
       │   FAST PATH (PARALELO)  │   │   MUSIC PATH            │   │  SLOW PATH (SERIALIZADO)│
       ├─────────────────────────┤   ├─────────────────────────┤   ├─────────────────────────┤
       │                         │   │                         │   │                         │
       │  ┌───────────────────┐  │   │  ┌───────────────────┐  │   │  ┌───────────────────┐  │
       │  │   Vector Search   │  │   │  │   Spotify Fast    │  │   │  │  Context Manager  │  │
       │  │   (Domotica)      │  │   │  │   (Busqueda)      │  │   │  │  (por usuario)    │  │
       │  └───────────────────┘  │   │  └───────────────────┘  │   │  └─────────┬─────────┘  │
       │                         │   │                         │   │            │            │
       │  ┌───────────────────┐  │   │  ┌───────────────────┐  │   │  ┌─────────▼─────────┐  │
       │  │   Router 7B       │  │   │  │   Spotify Slow    │  │   │  │  Priority Queue   │  │
       │  │   (Preg. simples) │  │   │  │   (LLM + Mood)    │  │   │  │  P0 > P1 > P2     │  │
       │  └───────────────────┘  │   │  └───────────────────┘  │   │  └─────────┬─────────┘  │
       │                         │   │                         │   │            │            │
       │  ┌───────────────────┐  │   │  Latencia:              │   │  ┌─────────▼─────────┐  │
       │  │   Rutinas         │  │   │  • Fast: ~500ms         │   │  │   LLM 72B (CPU)   │  │
       │  │   (Predefinidas)  │  │   │  • Slow: ~3-5s          │   │  │   + Buffered TTS  │  │
       │  └───────────────────┘  │   │                         │   │  └───────────────────┘  │
       │                         │   │                         │   │                         │
       │  ┌───────────────────┐  │   │  Ver: docs/SPOTIFY.md   │   │  Latencia: 5-30s        │
       │  │   Lists (CRUD)    │  │   │                         │   │                         │
       │  └───────────────────┘  │   │                         │   │                         │
       │                         │   │                         │   │                         │
       │  ┌───────────────────┐  │   │                         │   │                         │
       │  │   Reminders       │  │   │                         │   │                         │
       │  │   (CRUD+Schedule) │  │   │                         │   │                         │
       │  └───────────────────┘  │   │                         │   │                         │
       │                         │   │                         │   │                         │
       │  Latencia: < 300ms      │   │                         │   │                         │
       └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

## Componentes

### 1. ContextManager (`context_manager.py`)

Mantiene el historial de conversacion por usuario.

```python
from src.orchestrator import ContextManager

manager = ContextManager(max_history=10, inactive_timeout=300)

# Crear contexto para usuario
ctx = manager.get_or_create("user_juan", "Juan", "zona_living")

# Construir prompt con historial
prompt = manager.build_prompt("user_juan", "¿Y eso para qué sirve?")
# Incluye turnos anteriores automaticamente

# Agregar respuesta al historial
manager.add_turn("user_juan", "assistant", "Sirve para...")
```

**Caracteristicas:**
- Historial separado por usuario
- Limpieza automatica de contextos inactivos
- Construccion de prompts con historial
- Thread-safe

### 2. PriorityRequestQueue (`priority_queue.py`)

Cola priorizada para peticiones al LLM.

```python
from src.orchestrator import PriorityRequestQueue, Priority

queue = PriorityRequestQueue(auto_cancel_previous=True)

# Encolar peticion
request = queue.enqueue(
    user_id="juan",
    text="Explicame la relatividad",
    priority=Priority.LOW
)

# Peticion de mayor prioridad (domotica) entra primero
queue.enqueue(
    user_id="maria",
    text="Prende la luz",
    priority=Priority.HIGH  # Se procesa antes
)

# Obtener siguiente (por prioridad)
next_req = await queue.dequeue_async()  # -> Maria primero
```

**Prioridades:**
| Nivel | Nombre | Uso | Timeout |
|-------|--------|-----|---------|
| P0 | CRITICAL | Seguridad, alarmas | 60s |
| P1 | HIGH | Domotica | 30s |
| P2 | MEDIUM | Rutinas, consultas | 60s |
| P3 | LOW | Conversacion | 120s |

### 3. CancellationToken (`cancellation.py`)

Permite cancelar peticiones en curso.

```python
from src.orchestrator import CancellationToken, CancellationReason

token = CancellationToken(request_id="req_123", user_id="juan")

# En el generador del LLM
for chunk in llm.generate_stream(prompt):
    if token.is_cancelled:
        break
    yield chunk

# Desde otro lugar, cancelar
token.cancel(CancellationReason.USER_NEW_REQUEST)
```

**Razones de cancelacion:**
- `USER_NEW_REQUEST`: Usuario hizo otra pregunta
- `HIGHER_PRIORITY`: Llego peticion mas urgente
- `TIMEOUT`: Tiempo agotado
- `USER_EXPLICIT`: Usuario dijo "cancela"

### 4. RequestDispatcher (`dispatcher.py`)

Enruta peticiones al path correcto. Usa `PathType` (StrEnum) para clasificar.

**PathType enum completo:**

| PathType | Prioridad | Descripcion |
|----------|-----------|-------------|
| FAST_DOMOTICS | HIGH | Vector search + HA action |
| FAST_ROUTINE | MEDIUM | Rutinas predefinidas |
| FAST_ROUTER | MEDIUM | Router 7B para respuestas simples |
| FAST_MUSIC | HIGH | Spotify busqueda directa |
| SLOW_MUSIC | LOW | Spotify con LLM + mood mapping |
| SLOW_LLM | LOW | LLM 72B para razonamiento |
| SYNC | MEDIUM | Comandos de sincronizacion |
| ENROLLMENT | MEDIUM | Registro de usuarios |
| FEEDBACK | LOW | Feedback sobre respuestas |
| FAST_LIST | HIGH | List CRUD (src/lists/) |
| FAST_REMINDER | HIGH | Reminder CRUD (src/reminders/) |

**Confirmation ordering fix:** El dispatcher verifica `ctx.pending_confirmation` ANTES de procesar cancel keywords, para que "no cancela" sea tratado como rechazo de confirmacion y no como un comando de cancelacion.

```python
from src.orchestrator import RequestDispatcher

dispatcher = RequestDispatcher(
    chroma_sync=chroma,
    ha_client=ha,
    routine_manager=routines,
    router=router_7b,       # Para fast path
    llm=llm_72b,            # Para slow path
    context_manager=context_manager,
    priority_queue=queue
)

# Dispatch automatico al path correcto
result = await dispatcher.dispatch(
    user_id="juan",
    text="Prende la luz",     # -> FAST_DOMOTICS
    zone_id="living"
)

result = await dispatcher.dispatch(
    user_id="maria",
    text="Agrega leche a la lista de compras",  # -> FAST_LIST
    zone_id="cocina"
)

result = await dispatcher.dispatch(
    user_id="juan",
    text="Recuerdame regar las plantas a las 8",  # -> FAST_REMINDER
    zone_id="living"
)

result = await dispatcher.dispatch(
    user_id="maria",
    text="Explicame fisica",  # -> SLOW_LLM
    zone_id="cocina"
)
```

## Flujos de Ejemplo

### Escenario 1: Domotica No Espera

```
t=0.0s  Juan: "Explicame la fotosintesis"
        -> Slow path, P3, entra a cola
        -> "Dejame ver..." (filler)

t=0.5s  Maria: "Prende la luz del living"
        -> Fast path, P1
        -> Vector search (20ms)
        -> Home Assistant (50ms)
        -> "Listo" (instantaneo)

t=3.0s  Juan recibe respuesta del LLM
        -> "La fotosintesis es el proceso..."
```

### Escenario 2: Conversaciones Paralelas

```
t=0.0s  Juan (living): "¿Que es Python?"
        -> Slow path, contexto de Juan

t=1.0s  Maria (cocina): "¿Como esta el clima?"
        -> Fast path (Router 7B responde)
        -> "Soleado, 22 grados"

t=5.0s  Juan: respuesta sobre Python

t=6.0s  Juan: "¿Y para que sirve?"
        -> Slow path, con contexto previo
        -> LLM sabe que hablan de Python
```

### Escenario 3: Interrupcion por Prioridad

```
t=0.0s  Juan: "Cuentame la historia de Roma"
        -> Slow path, P3
        -> LLM empieza a generar

t=2.0s  ALARMA: Sensor de humo activado
        -> P0 CRITICAL
        -> Interrumpe generacion de Juan
        -> "¡Alerta! Humo detectado en cocina"

t=2.5s  Juan: peticion cancelada
        -> "Lo siento, hubo una alerta urgente"
```

### Escenario 4: Mismo Usuario, Nueva Pregunta

```
t=0.0s  Juan: "Explicame la teoria de cuerdas"
        -> Slow path, empieza a generar

t=3.0s  Juan: "Olvida eso, dime la hora"
        -> Cancela peticion anterior
        -> Fast path (Router)
        -> "Son las 3:45 PM"
```

## Configuracion

```yaml
# settings.yaml
orchestrator:
  enabled: true

  context:
    max_history: 10           # Turnos por usuario
    inactive_timeout: 300     # 5 min sin actividad = limpiar

  queue:
    max_size: 100
    auto_cancel_previous: true  # Nueva pregunta cancela anterior

  priorities:
    domotics: 1      # HIGH
    routines: 2      # MEDIUM
    conversation: 3  # LOW
```

## Uso con el Pipeline

```python
from src.orchestrator import MultiUserOrchestrator

orchestrator = MultiUserOrchestrator(
    chroma_sync=chroma,
    ha_client=ha,
    routine_manager=routines,
    router=router_7b,
    llm=llm_72b,
    tts=tts,
    speaker_identifier=speaker_id,
    user_manager=user_manager
)

await orchestrator.start()

# Procesar peticion
result = await orchestrator.process(
    user_id=None,        # Identificar por voz
    text="Prende la luz",
    audio=audio_data,    # Para speaker ID
    zone_id="living"
)

# result.response -> "Listo"
# result.path -> PathType.FAST_DOMOTICS
# result.timings -> {"vector_search": 18, "home_assistant": 45}
```

## Estadisticas

```python
stats = orchestrator.get_stats()
# {
#   "total_requests": 150,
#   "fast_path": 120,
#   "slow_path": 30,
#   "queue": {
#     "queue_size": 2,
#     "total_cancelled": 5,
#     "total_timeout": 1
#   },
#   "contexts": {
#     "active_contexts": 3,
#     "total_turns": 45
#   }
# }
```

## Lists y Reminders (Fast Paths)

Desde marzo 2026, el dispatcher soporta dos paths adicionales para listas y recordatorios:

### FAST_LIST (src/lists/)
- Detecta keywords como "agrega", "quita", "lista", "compras"
- Enruta a `ListManager` para CRUD de items
- `ListStore` persiste en JSON con sync a HA via `ha_sync.py`
- Prioridad HIGH, fast path (<300ms)

### FAST_REMINDER (src/reminders/)
- Detecta keywords como "recuerdame", "recordatorio", "alarma"
- Enruta a `ReminderManager` para CRUD + scheduling
- `ReminderScheduler` ejecuta recordatorios con `recurrence.py` para patrones repetidos
- Prioridad HIGH, fast path (<300ms)

## Beneficios

1. **Domotica siempre rapida**: Comandos de luces/clima nunca esperan
2. **Contexto por usuario**: Cada persona tiene su conversacion
3. **Cancelacion inteligente**: Nueva pregunta cancela la anterior
4. **Interrupciones**: Prioridades altas interrumpen las bajas
5. **Feedback**: Usuarios saben cuando deben esperar
6. **Escalable**: Fast path es paralelo, slow path es serializado eficientemente
7. **Listas y recordatorios**: Operaciones locales rapidas sin necesitar LLM
