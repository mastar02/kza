# KZA — Asistente de Voz Local para Domótica

Sistema de control por voz local para Home Assistant. Latencia <300ms para domótica (fast path 100% local); razonamiento complejo delegado al gateway LLM `:8200` (MiniMax cloud, decisión 2026-05-30 con `cloud.consent`). **2x RTX 3070 hoy** (cuda:1 audio completo + TTS + llama-server 7B fast-path NLU `:8101`; cuda:0 casi libre — Emotion deshabilitado; BGE-M3 vive en CPU, no en GPU) — se irán conectando más GPUs; cualquier reasignación se discute primero. Python 3.13, async/await, ~38K líneas, 2740+ tests.

## Source of truth cross-project

Este proyecto cubre **solo** el pipeline de voz. Para las convenciones del servidor compartido (usuarios/UID, sub-rangos de puertos, Podman rootless + Quadlets, GPU por CDI, onboarding), consultar **primero** el espejo local `docs/SERVER_CONVENTIONS.md`. **Notion** (workspace KZA, root page_id `345ab24f-c493-80b2-b6f4-ef917e865f26`, vía MCP `mcp__notion__*`) queda como referencia secundaria y fuente canónica si difieren.

- **No** leer la memoria de `~/Documents/homelab-infra/` ni `~/Documents/homelab-services/`.
- Para temas de plataforma sin espejo local, ir a Notion: pág 8 (contrato compartido), pág 11 (red), pág 10 (HA), pág 12 (mail), pág 14 (obs).
- Cuando el código de KZA depende de algo compartido (gateway LLM :8200, sub-rangos puertos 9500-9599), seguir `docs/SERVER_CONVENTIONS.md` y validar contra Notion pág 8 ante dudas.

## Reglas para Claude — LEER SIEMPRE

### SIEMPRE hacer
- Usar `async/await` para toda operación I/O (HA, audio, red, disco)
- Inyectar dependencias por constructor (patrón dominante del proyecto)
- Usar `@dataclass` para DTOs y estructuras de datos
- Usar `Enum` para estados y tipos (`ModelState`, `PathType`, `AlertPriority`)
- Escribir docstrings Google-style en clases y métodos públicos
- Type hints en parámetros y return de funciones públicas
- Imports: stdlib → third-party → `from src.modulo import Clase`
- Logging con `logger = logging.getLogger(__name__)` y prefijos descriptivos
- Mensajes de voz y UI en español, código/logs en inglés
- Tests con pytest + fixtures en `conftest.py`, mocks en `tests/mocks/`
- Respetar asignación de GPUs (2 hoy): cuda:1 = STT + SpeakerID + TTS + llama-server 7B `:8101` (todo el audio + fast-path NLU, comparten GPU); cuda:0 = Emotion (deshabilitado) + fallback whisper del ambient path, casi libre. BGE-M3 corre en CPU, no en GPU. Al conectar GPUs nuevas, la reasignación se discute primero

### NUNCA hacer
- Herencia profunda (usar composición siempre)
- Imports relativos (usar `from src.modulo`)
- Bloquear el event loop con llamadas síncronas
- Cambiar asignación de GPUs sin discutir primero
- Agregar dependencias sin justificar (8GB VRAM por GPU es limitado)
- Crear archivos de configuración nuevos (todo va en `config/settings.yaml`)
- Modificar `src/main.py` sin entender la cadena de DI completa
- Usar `print()` en lugar de `logger`
- Pedir confirmación para continuar cuando el siguiente paso es obvio. Si hay un plan con pasos secuenciales (BL-001, BL-002, etc.), avanzar al siguiente sin preguntar "¿seguimos?". Solo detenerse si hay una decisión de diseño ambigua, un bloqueante real, o se necesita input del usuario que no se puede inferir

### Estilo de código
```python
# Naming
class MiClase:              # PascalCase
def mi_funcion():            # snake_case
CONSTANTE_GLOBAL = "valor"   # UPPER_SNAKE_CASE
self._privado = None         # prefijo _

# Estructura de clase típica
@dataclass
class ResultadoAlgo:
    campo: str
    confianza: float = 0.0
    datos: dict = field(default_factory=dict)

class MiServicio:
    """Descripción breve del servicio."""

    def __init__(self, dependencia_a, dependencia_b, config: dict = None):
        self.dep_a = dependencia_a
        self.dep_b = dependencia_b
        self._config = config or {}
        self._running = False

    async def process(self, input_data: dict) -> ResultadoAlgo:
        """Procesar datos de entrada."""
        ...
```

## Arquitectura

```
Mic → WakeWord(CPU) → STT(GPU1) → Router 7B(GPU1 :8101) → TTS(GPU1) → Speaker
                         ↕                  ↕
                   SpeakerID(GPU1)   Reasoner cloud (gateway :8200)
                   Emotion(GPU0)     ChromaDB
                   BGE-M3(CPU)       HomeAssistant
```

**Paths de ejecución:**
- **Fast path** (<300ms): Domótica → VectorSearch → HA action → TTS
- **Music path** (~500ms): Spotify → MoodMapper → ZoneController → TTS
- **Slow path** (segundos): Reasoner cloud (gateway `:8200` → MiniMax) → Memory → TTS

**Orquestación multi-usuario:** `MultiUserOrchestrator` → `PriorityRequestQueue` → `ContextManager` (contexto por usuario) → `CancellationManager`

## Mapa de Archivos Clave

| Ruta | Qué hace | Cuándo tocarlo |
|------|----------|----------------|
| `src/main.py` | Entry point, DI de todos los servicios | Solo al agregar nuevo servicio top-level |
| `src/pipeline/voice_pipeline.py` | Pipeline completo de voz | Cambios en flujo de audio |
| `src/pipeline/command_processor.py` | Audio → texto + speaker + emotion | Cambios en procesamiento |
| `src/pipeline/response_handler.py` | Texto → audio con streaming | Cambios en respuesta |
| `src/orchestrator/request_dispatcher.py` | Routing fast/slow path | Agregar nuevos paths |
| `src/orchestrator/context_manager.py` | Contexto conversacional por usuario | Cambios en memoria |
| `src/llm/reasoner.py` | HttpReasoner (gateway :8200) + FastRouter 7B | Cambios en inferencia |
| `src/home_assistant/ha_client.py` | Cliente HA REST + WebSocket | Nuevas integraciones HA |
| `src/spotify/music_dispatcher.py` | Routing de comandos musicales | Nuevos comandos Spotify |
| `src/spotify/speaker_groups.py` | Gestión de bocinas y zonas | Cambios en multi-room |
| `src/users/speaker_identifier.py` | Identificación por voz ECAPA-TDNN | Cambios en speaker ID |
| `src/users/emotion_detector.py` | Detección de emociones wav2vec2 | Cambios en emotion |
| `src/alerts/alert_manager.py` | Sistema de alertas proactivas | Nuevos tipos de alerta |
| `src/vectordb/chroma_sync.py` | Sync HA entities → ChromaDB | Cambios en búsqueda |
| `src/code_index/` | Servicio índice semántico del codebase (:9515) | Cambios en búsqueda de código para agentes |
| `src/audio/zone_manager.py` | Multi-zona + MA1260 | Cambios en zonas |
| `src/rooms/room_context.py` | Contexto por habitación (mic+BT) | Nuevas habitaciones |
| `src/ambient/textual_wake.py` | Wake textual "nexa" sobre stream ambient (red de seguridad) | Cambios en disparo textual |
| `config/settings.yaml` | TODA la configuración centralizada | Cualquier config nueva |
| `tests/conftest.py` | Fixtures globales de tests | Nuevos mocks/fixtures |

## Módulos del Sistema (20+)

| Módulo | Líneas | Función principal |
|--------|--------|-------------------|
| spotify | 4,568 | Multi-room, mood mapping, enrollment |
| orchestrator | 3,075 | Multi-usuario, routing, prioridades |
| alerts | 3,159 | Alertas seguridad/patrones/dispositivos |
| training | 2,600 | LoRA nocturno, personalidad |
| pipeline | 2,492 | Voice pipeline, command processor |
| users | 1,511 | Speaker ID, emociones, permisos |
| audio | 1,271 | Multi-zona, MA1260, captura |
| llm | 948 | HttpReasoner (cloud gateway) + Router 7B |
| memory | 721 | Short/long term, preferencias |
| presence | ~600 | BLE scanning, tracking por zona |
| rooms | ~400 | Contexto por habitación |

## Comandos de Desarrollo

```bash
# Tests
pytest tests/                              # Todos los tests (2740+)
pytest tests/unit/spotify/                 # Tests de un módulo
pytest tests/ -k "test_speaker"            # Tests por nombre
pytest tests/ --cov=src --cov-report=html  # Coverage

# Ejecutar
python -m src.main                         # Iniciar sistema
python -m src.rooms.room_context --detect  # Detectar dispositivos USB

# Benchmark
python tools/benchmark_latency.py --iterations 20

# Búsqueda semántica del codebase (requiere kza-code-index en el server)
python tools/code_search.py "cómo se maneja el timeout de HA al boot"

# Modelos
./scripts/download_models.sh               # Descargar todos los modelos
```

## Worktrees

- **Regla: nunca trabajar en este repo y en un worktree a la vez** (comparten `.git`: locks de index y refs pueden colisionar; ya pasó con un `index.lock` colgado). **Vale igual para dos sesiones de Claude sobre el mismo checkout**: el 2026-08-03 una hizo `checkout main` y le sacó el árbol de abajo a la otra a mitad de trabajo. Antes de arrancar, `lsof -a -p <pid> -d cwd` sobre los `claude` vivos.
- **Cómo se integra**: el trabajo de un worktree se mergea a `main` desde la laptop (PR o merge local + push); el worktree NO pushea por su cuenta nada que no esté coordinado con la rama principal.
- **Cuándo se elimina — NO usar `git log <rama> ^main`.** Ese conteo miente: cuenta commits, y el trabajo se integra reescrito, squasheado o reimplementado. El 2026-08-03 daba 10 y 15 commits "sin integrar" en dos ramas cuyo contenido estaba **entero** en main. Ni `git cherry` (patch-id) alcanza: seguía marcando 2 y 13 ausentes. **Comparar contenido:**
  ```bash
  comm -23 <(git ls-tree -r --name-only <rama> -- src/ tests/ | sort) \
           <(git ls-tree -r --name-only main  -- src/ tests/ | sort)   # archivos que main NO tiene
  git diff --shortstat main <rama> -- src/ tests/                      # si es casi todo deleciones, la rama está atrás
  ```
  Vacío + deleciones masivas ⇒ contenida en main: `git tag archive/<rama> <rama>` (rescate), `git worktree remove <path>`, `git branch -D <rama>`.
- Worktrees activos: ninguno. `kza-dashboard` y `llm-failover-cooldown` se eliminaron el 2026-08-03 tras verificar por contenido; recuperables en los tags `archive/*`.

## Hardware Resumen (detalle en docs/architecture/HARDWARE.md)

- **CPU**: Threadripper PRO 7965WX — 24c/48t (el LLM 72B local en CPU se retiró 2026-05-30; el reasoner es cloud vía gateway :8200)
- **RAM**: 128GB DDR5-5600 RDIMM (8x16GB, 8 canales, ~358 GB/s)
- **GPUs**: **2x RTX 3070 8GB hoy** — cuda:1 = audio completo + llama-server 7B :8101 (STT/SpeakerID/TTS/fast-path NLU, ~7.4GB), cuda:0 = Emotion (deshabilitado) + fallback ambient, casi libre (~1.9GB). BGE-M3 corre en CPU. Se irán conectando más GPUs a futuro; ver contrato en `docs/SERVER_CONVENTIONS.md`
- **Audio**: ReSpeaker XVF3800 por habitación + extensores USB Cat5e
- **Amplificador**: Dayton Audio MA1260 Multi-Zone (12 canales / 6 zonas estéreo, control RS-232)
- **BLE**: UGREEN BT 5.3 por habitación para presencia

## Estado Actual

**Funcionando:** Pipeline completo de voz, multi-usuario, Spotify multi-room, alertas, identificación por voz, emociones, presencia BLE, memoria contextual, entrenamiento LoRA nocturno, personalidad configurable.

**Latencia actual:** ~150-280ms (objetivo <300ms cumplido).

**Pendiente:** Wake word personalizado entrenado, dashboard web, fine-tuning real con datos de usuario, integración calendarios, cámaras (Frigate).

## Variables de Entorno (.env)

```bash
HOME_ASSISTANT_URL=http://192.168.1.100:8123
HOME_ASSISTANT_TOKEN=<token>
SPOTIFY_CLIENT_ID=<id>
SPOTIFY_CLIENT_SECRET=<secret>
CONFIG_PATH=config/settings.yaml
```

## Persistencia de Datos

| Qué | Formato | Path |
|-----|---------|------|
| Embeddings/búsqueda | ChromaDB | `./data/chroma_db/` |
| Eventos/latencia | SQLite | `./data/events.db`, `./data/latency.db` |
| Usuarios/preferencias | JSON | `./data/users.json`, `./data/preferences.json` |
| Contextos de usuario | JSON | `./data/contexts/` |
| Modelos LLM | GGUF | `./models/` |
| Modelos LoRA | Safetensors | `./models/lora_adapters/` |
| Config completa | YAML | `config/settings.yaml` |
