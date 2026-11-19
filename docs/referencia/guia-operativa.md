# KZA — Guía Operativa y Referencia Técnica

Referencia detallada extraída de `CLAUDE.md` para mantener el contexto base ligero (`<= 14 KB`) sin perder ninguna convención ni enlace.

## Source of truth cross-project (Notion bajo demanda)

Este proyecto cubre **solo** el pipeline de voz. Para las convenciones del servidor compartido (usuarios/UID, sub-rangos de puertos `9500-9599`, Podman rootless + Quadlets, GPU por CDI, onboarding), la fuente operativa es el espejo local **`docs/SERVER_CONVENTIONS.md`**. **Notion** (workspace KZA, root `page_id` `345ab24f-c493-80b2-b6f4-ef917e865f26`) sigue siendo la fuente canónica si ambas difieren, pero se consulta **bajo demanda**: su MCP ya no se carga en KZA.

- **El MCP de Notion NO está disponible en este proyecto** (desde 2026-08-03; cargaba 24 tools / 20,485 tokens por sesión sin usarse). No escribir código ni instrucciones que asuman `mcp__notion__*`.
- **Para consultar Notion:** usar el repo de `homelab-infra` que ya lo declara, o agregarlo temporalmente en scope local tomando el token desde el archivo `.mcp.json` fuera del repo (nunca pegar tokens en archivos commiteados) y removerlo al terminar (`claude mcp remove -s local notion`).
- **No** leer la memoria de `homelab-infra` ni `homelab-services`.
- **Páginas de Notion por tema:** pág. 8 (contrato compartido), pág. 10 (HA), pág. 11 (red), pág. 12 (mail), pág. 14 (obs).

## Estilo de código Python

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

    def __init__(self, dependencia_a, dependencia_b, config: dict | None = None):
        self.dep_a = dependencia_a
        self.dep_b = dependencia_b
        self._config = config or {}
        self._running = False

    async def process(self, input_data: dict) -> ResultadoAlgo:
        """Procesar datos de entrada."""
        ...
```

## Mapa de Archivos Clave

| Ruta | Qué hace | Cuándo tocarlo |
|------|----------|----------------|
| `src/main.py` | Entry point, DI de todos los servicios | Solo al agregar nuevo servicio top-level |
| `src/pipeline/voice_pipeline.py` | Pipeline completo de voz | Cambios en flujo de audio |
| `src/pipeline/command_processor.py` | Audio → texto + speaker + emotion | Cambios en procesamiento |
| `src/pipeline/response_handler.py` | Texto → audio con streaming | Cambios en respuesta |
| `src/orchestrator/request_dispatcher.py` | Routing fast/slow path | Agregar nuevos paths |
| `src/orchestrator/context_manager.py` | Contexto conversacional por usuario | Cambios en memoria |
| `src/llm/reasoner.py` | HttpReasoner (gateway `:8200`) + FastRouter 7B | Cambios en inferencia |
| `src/home_assistant/ha_client.py` | Cliente HA REST + WebSocket | Nuevas integraciones HA |
| `src/spotify/music_dispatcher.py` | Routing de comandos musicales | Nuevos comandos Spotify |
| `src/spotify/speaker_groups.py` | Gestión de bocinas y zonas | Cambios en multi-room |
| `src/users/speaker_identifier.py` | Identificación por voz ECAPA-TDNN | Cambios en speaker ID |
| `src/users/emotion_detector.py` | Detección de emociones wav2vec2 | Cambios en emotion |
| `src/alerts/alert_manager.py` | Sistema de alertas proactivas | Nuevos tipos de alerta |
| `src/vectordb/chroma_sync.py` | Sync HA entities → ChromaDB | Cambios en búsqueda |
| `src/code_index/` | Servicio índice semántico del codebase (`:9515`) | Cambios en búsqueda de código |
| `src/audio/zone_manager.py` | Multi-zona + MA1260 | Cambios en zonas |
| `src/rooms/room_context.py` | Contexto por habitación (mic+BT) | Nuevas habitaciones |
| `src/ambient/textual_wake.py` | Wake textual `"nexa"` sobre stream ambient | Cambios en disparo textual |
| `config/settings.yaml` | TODA la configuración centralizada | Cualquier config nueva |
| `tests/conftest.py` | Fixtures globales de tests | Nuevos mocks/fixtures |

## Módulos del Sistema (20+)

| Módulo | Líneas aprox. | Función principal |
|--------|---------------|-------------------|
| `spotify` | 4,568 | Multi-room, mood mapping, enrollment |
| `alerts` | 3,159 | Alertas seguridad/patrones/dispositivos |
| `orchestrator` | 3,075 | Multi-usuario, routing, prioridades |
| `training` | 2,600 | LoRA nocturno, personalidad |
| `pipeline` | 2,492 | Voice pipeline, command processor |
| `users` | 1,511 | Speaker ID, emociones, permisos |
| `audio` | 1,271 | Multi-zona, MA1260, captura |
| `llm` | 948 | HttpReasoner (cloud gateway) + Router 7B |
| `memory` | 721 | Short/long term, preferencias |
| `presence` | ~600 | BLE scanning, tracking por zona |
| `rooms` | ~400 | Contexto por habitación |

## Hardware Resumen (detalle en `docs/architecture/HARDWARE.md`)

- **CPU:** Threadripper PRO 7965WX — 24c/48t (el LLM 72B local en CPU se retiró 2026-05-30; reasoner cloud vía gateway `:8200`).
- **RAM:** 128 GB DDR5-5600 RDIMM (8×16 GB, 8 canales, ~358 GB/s).
- **GPUs:** **2× RTX 3070 8 GB hoy** — `cuda:1` = audio completo + `llama-server` 7B `:8101` (STT/SpeakerID/TTS/fast-path NLU, ~7.4 GB); `cuda:0` = Emotion (deshabilitado) + fallback ambient, casi libre (~1.9 GB). BGE-M3 corre en CPU. Ver contrato en `docs/SERVER_CONVENTIONS.md`.
- **Audio / Amp / BLE:** ReSpeaker XVF3800 por habitación (USB Cat5e), Dayton Audio MA1260 (12 ch / 6 zonas, RS-232), UGREEN BT 5.3.

## Variables de Entorno (`.env`) y Persistencia

```bash
HOME_ASSISTANT_URL=http://192.168.1.100:8123
HOME_ASSISTANT_TOKEN=<token>
SPOTIFY_CLIENT_ID=<id>
SPOTIFY_CLIENT_SECRET=<secret>
CONFIG_PATH=config/settings.yaml
```

| Qué | Formato | Path |
|-----|---------|------|
| Embeddings/búsqueda | ChromaDB | `./data/chroma_db/` |
| Eventos/latencia | SQLite | `./data/events.db`, `./data/latency.db` |
| Usuarios/preferencias | JSON | `./data/users.json`, `./data/preferences.json` |
| Contextos de usuario | JSON | `./data/contexts/` |
| Modelos LLM / LoRA | GGUF / Safetensors | `./models/`, `./models/lora_adapters/` |
| Config completa | YAML | `config/settings.yaml` |

## Comandos de Desarrollo

```bash
# Tests (en Mac usar .venv/bin/pytest con CPython 3.13)
.venv/bin/pytest tests/                              # Suite completa
.venv/bin/pytest tests/unit/spotify/                 # Módulo específico
.venv/bin/pytest tests/ -k "test_speaker"            # Por nombre
.venv/bin/pytest tests/ --cov=src --cov-report=html  # Coverage

# Ejecutar / Benchmark / Búsqueda semántica
python -m src.main
python -m src.rooms.room_context --detect
python tools/benchmark_latency.py --iterations 20
python tools/code_search.py "cómo se maneja el timeout de HA al boot"
./scripts/download_models.sh
```

## Worktrees, Podas y Verificación por Contenido

- **Regla de aislamiento:** nunca ejecutar dos agentes concurrentes sobre el **mismo** checkout o el **mismo** worktree (evita `index.lock` y cambios de rama cruzados). Cada worker multi-consola recibe su propio worktree aislado en `~/.herdr/worktrees/kza/<alias>` gestionado por `scripts/kza_seat.py`.
- **Verificación antes de eliminar una rama histórica — NO confiar solo en `git log <rama> ^<base>`:** el trabajo puede haberse integrado squasheado o reescrito. Comparar contenido real:
  ```bash
  comm -23 <(git ls-tree -r --name-only <rama> -- src/ tests/ | sort) \
           <(git ls-tree -r --name-only <base> -- src/ tests/ | sort)
  git diff --shortstat <base> <rama> -- src/ tests/
  ```
  Si está contenida y se decide archivar: `git tag archive/<rama> <rama>`, luego `git worktree remove <path>` y `git branch -D <rama>`.
- **Worktree hermano `bench-ambient`:** tiene checkout, memoria Serena y colección RAG propios (`feat/ambient-multipista`). No modificar ni podar sin confirmación explícita.

## Workflow Multi-Consola KZA (`herdr-app` + `hc` + `gpt-6-luna`)

- **Orquestador agnóstico:** cualquier consola principal (`claude`, `agy`, `codex`, `muse`) en el space `kza` actúa como Orquestador: mantiene roadmap, cierra decisiones con el usuario, despacha especialistas vía `scripts/kza_seat.py`, juzga gates y mergea. Nunca implementa cambios de aplicación desde el checkout raíz mientras hay workers activos.
- **Pre-Task Spec Gate:** para tareas medianas/complejas, despachar primero `kza-analista` (read-only) que emite una Spec `<= 2 KB` con alcance, invariantes y decisiones abiertas.
- **Comunicación push sin polling:** workers escriben su reporte completo a `~/.local/state/kza-router/reports/<alias>.md` (o inline si `--disable-write` en Muse) y notifican con `<= 6` líneas vía `hc send $HC_TERMINAL_ID`. El orquestador espera con `hc wait --children any` y verifica archivo de reporte, `HEAD` exacto y estado de PR antes de aceptar `done`.
- **Confianza inicial en tabs visibles:** `agy` (primera vez en un worktree nuevo), `muse` (workspace nuevo) y `codex` (primera vez con `.codex/hooks.json`) pausan en `blocked` pidiendo confirmación humana en el tab visible de `herdr-app`. **Nunca bypassear diálogos de confianza programáticamente.**
- **Scripts de orquestación:**
  - `scripts/kza_quota.py` — telemetría real de cuotas de las 4 suscripciones (`claude`, `codex`, `muse`, `agy`).
  - `scripts/kza_router.py` — enrutamiento dinámico vía OpenAI Decisions API (`gpt-6-luna`, 0 output tokens) con fallback determinista local, exclusión cross-model por **familia real** (`anthropic`|`openai`|`meta`|`google`), `outcome` y `report`.
  - `scripts/kza_seat.py` — `open`, `status`, `close`, `sync-templates` (solo enlaza `kza-*.md`; jamás toca `tu-*` ni `gsm-*`).
- **Carga de API Key en macOS Keychain (si se crea una entrada nueva):** usar siempre `read -rs` (nunca `security add-generic-password -w` interactivo, porque `getpass` BSD trunca silenciosamente a 128 bytes):
  ```bash
  read -rs OPENAI_KEY && security add-generic-password -U -a "$USER" -s "kza-openai-decisions" -w "$OPENAI_KEY" && unset OPENAI_KEY
  ```
- **Scripts legacy de despliegue previo:** `scripts/kza-push`, `scripts/kza-sync` y `scripts/install_code_index_hook.sh` corresponden al flujo histórico en `/home/kza/app` (reemplazado por `scripts/build_runtime.py`). Se conservan como referencia sin interferir en el desarrollo Mac.
