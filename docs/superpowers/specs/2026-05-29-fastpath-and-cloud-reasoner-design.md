# Diseño: Optimización del fast path + Reasoner cloud (MiniMax-M2.7-highspeed)

- **Fecha:** 2026-05-29
- **Autor:** Gabriel Asuaga (con Claude)
- **Estado:** Propuesto — pendiente de aprobación del dueño
- **Origen:** Auditoría multi-agente read-only del proyecto (49 agentes, 31 hallazgos verificados). Ver `tasks/wsh0gi357`.

---

## 1. Objetivo y alcance

Dos ejes independientes, en orden:

1. **FASE A — Fast path (<300ms):** reducir latencia percibida sin tocar la premisa "100% local" ni la asignación de GPU.
2. **FASE B — Reasoner cloud:** reemplazar el reasoner local del **slow path** (GLM-4.5-Air `:8200`) por **MiniMax-M2.7-highspeed** servido en el cloud de MiniMax (`https://api.minimax.io/v1`, OpenAI-compatible).

### Fuera de alcance (decisiones explícitas del dueño)

- **A4 — mover BGE-M3 a GPU:** descartado por ahora (toca asignación de GPU; `CLAUDE.md` exige discutir). Se puede retomar más adelante; el swap del reasoner libera presión de `cuda:1` (GLM-Air local deja de cargarse) y podría habilitarlo.
- **Mantener GLM-Air como fallback:** descartado. El dueño eligió **reemplazo total** del reasoner del slow path. Riesgo aceptado (ver §6).

### No-objetivos

- No se toca el **fast path de domótica** funcionalmente: sigue siendo `Wake → STT → dispatcher → VectorSearch → acción HA → TTS`, sin LLM. El reasoner cloud entra **solo** en el slow path (5-30s).
- El **fast_router local** (`:8101`, Qwen2.5-7B Q4) **se mantiene**: cumple el rol de gate/router dentro del presupuesto de 300ms y no debe depender de internet.

---

## 2. Contexto verificado (estado actual del código)

| Componente | Estado hoy | Archivo |
|---|---|---|
| Reasoner slow path | GLM-4.5-Air Q4_K_M, HTTP `:8200`, instancia única `HttpReasoner` | `settings.yaml:247-309`, `reasoner.py` |
| API call del reasoner | **`client.completions.create(prompt=...)`** (endpoint legacy `/v1/completions`) | `reasoner.py:471,510,530,658,703` |
| Path chat-style (existe) | `messages=...` usado en otro método (probable FastRouter) | `reasoner.py:360` |
| Resolución de API key | **Por puerto** (`urlparse(base_url).port`): 8100→VLLM, 8101/8200→LLAMA | `reasoner.py:17-55` |
| `EndpointKind.CLOUD` | **Declarado pero nunca instanciado** | `types.py:20` |
| Failover | Cadena por `priority` ascendente; cooldown exponencial 1m→5m→25m→1h | `router.py:72`, `settings.yaml:259-292` |
| `ep.timeout_s` / `ep.idle_timeout_s` | **Campos muertos**: definidos en `types.py:53` pero no se propagan en `complete()` | `router.py:100`, `reasoner.py:419` |
| Construcción de clients | **Hardcodeada**: dict `{fast_router_7b, reasoner_72b}` | `main.py:320-322`, `router_factory.py:87` |
| Clasificación de errores | `ssl.SSLError`/`socket.gaierror` **no** mapeados → caen como PERMANENT y **bloquean failover** | `reasoner.py:453`, `classify_error()` |
| Construcción del prompt | texto verbatim + historial 10 turnos + nombre + zona + estado del hogar | `context_manager.py:538-607` |
| Sanitización de logs | solo `Bearer xyz`, no query params `?api_key=` | `core/logging.py:107-110` |

### Datos de integración MiniMax (verificados contra la doc oficial)

- **Base URL:** `https://api.minimax.io/v1`
- **Endpoint:** `POST /v1/chat/completions` (OpenAI-compatible) — **no** hay garantía de `/v1/completions` legacy.
- **Auth:** `Authorization: Bearer <API_key>`
- **Model id:** `MiniMax-M2.7-highspeed` (también `MiniMax-M2.7`, `MiniMax-M2.5`, `MiniMax-M2.1`)
- **Streaming:** soportado (`stream: true`)
- **Anthropic-compat:** también ofrece formato Anthropic, pero usamos OpenAI-compat por consistencia con el código actual.

---

## 3. FASE A — Diseño del fast path

Tres cambios, todos seguros (no tocan GPU ni premisa local). Orden por ROI/riesgo.

### A1 — Desbloquear el event loop en la búsqueda vectorial

**Problema:** `search_command()` es síncrono y `embedder.encode()` (BGE-M3 en CPU, ~48ms) corre en el hilo del event loop, bloqueando todo el pipeline async durante la query.

**Cambio:**
- Envolver `embedder.encode(...)` en `await asyncio.to_thread(...)` dentro de `chroma_sync.py:300`.
- Convertir `search_command()` en `async def`.
- Actualizar el call site `dispatcher.py:590` a `await`.

**Interfaz:** `async def search_command(self, text: str, ...) -> SearchResult`. Consumidores existentes que la llamen sync deben migrar a `await` (auditar call sites con grep antes de editar).

**Beneficio:** libera el event loop (~35ms de no-bloqueo), permite concurrencia real. No reduce el tiempo absoluto del encode, pero deja de serializar el pipeline.

### A2 — Arreglar el warmup de embeddings

**Problema:** el guard `if getattr(chroma, "_embedder", None) is not None` (`main.py:155-161`) da `False` porque el embedder se crea lazy → el warmup se saltea **en silencio** → el primer comando de la sesión paga ~48ms de cold start.

**Cambio:**
- Exponer un método explícito `ChromaSync.warmup_embedder()` que fuerce la inicialización lazy y haga un `encode` dummy.
- Llamarlo en el arranque (`main.py`) sin depender del guard frágil.
- Loguear el resultado del warmup (éxito + latencia del encode dummy) para que un warmup roto vuelva a ser visible.

**Beneficio:** elimina ~43ms del primer comando.

### A3 — Diferir Speaker ID (ECAPA) del path crítico para domótica simple

**Problema:** ECAPA-TDNN (speaker ID) corre en la línea crítica (`command_processor.py:119-225`). Para transcripciones cortas (<100ms, típico de "prendé la luz"), ECAPA es el cuello de botella (−50/−80ms).

**Cambio:**
- Guard por intent: para intents de domótica simple (`turn_on`/`turn_off`/etc.), **no** ejecutar speaker ID en el path crítico; despacharlo a background post-dispatch (la acción HA no necesita identidad).
- Mantener ECAPA **sincrónico** solo cuando el intent requiere `voice_auth` (acciones sensibles) o cuando el contexto multi-usuario lo exija.
- Reusar el cache de embedding existente (TTL=60s) para no recomputar.

**Interfaz:** introducir un predicado `_requires_speaker_id(intent) -> bool` en `command_processor.py`; el resto del pipeline ya tolera `speaker=None` (verificado en `command_processor.py:44-46`).

**Beneficio:** −50/−80ms en el caso dominante (domótica), sin perder auth donde importa.

### Testing FASE A

- `tests/unit/vectordb/`: `search_command` async devuelve mismos resultados; no bloquea (test con `asyncio` + `to_thread` mockeado).
- `tests/unit/`: warmup inicializa el embedder y un comando posterior no paga cold start (medir con `time` mockeado o flag).
- `tests/unit/pipeline/`: speaker ID se difiere para intent de domótica y se ejecuta sync para intent con `voice_auth`. `speaker=None` no rompe el dispatch.

---

## 4. FASE B — Diseño del reasoner cloud

### B1 — Path chat-completions para endpoints cloud *(blocker)*

**Problema:** el `HttpReasoner` usa `completions.create(prompt=...)` → `/v1/completions`. MiniMax solo documenta `/v1/chat/completions` (`messages=...`).

**Cambio:**
- Agregar a `HttpReasoner` un modo `api_style: "chat" | "completions"` (default `"completions"` para backward-compat con ik_llama local).
- Cuando `api_style == "chat"`: construir `messages=[{"role": "user", "content": prompt}]` (y `system` si el prompt ya trae system separable) y llamar `self._client.chat.completions.create(messages=..., stream=...)`. Parsear `resp.choices[0].message.content` (y `delta.content` en streaming) en lugar de `resp.choices[0].text`.
- Aislar el parseo de respuesta en un helper (`_extract_text(resp, api_style)`) para no duplicar la lógica en los 5 call sites de completin.

**Interfaz:** `HttpReasoner(base_url, model, api_style="completions", api_key_env=None, idle_timeout_s=8.0, timeout_s=..., verify_ssl=True)`.

### B2 — Resolución de API key explícita

**Problema:** `_resolve_api_key()` infiere la env var por puerto. `https://api.minimax.io/v1` tiene `port=None` → cae al fallback `LLAMA_API_KEY or VLLM_API_KEY` → usaría una key local equivocada o el sentinel `"not-used"` (con warning, pero igual fallaría con 401).

**Cambio:**
- Firma nueva: `_resolve_api_key(base_url: str, api_key_env: str | None = None) -> str`.
- Si `api_key_env` está seteado → `os.getenv(api_key_env)` directo; si la env var no existe, **fallar ruidosamente al load-time** (no sentinel silencioso para endpoints cloud).
- Si `api_key_env is None` → heurística legacy por puerto (backward-compat con `:8101`/`:8200`).
- `HttpReasoner` y `FastRouter` reciben y guardan `api_key_env`, lo pasan a `_resolve_api_key`.

**Nota:** se ejecuta en load-time (`_try_connect()` `:433`, `FastRouter.load()` `:618`), **no** en request-time → cero impacto en el fast path de 300ms.

### B3 — Config del endpoint cloud

En `settings.yaml`, `llm.failover.endpoints`. **Reemplazo total:** el endpoint `reasoner_72b` (GLM-Air `:8200`) sale de la cadena del reasoner y entra `reasoner_cloud` como **priority 1** del slow path.

```yaml
llm:
  failover:
    endpoints:
      # Fast router local — SE MANTIENE (gate/router del fast path, no depende de internet)
      - id: fast_router_7b
        kind: fast_router
        priority: 1
        base_url: "http://127.0.0.1:8101/v1"
        model: "/home/kza/kza/models/Qwen2.5-7B-Instruct-Q4_K_M/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

      # Reasoner del slow path — REEMPLAZADO: GLM-Air local → MiniMax cloud
      - id: reasoner_cloud
        kind: http_reasoner
        priority: 1                         # primary del slow path (reemplazo total)
        base_url: "https://api.minimax.io/v1"
        api_style: "chat"                   # B1
        api_key_env: "MINIMAX_API_KEY"      # B2
        model: "MiniMax-M2.7-highspeed"
        timeout_s: 60.0                      # B5: ahora SÍ se respeta
        idle_timeout_s: 25.0                 # B4: cloud cold ~10-20s al primer token
        verify_ssl: true                     # B5
```

- `.env.example` y `/home/kza/secrets/.env` (chmod 600): documentar `MINIMAX_API_KEY`. **No** se crea archivo de config nuevo (regla de `CLAUDE.md`).
- El bloque `reasoner` global (`settings.yaml:247-309`) y el path local GLM-Air quedan **comentados con fecha y razón** (no borrados), por si se necesita rollback de emergencia.

### B4 — `idle_timeout_s` por endpoint + construcción dinámica de clients

**Problema:** `HttpReasoner` toma `idle_timeout_s` del bloque global (8.0s fijo); el watchdog abortaría un cloud que tarda 10-20s al primer token. Además los clients están hardcodeados en `main.py:320-322`.

**Cambio:**
- En `router_factory.py` / `main.py`: iterar `failover.endpoints` y construir un `HttpReasoner`/`FastRouter` **por endpoint**, cada uno con su `base_url`, `api_key_env`, `api_style`, `idle_timeout_s`, `timeout_s`, `verify_ssl`. Eliminar el dict hardcodeado.
- El `LLMRouter` recibe el dict construido dinámicamente (mismo contrato que hoy, pero poblado por factory).

**Interfaz:** `build_clients(config) -> dict[str, BaseLLMClient]`. Mantiene el patrón de DI por constructor.

### B5 — Respetar `ep.timeout_s` + clasificar errores de red

**Problema:**
- `ep.timeout_s` (`types.py:53`) no se usa en `router.py:100` → el slow path siempre usa el 5.0s hardcodeado de `dispatcher.py:695`, insuficiente para TLS handshake (100-300ms) + TTFT cold (hasta 2s).
- `ssl.SSLError`/`socket.gaierror` no están en `classify_error()` → se marcan PERMANENT → **bloquean el failover** en vez de permitir cooldown/retry.

**Cambio:**
- En `LLMRouter.complete()`: envolver `ep.client.complete(...)` en `asyncio.wait_for(..., timeout=ep.timeout_s)` cuando esté definido.
- Subir el timeout del slow path en `dispatcher.py:695` a un valor coherente con red (configurable, p.ej. `slow_path_timeout_s` ~ `ep.timeout_s + headroom`), o derivarlo del `ep.timeout_s` del endpoint activo.
- `classify_error()`: mapear `ssl.SSLError`, `socket.gaierror`, y errores de conexión httpx (`httpx.ConnectError`, `httpx.ReadTimeout`) a categoría **TIMEOUT/transient** (failover-worthy).
- Health-check de startup: ping al endpoint cloud antes de meterlo a la cadena; si falla, loguear WARN claro (no crashear el arranque — el fast path debe seguir vivo).

### B6 — Privacidad y secrets (obligatorio)

**Problema:** mover el reasoner a cloud manda al tercero: texto verbatim del usuario, historial (10 turnos), nombre, zona y estado del hogar (`context_manager.py:538-607`). Rompe la premisa "100% local". Los contextos se guardan plaintext en SQLite.

**Cambio (mínimo viable, sin mutilar el reasoning):**
- **Consentimiento explícito en config:** `reasoner.cloud.consent: true` requerido para que el endpoint cloud se active. Sin consent → no se instancia el cliente cloud y se loguea por qué.
- **Log prominente de salida de datos:** cada request al cloud loguea a nivel INFO/WARN: `"Cloud reasoning: enviando ~N chars, M turnos de historial, estado_hogar=<bool> a MiniMax (api.minimax.io)"`. Transparencia, no spam (un log por request de slow path; el slow path es raro).
- **Opción `strip_home_state_for_cloud: true`:** no enviar el estado del hogar al cloud (la parte más sensible) salvo que el comando lo requiera.
- **Sanitización de logs:** extender `core/logging.py:107-110` para redactar también query params de auth (`?api_key=`, `&token=`) además de `Bearer`. Test en `tests/safety/test_api_no_secret_exposure.py`.
- **(Diferible, anotado como follow-up):** cifrado at-rest de `./data/contexts/` (Fernet + `CONTEXT_ENCRYPTION_KEY`). No bloquea esta entrega.

### Testing FASE B

- `tests/unit/llm/test_reasoner.py`: `api_style="chat"` arma `messages` y parsea `message.content`; `api_style="completions"` mantiene comportamiento legacy. Mock del cliente OpenAI.
- `tests/unit/llm/test_reasoner.py`: `_resolve_api_key` con `api_key_env` explícito usa la env var correcta; falla ruidoso si falta; legacy por puerto intacto.
- `tests/unit/llm/test_router.py`: `ep.timeout_s` se respeta (mock que tarda > timeout → `asyncio.TimeoutError` → failover). `ssl.SSLError`/`gaierror` → categoría transient → rotación, no PERMANENT.
- `tests/unit/llm/`: `build_clients` construye un cliente por endpoint con sus params; cloud sin `consent` no se instancia.
- `tests/safety/test_api_no_secret_exposure.py`: logs no contienen la API key ni en Bearer ni en query params.
- `tests/integration/` (mock del endpoint MiniMax con un servidor OpenAI-compat fake): slow path completa contra el cloud mockeado; con cloud caído, el slow path falla de forma controlada y el dispatcher **notifica al usuario** (issue conocido C).

---

## 5. Flujo de datos (resumen)

```
Fast path (sin cambios funcionales, A1-A3 reducen latencia):
  Mic → Wake(CPU) → STT(cuda:0) → [Speaker ID diferido A3] → dispatcher
       → VectorSearch async (A1, warmup A2) → acción HA → TTS

Slow path (B1-B6, reemplazo del reasoner):
  dispatcher → LLMRouter → reasoner_cloud (MiniMax, chat-completions, Bearer)
       → [consent gate + log de salida B6] → respuesta → context → TTS
       (si cloud falla: error clasificado transient → cooldown; sin fallback local
        → dispatcher notifica al usuario)
```

---

## 6. Riesgos y mitigaciones

| Riesgo | Severidad | Mitigación |
|---|---|---|
| **Reemplazo total: sin reasoner si cae internet/MiniMax** | Alta (aceptado) | El fast path (domótica) sigue 100% local y funcional. El slow path falla de forma controlada y notifica al usuario (no cuelga). Rollback: descomentar GLM-Air en config. |
| **Privacidad: datos de voz a un tercero** | Alta | Consent gate + log de salida + `strip_home_state` (B6). Decisión informada del dueño. |
| **MiniMax no soporta `/v1/completions` legacy** | Alta (blocker) | B1: path chat-completions. |
| **Timeouts de red revientan el slow path** | Media | B5: respetar `ep.timeout_s` + headroom de red. |
| **Errores TLS/DNS bloquean failover** | Media | B5: clasificarlos como transient. |
| **API key filtrada en logs** | Media | B6: sanitización + test. |
| **`search_command` async rompe call sites sync** | Baja | Auditar call sites con grep antes de editar; migrar todos a `await`. |

---

## 7. Secuencia de implementación

1. **FASE A** (independiente, seguro): A1 → A2 → A3, con tests, en branch propio. Validar latencia con `tools/benchmark_latency.py`.
2. **FASE B** (depende de tener `MINIMAX_API_KEY`): B2 → B1 → B4 → B5 → B3 → B6, con tests. Validar contra endpoint MiniMax real (o mock OpenAI-compat) antes de deployar al server.
3. Deploy al server `kza@192.168.1.2` siguiendo el patrón de drift (comparar hashes, verificar `kza-voice.service` activo post-deploy).

Cada fase es un PR independiente; FASE A puede mergear sin esperar a FASE B.
