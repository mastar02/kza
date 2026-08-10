# Reasoner slow path vía Hermes Agent + ChatGPT (Codex OAuth) — Pieza 1: cambio de backend

## 1. Motivación

El slow path de KZA (razonamiento complejo, `src/llm/reasoner.py::HttpReasoner`) manda hoy sus
prompts al gateway `:8200` → MiniMax cloud, con facturación por token. Se propone reemplazarlo
por **Hermes Agent** (Nous Research, MIT), autenticado contra la cuenta de ChatGPT del usuario vía
el flujo OAuth de OpenAI Codex (`hermes auth add openai-codex`) — el mismo device-code flow que
usa el propio Codex CLI de OpenAI. Motivación combinada: costo (cuota de la suscripción ya pagada
en vez de facturación por token), calidad de razonamiento (GPT-5.x vs MiniMax-M2.7) y, a futuro
(Pieza 2, fuera de alcance de este documento), el *wiring* de KZA hacia las herramientas nativas
del agente — las herramientas en sí ya están activas desde esta Pieza 1, ver corrección en §2.

## 2. Objetivo / No-objetivos

**Objetivo:** el slow path sigue mandando el mismo prompt/contexto que manda hoy y recibe texto de
vuelta — ningún cambio en `MultiUserOrchestrator._process_llm_request` ni en cómo se arma el
prompt. Solo cambia quién genera la respuesta.

**No-objetivos (explícitos):**
- **KZA *wiring into* el tool-calling de Hermes** (manejo estructurado de request/response de
  herramientas del lado de KZA) — Pieza 2, spec aparte, después de ver esta pieza funcionando en
  producción. ⚠️ **Corrección post-review final (2026-08-10):** esto NO significa que `-z` corra
  sin herramientas hoy. `hermes -z` invoca el **mismo agente completo**, con **todo su toolset por
  defecto activo** (acceso a archivos, terminal/shell, browsing web) — la doc de Nous Research lo
  dice explícito: *"Same agent, same tools, same skills — just strips every interactive/cosmetic
  layer"*. No hay flag `--toolsets`/`--no-tools`/`--sandbox` documentado que aplique a `-z`
  específicamente (esos flags existen solo para `hermes chat`). Lo que "Pieza 2" agregaría es que
  KZA *use* ese tool-calling de forma estructurada — no la mera existencia de las herramientas, que
  ya está activa desde Pieza 1. **Riesgo aceptado explícitamente por el usuario** dado que el
  deploy es un server doméstico privado sin exposición a red externa; no hay mecanismo en código
  para restringir el toolset de `-z` (no existe ese flag), así que no se intenta.
- Fallback automático a MiniMax si Hermes falla — decisión explícita: reemplazo total. Ver §6 y §9.
- Streaming token-por-token real — no existe en el mecanismo elegido (§3). El slow path ya tolera
  reasoners no-streaming.

## 3. Contexto: por qué no `hermes proxy`

Hermes Agent tiene dos mecanismos distintos, y solo uno de los dos sirve acá:

- **`hermes proxy start --provider <nous|xai>`**: servidor HTTP local OpenAI-compatible. Hubiera
  sido el enganche más limpio (`HttpReasoner` tal cual, solo cambiando `http_base_url`), pero la
  doc oficial de Nous Research (`/docs/user-guide/features/subscription-proxy`) es explícita:
  *"Currently shipped: `nous` (Nous Portal) and `xai` (xAI / Grok)"* — **no hay adaptador para
  `openai-codex`/ChatGPT**. Es extensible (interfaz `UpstreamAdapter`), pero construirlo es
  trabajo nuestro no documentado como existente. Descartado.
- **`hermes -z "<prompt>" --provider openai-codex`**: el "purest scripted entry point" del CLI —
  *"single prompt in, final response text out, nothing else on stdout or stderr"*. No expone HTTP,
  es invocación de proceso por request. Es el mecanismo elegido.

  ⚠️ **Aclaración post-review final (2026-08-10):** esa cita describe la *forma del I/O*
  (stdout/stderr limpios, un prompt entra, un texto final sale) — **no** dice nada sobre qué hace
  el agente puertas adentro para llegar a esa respuesta. No hay que leerla como "text in, text
  out, sin herramientas". Como se corrige en §2, `-z` corre el agente completo con su toolset
  normal activo (archivos, terminal, web); lo único que cambia frente a `hermes chat` es la capa
  interactiva/cosmética del CLI, no el acceso a herramientas.

Consecuencia: no hay puerto nuevo que reservar en el sub-rango 9500-9599, no hay servicio HTTP que
correr — pero tampoco hay streaming, y cada request paga el overhead de arrancar el proceso
`hermes`.

## 4. Arquitectura

Clase nueva `HermesCliReasoner` en `src/llm/hermes_reasoner.py`, duck-typed idéntica a
`HttpReasoner`/`LLMReasoner` (mismo patrón "drop-in" ya establecido en el proyecto):

- `load()` — **síncrono** (igual que `HttpReasoner.load()`; `main.py` lo llama sin `await`, dentro
  de un `try/except` en el boot). Corre `subprocess.run(["hermes", "auth", "status"], ...)`
  bloqueante — es un chequeo único al arranque, no en el hot path — y valida que reporta
  `openai-codex` con credenciales válidas. Falla ruidosamente si no (mismo espíritu que
  `_resolve_api_key`: un deploy mal configurado se ve al boot, no como un 401 opaco en producción).
- `__call__` / `generate()` / `generate_stream()` — **síncronos** (el slow path ya tolera un worker
  bloqueado, mismo espíritu que `HttpReasoner.generate_stream()`). Por dentro arman el comando
  `hermes -z "<prompt>" --provider openai-codex [-m <hermes_model>] --usage-file <tmpfile>` y lo
  corren con `subprocess.Popen(...)` + `proc.communicate(timeout=...)` **bloqueante** (no
  `asyncio.create_subprocess_exec` — ese API async se evaluó y se descartó durante la
  implementación a favor de reusar el mecanismo de process-group-kill de forma síncrona; ver
  `complete()` abajo para cómo se evita bloquear el event loop en el único call site que lo
  necesita). Devuelven el stdout decodificado y trimeado como texto de respuesta.
  `generate_stream()` en particular yield-ea un único chunk con el texto completo una vez que el
  subproceso termina — sin streaming real debajo (mismo mecanismo síncrono que `__call__`).
  `MultiUserOrchestrator._process_llm_request` ya tiene el fallback a `generate()` para reasoners
  sin streaming real (documentado en el docstring actual de `HttpReasoner.generate`) — no hace
  falta tocar el orchestrator.
- `complete()` — la única variante **async** de las cuatro (es la que usa `LLMRouter`/el path que
  sí corre dentro del event loop). Envuelve la llamada síncrona completa (`_run()`, que hace
  `Popen`+`communicate`) en `asyncio.to_thread(...)` — mismo patrón que ya usa
  `HttpReasoner.complete()` — para no bloquear el loop mientras el subproceso corre en un thread
  aparte.
- Métricas: `--usage-file` escribe un JSON (tokens, costo, modelo, provider, session_id,
  completed/failed) por corrida, incluso en fallo. Se parsea a `_last_metrics` / se reenvía a
  `_metrics_tracker` — mismo patrón que ya usan `HttpReasoner`/`FastRouter`.
- Timeout: `proc.communicate(timeout=self.timeout_s)` **síncrono** (no `asyncio.wait_for` — ese
  mecanismo async no aplica a un `Popen.communicate()` bloqueante); al vencer, captura
  `subprocess.TimeoutExpired` y mata el process group completo vía `os.killpg` (`proc.kill()` no
  alcanza si `hermes` forkea hijos) para no dejar procesos huérfanos colgados del slow path.

## 5. Config (`reasoner:` en `settings.yaml`)

```yaml
reasoner:
  mode: "hermes_cli"           # nuevo, junto a "http"/"local" existentes
  hermes_binary_path: "hermes" # override a ruta absoluta si el systemd --user no hereda PATH
  hermes_provider: "openai-codex"
  hermes_model: null            # sin pin — usa el default de Hermes/Codex
  hermes_timeout_s: 90          # vs 60s de MiniMax hoy — suma el arranque del proceso
  cloud:
    consent: true                # misma key, reusada — ver §6
```

## 6. Gate de privacidad — branch nuevo, no cosmético

`is_cloud_endpoint` (`src/llm/cloud_consent.py`) clasifica por host de una URL. Un subproceso CLI
no tiene URL — `hermes_cli` tiene que tratarse como **cloud incondicional**, gateado solo por
`reasoner.cloud.consent`, sin heurística de puerto/host de por medio.

Cambio en `resolve_reasoner_gate`:

```python
if reasoner_mode == "http":
    return resolve_http_reasoner_base_url(reasoner_config, default_local_url)
if reasoner_mode == "hermes_cli":
    return cloud_reasoner_allowed(reasoner_config), None  # no hay base_url que resolver
return (
    cloud_reasoner_allowed(reasoner_config),
    reasoner_config.get("http_base_url", default_local_url),
)
```

`cloud_reasoner_allowed` ya funciona por `is_cloud_endpoint(base_url)` — para `hermes_cli` esa
función nunca se llama; el nuevo branch consulta `reasoner.cloud.consent` directo, fail-closed por
default (mismo comportamiento que hoy si la key falta). Tests explícitos con el mismo rigor que el
gate actual (PR #12: *"el fail-closed lo sostiene SOLO el orden gate→fallback"*).

## 7. Wiring en `main.py`

Tercer branch junto a `mode == "http"` / el `else` de `LLMReasoner` local:

```python
elif reasoner_mode == "hermes_cli":
    if gate_allowed:
        llm = HermesCliReasoner(
            binary_path=reasoner_config.get("hermes_binary_path", "hermes"),
            provider=reasoner_config.get("hermes_provider", "openai-codex"),
            model=reasoner_config.get("hermes_model"),
            timeout_s=reasoner_config.get("hermes_timeout_s", 90),
        )
        try:
            llm.load()
        except Exception as e:
            logger.error(f"HermesCliReasoner no pudo inicializarse: {e}")
            llm = None
    else:
        logger.warning("Reasoner cloud bloqueado por falta de consent — slow path sin reasoner.")
        llm = None
```

**Compactor** (`compaction_reasoner`, hoy siempre `HttpReasoner`): con `mode == "hermes_cli"` y
`gate_allowed`, el compactor también usa `HermesCliReasoner` (instancia propia, mismo patrón de
pool separado que ya existe) — decisión explícita: se prioriza calidad de compactación pareja
sobre el costo de un arranque de proceso extra por turno compactado. Si el gate está bloqueado,
degrada al fallback local existente (`:8101`), sin cambios ahí.

## 8. Degradado / manejo de errores

Sin fallback a MiniMax (reemplazo total, decisión explícita del usuario). Exit code ≠ 0 o timeout
del subprocess → excepción propagada tal cual llega — el orchestrator ya maneja "reasoner cloud no
responde" de forma provider-agnóstica, no hace falta código nuevo ahí.

⚠️ **Riesgo aceptado sin red de contención:** si el token OAuth de `hermes auth add openai-codex`
expira y necesita re-login manual, o Codex devuelve rate-limit, **el slow path completo se cae**
hasta que alguien lo note y corrija. Mismo espíritu que `cloud.consent` hoy — una decisión
informada, no un bug. Mitigación mínima: loguear con nivel ERROR distinguible (`hermes auth
status` en el mensaje de excepción) para que sea diagnosticable rápido, no una degradación
silenciosa.

## 9. Deployment

- Instalar `hermes` en el server kza (`192.168.1.2`), usuario `kza`, vía el instalador oficial
  (`curl -fsSL .../install.sh | bash`).
- Bootstrap de auth, **una sola vez, a mano, por SSH**: `hermes auth add openai-codex` abre un
  device-code flow (URL + código) que se completa desde un browser aparte (celular o laptop).
  Credenciales quedan en `~/.hermes/auth.json` bajo el usuario `kza`.
- **Sin systemd service nuevo** — es subproceso por request, no daemon. Sin puerto nuevo en el
  sub-rango 9500-9599.
- `hermes_binary_path` en config apunta a la ruta absoluta si el entorno de `kza-voice.service`
  (systemd --user) no hereda el `PATH` del shell interactivo donde corrió el instalador.

## 10. Testing

TDD por convención del proyecto:

- `tests/unit/llm/test_hermes_cli_reasoner.py` — mockea `asyncio.create_subprocess_exec`
  (patrón ya usado en `test_httpreasoner_generate.py` para el cliente HTTP mockeado). Cubre:
  parseo de `--usage-file`, timeout + kill de process group, texto vacío / exit≠0 → excepción,
  construcción correcta del comando con/sin `hermes_model`.
- `tests/unit/llm/test_cloud_consent.py` — nuevo caso para el branch `hermes_cli` de
  `resolve_reasoner_gate` (consent=true/false, sin URL de por medio).
- `tests/unit/llm/test_main_cloud_client_wiring.py` — cubre el wiring nuevo en `main.py` (llm=None
  si gate bloqueado, instancia correcta si permitido).
- **Smoke test pre-flip en producción** (no automatizado, manual): comparar un set chico de
  consultas es-AR de domótica representativas contra la respuesta actual de MiniMax — calidad y
  latencia — antes de cambiar `mode` en el `settings.yaml` del server. Lección ya aprendida en este
  proyecto: no evaluar sobre el mismo set con el que se construyó el criterio.
- Rollback: `mode: "http"` de vuelta — un solo valor de config, mismo patrón que el bloque de
  rollback ya comentado en `settings.yaml` para el GLM-Air local.

## 11. Criterios de éxito

- Suite completa verde con el nuevo módulo + branches de config.
- Smoke test manual: calidad de respuesta en español (es-AR) percibida como igual o mejor que
  MiniMax-M2.7-highspeed en al menos las consultas del set de humo.
- Latencia end-to-end del slow path (arranque de proceso incluido) dentro de un orden de magnitud
  similar al actual (no hay objetivo numérico duro — el slow path ya es "segundos", no <300ms).
- Al menos una corrida real en producción sin fallos de auth/timeout antes de considerar esto
  cerrado (no solo tests unitarios con subprocess mockeado).

## 12. Riesgos y dependencias

- **Sin fallback** (§8) — riesgo aceptado explícitamente, no accidental.
- **Refresh de token OAuth no verificado empíricamente** — la doc menciona que un refresh fallido
  "se pone en cuarentena" pero no está claro cuánto dura una sesión antes de pedir re-login manual.
  Primera semana de producción es también el período de aprendizaje de este comportamiento.
- **Cuota de Codex bajo uso real 24/7** — no hay datos de cuánto consume un asistente de voz
  doméstico contra el límite de la suscripción ChatGPT del usuario; puede aparecer rate-limit en
  producción que el smoke test (volumen bajo) no capture.
- **Overhead de arranque de proceso por request** — no medido todavía; si resulta alto (el binario
  `hermes` puede cargar skills/plugins/config al arrancar), puede empujar la latencia del slow path
  más de lo esperado. A medir en el smoke test.

## 13. Relacionado

Pieza 2 (fuera de alcance): cablear el manejo estructurado del tool-calling nativo de Hermes
(búsqueda web, ejecución de código) del lado de KZA — las herramientas en sí ya corren activas
desde Pieza 1 (§2), esto es sobre *usarlas* desde el orchestrator, no sobre habilitarlas. Requiere
esta pieza funcionando y midiendo bien en producción primero, mismo patrón secuencial que Pieza A →
Pieza B en la campaña de fidelidad del ambient
(`docs/superpowers/specs/2026-08-05-fidelidad-transcripcion-ambient-design.md`).
