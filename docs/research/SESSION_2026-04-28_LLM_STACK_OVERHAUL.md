# Sesión 2026-04-28 — Overhaul del stack LLM y consolidación de branches

> **Resumen:** Sesión larga que tocó tres ejes simultáneos:
> 1. Bug operativo: "no se apaga la luz del escritorio" → diagnóstico + fix + cache stale
> 2. Consolidación de 3 branches divergentes (main, feature/wake-tv-filter-fixes, wip/server-snapshot) → main único
> 3. Reemplazo del LLM 72B (caído, modelo no descargado) por Qwen3-30B-A3B con engine ik_llama.cpp
>
> **Estado final:** server `kza@192.168.1.2` corriendo `origin/main` con Qwen3-30B-A3B Q5_K_M en ik_llama.cpp a ~58 tok/s, fix del cache idempotent, 22 commits del wake/TV regression mergeados, plan #1 OpenClaw (LLMRouter failover) en producción.

---

## 1. Setup e infraestructura

### 1.1 Acceso al server (faltaba en config)

```bash
# Agregado a ~/.ssh/config
Host kza
    HostName 192.168.1.2
    User kza
```

Memoria persistente: `reference_kza_server_ssh.md` para que próximas sesiones encuentren el host directo.

### 1.2 Estado inicial encontrado

- `kza-72b.service` en restart-loop **10 665 veces** — modelo Qwen2.5-72B Q8_0 nunca terminó de descargar (`.lock` y `.metadata` en HF cache, sin `.gguf` final)
- `kza-voice.service` activo con WIP no commiteado (~1300 LOC vs origin/main)
- 3 branches divergentes:
  - `origin/main` (canonical)
  - `origin/feature/wake-tv-filter-fixes` (22 commits limpios sin mergear)
  - Server con WT divergente del feature (en vivo)

---

## 2. Bug operativo: "no apaga la luz del escritorio"

### 2.1 Diagnóstico

**Trace del comando (10:12:55):**
```
🔥 Wake word 'nexa' detectado en: 'Nexa apagá la luz del escritorio.'
[LLMRouter SKIP] partial confiable: intent=turn_off entity=light room=escritorio
[OK] Total: 11ms ... home_assistant=0
WebSocket HA[calls] conectado y autenticado    ← reconexión sospechosa
```

**Root cause encontrado** en `dispatcher.py:507-539` (S6 cache check):

```python
# El skip idempotent saltaba la HA call si cache decía "ya está en target_state"
if cached.get("state") == target_state:
    return DispatchResult(action={**command, "already_in_state": True}, ...)
    # ↑ silent skip + sin TTS feedback
```

**Combo de fallos:**
1. Hue bridge tiene lag de varios segundos entre estado real y cache HA via WS
2. Cache decía `light.escritorio: off` mientras la luz física estaba ON
3. Skip idempotent → KZA no manda call HA → luz no cambia
4. Log del skip era `logger.debug` → invisible al user
5. Fire-and-forget no se ejecuta porque hubo skip antes → sin TTS feedback

### 2.2 Fix aplicado

**Eliminado el skip idempotent.** La HA call es idempotent server-side, no hay costo, garantiza sync con realidad.

```python
# dispatcher.py — versión nueva
if command:
    asyncio.create_task(self._fire_and_reconcile_ha(command))
    timings["home_assistant"] = 0.0
    return DispatchResult(...fire_and_forget=True...)
```

Commit: `1c2afbd fix(dispatcher): remove idempotent cache skip for HA actuators`

### 2.3 Análisis de latencia Hue (lo que sentías como "tarda mucho")

Mediciones reales del bridge:

| Capa | Latencia |
|---|---:|
| LAN ping bridge `192.168.1.101` | 0.26 ms |
| HA → bridge HTTP | ~95 ms |
| Bridge → bombilla Zigbee | ~1.0-1.2 s |
| **Total comando único** | **~1.5 s** |

**Cuello físico está en Hue Bridge → Zigbee → bombilla.** Confirmado normal. La sensación de "antes era instantáneo" venía del cache skip (silent + sin call), no de cambio real de hardware.

Mitigación futura: migrar bombillas a Zigbee2MQTT directo (Notion 9.6 ya tiene infra), saltar bridge Hue. Estimado: ~150-300ms vs 1.2s actual.

---

## 3. Performance del pipeline

### 3.1 WS [calls] persistent

`ha_client.start_state_sync` solo pre-conectaba WS [events]. WS [calls] era **lazy** — primer comando pagaba ~90ms de SSL+auth handshake.

**Fix:** pre-conectar [calls] en startup también. Heartbeat=30s mantiene viva.

```python
await self._wait_for_ws_ready(max_attempts=3, backoff_s=2.0)
try:
    await self.connect_websocket()  # NUEVO: pre-conectar [calls]
except Exception as e:
    logger.warning(f"WS calls warmup falló (seguirá lazy): {e}")
```

Verificable en logs:
```
WebSocket HA[events] conectado y autenticado
WebSocket HA[calls] conectado y autenticado    ← ambos al startup
HA state prefetch cache: sync loop arrancado
```

### 3.2 silence_end_ms 700 → 500

Reverte ajuste 2026-04-23. Para grammar match el early_dispatch corta antes y este wait no aplica. Para queries sin grammar (LLM libre), 500ms es suficiente y el follow-up window 4s cubre pausas naturales.

Commit: `006f1c5 perf(ha,wake): WS calls persistente al startup + silence_end_ms 700→500`

---

## 4. Consolidación de branches

### 4.1 Topología antes

```
main (8820c84)
├── feature/wake-tv-filter-fixes (25836d0)  ← 22 commits limpios
└── server WT (3cefd80 + 1300 LOC sin commitear)
```

### 4.2 Snapshot del server preservado

Capturé el WT del server como branch antes de tocar nada:
- `git diff HEAD > /tmp/server.patch` desde server
- Apliqué el patch a una branch nueva en laptop
- Push como `wip/server-snapshot-2026-04-28` (commit `173181f`) para no perder trabajo en vivo

### 4.3 Merge consolidado

```
main + plan #1 OpenClaw (5 commits) ──────┐
                                           ├──→ merge/consolidate
feature/wake-tv-filter-fixes (22 commits)──┘
                                           │
wip/server-snapshot fixes ─────────────────┘ (cherry-pick deltas)
```

**Conflictos resueltos:**
- `src/llm/reasoner.py`: feature agregaba `fallback_base_url` simple en `HttpReasoner.load()`. Plan #1 LLMRouter supersede esa funcionalidad — descarté el simple fallback.
- `src/main.py`: feature agregaba `LLMCommandRouter` (NLU Opción 2), plan #1 agregaba `LLMRouter` (failover). Mantuve **ambos** — son features ortogonales.
- `dispatcher.py`: cache skip removal aplicado manualmente + agregué método `_fire_and_reconcile_ha` que el wip tenía pero feature no.

### 4.4 Resultado

50/50 tests verdes, push a main. Branches `wip/server-snapshot-2026-04-28` y `feature/wake-tv-filter-fixes` borradas (origin + local).

---

## 5. Plan #1 OpenClaw — LLM failover finalizado

Continuación del trabajo iniciado el 2026-04-26. Faltaban:

### 5.1 Componentes nuevos

| Archivo | Responsabilidad | Tests |
|---|---|---|
| `src/llm/idle_watchdog.py` | Aborta stream si no llega chunk en N seg | 6/6 ✓ |
| `src/llm/router_factory.py` | Construye LLMRouter desde dict de config | 5/5 ✓ |
| `tests/unit/llm/test_types.py` | Smoke tests de types | 7/7 ✓ |
| `tests/integration/test_llm_failover_e2e.py` | E2E: primary falla → secondary succeeds | 4/4 ✓ |

### 5.2 Wire-up

- `src/llm/__init__.py`: exports actualizados (`LLMRouter`, `build_llm_router`, `IdleTimeoutError`, etc.)
- `src/llm/reasoner.py`:
  - `FastRouter.complete()` — async, propaga excepciones (vs `generate()` que las swallowea)
  - `HttpReasoner.complete()` — soporta `idle_timeout_s` con stream + `idle_watchdog`
  - `HttpReasoner.__init__(idle_timeout_s=...)` parámetro nuevo
- `src/main.py:308-327`: `build_llm_router(failover_cfg, clients)` post-creación del FastRouter + LLM
- `src/orchestrator/dispatcher.py:561`: detecta `hasattr(router, 'complete')` y prefiere API nueva, mantiene fallback `generate()` para compat
- `config/settings.yaml`: bloque `llm.failover` con 2 endpoints (`fast_router_7b` prio 1, `reasoner_72b` prio 2), cooldowns persist en disco

### 5.3 Robustez ante endpoint caído

Bug encontrado durante deploy: cuando el 72B no estaba disponible al startup (modelo sin descargar), `build_llm_router()` fallaba duro porque la config refería a `reasoner_72b` sin cliente.

**Fix:** factory ahora skipea endpoints sin cliente con warning y solo falla si **ningún** endpoint queda disponible.

```python
if client is None:
    logger.warning(f"[LLMRouter] skip endpoint {ep_id!r} — sin cliente")
    skipped.append(ep_id)
    continue
```

Commit: `4fe89fa fix(llm): tolerate missing clients — skip endpoint instead of fail`

---

## 6. Reemplazo del LLM: Qwen2.5-72B (caído) → Qwen3-30B-A3B

### 6.1 Decisión de modelo

72B Q8_0 estaba **caído desde días atrás** (modelo no descargado). El usuario decidió no re-descargar (37 GB) y buscar alternativa más rápida.

Research detallado en `docs/superpowers/plans/2026-04-26-openclaw-roadmap.md` y memorias `feedback_*` y `project_*`.

**Pick: Qwen3-30B-A3B-Instruct-2507** (MoE 30B totales, 3B activos por token).

Razón clave: en CPU el bottleneck es **bandwidth × params_activos**, no params totales. MoE con 3B activos por token usa la misma banda de memoria que un dense 3B, pero tiene la calidad de un dense 30B+. Calculado teórico:

```
Q5_K_M weights por token: ~2.27 GB (3B activos × 5.5 bits ÷ 8)
DDR5-5600 8ch realista:   270 GB/s
Techo bandwidth-bound:    270 / 2.27 = ~119 tok/s
Realista (60-75% efic):    ~75-90 tok/s
```

### 6.2 Bench en CPU

Con `llama-cpp-python` (legacy):

| Modelo | TG sostenido |
|---|---:|
| Qwen2.5-72B Q8_0 (anterior) | ~2 tok/s (37s respuesta 60 tok) |
| Qwen3-30B-A3B Q5_K_M | **42.6 tok/s** (1.4s respuesta 60 tok) |
| Qwen3-30B-A3B Q4_K_M | 46.4 tok/s (+9% pero ❌ falla razonamiento) |

**~21× speedup** vs el 72B. Pero el Q4 falla matemática multi-paso (ver A/B abajo).

### 6.3 Build ik_llama.cpp y migración

`ik_llama.cpp` (fork por ikawrakow) tiene optimizaciones específicas para MoE en CPU. Build:

```bash
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_CUDA=OFF \
  -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON \
  -DLLAMA_BUILD_SERVER=ON
make -j24 llama-server llama-bench
```

Bench A/B con MISMO modelo Q5_K_M:

| Engine | TG bench | TG API real |
|---|---:|---:|
| llama-cpp-python | 42.6 tok/s | 36-46 tok/s |
| **ik_llama.cpp** | **58.6 tok/s** | **63 tok/s** |
| **Mejora neta** | **+38%** | **+36-75%** |

Service swap: `kza-72b.service` (legacy) → `kza-llm-ik.service` (mismo puerto :8200, OpenAI compat). Unit en `scripts/kza-llm-ik.service`.

### 6.4 Sweep de flags ik_llama (qué SÍ y qué NO ayuda)

Probadas todas las combinaciones. Resultados:

| Flag combo | TG tok/s | Verdict |
|---|---:|---|
| Baseline (24 threads) | 58.6 | — |
| `+ -fa on -ctk q8_0 -ctv q8_0` | 58.7 | ✅ aplicado (KV cache 50% más chico) |
| `+ -rtr 1` (runtime row-interleave) | 52.8 | ❌ degrada |
| `+ -fmoe 1` (fused MoE) | 59.4 | ~empate |
| `+ -rtr 1 -fmoe 1` | 57.3 | ❌ peor que baseline |
| `+ -ser 2,1` (smart expert reduction agresiva) | 65.0 | ~marginal, riesgo calidad |
| `+ -ser 4,1` | 65.3 | ~marginal |
| `--mlock --use_mlock` | igual | ✅ aplicado (no-swap) |
| **NUMA pinning** (`taskset -c 0-23`) | 58.65 | ❌ +0.2% irrelevante (single-NUMA) |
| `taskset -c 0-11` (12 threads) | 35.9 | ❌ -39% (menos cores) |
| `numactl --interleave=all` | 58.4 | ~empate |

**Conclusión:** flag tuning agotado. ~58 tok/s es el techo de Qwen3-30B-A3B Q5 en este hardware con ik_llama.cpp.

### 6.5 A/B blind: Q5 vs Q4_K_M (calidad razonamiento)

Mismas 8 preguntas en español rioplatense. Datos críticos:

| Caso | Q4_K_M (46 tok/s) | Q5_K_M (50 tok/s) |
|---|---|---|
| Math tanque 200L (canilla 8L/min, pérdida 3L/10min) | **❌ 40 min (mal: hizo 80-30)** | **✅ 26 min (bien: 80-3)** |
| Conversión 350°F → C | ✅ 177°C | ✅ 175°C |
| Domótica prudente | ✅ corto | ✅ + sugiere troubleshooting |
| Física hervor | ✅ correcto | ✅ + ejemplo concreto |
| Comparativa Spotify/YT | ⚠️ "más música local" (dudoso) | ✅ más prudente |
| Negación honesta | ✅ | ✅ |
| Termostato frío | ✅ práctico | ✅ empático |
| Apagar vs desconectar | ⚠️ ambiguo | ⚠️ ambiguo |

**Veredicto:** Q4 falla razonamiento multi-paso (predicho: +1.2% perplejidad vs FP16, vs Q5 +0.4-0.6%). El +9% de velocidad NO compensa el riesgo. **Quedamos con Q5_K_M.**

### 6.6 A/B blind: Q5_K_M vs ubergarm/mix-IQ4_K

Research recomendaba `ubergarm/Qwen3-30B-A3B-mix-IQ4_K` (PPL 9.118 vs Q5 9.07x — supuestamente MEJOR calidad y MÁS rápido).

**Resultado real:**

| Métrica | Q5_K_M | mix-IQ4_K |
|---|---:|---:|
| TG tok/s | 58.6 | 59.6 (+1.7%) |
| Math tanque (latencia respuesta) | 1.5s, 73 tok | **8.9s, 507 tok** ❌ |
| Domótica corta | 0.7s | **6.2s** ❌ |

**Cause root:** `mix-IQ4_K` es la variante **Thinking** (no Instruct). Genera 300-600 tokens de `<think>...</think>` antes de responder. Acierta razonamiento pero **5-10× más latencia**.

**Inaceptable para voice domotics.** Reverte a Q5_K_M.

Memoria: `project_ik_llama_deployed.md` documenta el descarte para evitar repetir.

### 6.7 Optimización TTFT con prompt cache RAM

Aplicado al unit:
```
-cram 4096                                            # 4GB cache RAM
-crs 0.3                                              # similarity 0.3 (default 0.5)
--prompt-cache /home/kza/cache/llm-prompt.bin        # disk persist (no efectivo en
                                                      # ik_llama-server, dejado por compat)
```

Bench con system prompt ~350 tokens:

| Llamada | TTFT |
|---|---:|
| Cold (post-restart) | 204 ms |
| Warm (mismo system prompt) | 161 ms (-22%) |
| Post-restart cold | 205 ms (cache disco no persiste) |

Win modesto hoy (~40ms ahorrados por query). **Crítico cuando integremos history conversacional largo** (sin cache, prompt 1000 tok = 1.8s prefill; con cache, ~50ms).

Commit: `eaaca5e perf(kza-llm-ik): add prompt cache RAM persistence + similarity match`

---

## 7. Performance final del slow path

```
Stack actual (post-sesión):
  Engine:        ik_llama.cpp 453a027
  Modelo:        Qwen3-30B-A3B-Instruct-2507 Q5_K_M (20 GB)
  Service:       kza-llm-ik.service en :8200
  TG:            ~58 tok/s sostenidos
  TTFT cold:     204 ms (system prompt ~350 tok)
  TTFT warm:     161 ms (prompt cache hit)
  RAM peak:      21 GB
```

**Comparativa total vs estado inicial:**

| Métrica | Inicial (72B caído) | Final (Qwen3 ik_llama) |
|---|---|---|
| Slow path disponible | ❌ servicio dead | ✅ activo |
| TG | ~2 tok/s (cuando funcionaba) | **58 tok/s (29×)** |
| Respuesta 60 tok | ~30s | **1.4s (21×)** |
| Calidad razonamiento | (ok) | comparable, en español |
| RAM uso LLM | 73 GB | 21 GB (-71%) |

---

## 8. Caminos a 100 tok/s — investigado y descartado

Research extensivo (en memorias). 100 tok/s NO alcanzable en este hardware manteniendo calidad Q5:

```
Techo bandwidth-bound físico: ~75-90 tok/s
Estamos al 75% del techo (58 / 78)
Para 100 tok/s harían falta:
  - DDR5-6400 (no upgradeable en Threadripper PRO 7965WX)
  - O EPYC 9004 (12 canales, ~460 GB/s)
  - O degradar calidad
```

**Descartado con datos:**
- `-rtr/-fmoe/-ser` — empate o peor en este modelo
- NUMA pinning — single-NUMA, +0.2% irrelevante
- Q4_K_M plano — falla razonamiento (A/B blind)
- mix-IQ4_K — variante thinking, 5-10× más latencia
- Speculative decoding clásico — net-negativo en MoE A3B (issue #21886)
- EAGLE/Medusa/Lookahead — implementaciones CUDA-only en 2026
- AMX — solo Intel Sapphire Rapids+, no AMD Zen 4

**Caminos válidos pendientes (no aplicados):**
- N-gram lookup decoding (`-lcd`) sobre corpus HA: +15-40% en outputs estructurados (JSON), 0% en chat libre
- Migrar Hue → Zigbee2MQTT directo: -800ms en latencia bombilla (no del LLM)
- Esperar Qwen3.6-35B-A3B Instruct (sale Q1-Q2 2026): mejor calidad mismo perfil

---

## 9. Bugs / lecciones aprendidas

### 9.1 `git stash -u` puede borrar modelos

Durante el deploy del merge consolidado, `git stash push -u` capturó archivos untracked de `models/` (Whisper, ECAPA-TDNN). Posterior `git reset --hard` los borró del WT. Service `kza-voice` entró en restart-loop **3 horas** antes de notarlo.

Recovery: `git checkout stash@{0}^3 -- models/<path>` — los untracked stasheados van al `^3` parent.

Memoria: `feedback_git_stash_u_models.md` para evitar repetir.

### 9.2 Nombres de quants ambiguos

`mix-IQ4_K.gguf` sin sufijo `Instruct/Thinking` puede ser cualquiera de las dos variantes. ubergarm publicó la Thinking. Para domotics conversacional eso es show-stopper.

**Lección:** verificar metadata del GGUF (`general.name` field) antes de deploy. O bench corto con prompt simple — si ves `<think>` blocks, es la variante thinking.

### 9.3 SSH bg + nohup + redirección no funciona reliably

Intentos múltiples de `nohup ... > /tmp/log 2>&1 &` via SSH non-interactive fallaron silenciosamente. La solución: **systemd unit dedicado**. Más robusto que cualquier wrapping de shell.

---

## 10. Commits clave de la sesión

```
eaaca5e perf(kza-llm-ik): add prompt cache RAM persistence + similarity match
59ac091 fix(kza-llm-ik): revertir Q4_K_M a Q5_K_M tras A/B blind real
a2aa12e infra(kza-llm-ik): switch primary LLM service a ik_llama.cpp
9896bbd infra(kza-llm): unit con flash_attn + KV cache Q8 + n_threads_batch=24
4fe89fa fix(llm): tolerate missing clients — skip endpoint instead of fail
d741788 fix: dispatcher fire-and-forget + WS calls persistent + silence_end_ms 500
d725edb merge feature/wake-tv-filter-fixes — wake/TV regression fixes (22 commits)
a3e788b docs: planes openclaw + cleanup obs-ai-analyzer obsoleto
e9dd123 feat(llm): wire LLMRouter into main + dispatcher (plan #1 OpenClaw)
a47e10b feat(llm): idle watchdog + router factory (plan #1 OpenClaw)
0ccbc45 feat(llm): foundation types + error classifier (plan #1 OpenClaw)
1c2afbd fix(dispatcher): remove idempotent cache skip for HA actuators
006f1c5 perf(ha,wake): WS calls persistente al startup + silence_end_ms 700→500
```

---

## 11. Pendientes para próxima sesión

### Alto valor

- **N-gram lookup decoding** (`-lcd`) — generar cache desde corpus de tool-calls HA, evaluar +15-40% en outputs estructurados
- **OpenClaw plan #2: auto-compaction de contexto** — el prompt cache va a brillar cuando integremos history conversacional rolling
- **Validación end-to-end con voz**: probar slow path real ("Nexa ¿por qué tarda en hervir el agua?") con el nuevo stack

### Operativo

- **Renombrar `kza-72b.service` → `kza-llm.service`** definitivamente (cosmético, evitar confusión con modelo viejo)
- Verificar si `--prompt-cache` disco realmente funciona en `ik_llama-server` (parece no estar persistiendo)

### Backlog OpenClaw roadmap

- Plan #3: hooks system tipados
- Plan #4: file-based session write lock
- Plan #5: session transcript JSONL + idempotency keys HA
- (y resto del top-10)
