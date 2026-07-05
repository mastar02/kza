# Plan: MiniMax M2.7-highspeed + hermes-agent en sandbox — etapas y prompts por proyecto

**Fecha:** 2026-06-06
**Decisión del usuario:** suscripción MiniMax ya paga (costo por token no es factor) → habilitar `-highspeed`; hermes-agent en **etapas: asistente paralelo primero → delegación del slow path solo si los números validan**; sandbox = **Quadlet/podman rootless en el server**.
**Contexto previo:** `docs/research/2026-06-06_HERMES4_RAG_TOTAL_ANALISIS.md` (veredicto Hermes-modelo + addendum MiniMax).

---

## Mediciones reales (2026-06-06, desde el server, camino de producción)

Ruta: server → gateway LiteLLM `:8200` → MiniMax cloud. Virtual key del consumer KZA (la env `MINIMAX_API_KEY` del server es la **virtual key del gateway** `sk-c…`, NO la key real de MiniMax — la real vive en la config del gateway, lado infra).

| Medición | Valor |
|---|---|
| Transporte a api.minimax.io | DNS 2ms · TCP connect **142ms** (RTT ~140ms) · TLS 440ms · TTFB GET 583ms |
| Modelos expuestos por el gateway | `MiniMax-M2.7`, `tts`, `tts-turbo` — **`-highspeed` NO registrado** (explica el 429 de mayo: nunca hubo ruta) |
| M2.7 prompt corto (49 tok) | TTFT 0.87-1.56s · gen **37-41 t/s** · total 1.9-2.9s |
| M2.7 prompt RAG 4.7K tok | TTFT 1.40-2.49s · gen **41-42 t/s** · total ~9s (300 tok generados) |

**Lecturas:**
1. **Prefill cloud es despreciable**: 4.7K tokens de contexto agregaron <1s de TTFT → el RAG total no penaliza el slow path cloud.
2. **El transporte no es el problema** (~0.6s una vez por conexión; keep-alive lo amortiza).
3. **Gen real ~41 t/s vía gateway** vs ~77.7 t/s que Artificial Analysis mide para MiniMax direct → investigar si el gateway agrega overhead de streaming o es región/carga (ítem infra). Con `-highspeed` (~100 t/s prometido): una respuesta con thinking de ~1000 tok pasa de ~24s a ~10s — **ahí está la ganancia real**, no en los 150 tok hablados (3.7s → 1.5s).
4. El E2E actual con RAG 4.7K + 150 tok hablados ya daría **~5-6s** sin thinking largo. Con thinking complejo es donde duele y donde highspeed paga.

---

## Arquitectura objetivo (etapas)

```
ETAPA A (ahora)
  KZA slow path ──► gateway :8200 ──► MiniMax-M2.7-highspeed   (config-only en KZA)
  hermes-agent (Quadlet rootless, sandbox) ──► gateway :8200 (virtual key propia)
                                          ──► HA :8123 (token scoped de domo)
  └── se miden latencias E2E reales de hermes con tools HA

ETAPA B (solo si A valida con números)
  KZA slow path ──feature flag──► hermes-agent (shadow mode primero) ──► gateway ──► MiniMax
```

Por qué este orden: hermes-agent resuelve internamente la **mina del thinking multi-turno de M2.x** (hay que preservar `<think>` entre turnos de tool-calling: sin él Tau2 -36%) — delegar a hermes evita implementar tool-calling propio contra M2.x. Pero su overhead (~13.9K tok/call) y madurez (bugs activos) exigen validación con números antes de tocar el pipeline de voz.

---

## PROMPT 1 — proyecto **homelab-infra** (pegar en sesión Claude de infra)

```
Contexto: el gateway LiteLLM en :8200 del server (192.168.1.2) es el punto único de acceso
a MiniMax para consumers (KZA usa virtual key). Medición 2026-06-06 desde el server:
modelos expuestos hoy = [MiniMax-M2.7, tts, tts-turbo]; M2.7 vía gateway rinde TTFT
0.9-2.5s y ~41 t/s de generación (Artificial Analysis mide ~77.7 t/s contra MiniMax
direct — puede haber overhead de streaming del gateway o ser región/carga). El usuario
paga suscripción MiniMax y quiere habilitar MiniMax-M2.7-highspeed (mismo modelo, serving
2x: ~100 t/s; en PAYG cuesta el doble pero la suscripción lo cubriría — verificar).

Tareas (server de PRODUCCIÓN del hogar — no tocar kza-voice ni el deployment M2.7 existente;
reload del gateway sin downtime):

1. Con la key REAL de MiniMax (en la config del gateway), probar si la suscripción actual
   habilita el modelo "MiniMax-M2.7-highspeed" (request mínima a api.minimax.io). Si da
   429/403 → el plan actual no lo incluye; reportar qué plan Highspeed hace falta
   (platform.minimax.io) y frenar ahí.
2. Si responde: registrar "MiniMax-M2.7-highspeed" en el model_list del LiteLLM,
   misma key real. Verificar que `curl :8200/v1/models` lo liste.
3. Benchmark comparativo documentado: TTFT y gen t/s de base vs highspeed vía gateway,
   y base vía api.minimax.io directo (para aislar el overhead del gateway). Si el gateway
   agrega >15% de overhead de streaming, investigar (buffering de SSE en LiteLLM).
4. Agregar un segundo deployment con el backend ANTHROPIC-COMPAT de MiniMax
   (api.minimax.io/anthropic) — es la única ruta donde el thinking se puede controlar
   (body.thinking={type:"disabled"}, reasoning_split); el endpoint OpenAI-compat IGNORA
   reasoning_effort silenciosamente. Exponerlo como alias (ej. "MiniMax-M2.7-nothink")
   si LiteLLM lo soporta (provider anthropic con api_base custom).
5. Crear una virtual key NUEVA y separada para hermes-agent (budget/rate límites propios,
   revocable sin tocar la de KZA).
6. Métricas de latencia por request en el gateway (TTFT, total, tokens in/out) hacia la
   observabilidad existente (dashboard :9500/obs).
7. Desplegar hermes-agent (github.com/NousResearch/hermes-agent) en Quadlet/podman
   ROOTLESS bajo infra:
   - Container con HERMES_HOME en volumen persistente; PINEAR versión (no auto-update:
     issue #32384 — "hermes update" corrompe el repo git).
   - Provider: OpenAI-compatible → http://192.168.1.2:8200/v1 con su virtual key,
     modelo MiniMax-M2.7 (205K ctx; NO apuntarlo a modelos de contexto chico:
     issue #23767, oversized prompts).
   - Red: salida SOLO al gateway :8200 y a HA :8123 (HA recién cuando homelab-domo
     entregue token scoped — arrancar sin HA). Sin acceso al filesystem del host fuera
     de su volumen (el agente ejecuta tools de terminal/archivos por diseño).
   - Smoke test: respuesta en CLI dentro del container vía gateway.
   - Esperado/conocido: overhead ~13.9K tokens fijos por call (issue #4379) — con
     suscripción no es costo marginal; memoria limitada a ~2200 chars (#16831).

Criterios de aceptación: highspeed listado y benchmarkeado (o reporte de que el plan no
lo cubre); alias nothink operativo o descartado con razón; virtual key hermes activa;
hermes-agent respondiendo en sandbox; kza-voice intacto (verificar service activo al final).
```

## PROMPT 2 — proyecto **homelab-domo** (pegar en sesión Claude de domo)

```
Contexto: infra está desplegando hermes-agent (framework agéntico de Nous Research) en un
container rootless del server, como asistente del homelab en sandbox. Necesita acceso a
Home Assistant vía REST/WebSocket con el MENOR privilegio posible. Sus tools son:
ha_list_entities, ha_get_state, ha_list_services, ha_call_service.

Tareas (config-as-code en este repo):
1. Crear usuario HA dedicado "hermes" (NO admin, no en grupo de administradores).
2. Generar long-lived access token para ese usuario.
3. HA no tiene scoping fino de API por usuario para REST: documentar el blast radius real
   del token (qué puede tocar un usuario no-admin) y, si es viable, restringir con
   policies/grupos a dominios light/switch/scene/climate. Si no es viable, dejarlo
   explícito en el doc de decisión.
4. Entregar el token a infra por el canal de secrets (NO commitearlo); infra lo inyecta
   al container como secret.
5. Registrar la decisión en el repo (quién es el usuario hermes, para qué, cómo revocarlo).
```

## PROMPT 3 — proyecto **kza** (este repo, sesión futura — "la migración")

```
Contexto: infra registró MiniMax-M2.7-highspeed en el gateway :8200 (verificar con
curl :8200/v1/models antes de empezar). El slow path de KZA usa HttpReasoner
(src/llm/reasoner.py:446, OpenAI-compat) → gateway → MiniMax-M2.7 base. Mediciones
2026-06-06: base vía gateway = TTFT 0.9-2.5s, gen ~41 t/s; prefill cloud de 4.7K tokens
<1s (el RAG total casi no agrega latencia cloud). Doc de referencia:
docs/plans/2026-06-06_PLAN_MINIMAX_HIGHSPEED_HERMES_AGENT.md y
docs/research/2026-06-06_HERMES4_RAG_TOTAL_ANALISIS.md.

ETAPA A — migrar el reasoner a highspeed (config-only + medición):
1. config/settings.yaml reasoner.http_model: "MiniMax-M2.7" → "MiniMax-M2.7-highspeed"
   (mismo base_url :8200; HttpReasoner no cambia).
2. Re-correr el benchmark TTFT/t/s (script en el doc del plan) antes y después; objetivo:
   gen ≥80 t/s. Si no mejora ≥1.5x, reportar a infra (overhead del gateway).
3. Si TTFT warm queda consistentemente <5s, bajar reasoner.idle_timeout_s 25→15 y validar
   que no dispare el watchdog en cold start (probar tras 30+ min de inactividad).
4. Si infra expuso el alias "MiniMax-M2.7-nothink" (backend Anthropic-compat): evaluar
   rutearle las respuestas conversacionales simples (segments con needs_reasoning=false
   que igual caen al slow path) — menos latencia; mantener thinking para razonamiento
   real. Implementar como selección de modelo por request en HttpReasoner si el ahorro
   medido lo justifica.
5. Verificación: pipeline de voz end-to-end con kza-voice activo; el strip de <think>
   (reasoner.py:_strip_reasoning) sigue cubriendo el caso single-turn.

ETAPA B — experimento de delegación a hermes-agent (SOLO si la etapa sandbox de infra
validó latencias y el usuario lo aprueba):
6. Feature flag reasoner.hermes_delegate (default false) + cliente al endpoint de
   hermes-agent en SHADOW MODE: duplicar N requests reales del slow path hacia hermes,
   loggear latencia E2E/respuesta SIN servirla al usuario (mismo patrón shadow que
   CommandGate). Comparar calidad/latencia vs camino actual con logs de ambos.
7. Decisión con datos: si hermes E2E entra en presupuesto (5-30s) y la calidad es
   superior (tool-calling HA real, memoria propia), cablear como path opcional detrás
   del flag. Nota: delegar a hermes evita implementar a mano la preservación de <think>
   multi-turno que M2.x exige para tool-calling (sin ella Tau2 cae -36%).
```

---

## Riesgos y notas

- **El 429 de mayo puede repetirse**: si la suscripción actual no incluye el tier Highspeed, el paso 1 de infra lo detecta antes de tocar nada (los planes Highspeed son SKUs separados: ~$40/$80/$150/mes según agregadores, sin verificar en la página oficial).
- **~41 t/s vía gateway vs ~78 direct**: hasta no aislar el overhead (ítem 3 de infra), la ganancia real de highspeed podría ser mayor o menor que 2×.
- **hermes-agent es joven** (feb-2026): versión pineada, sandbox sin host filesystem, key revocable. No tocar el pipeline de voz hasta la Etapa B con números.
- **Privacidad sin cambios**: todo esto sigue mandando slow path a MiniMax (China). La tensión §5.5 del análisis (RAG total + datos sensibles → cloud) queda abierta; el endgame local en `:8200` sigue vigente como norte. El RAG total (F1→F3) avanza en paralelo, es independiente de todo esto.
