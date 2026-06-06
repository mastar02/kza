# Tarea KZA: migrar el reasoner a MiniMax-M2.7-highspeed (+ Etapa B gated: delegación a hermes-agent)

> **ESTADO 2026-06-06 — Etapa A ejecutada en local, pendiente deploy al server.**
> - Gateway verificado: lista `MiniMax-M2.7-highspeed` y `MiniMax-M2.7-nothink`; la key de KZA accede a ambos.
> - Re-bench sostenido (600 tok, vía gateway, key KZA): base 32.3/77.3 t/s, highspeed 58.2/47.5 t/s — confirma el hallazgo de infra: **+15% mediano con variancia de serving que domina** (un run de base superó al highspeed). El objetivo "≥80 t/s" del punto 2 NO se cumple; no es overhead del gateway (infra lo midió ≈0%) sino el serving de MiniMax desde esta región. Se migra igual: costo marginal 0 (suscripción), mejora mediana, rollback de 1 línea.
> - `settings.yaml` actualizado: `reasoner.http_model: "MiniMax-M2.7-highspeed"` (comentario con bench y rollback).
> - **Decisión punto 3**: `idle_timeout_s` queda en 25.0 — TTFT warm ~1s pero cold no caracterizado y variancia alta; el watchdog es protección, no latencia: bajar no gana nada.
> - **Decisión punto 4**: NO adoptar `nothink` — infra verificó que MiniMax ignora `thinking:disabled` server-side (6/6): el alias solo separa el reasoning del content, y `_strip_reasoning()` ya cubre eso. Queda como opción si el parsing de tags da problemas.
> - **Pendiente de infra para coordinar en la ventana de deploy**: KZA usa la MASTER key del gateway → migrar a virtual key propia + rotación de la master (cambio en `/home/kza/secrets/.env`, lado server).
> - **Falta**: commit + `kza-push` + pull en server + restart `kza-voice` + verificación de voz end-to-end (punto 5).

**Fecha de creación:** 2026-06-06.
**Prerequisito:** infra registró `MiniMax-M2.7-highspeed` en el gateway — **verificar antes de empezar**: `curl -s -H "Authorization: Bearer $MINIMAX_API_KEY" http://192.168.1.2:8200/v1/models` debe listarlo (la env del server es la virtual key del gateway).
**Contexto completo:** `docs/2026-06-06_PLAN_MINIMAX_HIGHSPEED_HERMES_AGENT.md` (mediciones) y `docs/2026-06-06_HERMES4_RAG_TOTAL_ANALISIS.md` (análisis de fondo).

## Contexto

El slow path usa `HttpReasoner` (`src/llm/reasoner.py:446`, OpenAI-compat) → gateway LiteLLM `:8200` → MiniMax-M2.7 base. `-highspeed` es el mismo modelo con serving 2× (~100 t/s vs ~60 nominal).

Línea base medida 2026-06-06 (vía gateway, desde el server):
- Prompt corto: TTFT 0.87-1.56s, gen **~37-42 t/s**, total <3s.
- Prompt con RAG 4.7K tokens: TTFT 1.4-2.5s (**prefill cloud <1s** — el RAG total casi no agrega latencia), ~41 t/s, ~9s/300 tok.
- La ganancia real de highspeed está en queries con thinking largo: ~1000 tok de `<think>` pasan de ~24s a ~10s.

## Etapa A — migración a highspeed (config-only + medición)

1. `config/settings.yaml` → `reasoner.http_model: "MiniMax-M2.7"` → `"MiniMax-M2.7-highspeed"`. Mismo `http_base_url` (gateway); `HttpReasoner` no cambia.
2. **Re-bench antes/después** (mismo método de la línea base: streaming, TTFT + completion_tokens/tiempo; el script está documentado en el doc del plan). Objetivo: gen ≥80 t/s. Si la mejora es <1.5×, reportar a infra (posible overhead de streaming del gateway — ya se midió 41 vs 78 direct en base).
3. Si TTFT warm queda consistentemente <5s: bajar `reasoner.idle_timeout_s` 25→15 y validar que NO dispare el watchdog en cold start real (probar tras 30+ min de inactividad del endpoint).
4. **Opcional (si infra expuso el alias `MiniMax-M2.7-nothink`** vía backend Anthropic-compat): evaluar rutear ahí las respuestas conversacionales simples que caen al slow path (segments con `needs_reasoning=false` del LLMCommandRouter) — menos latencia y tokens; mantener el modelo con thinking para razonamiento real. Implementar como selección de modelo por request en `HttpReasoner` SOLO si el ahorro medido lo justifica.
5. **Verificación:** voz end-to-end con `kza-voice` activo; confirmar que `_strip_reasoning()` (`reasoner.py:82`) sigue limpiando el stream (cubre single-turn, que es el caso actual).

## Etapa B — delegación del slow path a hermes-agent (GATED)

> Ejecutar SOLO si: (a) el sandbox de hermes-agent de infra validó latencias E2E con tools HA, y (b) el usuario lo aprueba explícitamente.

6. Feature flag `reasoner.hermes_delegate` (default **false**) + cliente al endpoint de hermes-agent, en **shadow mode** (mismo patrón que CommandGate): duplicar N requests reales del slow path hacia hermes, loggear latencia E2E y respuesta SIN servirla al usuario. Comparar calidad/latencia contra el camino actual con logs de ambos lados.
7. Decisión con datos: si hermes entra en presupuesto (5-30s E2E) y la calidad es superior (tool-calling HA real + memoria propia del agente), cablear como path opcional detrás del flag.
8. Nota técnica a favor de delegar (si B se justifica): M2.x exige **preservar `<think>` entre turnos** de tool-calling multi-turno (sin eso Tau2 cae -36%, BrowseComp -40% — datos de MiniMax). hermes-agent lo maneja internamente; implementarlo a mano en KZA sería plumbing extra y frágil.

## Qué NO hace esta tarea

- No toca el fast path (<300ms) ni el FastRouter `:8101`.
- No cambia la privacidad: el slow path sigue saliendo a MiniMax (cloud). La tensión RAG-total-vs-cloud (§5.5 del análisis) queda abierta — el endgame local en `:8200` sigue vigente como norte.
- No depende del RAG total (F1→F3), que avanza en paralelo e independiente.
