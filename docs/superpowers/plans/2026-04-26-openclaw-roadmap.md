# OpenClaw-Inspired Roadmap (KZA)

> **Tipo de documento:** Roadmap de alto nivel, NO un implementation plan. Cada item linkea (o linkeará) a su propio plan implementable cuando se aborde.

**Origen:** Análisis del repo `github.com/openclaw/openclaw` el 2026-04-26. Detalles del repo en `~/.claude/projects/-Users-yo-Documents-kza/memory/reference_openclaw_repo.md`.

**Filosofía:** OpenClaw es un asistente generalista TS/Node multi-canal chat — distinto a KZA en stack y propósito. Lo transferible son **patrones arquitectónicos**, no código. Esta hoja de ruta extrae 10 mejoras concretas y las prioriza por ROI para KZA (voz local + multi-usuario + 2 GPUs).

---

## Top-10 priorizado

Orden = `(impacto + cobertura de bugs/limitaciones actuales) ÷ esfuerzo`. S=Small (~150 líneas), M=Medium (~250), L=Large (~500).

| # | Feature | Cat | Esfuerzo | Plan |
|---|---|---|---|---|
| 1 | **LLM failover + cooldown + idle timeout** (A1+A2+A5) — cooldowns con backoff exponencial 1m→5m→25m→1h, candidate chain 7B→72B→cloud, idle watchdog para 72B en CPU | Resiliencia | M+S+XS | [`2026-04-26-llm-failover-cooldown.md`](./2026-04-26-llm-failover-cooldown.md) |
| 2 | **Auto-compaction de contexto + identifier policy strict** (B1+B2+B3) — compactar contextos por usuario al cerca del límite, modelo dedicado (72B en background), preservar IDs HA opacos | Memoria | M+XS+XS | _pendiente_ |
| 3 | **Plugin hooks system tipados** (C1+C2) — 8-10 hooks formales (`before_tool_call`, `before_compaction`, `message_received`, etc.) con block/rewrite/requireApproval | Extensibilidad | L | _pendiente_ |
| 4 | **File-based session write lock** (B4) — lock process-aware para `data/contexts/<user>.json` | Memoria | S | _pendiente_ |
| 5 | **Session transcript JSONL + idempotency keys HA** (K4+K5) — transcript estructurado por sesión, idempotency en acciones HA side-effecting | Observabilidad | M | _pendiente_ |
| 6 | **Heartbeat agent turn + HEARTBEAT.md per-user** (D1+D2) — turn periódico que revisa estado, ack token `HEARTBEAT_OK` | Background | M | _pendiente_ |
| 7 | **Lane-aware queue con modes** (F1+F2) — per-session lane + modes collect/steer/followup/steer-backlog | Queue | M | _pendiente_ |
| 8 | **Active memory plugin** (B7) — sub-agent inyecta memoria relevante ANTES del reply | Memoria | M | _pendiente_ |
| 9 | **DM scope per-channel-peer + identity links** (E1+E2) — aislar contextos por (zona + speaker), linking cross-zone | Multi-zona | S | _pendiente_ |
| 10 | _(ya cubierto por #1)_ | — | — | — |

> Nota: la lista original tenía A5 (idle timeout) en el slot #10; lo movimos al plan #1 porque es cohesivo con cooldown/failover.

---

## Categorías cubiertas

- **A. Resiliencia**: #1
- **B. Memoria**: #2, #4, #8
- **C. Extensibilidad**: #3
- **D. Background**: #6
- **E. Multi-zona**: #9
- **F. Queue**: #7
- **K. Observabilidad**: #5

## Categorías NO cubiertas en top-10 (ver lista completa en este chat o memoria)

- G. Streaming/pacing (mejoras nice-to-have)
- H. System prompt / workspace bootstrap (alta valor pero re-arquitectura grande)
- I. Dreaming nocturno (encaja con LoRA nocturno actual, futuro)
- J. Multi-agent / per-zone agents (largo plazo)
- L. SOUL.md / personalidad (cosmético)
- M. Sandboxing / tool policy (no urgente; KZA aún no tiene tool execution agresivo)
- N. WebSocket gateway / companion app (proyecto en sí mismo)
- O. Skills system formal (refactor pesado)

---

## Cómo ejecutar este roadmap

1. **Por items, no en bloque.** Cada item es independiente y puede ir a producción solo. Eso evita branches huge y facilita rollback.
2. **Plan implementable se genera al inicio del item**, no todos al inicio del roadmap. Las decisiones del item N+1 dependen de cómo salió el N.
3. **Cada plan vive en `docs/superpowers/plans/2026-04-26-<feature>.md`** y se ejecuta con `superpowers:subagent-driven-development` o `superpowers:executing-plans`.
4. **Verificación de patrones**: cuando termine #1, validar que las abstracciones (CooldownManager, error classification, candidate chain) son las correctas antes de aplicarlas a #2 (auto-compaction usa fallback-style retry parecido).

---

## Lo que NO hacemos (intencional)

- **Multi-canal chat** (WhatsApp/Telegram/Slack) — fuera de KZA voz
- **Companion app móvil completa** — solo si decidimos ir a remoto
- **Sherpa-onnx TTS / voice-call Twilio** — Kokoro local es mejor; voice-call es cloud
- **Sandbox Docker/SSH** — KZA aún no tiene tool execution agresivo
- **Delegate architecture** — KZA es personal, no organizacional
- **ACP / Codex harness** — ajenos al pipeline KZA

---

## Próximo paso

Ejecutar plan #1 ([`2026-04-26-llm-failover-cooldown.md`](./2026-04-26-llm-failover-cooldown.md)). Cuando esté merged, abrir plan #2 (auto-compaction).
