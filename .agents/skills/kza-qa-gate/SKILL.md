---
name: kza-qa-gate
description: Checklist portable de auditoría cross-model con 4 lentes integradas y gate de privacidad/autoridad para KZA.
---

# KZA QA & Privacy Gate (4 Lentes)

Aplicable a `kza-qa`, `kza-pregate` y `kza-privacy` en cualquier runtime (`claude`, `codex`, `muse`, `agy`).

## Verificación Inicial (Fail-Closed)
1. Confirmar `HEAD` exacto: `git rev-parse HEAD`.
2. Confirmar árbol trackeado limpio (`dirty=0`): `git status --porcelain --untracked-files=no`.
3. Inspeccionar diff completo contra la rama base asignada.

## 4 Lentes Obligatorias
1. **Corrección / Diseño / Tipos:** contratos, edge cases, `async/await`, DI, rollback ante errores, manejo de timeouts y señales.
2. **Fallos silenciosos:** `except` amplios, retornos `None` silenciosos, fallbacks no reportados, exclusiones de familia por binario en vez de familia real.
3. **Rigor de tests:** correr tests reales en `.venv/bin/pytest`; detectar mocks que oculten regresiones o aserciones triviales.
4. **Comentarios / Docs / Afirmaciones:** contrastar promesas de la PR/docs con el código real.

## Invariantes KZA Adicionales
- Separación voz / persona / cuenta / roles / consentimiento / administración.
- Releases autocontenidas (`scripts/build_runtime.py`).
- Cero fuga de secretos, `.env`, audio, embeddings o rutas `/home/kza/*`.
- Cero interferencia con templates/worktrees de otros proyectos (`tu-*`, `gsm-*`).

## Formato de Dictamen
- Veredicto literal en la primera línea: **`APROBADO`** o **`BLOQUEA`**.
- Hallazgos ordenados por severidad (`CRÍTICO` / `ALTO` / `MEDIO` / `RESIDUO`) con `archivo:línea`.
- Solo `CRÍTICO` / `ALTO` introducido por el diff justifica `BLOQUEA`.
- Escribir reporte UNA vez en `report_path` y enviar sobre `<= 6` líneas vía `hc send`.
