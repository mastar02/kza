# PLAN_MEJORAS — kza
> Para ejecutar con Claude Code desde esta carpeta. Origen: auditoría del workspace 2026-06-09.
> ⚠️ No correr este plan en paralelo con una sesión en `kza-wt-escritorio/` (comparten `.git`).

## Contexto
Asistente de voz 100% local para Home Assistant: Faster-Whisper (STT), Piper/Kokoro (TTS), LLM 72B CPU + router 7B GPU (4× RTX 3070), speaker-ID, emociones. 617+ tests. Rama actual: `feat/nexa-command-detection-fixes`. Worktree activo: `kza-wt-escritorio` (rama `feat/escritorio-grupos-escenas`).

## Reglas
1. Rama `chore/plan-mejoras-2026-06` desde la rama principal. Un commit por tarea, Conventional Commits.
2. Correr `pytest` (subset rápido si la suite completa es pesada) antes de cada commit.
3. Tareas **[HUMANO]**: no ejecutar; listarlas al final. Marcá `[x]` al completar.
4. No imprimas valores de secretos ni toques `config/settings.yaml` productivo sin backup.

## P0 — Higiene urgente
- [x] Resolver `kza_tts_test.mp3` sin trackear: borrarlo y agregar `*.mp3` de prueba a `.gitignore` (o moverlo a `tests/fixtures/` si se usa). _(Sin referencias en el repo → borrado; `*.mp3` agregado a .gitignore. No había mp3/wav trackeados.)_
- [x] Revisar el `index.lock` colgado en `.git/worktrees/kza-wt-escritorio/`: si no hay proceso git activo, eliminarlo y verificar `git -C ../kza-wt-escritorio status`. _(El lock ya no existía al ejecutar (2026-06-09); `git status` del worktree OK: rama `feat/escritorio-grupos-escenas`, working tree clean, 4 commits behind origin/main. Sí se eliminó un `index.lock` huérfano distinto en `.git/` del repo principal — 0 bytes, 4 h de antigüedad, sin proceso dueño.)_
- [x] Commitear o descartar el cambio pendiente en `docs/plans/2026-06-06_TAREA_KZA_MIGRACION_HIGHSPEED.md`. _(Commiteado: era la actualización de estado "Etapa A deployada y verificada en server" — consistente con la sesión 2026-06-06/07.)_

## P1 — Documentación y convenciones
- [x] Reorganizar `docs/` en `docs/architecture/`, `docs/research/`, `docs/plans/` (mover los ~30 .md con fecha según su tipo) + `docs/INDEX.md` con una línea por documento. Usar `git mv`. _(25 .md movidos con git mv: 9→architecture, 14→research, 2→plans (11 ya estaban en plans/). Referencias internas actualizadas en CLAUDE.md, README.md, .gitignore, src/ambient/, config/settings.yaml y docs cruzados. runbooks/, examples/ y superpowers/ quedan como estaban.)_
- [x] Crear `docs/SERVER_CONVENTIONS.md`: espejo local de las convenciones del servidor compartido que hoy viven solo en Notion (pág. 8 y 9.x — usuarios/UID, sub-rangos de puertos, Podman rootless + Quadlets, GPU por CDI). **[HUMANO]** pegar/validar el contenido desde Notion si Claude no tiene acceso al MCP. _(Hecho vía MCP Notion 2026-06-09: pág. 8 completa (4 páginas de bloques + 3 tablas) y pág. 9 completa. Credenciales que figuran en Notion fueron redactadas. Las sub-páginas 9.x de proyectos ajenos se listan sin espejar. Queda a validación humana opcional, el acceso MCP funcionó.)_
- [x] Actualizar `CLAUDE.md`: apuntar primero a `docs/SERVER_CONVENTIONS.md` y dejar Notion como referencia secundaria. _(Sección "Source of truth cross-project" reescrita; Notion sigue siendo canónica ante diferencias.)_
- [x] Documentar en `CLAUDE.md` la estrategia del worktree `kza-wt-escritorio`: qué rama vive ahí, cómo y cuándo se mergea, regla de no tocar ambos a la vez. _(Sección "Worktrees" nueva. Verificado: la rama del worktree ya está contenida en origin/main → candidato a cleanup.)_

## P2 — Deuda técnica
- [x] Crear schema Pydantic para `config/settings.yaml` con validación al boot (`src/main.py`) y test que cargue el settings de ejemplo. _(`src/core/settings_schema.py`: schema permisivo (extra=allow) con las secciones núcleo + campos que main.py ya exigía a mano; `load_config` ahora fail-fast con detalle por campo. 6 tests TDD en `tests/unit/core/test_settings_schema.py`, incluido smoke contra el settings.yaml del repo.)_
- [x] Limpiar referencias legacy a systemd/docker-compose en docs si producción usa Quadlets+podman; dejar una sola fuente de verdad de deploy. _(El condicional era falso: producción KZA-voice es systemd --user NATIVO (excepción R10 del contrato), no Quadlets — systemd no es legacy acá; lo legacy es docker-compose. Creado `docs/architecture/DEPLOYMENT.md` como fuente de verdad única; banner histórico en KZA_ANALISIS_Y_ROADMAP.md; pointer en docker/README.md. Los plans/ 2026-02/03 quedan como históricos fechados.)_
- [x] Revisar los 8 paquetes de diferencia en `requirements.txt` vs el worktree y consolidar. _(Verificado con `comm`: el worktree NO tiene ningún paquete que falte acá — la diferencia son 3 adiciones legítimas de esta rama posteriores al worktree (`onnx-asr[hub]` swap Parakeet 2026-06-07, `py3langid` language-ID shadow, `pyusb` control XVF3800) + comentarios. La rama del worktree está 100% contenida en origin/main (`git log feat/escritorio-grupos-escenas ^origin/main` vacío). `requirements.txt` actual ES la versión consolidada; nada que cambiar. El worktree se actualiza solo cuando se haga su cleanup (ver CLAUDE.md § Worktrees).)_

## Verificación final
- [x] `git status` limpio, pytest del subset verde, `docs/INDEX.md` consistente. _(Subset core+pipeline+orchestrator+ambient: 657 passed, 5 failed — las MISMAS 5 fallas pre-existentes del baseline de la rama antes de empezar (verificado en el primer paso; `test_endpointing::test_voice_prob_uses_vad` ya documentada como falla del baseline al 2026-06-01). Cero regresiones nuevas. INDEX: 0 docs sin listar, 0 links muertos.)_
- [x] Resumen: hecho / pendiente / riesgos / tareas [HUMANO]. _(Ver sección "Cierre 2026-06-09" abajo.)_

## Cierre 2026-06-09 (ejecutado por Claude)

**Hecho** (11 commits, ejecutados en `chore/plan-mejoras-2026-06`): P0 completo (mp3 borrado + .gitignore; locks verificados/limpiados; doc highspeed commiteado), P1 completo (docs/ reorganizado en architecture/research/plans con git mv + INDEX.md; SERVER_CONVENTIONS.md espejado desde Notion pág 8+9 vía MCP; CLAUDE.md apunta primero al espejo local; estrategia de worktrees documentada), P2 completo (schema Pydantic + validación al boot + 6 tests TDD; DEPLOYMENT.md fuente de verdad única de deploy; requirements verificado ya-consolidado).

**Decisiones tomadas** (documentadas, revisables):
- La rama se creó desde `feat/nexa-command-detection-fixes` (que contiene main entero, ff-able) y NO desde main puro: el cambio pendiente de P0.3 y 5 de los docs a reorganizar solo existen en esa rama; desde main las tareas eran inejecutables.
- "Lint" = `py_compile` + suite pytest: el repo no tiene linter configurado (sin ruff/flake8/pyproject).

**Riesgos**:
- La reorganización de docs/ rompe paths memorizados fuera del repo (memoria de Claude, Notion, notas). Dentro del repo todas las referencias fueron actualizadas y verificadas con grep.
- SERVER_CONVENTIONS.md es un snapshot (2026-06-09); puede driftear de Notion. El doc declara que Notion gana ante conflicto.
- La validación Pydantic al boot es fail-fast: si el settings.yaml del SERVER difiere del local y le faltara una sección núcleo, kza-voice no arrancaría tras el próximo deploy+restart. Mitigado (post-review 2026-06-09): el schema replica el contrato exacto de main.py — embeddings.model/device y home_assistant.url incondicionales; speaker_id/emotion model+device solo cuando enabled (default true); el settings.yaml del repo pasa el smoke test.
- ~~Esta rama NO está mergeada ni pusheada~~ Actualización post-cierre: mergeada ff a `feat/nexa-command-detection-fixes` y pusheada a origin el 2026-06-09; la rama `chore/` se borró tras integrarse. El server aún no hizo pull.

**Tareas [HUMANO] pendientes**:
- P1.2: validar el contenido de `docs/SERVER_CONVENTIONS.md` contra Notion (el acceso MCP funcionó, así que el fallback [HUMANO] no se necesitó — queda solo la validación opcional de fidelidad).
- (Fuera del plan, detectado): ~~decidir cleanup del worktree `kza-wt-escritorio` y mergear/pushear esta rama~~ — hecho el 2026-06-09 (worktree removido, rama escritorio borrada, merge+push ejecutados). El worktree `kza-dashboard` se conserva (tiene commits propios no integrados).
