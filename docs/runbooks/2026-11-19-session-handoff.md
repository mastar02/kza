# Cierre de sesión de KZA — 2026-11-19

## Estado al cerrar

Roadmap **pausado por pedido del usuario** («luego de finalizar con estas dos tareas realizemos una pausa y gurdemos el avanze completo y lo restante del roadmap»), **no completado**. F1, U3b y las fases P2–P8 siguen abiertas; nada de lo de abajo cierra el roadmap ni F1.

- Checkout de trabajo del roadmap: `/Users/gabriel/Documents/Proyectos/kza`, rama `recovery/production-2026-10-25`. **Al cerrar, ese checkout lo usa otra sesión** en la rama `feat/kza-multi-console-workflow` (commits `ad9cfa69`, `97111bbe`, `e55e366b` de tooling de workflow y cambios sin commitear en `.gitignore`, `AGENTS.md`, `CLAUDE.md`): no hacer checkout ni merge ahí sin coordinar. Las integraciones de esta sesión se hicieron en clones bajo `/Users/gabriel/.claude/jobs/3a8b26c3/tmp/` y se empujaron por SHA.
- Rama integrada: `recovery/production-2026-10-25` = lote M `3dc3ef0a` + el commit de este handoff (solo documentación), empujada a origin. `main` (`0fd3d166`) es ancestro de esta rama (1287+ commits atrás, avanzable por fast-forward); la rama de integración del roadmap es `recovery/…`, no `main`.
- **Producción no cambió**: `kza-voice` sigue en la release `e9669ec` (slot `release_20261029T185008966364Z_e9669ec`, estado en el slot `b0866e8`). Todo lo integrado desde el 2026-10-29 (SC-*, U3b, P3/P4, T048-U1…) está **integrado sin release**; entra con R1/R2.
- No quedan workflows ni pruebas en segundo plano de esta sesión. En el server (`yo@192.168.1.2`) el runner es `run-server.sh` **v7** y `SOLO_WINDOW` quedó retirado.
- Estado detallado y máquina-legible: `data/autonomous-roadmap-20261102/checkpoint.json` (claves de 2026-11-19) e inventario completo con citas y críticas: `data/autonomous-roadmap-20261102/session-closeout-20261119/roadmap-inventory.json`.

## Trabajo realizado (2026-11-18 → 2026-11-19)

Lotes integrados y empujados (cada unidad: implementación → revisión independiente → correcciones → verificación independiente con mutantes → suite completa en el server):

| Lote | Head | Contenido |
|---|---|---|
| H | `c70a8c20` | R1E + «desactivá» |
| I | `dc3affcc` | T030-U2 (temporizadores), T037-U2 (clima), T035-U1 |
| J | `bf133a32` | T033-U1 (rutinas por voz, solo lectura), T037-U2b |
| K | `4701a43b` | T037-U6 (comandos compuestos), T044-U1 (`POST /api/voices/captures/query`) |
| L | `77518f97` | T033-U1b (preguntas de rutinas con wake propio claro saltean el veto 7B), T030-U2b («listo» corta una alarma solo mientras suena), SC6B-B-M1 (docs del presupuesto de la primera transición) |
| M | `3dc3ef0a` | RSG (sin skips silenciosos de systemd real), T048-U1 (auto-recuperación XVF3800), runbook del mic corregido; luego este handoff |

Además:
- **SC-6b medición solo** a tamaño de producción (`sc6b/solo-20261119/RESULT.md`): con B real ≈146 s la ruta A da D≈660 s contra 600 s; el usuario aceptó **un** exceso (SC6B-B-M1). Los docs ahora exigen recalcular `D ≈ 222,4 + 3 × B` con el B de W8 y volver a preguntar si supera ~660 s.
- **Smoke T046 en la Mac** terminado; lectura F7 registrada (`t046/smoke/f7-reading-20261119.*`). El fix del harness (timeouts CDP en `Tracing.start`/`Network.clearBrowserCache`) sigue pendiente como cambio de método revisado.
- **Incidente USB/journald del server (2026-11-19)** — resuelto a distancia, sin reboot, sin volver de kernel y sin reiniciar kza-voice:
  - Causa: tras el reboot en caliente (kernel 6.8.0-149) los 4 XVF3800 quedaron con el isócrono trabado; en el controlador `0000:8c:00.0` (Cocina, Hall) eso es un `xhci_warn` sin rate-limit (~8k/s por stream) que hizo que el watchdog de journald lo matara 111 veces. No es una regresión del kernel (mismo flood en -110 el 2026-10-27/28).
  - Las fallas de pruebas con systemd real de ese día se explican por esos cuelgues (el user manager bloquea hasta 10 s por línea de log): corridas antes de las 14:30 son inválidas.
  - Contención 14:30 (desautorizar Cocina/Hall) y **recuperación 16:27–16:31 con el `REBOOT` de firmware** del XVF3800 + re-aplicar el tuning; verificado 4/4 rooms con audio, 0 overruns, journald estable.
  - Evidencia: `data/autonomous-roadmap-20261102/usb-overrun-20261119/` y `env-regression-20261119/`; checkpoint `usb_overrun_20261119`, `env_regression_20261119`.
- **Runner del server**: v6 partió `b1b` (1362–1456 s de 1500) en `b1b1`/`b1b2`; v7 exporta `KZA_REQUIRE_REAL_SYSTEMD=1`. Antes de confiar en un lote se compara `NRestarts` de journald antes/después (los scripts `*-run.sh` lo registran).

## Validación y evidencia

- Resultados de lotes: `data/autonomous-roadmap-20261102/u3b/full-batch-{h,i,j,k,l,m}/RESULT.md`. **Lote M tuvo 1 falla sin explicar** en compiled-solo (`test_compiled_slot_rollback_preserves_latest_privacy_authority`: el hijo candidato abortó con `runtime_admission_delegation_missing`); pasó 6/6 aislada en M, L+RSG y L y 8/8 repitiendo la secuencia completa en M, y M no toca ese camino. Se integró registrándola como intermitente **sin causa**, a explicar antes de R2 (ver Pendientes). Todos verdes salvo fallas solo-bajo-carga que pasan aisladas (regla del lote H): lote L tuvo 2 en `a2d` (bloqueo de loop 403 ms; `WorkCloseTimeout` al cerrar en la prueba nueva de T033-U1b), 5/5 aisladas. **No se ampliaron límites, timeouts ni se debilitaron aserciones.**
- Evidencia por unidad: `data/autonomous-roadmap-20261102/<unidad>/evidence.json` (`t030-u2b`, `t033-u1b`, `sc6b-b-m1`, `rs-guard`, `t048-u1`, …), con revisión, correcciones y `fix_verification`.
- Decisiones del usuario de esta sesión: `t040-t041/user-decisions-20261110.json` claves `morning_questions_20261119` y `xvf_auto_recovery_20261119`.

## Pendientes exactos

### Primero al retomar
- **Explicar la falla intermitente de `test_compiled_slot_rollback_preserves_latest_privacy_authority`** (lote M, `RESULT.md` y evidencia en `/home/yo/projects/runs/full-batch-m/slot-rollback-failure/evidence/`): un registro de delegación se saltó como «stale» (`owner.live` falso o digest de `source_close` distinto) y la revalidación de admisión del candidato no encontró delegación. Es un ensayo de rollback de release: bloquea R2 hasta tener causa y, si corresponde, corrección revisada. Sin ampliar timeouts ni debilitar la aserción.

### Decisiones del usuario
1. **R1** (ventana con el usuario, OK explícito por cada cambio de producción): firma visual del panel R1-B2 «Olvidar grabaciones perdidas» y de las pantallas U3b-4a-3; fecha de la ventana; registrar autorización e ids P1=D y L4; drop-in sudo W7 el día; contingencias (detener servicio si el dry run da `torn`; regla revisada si `legacy_checkout > 0`, que cubre U3b-0e2 F2). Runbook: `docs/runbooks/2026-11-13-r1-adoption-window.md`; archivo fijado `releases-r1/r1-7752570.tar.gz`.
2. **R2**: cuándo, dentro de los 14 días de L4 tras R1; si el B medido en W8 da D > ~660 s, decidir de nuevo el presupuesto.
3. **T044-U1**: aceptar el scan de ~0,79 s a 4096 filas o exigir optimización antes de la release que lo lleve (RV3); confirmar la redacción del fence de revocación (RV6); autorizar el chequeo read-only de NGINX de `kza-dashboard-proxy` (R2-10).
4. **T048-U1**: mantener 2 recuperaciones fallidas antes del primer `REBOOT` (~9 min) o usar 1 tras un reboot del host.
5. Preguntas de producto: «listo» con wake débil se veta en silencio (T030-U2b RV2); turno que es pregunta de rutina y comando a la vez (T033-U1); temporizador homónimo creado mientras el viejo suena (T030-U2-FV-R1).
6. **T046**: aprobar el cambio de método del harness (reintento/objetivo nuevo ante timeouts CDP) con hash de preregistro nuevo.
7. **U3b-4c**: objetivo de falsos aceptados y ventana física P7.
8. **Server (admin)**: conservar o quitar `/etc/rsyslog.d/10-drop-xhci-overrun.conf` y `/root/fix-logs-20261119` (creados por una sesión root no atribuida); política de journald `ReadKMsg`; alerta de host por `NRestarts>0`.
9. Momento de release de los fixes de voz y la prueba en vivo de Cocina con parlantes.

### Release (R1 → R2)
- R2 lleva **todo desde SC-0** (SC-0/0b/1a/1b y lotes C–M), no solo F–M.
- Prerrequisitos en el commit de release: agregar el par `(5,1)` y su digest a `RELEASED_CONTROLLERS`; repetir SCR-P-3 (ensayo compilado CR-7 R1-H) solo en el server; regla del paso 0 (`resurrected != 0` ⇒ sin transición hasta implementar R5V-2/R5V-3).
- Medir sizes en el dry run de R1 (límite de binding de 16 MB, ~250k filas protegidas); U3b-0c: cronometrar una compactación sobre copia del estado de producción.
- SCHEMA-V2 (P3-F0) va después de R2 como release propia revisada.

### Unidades listas (dependencias cumplidas), en este orden sugerido
T043-U1 → T043-U2 (antes de la baseline T046, decisión QL), T033-U2, T033-U3a, T033-U6, T030-U4, T037-U7, W-OBS-2, T042-U4, T044-U2, T034-U3, docs OQ-8 pre-H6 (F11/F15/F16, precondición D7), salvedades T040/T041 de desarrollo. Cada una por impl → revisión → verificación → lote.

### Encadenadas / bloqueadas
- T033-U3b, T033-U3c, T033-U4, T033-U5, T035-U2, T031-U5 y siguientes: tras las listas.
- T036-U2: tras T021 estable y T023 (mute global), ambos `in_progress`.
- T042-U3, T045-U3, T042-U5, T035-U3: tras cerrar U3b.
- P4-D0/T046: medir la baseline T046-U1 (40 corridas por vista) en la Mac después de T043-U1/U2 y del fix del harness.
- U3b abierto: U3b-0d4b (cleaners faltantes y escaneo de resurrección), U3b-0e2 F2, U3b-0c con estado de producción, U3b-1b F1/F2, U3b-4b-3 (decidir con 4c), U3b-4c (P7).
- F1 (T024-F1: censo global/owners, cierre SDK genuino, migración legacy, guards, cleanup ACK) y T021–T023: re-planificar contra el head actual antes de retomar.
- P5 (T050–T054), P6 (T060–T064), P7 (T070–T075, ventana física con el usuario) y P8: sin empezar.

### Intermitentes e higiene
- Investigar bajo carga: `test_loop_hygiene::test_sync_commands_keeps_the_loop_responsive` (403 ms), `test_routine_voice_factory::…keep_the_7b_veto[untagged]` (cierre de erasure), `test_memory_home_denial[home-True]`.
- Pruebas compiladas con margen chico: tope de fase 55 s contra 46–52 s de `advance-N` (dividir o reducir trabajo, nunca subir el tope). `b2` (~1200 s) es la próxima tanda a partir.
- Salud de producción a revisar read-only: conexión a HA registrada como `auth_error` (cualquier arreglo de token requiere OK del usuario).
- Fuentes desactualizadas a reconciliar (no se tocaron): `DEPLOYMENT.md` («Living y Cocina sin captura útil», «diseño v4 propuesto»), `CLAUDE.md` (dice que producción corre `b0866e8`; corre `e9669ec` con estado en `b0866e8`), `readiness-r1-r2-20261118.md`, `p3-p4-plan.json` (Q9d ya respondida), `p4-d0/evidence.json` status.
- Limpieza: clones de la Mac en `/Users/gabriel/.claude/jobs/3a8b26c3/tmp` (~1,4 GB; borrar los integrados tras comparar contenido, regla de Worktrees de `CLAUDE.md`); clones/salidas en `/home/yo/projects/runs`; checkout viejo `/home/kza/kza` tras las ventanas de rollback.

## Prompt para sesión nueva

```text
Retomá el roadmap de KZA (docs/plans/2026-11-02-autonomous-roadmap.md) desde /Users/gabriel/Documents/Proyectos/kza sin reiniciarlo. La sesión del 2026-11-19 quedó pausada por mi pedido.

Leé AGENTS.md, CLAUDE.md, docs/runbooks/2026-11-19-session-handoff.md, data/autonomous-roadmap-20261102/checkpoint.json (claves de 2026-11-19 y resume_runbook) y session-closeout-20261119/roadmap-inventory.json. Confirmá usuario, ruta, Serena/proyecto, HEAD/origin de recovery/production-2026-10-25 (debe ser el lote M o posterior), cambios ajenos (otra sesión usa el checkout principal en feat/kza-multi-console-workflow: no hagas checkout ahí; integrá desde clones y empujá por SHA) y procesos reales en el server.

Producción sigue en e9669ec; no hubo release. Empezá por las decisiones pendientes que necesito tomar (sección «Decisiones del usuario») y seguí con las unidades listas en el orden del handoff, cada una por impl → revisión independiente → correcciones → verificación con mutantes → lote con suite completa (runner v7 en el server, journald antes/después). R1 lo hacemos juntos, con OK explícito por cada cambio de producción.

Pruebas solo en el server como yo (taskset -c 24-47 nice -n 10, flock feature-pytest.lock, respetar SOLO_WINDOW, hasta cuatro pytest), ninguna en la Mac. No ocultes fallos con GC forzado, aserciones debilitadas ni ampliaciones de timeouts. Conservá voz/persona separado de permisos. Sin release ni cambios de producción, credenciales o índices RAG como efecto secundario. No declares F1 ni el roadmap completos por un hito parcial.
```
