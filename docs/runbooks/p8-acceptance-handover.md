# P8 · Revisión, matriz y entrega final

T080 apoya la aceptación sobre la revisión independiente de P6/T060–T064 y
P7/T070–T075 en el mismo SHA Git y archivo sellado. T081 reúne la matriz
H0–H6, la guía de hogar y los runbooks aplicables. T082 se decide tras revisar
las siete áreas, todos los criterios C01–C21 y la aceptación humana final.

- [Roadmap y política de finalización](../plans/2026-11-02-autonomous-roadmap.json)
- [Ventana física y esquema de P7](p7-field-window.md)
- [Contrato de evaluación p6-v1](../plans/p6-evaluation.md)
- [Candidato, release y restauración](p6-release.md)

## Expediente privado

Usar el mismo candidato Epoch 2, el paquete exacto y los recibos Q16/Q17
emitidos antes de la ventana. El operador entrega los registros de campo y
el custodio entrega los resultados admitidos de la reserva. El revisor
comprueba fuente, cliente físico, orden, evidencia humana y ámbito
observable. Guardar la guía operativa, matriz, resultados y decisión final
con referencias verificables. Mantener los originales en custodia privada
con los plazos de su finalidad, borrado y revisión de restauraciones. Los
resúmenes usan IDs opacos, hashes y agregados sin contenido del hogar.

Preparar un directorio de artefactos que contenga cada registro revisable
bajo su SHA-256 en minúsculas, sin extensión. El auditor vuelve a calcular
los hashes. El contenido de los recibos JSON de Q16/Q17 y T082 también se
coteja con el expediente. Verificar la autoría y autoridad de cada decisión
por el canal privado acordado. Los hashes vinculan contenido y versión, y
la revisión humana acredita el origen de cada medición.

El JSON del expediente incluye estos campos.

| Campo | Contenido |
| --- | --- |
| `candidate` | `git_commit`, `archive_sha256`, `schema_contract_epoch: 2`. |
| `tasks` | Objeto indexado por T060–T064 y T070–T075. Cada fila P7 usa el JSON aceptado por su validador. Las filas P6 usan `task`, `kind: "local"`, `candidate`, `evidence_sha256` y los resultados de la tabla inferior. |
| `criterion_reviews` | Objeto por C01–C21. Cada fila incluye `verdict: "accepted"`, `independent: true`, `original_observations_reviewed: true`, `reviewer`, `review_sha256` y `sources`, mapa de tarea a SHA de su registro. |
| `handover` | Revisor, `independent_review_sha256`, `handover_sha256` y `human_acceptance_sha256`. Adjuntar los resultados `independent_review_complete`, `home_guide_updated`, `runbooks_updated`, `matrix_updated`, `remaining_limits_documented` y `human_final_acceptance`. |

T082 adjunta un recibo JSON de la decisión humana con `decision: "accepted"`,
`candidate`, `profile_sha256` y `handover_sha256` del paquete aprobado.
El candidato del expediente coincide con cada validación. El H3 de P7 se
recalcula sobre los mismos sellos y clips de calibración T071. La verificación
de P7 incluye inventario Q16, observaciones de las integraciones antes y
después de promoción, evidencia del iPhone y de accesibilidad, trazas de
recuperación, adopción, censo, restore, cierre limpio y ensayo L4.

| Tarea P6 | Resultados locales a adjuntar |
| --- | --- |
| T060 | Familias incluidas y `factory_families` con obligatorios/pasados desde la fábrica real, actor, política, persistencia, efecto, respuesta, reinicio y fallos. `negative_cases` ≥200, `invariant_violations` = 0, `h2` con diez repeticiones y sus límites del roadmap. Suite, Tier 2, compiled-solo, ausencia de xfail requerido y revisión de warnings. |
| T061 | `soak_seconds`, `useful_seconds` de KZA y cuatro entradas, `recovery_seconds` para HA, WS y mic. Candidato congelado, privacidad y mute durable, carreras de productores tardíos, H2, heartbeat de entradas sanas y fallos fuera del soak. |
| T062 | `protocol: "p6-v1"`, `protocol_sha256`, resultados del evaluador y plan sellado de reserva independiente. |
| T063 | Artefactos reproducibles (`rebuilt_archive_sha256`), dependencias y arranque verificados fuera de fuentes, estado de entrenamiento, restore compatible y rollback de runtime y datos sin resurrección ni reenvío. |
| T064 | `release_review_sha256` del JSON P6 original, comprobación `--verify` y revisión del guion de rollback. |

Fuentes requeridas por criterio en `criterion_reviews.sources`.

| Criterios | Fuentes |
| --- | --- |
| C01 | T060 |
| C02 | T060, T063 |
| C03 | T060, T061 |
| C04 | T063, T071, T075 |
| C05 | T060, T063, T064, T075 |
| C06 | T080 |
| C07–C11 | T073 |
| C12 y C18 | T062, T071, T072 |
| C13–C17 | T062, T072 |
| C19 y C20 | T061, T070, T074 |
| C21 | T070, T075 |

Para C06, `observability` incluye `included_components`, `components` con
fuente real, instante/caducidad y efecto de backend comprobado, y
`verified_scenarios` con 401, 403, 409, 503, timeout, desconexión,
reconexión, dato vencido, parcial e incierto.

## Comprobar y entregar

El roadmap de entrada es el registro revisado por el coordinador. Incluir
ambos recibos en `field_authorization_evidence`. Cada tarea P6/P7 debe
referenciar su `evidence_sha256` en `evidence`. Cada criterio aceptado añade
su `review_sha256`. Para T080/T081/T082 registrar los recibos de revisión,
entrega y aceptación humana respectivamente. Las tareas requeridas usan
`verified_local` o `accepted` según el vocabulario del roadmap. Para las
tareas con gate humano, `accepted` representa la aceptación de campo.
T082 puede documentar el expediente revisado en `verified_local` antes del
cambio autorizado a estado final. Registrar el cierre de cada hallazgo
incluido y el estado de aceptación de cada área.

Ejecutar desde la raíz con CPython 3.13. `ROADMAP` apunta al JSON revisado,
`PRIVATE` al directorio de trabajo privado y `PY` al intérprete del venv.

```bash
umask 077
"$PY" -B -m tools.p8_acceptance_audit report --roadmap "$ROADMAP" \
  --bundle "$PRIVATE/bundle.json" --artifacts "$PRIVATE/artifacts" > "$PRIVATE/p8-matrix.json"
"$PY" -B -m tools.p8_acceptance_audit check --roadmap "$ROADMAP" \
  --bundle "$PRIVATE/bundle.json" --artifacts "$PRIVATE/artifacts"
```

`report` emite la matriz y bloqueos para T081. `check` exige
`T082: "ready_for_closure"` para devolver 0. Un expediente legible y
bloqueado devuelve 1 en `check` y conserva su JSON de diagnóstico. Una
entrada ilegible o un contrato alterado devuelve 1 en ambos modos.
El auditor recalcula los dos hashes fijados del roadmap y verifica el
alcance base de tareas, hallazgos, fases, áreas y política de finalización.
Ambos modos son de sólo lectura. El coordinador registra el cierre tras
comprobar el resultado, la aceptación humana y la entrega. Un estrato
fallido, una dependencia física ausente o un criterio sin evaluar queda
como bloqueo y debe figurar en la matriz entregada.
