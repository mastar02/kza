# Roadmap ejecutable de Nexa v2

Estado al 2026-11-25. Base de la purga: `v2/main@18a9e8e4`. Esta es la referencia del repositorio para las olas de Nexa. Los planes P6/P7 del monolito corresponden a otra línea de desarrollo.

## Olas y evidencia de cierre

| Ola | Estado | Resultado o criterio para cerrarla |
| --- | --- | --- |
| 0 · Base | Completada | Workspace Rust, contratos gRPC, configuración tipada, cuenta dedicada, primeras fixtures y CI integrados en `7d9a8800`. |
| 1 · Componentes | Completada | DSP y captura, worker ML y motores de NLU, HA, hogar, LLM y memoria implementados. Revisar T2.1–T4.5 en el historial y los contratos en [crates](../../crates/). |
| 2 · Integración | Completada | Adaptadores de hardware T2.2, pipeline de voz T4.6, plano de control Axum T5.1 y SPA T5.2 integrados. Evidencia en `2d47215b`, `e99446b0`, `1d004471` y `390d798c`. |
| 3 · Entrega y paridad | Completada | L4 Playwright, migración offline T6.1 y paquete/rollback T6.2 con smoke L5. La CI enlaza las suites en `f9f4e3e5`. |
| 4 · Validación y cutover | En curso | Validar el candidato Linux en hardware, ensayar migración y rollback, reconciliar perfil y units, obtener GO y documentar estabilidad en campo. |
| 5 · Retiro y opcionales | Fase A integrada en `v2/main` | T7.0b aprobada por QA cross-model e integrada en `v2/main`. El retiro del runtime antiguo en servidor sigue sujeto al gate de Fase B. Priorizar los opcionales por separado. |

Los bloques A y B de paridad y los fixes Linux se reportan completados en la línea de trabajo. En esta base se pueden inspeccionar el protocolo (`046c69c0`), el endurecimiento de NLU/LLM/ops (`732f2551`, `50d4b334`, `8a97c0b7`), memoria y hogar (`98dc3693`, `cf33fb38`), web, migrador y brain (`0355975a`, `5ede0220`, `dab0b945`). Antes de usar un paquete, verificar que incluya los fixes Linux del perfil operativo; hay diferencias concretas en las units de esta base. [Contrato y diferencias de despliegue](DEPLOYMENT.md).

### Secuencia de la Ola 4

1. **Ops** identifica SHA y manifiesto limpios. Contrasta daemon de audio compilado con hardware, ORT, sello escribible, PYTHONPATH, caché offline y autenticación Bearer del LLM con las units efectivas. Registra PCI, VRAM y puertos según [hardware](HARDWARE.md).
2. **QA y privacidad** revisan L3, L4 y L5 en el candidato, con QA independiente por familia real. Verifican cuatro entradas físicas, permisos UDS, denegación de red en ML/memoria, revocaciones, RBAC y ausencia del modo de pruebas en release. Registrar MA1260 en simulación como tal.
3. **Ops** ensaya migración offline en copia consistente y rollback cronometrado (objetivo < 60 s). Entrega hashes, conteos y resultado sin datos privados. Si falla una condición, conservar el candidato anterior y corregir en la rama de trabajo.
4. **Coordinación** presenta evidencia y solicita GO de cutover. Tras la promoción autorizada, registrar observación estable, incidencias y disponibilidad de rollback. La retirada del host requiere una decisión posterior.

## T7.0b · Purga Fase A · HECHO e integrado en `v2/main`

El árbol de Nexa v2 conserva el workspace Rust, el worker ML, la SPA, el despliegue nativo, la configuración TOML y el migrador offline. Las pruebas del orquestador viven en `tests-v2/workflow/`. La CI las ejecuta junto a los gates de v2. Los cinco scripts de orquestación y hooks, sus skills y los templates de asientos siguen en el checkout.

Se retiraron del árbol de desarrollo el runtime Python del monolito, sus pruebas, herramientas, infraestructura y documentación histórica. Para consultar el árbol inmediatamente anterior, usar el tag [archive/pre-purge-v2-2026-11](https://github.com/mastar02/kza/tree/archive/pre-purge-v2-2026-11) en `18a9e8e49f719e37c76f1e7f1c2013768e8d5c42`. El tag `archive/legacy-monolith-2026-11` y la rama `recovery/production-2026-10-25` identifican el rescate en `d0defada`. Se conserva el [manifiesto de rescate](../research/2026-10-25-production-recovery.json).

El código del índice (`src/code_index/`), su unit, el instalador del hook y `tools/code_search.py` quedan en el archivo previo a la purga. El contenedor del servidor `kza-code-index-ctr` sigue fuera del alcance de ambas fases. Aprobado por QA cross-model independiente e integrado en `v2/main`.

### Fase B · Servidor 192.168.1.2

Requiere cutover estable documentado **y GO explícito de retiro en el servidor**. El dueño de operación identifica cada ruta, unit, proceso, volumen y dependencia del runtime sustituido, verifica el respaldo privado y el rollback y presenta la lista exacta al aprobador. Conservar estado, ledgers y revocaciones durante toda la transición; aplicar las políticas de retención y borrado también a las copias.

Excluir expresamente `kza-code-index`, `kza-chroma`, `postpilot-embeddings-*`, `infra`, `ha-core`, cualquier servicio de otro dueño y todos los recursos nuevos de `nexa`. No inferir propiedad por prefijo, puerto, GPU o cuenta. Una vez autorizado el inventario, retirar solo los elementos aprobados, verificar salud de Nexa y servicios compartidos y guardar evidencia sin PII. Si la atribución es ambigua o el rollback deja de estar disponible, suspender el retiro.
