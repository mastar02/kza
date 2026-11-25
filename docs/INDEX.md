# Índice de documentación

## Nexa v2 vigente

| Consulta | Fuente |
| --- | --- |
| Qué corre y cómo se organiza el repositorio | [Introducción](../README.md) y [guía operativa](referencia/guia-operativa.md) |
| Qué está integrado, cómo cerrar la Ola 4 y qué se puede retirar después | [Roadmap y cierre de T7.0b](architecture/NEXA_V2_ROADMAP.md) |
| Cuenta, configuración, migración, empaquetado y rollback | [Despliegue](architecture/DEPLOYMENT.md) y [contrato de migración](../tools/nexa-migrate/README.md) |
| Presupuesto de VRAM y puertos físicos de los micrófonos | [Hardware](architecture/HARDWARE.md) |
| Convivencia con otros servicios | [Nota operativa y espejo histórico de Notion](SERVER_CONVENTIONS.md) |
| Desarrollo, pruebas y workflow multi-consola | [AGENTS](../AGENTS.md), [CLAUDE](../CLAUDE.md) y [CI de v2](../.github/workflows/v2-ci.yml) |

## Runbooks de hardware y entorno

- [Firmware XVF3800 de seis canales](runbooks/2026-06-06-xvf3800-flasheo-6ch.md) y [diagnóstico de micrófono mudo](runbooks/2026-07-25-diagnostico-mic-mudo-isocronos.md). Aplicar al dispositivo identificado por puerto físico.
- [Entorno de desarrollo en Mac](runbooks/2026-11-05-mac-development-handoff.md). Sus resultados de pruebas corresponden al checkout y a la fecha registrados.

## Archivo del monolito

**T7.0b · HECHO e integrado en `v2/main`.** El [roadmap](architecture/NEXA_V2_ROADMAP.md) identifica los dos tags de archivo y el gate de retiro en el servidor. Las pruebas del orquestador se mantienen en [tests-v2/workflow](../tests-v2/workflow/). El [manifiesto de rescate](research/2026-10-25-production-recovery.json) conserva la trazabilidad de producción.
