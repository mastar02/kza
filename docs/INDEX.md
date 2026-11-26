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

[Árbol previo a la purga](https://github.com/mastar02/kza/tree/archive/pre-purge-v2-2026-11) · [Rescate del monolito](https://github.com/mastar02/kza/tree/archive/legacy-monolith-2026-11).
