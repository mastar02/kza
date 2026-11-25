# Índice de documentación

## Nexa v2 vigente

| Consulta | Fuente |
| --- | --- |
| Qué corre y cómo se organiza el repositorio | [Introducción](../README.md) y [guía operativa](referencia/guia-operativa.md) |
| Qué está integrado, cómo cerrar la Ola 4 y qué se puede retirar después | [Roadmap y manifiesto T7.0a](architecture/NEXA_V2_ROADMAP.md) |
| Cuenta, configuración, migración, empaquetado y rollback | [Despliegue](architecture/DEPLOYMENT.md) y [contrato de migración](../tools/nexa-migrate/README.md) |
| Presupuesto de VRAM y puertos físicos de los micrófonos | [Hardware](architecture/HARDWARE.md) |
| Convivencia con otros servicios | [Nota operativa y espejo histórico de Notion](SERVER_CONVENTIONS.md) |
| Desarrollo, pruebas y workflow multi-consola | [AGENTS](../AGENTS.md), [CLAUDE](../CLAUDE.md) y [CI de v2](../.github/workflows/v2-ci.yml) |

## Runbooks de hardware y entorno

- [Firmware XVF3800 de seis canales](runbooks/2026-06-06-xvf3800-flasheo-6ch.md) y [diagnóstico de micrófono mudo](runbooks/2026-07-25-diagnostico-mic-mudo-isocronos.md). Aplicar al dispositivo identificado por puerto físico.
- [Diagnóstico del flood USB en journald](runbooks/2026-10-28-journald-usb-flood.md). Conservar el aislamiento de los demás servicios del host.
- [Entorno de desarrollo en Mac](runbooks/2026-11-05-mac-development-handoff.md). Sus resultados de pruebas corresponden al checkout y a la fecha registrados.
- [Índice de código](runbooks/CODE_INDEX.md). Servicio compartido con ciclo de vida independiente.

## Material histórico previsto para T7.0b

Los planes, investigaciones, documentos del monolito y procedimientos de despliegue Python describen versiones anteriores. El archivo base es el tag `archive/legacy-monolith-2026-11` en `d0defada`. Consultar las exclusiones, diferencias posteriores y condiciones de aprobación en el [manifiesto](architecture/NEXA_V2_ROADMAP.md) antes de retirar archivos.
