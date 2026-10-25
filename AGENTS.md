# Trabajo en KZA

## Cuenta y directorio de desarrollo

- Desarrollar, revisar, probar, construir, commitear y pushear como usuario `yo`,
  desde `/home/yo/projects/kza`.
- `kza` es la cuenta administrativa y de ejecución del servidor. Tiene UID 1000
  y pertenece a `sudo`; no usar sus privilegios para tareas de desarrollo.
- No editar el código ni instalar dependencias en `/home/kza/kza` o
  `/home/kza/app`. Son el despliegue existente, no un checkout de desarrollo.
- Mantener secretos, datos de ejecución y modelos fuera de Git.
- Antes de cerrar una sesión, guardar el trabajo autorizado en commits y subir
  la rama a GitHub. Indicar expresamente cualquier trabajo que quede sin subir.

## Despliegue acordado

- Construir y probar el artefacto de release bajo `yo`; transferir a `kza`
  únicamente el artefacto compilado y los recursos necesarios para ejecutarlo.
- No desarrollar, compilar ni hacer `git pull` en la cuenta `kza` como parte del
  nuevo flujo. La configuración privada y los datos persistentes siguen siendo
  responsabilidad del entorno de ejecución.
- El proyecto actual es Python. El mecanismo de compilación y promoción de
  artefactos todavía está pendiente de implementación; no presentar el
  despliegue existente como si ya cumpliera este modelo.
## Contexto

- Rescate de producción: rama `recovery/production-2026-10-25` y manifiesto
  `docs/research/2026-10-25-production-recovery.json`.
- Desarrollo y despliegue: `docs/architecture/DEPLOYMENT.md`.
- Convenciones del servidor compartido: `docs/SERVER_CONVENTIONS.md`.
- Arquitectura y convenciones de código: `CLAUDE.md`.
