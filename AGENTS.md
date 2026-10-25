# Trabajo en KZA

## Cuenta y directorio de desarrollo

- Desarrollar, revisar, probar, construir, commitear y pushear como usuario `yo`,
  desde `/home/yo/projects/kza`.
- `kza` es la cuenta administrativa y de ejecución del servidor. Tiene UID 1000
  y pertenece a `sudo`; no usar sus privilegios para tareas de desarrollo.
- No editar el código ni instalar dependencias en `/home/kza/kza` o
  `/home/kza/app`. Son el despliegue existente, no un checkout de desarrollo.
- El servidor y sus servicios siguen ejecutándose como `kza`. Los archivos de
  la versión instalada se conservan hasta que una release los sustituya.
- Mantener secretos, datos de ejecución y modelos fuera de Git.
- Antes de cerrar una sesión, guardar el trabajo autorizado en commits y subir
  la rama a GitHub. Indicar expresamente cualquier trabajo que quede sin subir.

## Despliegue acordado

- Publicar en `kza` únicamente al hacer una release. Las tareas de desarrollo
  bajo `yo` no implican instalaciones, cambios de usuario ni reinicios de
  producción.
- Construir y probar el artefacto de release bajo `yo`; transferir a `kza`
  únicamente el artefacto compilado y los recursos necesarios para ejecutarlo.
- No desarrollar, compilar ni hacer `git pull` en la cuenta `kza` como parte del
  nuevo flujo. La configuración privada y los datos persistentes siguen siendo
  responsabilidad del entorno de ejecución.
- El proyecto actual es Python. `scripts/build_runtime.py` prepara un artefacto
  de bytecode CPython 3.13 sin fuentes `.py`; la instalación, promoción y
  retirada del material de desarrollo se harán con la próxima release.
  No borrar los archivos que necesita la versión instalada para adelantar esa
  limpieza, ni solicitar acceso administrativo mientras sólo se desarrolla.

## Contexto

- Rescate de producción: rama `recovery/production-2026-10-25` y manifiesto
  `docs/research/2026-10-25-production-recovery.json`.
- Desarrollo y despliegue: `docs/architecture/DEPLOYMENT.md`.
- Convenciones del servidor compartido: `docs/SERVER_CONVENTIONS.md`.
- Arquitectura y convenciones de código: `CLAUDE.md`.
