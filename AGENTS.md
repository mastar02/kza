# Trabajo en KZA

## Cuenta y directorio de desarrollo

- En el entorno Linux, desarrollar, revisar, probar, construir, commitear y
  pushear como usuario `yo`, desde `/home/yo/projects/kza`.
- La migración a Mac fue autorizada el 2026-11-05. En ese equipo, usar la
  cuenta habitual sin privilegios y su checkout local; confirmar usuario,
  ruta y arquitectura reales al retomar, sin reproducir la cuenta ni la ruta
  Linux. El estado verificado de dependencias y pruebas locales en Mac está
  registrado en `docs/runbooks/2026-11-05-mac-development-handoff.md`.
- `kza` es la cuenta administrativa y de ejecución del monolito histórico. Tiene UID 1000
  y pertenece a `sudo`; no usar sus privilegios para tareas de desarrollo.
- No editar el código ni instalar dependencias en `/home/kza/kza` o
  `/home/kza/app`. Son el despliegue existente, no un checkout de desarrollo.
- Los servicios legacy a cargo de `kza` conservan su cuenta. Los archivos de
  la versión instalada se conservan hasta que una release los sustituya.
- Mantener secretos, datos de ejecución y modelos fuera de Git.
- Antes de cerrar una sesión, guardar el trabajo autorizado en commits y subir
  la rama a GitHub. Indicar expresamente cualquier trabajo que quede sin subir.

## Despliegue acordado

- Ejecutar Nexa v2 bajo el usuario dedicado `nexa`, sin privilegios administrativos. En Linux construir y probar como `yo`; transferir únicamente el paquete para la arquitectura destino y los recursos necesarios. Separar configuración privada, modelos y estado de la release.
- Empaquetar con `deploy/nexa/package-release.py` y promover o revertir con `deploy/nexa/install-release.sh`. Requerir GO explícito antes de publicar, instalar o reiniciar en producción. Nunca desarrollar, compilar ni hacer pull en la cuenta `kza`. Conservar la versión instalada y su rollback; la limpieza del servidor requiere GO separado. No solicitar privilegios para tareas de desarrollo.
- Aplicar todas las prohibiciones de Git y privacidad de [CLAUDE.md](CLAUDE.md). Seleccionar rutas explícitas y subir solo la rama de trabajo. Mantener audio, PII y `casa.glb` fuera del repositorio.

## Delegación de agentes y workflow multi-consola

- Podemos usar más de cuatro agentes cuando haya trabajo útil para paralelizar.
  El máximo acordado es de ocho agentes activos en total, incluido el
  coordinador.
- La cantidad efectiva respeta las plazas disponibles en cada sesión. Si el
  entorno admite menos de ocho, usar ese límite sin interpretar cuatro como un
  máximo permanente del proyecto.
- En Mac (`herdr-app` + `hc`), los especialistas se abren en worktrees aislados
  (`~/.herdr/worktrees/kza/<alias>`) mediante `scripts/kza_seat.py`, con
  enrutamiento dinámico ponderado por cuotas vía `scripts/kza_router.py`
  (`gpt-6-luna`) y `scripts/kza_quota.py`. Jamás tocar terminales, templates
  (`tu-*`, `gsm-*`) ni worktrees de otros proyectos.
- Las pruebas que requieran interacción con la PC, mediante el navegador o
  aplicaciones de escritorio (rol `kza-ui`), se delegan a un especialista
  seleccionado dinámicamente vía `scripts/kza_router.py` según capacidades,
  documentación/benchmarks vigentes y cuotas reales (sin fijar un modelo único
  en duro; p. ej. `gpt-6.1-terra` y modelos multimodales/computer-use futuros).
  El coordinador define qué verificar y revisa la evidencia obtenida.

- Exigir QA cross-model por familia real. Resolver los diálogos de confianza en el tab visible de herdr-app. Nunca omitirlos.

## Contexto

- Rescate de producción: rama `recovery/production-2026-10-25` y manifiesto
  `docs/research/2026-10-25-production-recovery.json`.
- Desarrollo y despliegue: `docs/architecture/DEPLOYMENT.md`.
- Convenciones del servidor compartido: `docs/SERVER_CONVENTIONS.md`.
- Arquitectura y convenciones de código: `CLAUDE.md` y
  `docs/referencia/guia-operativa.md`.
- Roadmap y condiciones de retiro: [manifiesto de Nexa v2](docs/architecture/NEXA_V2_ROADMAP.md).
