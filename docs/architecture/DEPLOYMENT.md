# Deploy de KZA — fuente de verdad

> Si otro documento de este repo contradice a este, gana este (y ante dudas de
> contrato compartido, `docs/SERVER_CONVENTIONS.md` / Notion pág. 8).

## Desarrollo y despliegue acordados — 2026-10-25

El desarrollo, los tests, la construcción y Git se ejecutan como `yo` en
`/home/yo/projects/kza`. `kza` es la cuenta administrativa y de ejecución del
servidor (UID 1000, grupo `sudo`); no se usa para desarrollar ni compilar.

El próximo modelo de despliegue transferirá a `kza` únicamente el artefacto
compilado y sus recursos de ejecución, construidos y probados bajo `yo`.
Secretos y datos persistentes permanecen fuera del artefacto y de Git. El
proyecto es Python: el mecanismo de compilación, instalación y promoción aún
debe implementarse. No se cambió el servicio durante el rescate.

La sección siguiente describe el despliegue existente. El flujo histórico de
edición en producción y puente desde la laptop queda retirado para trabajo
nuevo. Consultar `AGENTS.md` en la raíz para las instrucciones operativas.

## Cómo corre producción (server `192.168.1.2`, usuario `kza`)

- **kza-voice.service** — pipeline de voz **nativo** (NO contenedor), systemd `--user` en
  `/home/kza/.config/systemd/user/`. Es la **excepción R10 documentada** del contrato del
  servidor (acceso USB ReSpeaker + MA1260 serial + presupuesto <300ms hacen al contenedor
  inviable). Ver `docs/SERVER_CONVENTIONS.md` § Excepciones a R10.
- **llama-server :8101** (Qwen2.5-7B Q4_K_M, cuda:1) — fast-path NLU, bajo `kza`.
- **Gateway LiteLLM :8200 → MiniMax cloud** — reasoner slow-path, bajo `infra` (no lo
  administra KZA; consumir con virtual key).
- **ChromaDB** — hoy in-process dentro de kza-voice. Existe el Quadlet
  `kza-chroma.container` para migrarlo a :9500, pero NO está operativo: la auditoría
  2026-05-30 lo encontró en crash-loop (monta `chroma` en vez de `chroma_db` —
  fix pendiente; ver `docs/SERVER_CONVENTIONS.md` § deudas 2026-05-30).
- Secrets en `/home/kza/secrets/` (`.env`, `*-api-key.env`), inyectados vía
  `EnvironmentFile=` del unit.

## Cómo se deployaba — pendiente de reemplazo por artefactos

Flujo git local ↔ server ↔ GitHub histórico documentado en
[`ECOSISTEMA_LOCAL_SERVER.md`](ECOSISTEMA_LOCAL_SERVER.md): el server es repo git y
deploya **in-place** (el código que corre es el working tree), no pushea; la laptop es el
puente (`scripts/kza-push` / `kza-sync`). Restart manual del service cuando el cambio lo
requiere — chequear VRAM libre antes (`nvidia-smi`; preflight integrado avisa si <1500MB).

## Qué NO es producción

- **`docker/` (Dockerfiles + services) — EXPERIMENTAL, sin paridad** con el monolito
  canónico (`python -m src.main`). Ver `docker/README.md` para el detalle de gaps.
  No deployar producción con esto.
- **docker-compose** — el `docker-compose.yml` de la raíz del repo pertenece al modo
  experimental de `docker/` (mismo estatus: sin paridad, no producción); las menciones
  en `docs/research/` y `docs/plans/` de 2026-02/03 son propuestas históricas. Nada de
  compose corre en producción.
- Units de sistema en `/etc/systemd/system/` — prohibido por contrato (todo va en
  systemd `--user`).
