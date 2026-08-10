# Satélites de KZA en Podman rootless — design spec

> 2026-08-10. Continuación de `docs/architecture/ROOTLESS_MIGRATION.md`: cerrada la parte de
> cuenta/sudo, esta pieza contiene la huella de los componentes satélite en contenedores
> rootless, para que dependencias pesadas (el caso Hermes) nunca más se instalen sueltas en la
> cuenta compartida. Convenciones: `docs/SERVER_CONVENTIONS.md` (R4 naming, R8 restart, R10
> rootless, R12 tags inmutables).

## Alcance

**Se contenerizan (bajo el usuario `kza`, Podman rootless + Quadlet):**

1. **`kza-hermes`** — Hermes CLI con su stack de dependencias (Xvfb/browser) aislado.
2. **`kza-chroma`** — ChromaDB como servicio en `127.0.0.1:9516` (reactivación del quadlet
   `kza-chroma.container.disabled-2026-05-30`).
3. **`kza-code-index`** — el servicio `:9515`, hoy nativo, pasa a imagen propia y consume el
   Chroma servicio.

> **Corrección post-aprobación (exploración del plan):** el spec aprobado decía `:9500`, pero
> ese puerto está ocupado por el dashboard de obs (verificado con `ss -ltnp`; el quadlet
> deshabilitado de 05-30 quedó stale). Se usa **`:9516`** (libre, contiguo al 9515).

**Quedan nativos, con justificación documentada (NO son deuda):**

- **`kza-voice`** — excepción R10 #4: USB ReSpeaker + serial MA1260 + presupuesto <300ms.
- **`kza-llm-fast` (llama-server :8101)** — cuda:1 está en ~7.4/8 GB; el overhead de CDI
  (~0.34 GB medido en la excepción vLLM 2026-04-20) no cabe. Mismo criterio que vLLM.

**Non-goals de esta pieza:**

- El flip `reasoner.mode=hermes_cli` — sigue gateado por su checklist pre-flip
  (`config/settings.yaml:420-465`). Acá solo se deja el binario invocable.
- Migrar el vectordb del pipeline de voz (`src/vectordb/`) al Chroma servicio — el fast path
  conserva su ChromaDB in-process.
- Contenerizar `open-webui`/`searxng` (ya corren como containers bajo kza, no se tocan).

## Arquitectura

```
kza-voice (nativo)
  └─ HermesCliReasoner ── subprocess ──> /home/kza/bin/hermes-ctr (wrapper host)
                                            └─ podman exec ──> [kza-hermes] ── HTTPS ──> Codex
tools/code_search.py / agentes
  └─ HTTP :9515 ──> [kza-code-index] ── kza-internal DNS (kza-chroma:8000) ──> [kza-chroma]
                         └─ /home/kza/app montado read-only (fuente a indexar)
```

- Red: `kza-internal.network` (ya existe). `kza-chroma` y `kza-code-index` se resuelven por
  DNS interno de Podman; solo `:9515` y `:9516` se publican, ambos bind `127.0.0.1`.
- `kza-hermes` no publica puertos ni usa GPU; solo salida HTTPS (slirp/pasta).
- Nada de esto requiere sudo, CDI ni grupos: 100% rootless bajo `kza`.

## Componentes

### 1. `kza-hermes` (container residente + wrapper)

- **Imagen**: `localhost/kza/hermes:<git-sha>`, `containers/hermes/Containerfile` en el repo.
  Base Ubuntu 24.04 + installer oficial de Hermes + el stack que su instalador requiere
  (Xvfb, mesa, libs de browser) — todo DENTRO de la imagen, nunca más `apt install` en el host.
- **Quadlet**: `containers/quadlets/kza-hermes.container` →
  `~/.config/containers/systemd/`. `Restart=on-failure`, `RestartSec=5s`,
  `StartLimitBurst=5` (R8). Proceso principal: `sleep infinity` (el container es un runtime
  residente; el trabajo entra por `podman exec`).
- **Estado/auth**: volumen `/home/kza/data/hermes-auth/` montado como `$HOME` del proceso
  hermes dentro del container. La auth se hace UNA vez, interactiva:
  `podman exec -it kza-hermes hermes auth add openai-codex` (device-code flow, igual que
  hermesdev con `hermes-t-t1`). Sobrevive restarts y rebuilds de imagen.
- **Wrapper host** `/home/kza/bin/hermes-ctr` (en el repo: `containers/hermes/hermes-ctr`):

  ```bash
  #!/bin/bash
  # Puente host→container para HermesCliReasoner. El timeout corre DENTRO del
  # container: si este cliente muere (kill del reasoner, ssh cortado), el
  # proceso hermes igual muere server-side — sin huerfanos.
  exec podman exec -i kza-hermes timeout -k 5 95 hermes "$@"
  ```

  `95 = hermes_timeout_s (90) + 5` de margen: el timeout local del reasoner dispara primero;
  el interno es la red de seguridad. Si `hermes_timeout_s` cambia en settings, este número se
  actualiza a mano (documentado en el propio wrapper).
- **Config**: `hermes_binary_path: /home/kza/bin/hermes-ctr` en el bloque comentado de
  `settings.yaml` (el flip sigue siendo decisión aparte).
- **Cero cambios en `src/llm/hermes_reasoner.py`**: el contrato subprocess se mantiene
  (exit codes, stdout/stderr, timeout local + kill de process group).

### 2. `kza-chroma` (servicio :9516)

- **Imagen**: `docker.io/chromadb/chroma` **pinneada por versión exacta** (elegir la última
  estable al implementar y fijarla en el quadlet; R12).
- **Quadlet**: reactivar/reescribir `kza-chroma.container`: `PublishPort=127.0.0.1:9516:8000`,
  `Volume=/home/kza/data/chroma-svc:/chroma/chroma:Z`, `Network=kza-internal.network`, R8.
- **Datos**: directorio nuevo `~/data/chroma-svc/` — no comparte nada con
  `~/app/data/chroma_db/` (vectordb del pipeline, intocado).

### 3. `kza-code-index` (:9515 contenerizado)

- **Imagen**: `localhost/kza/code-index:<git-sha>`, `containers/code-index/Containerfile`:
  python 3.13 slim + subset de requirements del servicio (BGE-M3 en CPU — sin CUDA, imagen
  chica). El modelo de embeddings NO se hornea en la imagen: se monta read-only el cache de
  HuggingFace del host (`~/.cache/huggingface` → `HF_HOME` del container) — ahí es donde
  `SentenceTransformer("BAAI/bge-m3")` lo resuelve hoy; la imagen queda chica.
- **Quadlet**: `kza-code-index.container`: `PublishPort=127.0.0.1:9515:9515`,
  `Volume=/home/kza/app:/app:ro,Z` (fuente a indexar, read-only),
  `Network=kza-internal.network`, R8. El plan debe verificar qué variables de entorno lee
  `src/code_index` hoy bajo la unit nativa (p.ej. la API key que exige el sync de Chroma) y
  replicarlas vía `EnvironmentFile=/home/kza/secrets/.env` — no asumir que no necesita ninguna.
  **Reemplaza** a la unit nativa `kza-code-index.service` (que se deshabilita, no se borra,
  hasta validar).
- **Único cambio de código Python del proyecto**: `src/code_index` gana un selector de
  backend Chroma por config (patrón DI del proyecto — el cliente se inyecta):

  ```yaml
  code_index:
    chroma:
      mode: embedded        # embedded (hoy, default) | http
      url: "http://kza-chroma:8000"   # solo mode=http
  ```

  `embedded` = `PersistentClient` (conducta actual, default: sin flip, nada cambia).
  `http` = `chromadb.HttpClient`. El deploy contenerizado arranca con `mode: http`.
- **Datos**: el índice se **reconstruye** contra el Chroma nuevo en el primer arranque
  (reindexar es barato); no se migra la colección embebida.

## Manejo de errores

| Fallo | Comportamiento |
|---|---|
| `kza-hermes` caído/no arranca | `podman exec` sale rc≠0 → `HermesCliReasoner` ya lo trata como fallo → fallback a `HttpReasoner` (:8200). El orden gate→fallback del consent NO se toca. |
| hermes colgado | Doble capa: timeout local del reasoner (mata process group host) + `timeout -k` interno (mata el proceso en el container aunque el cliente muera). |
| `kza-chroma` caído | code-index responde unhealthy en sus queries; `Restart=on-failure` lo levanta; los agentes ven error explícito en `:9515` (no silencioso). |
| Imagen ausente (typo de tag) | `systemctl --user start` falla ruidoso; las imágenes son `localhost/` — sin dependencia de registry externo en runtime. |
| Reindex incompleto al arrancar | El servicio expone su estado (según su health actual); los consumidores ya toleran STALE (memoria: ⚠ STALE ⇒ leer el archivo real). |

## Testing y verificación

- **TDD** para el selector embedded/http de `src/code_index` (mocks de ambos clientes;
  fixtures en `tests/`, patrón del proyecto). La suite completa queda verde.
- **Smoke post-deploy** (además de `tools/smoke_test.py`):
  1. `podman ps` muestra `kza-hermes`, `kza-chroma`, `kza-code-index` healthy.
  2. `/home/kza/bin/hermes-ctr --version` responde.
  3. Query conocida contra `:9515` devuelve resultados del repo real.
  4. `curl 127.0.0.1:9516/api/v2/heartbeat` responde.
  5. `systemctl --user is-active kza-voice kza-llm-fast` = active y
     `tools/benchmark_latency.py` dentro del rango (~150-280ms): la voz no se enteró.
- **Rollback por componente**: quadlet → tag de imagen anterior; code-index → re-habilitar la
  unit nativa (`mode: embedded`); hermes → el reasoner ya degrada solo a HttpReasoner.

## Estructura nueva en el repo

```
containers/
  hermes/Containerfile
  hermes/hermes-ctr
  code-index/Containerfile
  quadlets/kza-hermes.container
  quadlets/kza-chroma.container
  quadlets/kza-code-index.container
```

Deploy (runbook a detallar en el plan): build en server con tag `<git-sha>`, copiar quadlets a
`~/.config/containers/systemd/`, `systemctl --user daemon-reload`, arrancar en orden
chroma → code-index → hermes, correr smoke, deshabilitar la unit nativa de code-index.

## Decisiones tomadas (con quién y cuándo)

- Alcance "satélites + code-index"; voz y llama-server nativos — usuario, 2026-08-10.
- Chroma servicio consumido SOLO por code-index; pipeline intocado — usuario, 2026-08-10.
- Enfoque A: container residente + wrapper `podman exec` con timeout interno — usuario,
  2026-08-10.
- El flip de `reasoner.mode` y la auth interactiva de Hermes requieren al humano y quedan
  fuera del alcance automatizable.
