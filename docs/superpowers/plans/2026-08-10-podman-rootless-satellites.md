# Satélites de KZA en Podman rootless — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Contenerizar Hermes CLI, ChromaDB (:9516) y kza-code-index (:9515) bajo el usuario `kza` con Podman rootless + Quadlets, dejando kza-voice y llama-server nativos.

**Architecture:** Tres containers vía Quadlet en la red `kza-internal` existente. El pipeline nativo invoca Hermes por un wrapper host (`podman exec` + `timeout` interno). `src/code_index` gana un selector de backend Chroma (embedded|http) por config; el resto es empaquetado y deploy. Spec: `docs/superpowers/specs/2026-08-10-podman-rootless-satellites-design.md`.

**Tech Stack:** Podman 4.9.3 rootless, Quadlet, chromadb (client 1.5.9 instalado), aiohttp, sentence-transformers (BGE-M3 CPU), pytest.

## Global Constraints

- Convenciones server: R4 naming (`localhost/kza/<comp>:<version>`, container `kza-<comp>`, quadlet en `~/.config/containers/systemd/`), R8 restart (`Restart=on-failure`, `RestartSec>=5s`, `StartLimitBurst=5`), R10 rootless (cero sudo), R12 tags inmutables (nunca `:latest`).
- Puertos: `:9515` code-index (existente), `:9516` chroma (nuevo — **`:9500` está ocupado por obs**, verificado 2026-08-10). Sub-rango KZA 9500-9599; ocupados hoy: 9500, 9501, 9510, 9515, 9521, 9587.
- El server es **producción** (memoria: consultar antes de acción delicada). Los pasos de deploy tocan SOLO los servicios nuevos + deshabilitan `kza-code-index.service` nativo. `kza-voice` y `kza-llm-fast` NO se tocan.
- `ssh kza 'sudo …'` no funciona (`requiretty`, 2026-08-10). Nada de este plan usa sudo.
- Tests locales SIEMPRE con el venv 3.13: `/Users/yo/Documents/kza/.venv/bin/python -m pytest` (el python3 del sistema es 3.9 y rompe).
- Config: TODO en `config/settings.yaml`, nunca archivos de config nuevos. Código/comentarios en inglés en `src/`, docs en español.
- El flip `reasoner.mode=hermes_cli` NO es parte de este plan (checklist pre-flip aparte).
- Laptop = puente git: el server hace `git pull` pero no push (memoria ecosistema local↔server).

---

### Task 1: Selector de backend Chroma en `src/code_index` (TDD)

**Files:**
- Modify: `src/code_index/service.py:108-120` (función `build_indexer`, nueva función `build_chroma_client`)
- Test: `tests/unit/code_index/test_chroma_backend.py` (nuevo)

**Interfaces:**
- Produces: `build_chroma_client(cfg: dict) -> chromadb API client` — exportada desde `src.code_index.service`. `cfg` es el dict de `code_index` en settings; lee `cfg["chroma"]["mode"]` (`"embedded"` default | `"http"`), `cfg["chroma"]["url"]` (solo http) y `cfg["chroma_path"]` (solo embedded).
- Consumes: nada de otras tasks.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/code_index/test_chroma_backend.py`:

```python
"""Tests del selector de backend Chroma (embedded | http) del code-index."""

from unittest.mock import patch

import pytest

from src.code_index.service import build_chroma_client


def test_default_embedded_uses_persistent_client_with_chroma_path():
    with patch("chromadb.PersistentClient") as pc:
        client = build_chroma_client({"chroma_path": "/tmp/x"})
    pc.assert_called_once_with(path="/tmp/x")
    assert client is pc.return_value


def test_explicit_embedded_mode_uses_persistent_client():
    cfg = {"chroma_path": "/tmp/x", "chroma": {"mode": "embedded"}}
    with patch("chromadb.PersistentClient") as pc:
        build_chroma_client(cfg)
    pc.assert_called_once_with(path="/tmp/x")


def test_http_mode_uses_http_client_with_host_and_port():
    cfg = {
        "chroma_path": "/unused",
        "chroma": {"mode": "http", "url": "http://kza-chroma:8000"},
    }
    with patch("chromadb.HttpClient") as hc:
        client = build_chroma_client(cfg)
    hc.assert_called_once_with(host="kza-chroma", port=8000)
    assert client is hc.return_value


def test_http_mode_with_url_without_port_raises():
    cfg = {"chroma": {"mode": "http", "url": "http://kza-chroma"}}
    with pytest.raises(ValueError, match="chroma.url"):
        build_chroma_client(cfg)


def test_http_mode_without_url_raises():
    cfg = {"chroma": {"mode": "http"}}
    with pytest.raises(ValueError, match="chroma.url"):
        build_chroma_client(cfg)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="chroma.mode"):
        build_chroma_client({"chroma": {"mode": "grpc"}})
```

- [ ] **Step 2: Verificar que fallan**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/test_chroma_backend.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_chroma_client'`

- [ ] **Step 3: Implementar `build_chroma_client` y cablearla**

En `src/code_index/service.py`, agregar `from urllib.parse import urlparse` al bloque de imports stdlib del tope (junto a `from pathlib import Path`). Luego, encima de `build_indexer`, agregar:

```python
def build_chroma_client(cfg: dict):
    """Build the Chroma client per config: embedded (local dir) or http (service).

    ``embedded`` (default) preserves today's behavior (PersistentClient on
    ``chroma_path``). ``http`` targets the kza-chroma container via
    ``code_index.chroma.url`` (resolved by Podman's internal DNS).
    """
    import chromadb

    chroma_cfg = cfg.get("chroma") or {}
    mode = chroma_cfg.get("mode", "embedded")
    if mode == "embedded":
        return chromadb.PersistentClient(path=cfg["chroma_path"])
    if mode == "http":
        url = chroma_cfg.get("url", "")
        parsed = urlparse(url)
        if not parsed.hostname or not parsed.port:
            raise ValueError(f"code_index.chroma.url must be host:port, got: {url!r}")
        return chromadb.HttpClient(host=parsed.hostname, port=parsed.port)
    raise ValueError(f"code_index.chroma.mode unknown: {mode!r}")
```

Y en `build_indexer`, reemplazar:

```python
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=cfg["chroma_path"])
```

por:

```python
    from sentence_transformers import SentenceTransformer

    client = build_chroma_client(cfg)
```

- [ ] **Step 4: Verificar que pasan + suite del módulo**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/ -v`
Expected: PASS todos (los 6 nuevos + los existentes del módulo intactos).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/code_index/test_chroma_backend.py src/code_index/service.py
git commit -m "feat(code-index): selector de backend chroma embedded|http por config"
```

---

### Task 2: Config `code_index.chroma` en settings.yaml

**Files:**
- Modify: `config/settings.yaml` (bloque `code_index:`, ~línea 468; y el bloque comentado de hermes_cli, línea `# hermes_binary_path:`)

**Interfaces:**
- Consumes: claves `chroma.mode`/`chroma.url` definidas en Task 1.
- Produces: la config que el deploy (Task 7) y el quadlet (Task 4) asumen.

- [ ] **Step 1: Agregar el bloque chroma bajo `code_index:`**

Debajo de la línea `chroma_path: "/home/kza/code-index/chroma"` agregar:

```yaml
  chroma:
    # http = servicio kza-chroma en container (deploy actual, DNS interno podman).
    # embedded = PersistentClient local en chroma_path (rollback a la unit nativa;
    #   el hostname kza-chroma NO resuelve fuera de la red kza-internal).
    mode: "http"
    url: "http://kza-chroma:8000"
```

- [ ] **Step 2: Actualizar `hermes_binary_path` en el bloque comentado**

Reemplazar la línea:

```yaml
  # hermes_binary_path: "hermes"       # ruta absoluta si systemd --user no hereda PATH
```

por:

```yaml
  # hermes_binary_path: "/home/kza/bin/hermes-ctr"  # wrapper podman exec -> container kza-hermes
```

- [ ] **Step 3: Validar el YAML y correr la suite**

Run: `/Users/yo/Documents/kza/.venv/bin/python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))" && /Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/code_index/ -q`
Expected: sin excepción de YAML; tests PASS (el default embedded de Task 1 solo aplica si falta el bloque — acá queda http explícito para el server).

- [ ] **Step 4: Commit**

```bash
git add config/settings.yaml
git commit -m "config(code-index): chroma mode http hacia kza-chroma + path del wrapper hermes"
```

---

### Task 3: Quadlet de `kza-chroma` (:9516)

**Files:**
- Create: `containers/quadlets/kza-chroma.container`

**Interfaces:**
- Produces: servicio `kza-chroma.service` (generado por Quadlet) que Task 4 referencia en `After=/Wants=`; hostname interno `kza-chroma:8000` que la config de Task 2 asume.

- [ ] **Step 1: Crear el quadlet**

`containers/quadlets/kza-chroma.container`:

```ini
# Quadlet (podman rootless, usuario kza). Instalar en ~/.config/containers/systemd/
# Reemplaza al viejo kza-chroma.container.disabled-2026-05-30 (que publicaba :9500,
# hoy ocupado por obs — verificado 2026-08-10).
[Unit]
Description=KZA ChromaDB service — backend del code-index
After=network-online.target
Wants=network-online.target

[Container]
# Pin exacto (R12). 1.5.9 = version del client chromadb instalado en el venv;
# si el tag no existe en el registry, ver paso de verificacion del deploy.
Image=docker.io/chromadb/chroma:1.5.9
ContainerName=kza-chroma
Network=kza-internal.network
Volume=%h/data/chroma-svc:/chroma/chroma:Z
Environment=IS_PERSISTENT=TRUE
Environment=ANONYMIZED_TELEMETRY=FALSE
# Sub-rango KZA 9500-9599; :9516 libre (auditoria 2026-08-10). Bind solo localhost:
# los consumidores en la red kza-internal entran por DNS interno, no por este puerto.
PublishPort=127.0.0.1:9516:8000

[Service]
Restart=on-failure
RestartSec=10s
StartLimitBurst=5

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Commit**

```bash
git add containers/quadlets/kza-chroma.container
git commit -m "feat(containers): quadlet kza-chroma :9516 (reemplaza el .disabled de 05-30)"
```

---

### Task 4: Imagen y quadlet de `kza-code-index`

**Files:**
- Create: `containers/code-index/Containerfile`
- Create: `containers/code-index/requirements-code-index.txt`
- Create: `containers/quadlets/kza-code-index.container`

**Interfaces:**
- Consumes: `kza-chroma.service` (Task 3), config `mode: http` (Task 2).
- Produces: imagen `localhost/kza/code-index:<git-sha>` y servicio `kza-code-index.service` container que reemplaza a la unit nativa homónima.

- [ ] **Step 1: Crear requirements del servicio**

`containers/code-index/requirements-code-index.txt` — subset de `requirements.txt` que importa `src/code_index` (aiohttp, yaml, chromadb, sentence-transformers, openai para cards):

```
aiohttp>=3.9
PyYAML>=6.0
chromadb>=1.5,<2
sentence-transformers>=3.0
openai>=1.30
```

- [ ] **Step 2: Crear el Containerfile (imagen de deps; el código entra por mount ro)**

`containers/code-index/Containerfile`:

```dockerfile
# Imagen de runtime del code-index: SOLO dependencias (torch CPU explicito).
# El codigo NO se hornea: /home/kza/app se monta read-only en el mismo path que
# en el host, asi settings.yaml funciona sin cambios y el servicio corre el
# mismo commit que esta deployado en el checkout del server.
FROM python:3.13-slim

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements-code-index.txt /tmp/requirements-code-index.txt
RUN pip install --no-cache-dir -r /tmp/requirements-code-index.txt

WORKDIR /home/kza/app
CMD ["python", "-m", "src.code_index"]
```

- [ ] **Step 3: Crear el quadlet**

`containers/quadlets/kza-code-index.container`:

```ini
# Quadlet (podman rootless, usuario kza). Instalar en ~/.config/containers/systemd/
# con el tag de imagen reemplazado por el deploy: sed "s/@SHA@/$SHA/".
# Reemplaza a la unit nativa ~/.config/systemd/user/kza-code-index.service
# (deshabilitarla antes de arrancar este; mismo puerto :9515).
[Unit]
Description=KZA code-index :9515 (container) — indice semantico del codebase
After=network-online.target kza-chroma.service
Wants=network-online.target kza-chroma.service

[Container]
Image=localhost/kza/code-index:@SHA@
ContainerName=kza-code-index
Network=kza-internal.network
# Codigo + config, read-only, mismo path que el host (settings.yaml sin cambios)
Volume=%h/app:/home/kza/app:ro
# Estado del indice (manifest.json); chroma_path queda sin uso en mode=http
Volume=%h/code-index:/home/kza/code-index
# Cache HF del host: ahi vive BAAI/bge-m3. rw: HF puede escribir locks/refs.
Volume=%h/.cache/huggingface:/hf-cache
Environment=HF_HOME=/hf-cache
Environment=CONFIG_PATH=config/settings.yaml
Environment=PYTHONUNBUFFERED=1
# LLM_GATEWAY_URL + MINIMAX_API_KEY (cards del indexer los leen del entorno)
EnvironmentFile=%h/secrets/.env
# 0.0.0.0 deliberado: tools/code_search.py y agentes lo consumen desde la LAN
# (mismo contrato que la unit nativa; ver nosec B104 en src/code_index/__main__.py)
PublishPort=9515:9515

[Service]
Restart=on-failure
RestartSec=10s
StartLimitBurst=5

[Install]
WantedBy=default.target
```

- [ ] **Step 4: Commit**

```bash
git add containers/code-index/ containers/quadlets/kza-code-index.container
git commit -m "feat(containers): imagen deps-only + quadlet de kza-code-index"
```

---

### Task 5: Imagen, wrapper y quadlet de `kza-hermes`

**Files:**
- Create: `containers/hermes/Containerfile`
- Create: `containers/hermes/fake-sudo`
- Create: `containers/hermes/hermes-ctr`
- Create: `containers/quadlets/kza-hermes.container`

**Interfaces:**
- Produces: imagen `localhost/kza/hermes:<git-sha>`, container residente `kza-hermes`, y el wrapper host `/home/kza/bin/hermes-ctr` que `settings.yaml` referencia como `hermes_binary_path` (Task 2). Contrato del wrapper: mismo CLI que `hermes` (args passthrough, exit code passthrough, stdout/stderr passthrough) — es lo que `HermesCliReasoner._run()` espera de un binario.

- [ ] **Step 1: Crear `fake-sudo`**

`containers/hermes/fake-sudo`:

```bash
#!/bin/bash
# Shim para el build de la imagen: el installer de Hermes llama a sudo, pero en
# el build ya somos root. Ejecuta el comando tal cual, ignorando flags de sudo
# tipo -E/-H (no los usa el installer segun el incidente registrado en
# docs/architecture/ROOTLESS_MIGRATION.md; si aparecieran, fallara ruidoso).
exec "$@"
```

- [ ] **Step 2: Crear el Containerfile**

`containers/hermes/Containerfile`:

```dockerfile
# Hermes CLI aislado: TODO su stack (Xvfb/browser/mesa que su installer traiga)
# vive aca adentro. Nunca mas `sudo apt install` en el host por Hermes.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates coreutils curl \
    && rm -rf /var/lib/apt/lists/*

COPY fake-sudo /usr/local/bin/sudo
RUN chmod 0755 /usr/local/bin/sudo

# Instalar con HOME de build separado del HOME de runtime (/data es el volumen
# de auth y taparia lo instalado si el installer escribiera ahi).
ENV HOME=/opt/hermes-home
RUN mkdir -p /opt/hermes-home \
    && curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# El installer puede dejar el binario en $HOME/.local/bin o /usr/local/bin.
# Cubrimos ambos; el paso de verificacion del deploy corre `hermes --version`.
ENV PATH=/opt/hermes-home/.local/bin:/opt/hermes-home/bin:$PATH

# HOME de runtime = volumen de auth (persistente entre restarts y rebuilds)
ENV HOME=/data

CMD ["sleep", "infinity"]
```

- [ ] **Step 3: Crear el wrapper host `hermes-ctr`**

`containers/hermes/hermes-ctr`:

```bash
#!/bin/bash
# Puente host->container para HermesCliReasoner (hermes_binary_path apunta aca).
# Contrato: mismo CLI que `hermes` (args/exit code/stdout/stderr passthrough).
#
# El `timeout` corre DENTRO del container: si este cliente muere (kill del
# process group del reasoner, ssh cortado), el proceso hermes igual muere
# server-side — cierra la clase "huerfanos colgados" documentada en
# src/llm/hermes_reasoner.py.
#
# 95 = hermes_timeout_s (90 en settings.yaml) + 5 de margen. El timeout local
# del reasoner dispara primero; este es la red de seguridad. Si cambia
# hermes_timeout_s, actualizar este numero a mano.
exec podman exec -i kza-hermes timeout -k 5 95 hermes "$@"
```

- [ ] **Step 4: Crear el quadlet**

`containers/quadlets/kza-hermes.container`:

```ini
# Quadlet (podman rootless, usuario kza). Instalar en ~/.config/containers/systemd/
# con el tag reemplazado: sed "s/@SHA@/$SHA/". Container residente sin puertos:
# el trabajo entra por `podman exec` via /home/kza/bin/hermes-ctr.
[Unit]
Description=KZA Hermes CLI runtime — reasoner slow path (via wrapper hermes-ctr)
After=network-online.target
Wants=network-online.target

[Container]
Image=localhost/kza/hermes:@SHA@
ContainerName=kza-hermes
# Auth OAuth persistente (device flow, una vez): HOME del proceso hermes
Volume=%h/data/hermes-auth:/data
Environment=HOME=/data
# Sin PublishPort (solo salida HTTPS), sin GPU, sin red interna (no la necesita)

[Service]
Restart=on-failure
RestartSec=5s
StartLimitBurst=5

[Install]
WantedBy=default.target
```

- [ ] **Step 5: Commit**

```bash
git add containers/hermes/ containers/quadlets/kza-hermes.container
git commit -m "feat(containers): imagen hermes aislada + wrapper hermes-ctr + quadlet"
```

---

### Task 6: Runbook de deploy + mapa de puertos

**Files:**
- Create: `docs/runbooks/2026-08-10-deploy-satellites-podman.md`
- Modify: `docs/SERVER_CONVENTIONS.md` (tabla del mapa de puertos, ~línea 80)

**Interfaces:**
- Consumes: todos los artefactos de Tasks 1-5.
- Produces: el procedimiento exacto que Task 7 ejecuta.

- [ ] **Step 1: Agregar `:9516` al mapa de puertos**

En la tabla de `docs/SERVER_CONVENTIONS.md`, después de la fila de `:9515` (si no existe fila de :9515, insertar ambas en orden numérico dentro de la tabla):

```markdown
| :9516 | ChromaDB servicio (container kza-chroma, bind 127.0.0.1; backend del code-index) | kza |
```

- [ ] **Step 2: Escribir el runbook**

`docs/runbooks/2026-08-10-deploy-satellites-podman.md` con este contenido:

````markdown
# Deploy: satélites en Podman rootless (kza-chroma, kza-code-index, kza-hermes)

> Ejecutar como `kza` por SSH desde la laptop. Nada usa sudo. El server es
> producción: kza-voice y kza-llm-fast NO se tocan en ningún paso.
> Spec: docs/superpowers/specs/2026-08-10-podman-rootless-satellites-design.md

## 1. Sync del repo en el server

```bash
ssh kza 'cd ~/app && git pull --ff-only && git rev-parse --short HEAD'
SHA=<el sha impreso>
```

## 2. Directorios y wrapper

```bash
ssh kza 'mkdir -p ~/data/chroma-svc ~/data/hermes-auth ~/code-index ~/bin \
  && install -m 0755 ~/app/containers/hermes/hermes-ctr ~/bin/hermes-ctr'
```

## 3. Build de imágenes (tag = git SHA, R12)

```bash
ssh kza "cd ~/app && podman build -t localhost/kza/code-index:$SHA containers/code-index/"
ssh kza "cd ~/app && podman build -t localhost/kza/hermes:$SHA containers/hermes/"
```

Si el pull de `chromadb/chroma:1.5.9` (paso 4) fallara por tag inexistente:
`skopeo list-tags docker://docker.io/chromadb/chroma | grep '1\.5'` y editar el
quadlet instalado con el 1.5.x más cercano al client (1.5.9). NO usar latest.

## 4. Instalar quadlets y arrancar chroma

```bash
ssh kza "cd ~/app && for q in containers/quadlets/*.container; do \
    sed \"s/@SHA@/$SHA/\" \$q > ~/.config/containers/systemd/\$(basename \$q); done \
  && systemctl --user daemon-reload && systemctl --user start kza-chroma"
ssh kza 'curl -sf http://127.0.0.1:9516/api/v2/heartbeat && echo CHROMA_OK'
```

(Si `/api/v2/heartbeat` diera 404 en la versión pinneada, probar `/api/v1/heartbeat` —
cambió entre líneas de Chroma; anotar cuál respondió.)

## 5. Cutover del code-index (nativo → container)

```bash
ssh kza 'systemctl --user disable --now kza-code-index.service \
  && systemctl --user start kza-code-index && sleep 5 \
  && curl -sf -X POST http://127.0.0.1:9515/reindex'
# poll hasta reindex_running=false:
ssh kza 'for i in $(seq 1 60); do curl -sf http://127.0.0.1:9515/health && break; sleep 5; done'
```

Query de humo (desde la laptop): `python tools/code_search.py "timeout de HA al boot"` —
debe devolver resultados del repo real.

## 6. Hermes

```bash
ssh kza 'systemctl --user start kza-hermes && ~/bin/hermes-ctr --version'
```

Auth (HUMANO, una vez, interactivo): `ssh -t kza 'podman exec -it kza-hermes hermes auth add openai-codex'`
y completar el device-code flow desde un browser. Verificar:
`ssh kza '~/bin/hermes-ctr auth status openai-codex'`.

## 7. Verificación final

```bash
ssh kza 'podman ps --format "{{.Names}} {{.Status}}" | grep -E "kza-(chroma|code-index|hermes)"'
ssh kza 'systemctl --user is-active kza-voice kza-llm-fast'   # active, active — intocados
ssh kza 'cd ~/app && .venv/bin/python tools/smoke_test.py'
ssh kza 'cd ~/app && .venv/bin/python tools/benchmark_latency.py --iterations 20'
# esperado: dentro del rango histórico ~150-280ms — la voz no se enteró del deploy
```

## Rollback por componente

- **code-index** → `ssh kza 'systemctl --user stop kza-code-index'`; editar
  `~/app/config/settings.yaml` → `code_index.chroma.mode: "embedded"`;
  `systemctl --user enable --now kza-code-index.service` (unit nativa, sigue instalada).
- **chroma** → `systemctl --user stop kza-chroma` (solo lo consume code-index).
- **hermes** → `systemctl --user stop kza-hermes`; el reasoner degrada solo a
  HttpReasoner (:8200). El wrapper puede quedar.
- **Imagen rota** → re-apuntar el tag anterior en el quadlet instalado y
  `systemctl --user daemon-reload && systemctl --user restart <svc>`.
````

- [ ] **Step 3: Commit**

```bash
git add docs/runbooks/2026-08-10-deploy-satellites-podman.md docs/SERVER_CONVENTIONS.md
git commit -m "docs(runbook): deploy de los satelites podman + :9516 en el mapa de puertos"
```

---

### Task 7: Deploy en el server (ejecutar el runbook)

**Files:**
- Ninguno en el repo (acciones en el server, siguiendo `docs/runbooks/2026-08-10-deploy-satellites-podman.md` al pie de la letra).

**Interfaces:**
- Consumes: todo lo anterior, pusheado a main y pulleado en el server.

- [ ] **Step 1: Push desde la laptop** — `git push origin main` (verificar working tree limpio antes).
- [ ] **Step 2: Ejecutar §1-§4 del runbook** (sync, dirs, builds, chroma). Gate: `CHROMA_OK`.
- [ ] **Step 3: Ejecutar §5** (cutover code-index). Gate: query de humo con resultados reales.
- [ ] **Step 4: Ejecutar §6** hasta `hermes-ctr --version`. **PAUSA — el paso de auth es del humano** (device flow interactivo). No continuar hasta que `hermes-ctr auth status openai-codex` confirme.
- [ ] **Step 5: Ejecutar §7** (verificación final completa). Si cualquier gate falla: aplicar el rollback del componente y reportar — NO improvisar arreglos en producción.

---

### Task 8: Cierre documental

**Files:**
- Modify: `docs/architecture/ROOTLESS_MIGRATION.md` (sección "Política hacia adelante", punto 6 de "Cambios requeridos")

**Interfaces:**
- Consumes: resultado del deploy (Task 7).

- [ ] **Step 1: Marcar el punto 6 de "Cambios requeridos" como ejecutado**

En `docs/architecture/ROOTLESS_MIGRATION.md`, al final del punto 6 ("Política hacia adelante para componentes nuevos..."), agregar:

```markdown
   ✅ Ejecutado 2026-08-10+: Hermes CLI, ChromaDB (:9516) y code-index corren como
   containers rootless bajo kza (spec y runbook en docs/superpowers/ y docs/runbooks/).
```

- [ ] **Step 2: Suite completa + commit + push**

```bash
/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/ -q
git add docs/architecture/ROOTLESS_MIGRATION.md
git commit -m "docs(rootless): satelites contenerizados — cierre del punto 6"
git push origin main
```

Expected: suite verde (3050+ pass, 1 xfail conocido).
