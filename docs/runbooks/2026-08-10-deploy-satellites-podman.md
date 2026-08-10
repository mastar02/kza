# Deploy: satélites en Podman rootless (kza-chroma, kza-code-index, kza-hermes)

> Ejecutar como `kza` por SSH desde la laptop. Nada usa sudo. El server es
> producción: kza-voice y kza-llm-fast NO se tocan en ningún paso.
> Spec: docs/superpowers/specs/2026-08-10-podman-rootless-satellites-design.md

> ⚠️ Ejecutar §1→§5 de corrido, sin pausas largas entre pasos. Desde el `git pull`
> del paso 1, la unit nativa `kza-code-index.service` tiene en disco la config
> nueva (`chroma.mode: http`) pero el cutover al container recién pasa en §5 —
> un restart espontáneo de esa unit nativa en esa ventana (watchdog, reboot,
> systemd) la deja en crash-loop porque no hay chroma http disponible todavía.

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
Si se cambia el tag, back-portear el cambio a `containers/quadlets/kza-chroma.container`
en el repo (laptop) — si queda solo en el quadlet instalado en el server, el
próximo `git pull --ff-only` lo pisa con el 1.5.9 original.

## 4. Instalar quadlets y arrancar chroma

```bash
ssh kza "mkdir -p ~/.config/containers/systemd"
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
  && systemctl --user start kza-code-index-ctr'
# confirma que corre la unit generada por Quadlet (kza-code-index-ctr.service),
# NO la nativa homónima kza-code-index.service (la primera línea del `cat` es
# el path de la unit activa — debe apuntar a ~/.config/containers/systemd o al
# runtime generado del generator, no a ~/.config/systemd/user/):
ssh kza 'systemctl --user cat kza-code-index-ctr.service | head -1'
# build_indexer carga BGE-M3 en CPU antes de abrir el puerto (decenas de
# segundos) — pollear el puerto en vez de un sleep fijo, hasta ~120s:
ssh kza 'for i in $(seq 1 24); do curl -sf http://127.0.0.1:9515/health >/dev/null && break; sleep 5; done'
ssh kza 'curl -sf -X POST http://127.0.0.1:9515/reindex'
# poll hasta reindex_running=false explícito (cualquier 200 no alcanza: el
# health responde 200 también a mitad de reindex):
ssh kza 'for i in $(seq 1 60); do curl -sf http://127.0.0.1:9515/health | grep -q "\"reindex_running\": *false" && break; sleep 5; done'
# persistencia REAL de chroma (el heartbeat de §4 solo prueba que el server
# responde, no que escribió a disco):
ssh kza 'du -s ~/data/chroma-svc'   # no vacío tras el reindex
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

- **code-index** → `ssh kza 'systemctl --user stop kza-code-index-ctr'`; editar
  `~/app/config/settings.yaml` → `code_index.chroma.mode: "embedded"`;
  `systemctl --user enable --now kza-code-index.service` (unit nativa, sigue instalada).
  Ese edit de `settings.yaml` queda como cambio local sin commitear en el
  checkout del server — revertirlo (o commitear el flip desde la laptop y
  pushear) apenas se estabilice: un working tree sucio hace que el próximo
  `git pull --ff-only` en §1 se rehúse.
- **chroma** → `systemctl --user stop kza-chroma` (solo lo consume code-index).
- **hermes** → `systemctl --user stop kza-hermes`; el reasoner degrada solo a
  HttpReasoner (:8200). El wrapper puede quedar.
- **Imagen rota** → re-apuntar el tag anterior en el quadlet instalado y
  `systemctl --user daemon-reload && systemctl --user restart <svc>`.
