# Code-Index — Runbook

Servicio de búsqueda semántica del codebase para agentes (spec
`docs/superpowers/specs/2026-07-04-code-index-rag-design.md`).

## Qué es

- `kza-code-index.service` (systemd --user `kza`) en `:9510`.
- Chroma persistente propio: `/home/kza/code-index/chroma/` (colecciones
  `code_chunks` y `code_cards`) + manifest `/home/kza/code-index/manifest.json`.
- Embeddings BGE-M3 **en CPU** (cero VRAM). Cards por archivo con MiniMax vía
  gateway :8200 (`MINIMAX_API_KEY` del `.env` = virtual key del gateway).
- NO toca kza-voice ni el Chroma del pipeline.

## Deploy inicial (en el server)

```bash
ssh kza
cd /home/kza/app && git pull
mkdir -p /home/kza/code-index
cp scripts/kza-code-index.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kza-code-index
bash scripts/install_code_index_hook.sh   # hook post-merge → reindex por deploy

# primer indexado (full): ~196 archivos secuenciales contra MiniMax → estimar
# 30-60 min; seguí el avance con el watch de /health
curl -X POST localhost:9510/reindex -H 'Content-Type: application/json' -d '{"mode":"full"}'
watch -n 5 'curl -s localhost:9510/health'
```

## Uso (desde la laptop)

```bash
python tools/code_search.py "dónde se reintenta la conexión al gateway"
```

- `⚠ STALE` = el archivo local difiere del indexado (rama sin deployar) →
  leer el archivo real.
- Servicio caído → exit 1 con mensaje; fallback a Grep/Glob.

## Operación

| Acción | Comando |
|--------|---------|
| Estado | `curl -s localhost:9510/health` |
| Reindex incremental manual | `curl -X POST localhost:9510/reindex` |
| Reindex full (reconstruir) | `curl -X POST localhost:9510/reindex -d '{"mode":"full"}' -H 'Content-Type: application/json'` |
| Logs | `journalctl --user -u kza-code-index -f` |
| Reset total | parar servicio, borrar `/home/kza/code-index/`, arrancar, reindex full |

## Notas

- El reindex automático corre en el hook `post-merge` del repo del server
  (cada `git pull` de deploy). Si el servicio está caído el deploy NO se
  bloquea (el hook solo avisa).
- Cards fallidas (gateway caído) quedan `card_done: false` en el manifest y
  se reintentan solas en el próximo reindex.
- El hook post-merge dispara con `git pull` (merge/ff); un deploy por
  `git pull --rebase` o `reset` NO lo dispara — en ese caso correr el
  reindex manual.
