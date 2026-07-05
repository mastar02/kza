#!/usr/bin/env bash
# scripts/install_code_index_hook.sh
#
# Instala el hook post-merge que dispara el reindex incremental del
# code-index después de cada deploy (git pull) EN EL SERVER.
#
# Uso (en el server): bash scripts/install_code_index_hook.sh
set -euo pipefail

HOOK="$(git rev-parse --git-dir)/hooks/post-merge"

cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Dispara reindex incremental del code-index tras cada git pull (deploy).
# No bloquea el deploy si el servicio está caído.
curl -fsS -X POST -m 5 http://127.0.0.1:9515/reindex \
  -H 'Content-Type: application/json' -d '{"mode":"incremental"}' \
  || echo "[post-merge] code-index no disponible (reindex omitido)"
EOF

chmod +x "$HOOK"
echo "✓ hook post-merge instalado en $HOOK"
