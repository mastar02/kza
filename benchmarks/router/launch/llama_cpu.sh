#!/usr/bin/env bash
# Levanta un llama-server CPU-only en puerto temporal para benchmark.
# Usa -t 8 para dejar recursos al kza-llm-ik (24 threads en :8200).
#
# Uso:
#   ./llama_cpu.sh <port> <model.gguf> <served-name>
set -euo pipefail
PORT="${1:?port}"
MODEL="${2:?gguf path}"
NAME="${3:?served name}"

# Usamos el binario ik_llama.cpp que ya está compilado y testeado en server.
BIN=/home/kza/ik_llama.cpp/build/bin/llama-server

exec "$BIN" -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -t 8 -tb 8 \
  -c 4096 \
  -ctk q8_0 -ctv q8_0 \
  --alias "$NAME"
