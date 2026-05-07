#!/usr/bin/env bash
# Levanta los Qwen2.5-AWQ chicos en puertos temporales para benchmark.
# Correr en server (ssh kza). Cada vLLM en su propia GPU para evitar
# contienda con :8100 (infra) y voz (cuda:0).
#
# Uso:
#   ./vllm_qwen_small.sh 0.5b   # → :8210, GPU1
#   ./vllm_qwen_small.sh 1.5b   # → :8211, GPU1
#   ./vllm_qwen_small.sh 3b     # → :8212, GPU1
#
# IMPORTANTE: bajar :8100 antes (o subir gpu-mem-util muy bajo) si querés
# que estos quepan en GPU1. Alternativa: usar GPU0 fuera de horario de voz.
set -euo pipefail

SIZE="${1:?usage: $0 {0.5b|1.5b|3b}}"
case "$SIZE" in
  0.5b) PORT=8210; MODEL=/home/kza/kza/models/Qwen2.5-0.5B-Instruct-AWQ; UTIL=0.20 ;;
  1.5b) PORT=8211; MODEL=/home/kza/kza/models/Qwen2.5-1.5B-Instruct-AWQ; UTIL=0.30 ;;
  3b)   PORT=8212; MODEL=/home/kza/kza/models/Qwen2.5-3B-Instruct-AWQ;   UTIL=0.45 ;;
  *) echo "tamaño desconocido: $SIZE"; exit 1 ;;
esac

if [[ ! -d "$MODEL" ]]; then
  echo "Modelo no encontrado en $MODEL — bajalo primero"
  exit 1
fi

# Misma GPU que el 7B prod (cuda:1). Usar CUDA_VISIBLE_DEVICES=0 si querés
# evitar competencia, pero entonces no medís en condiciones reales.
export CUDA_VISIBLE_DEVICES=1
exec /home/infra/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "qwen2.5-${SIZE}-awq" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 4096 \
  --dtype half \
  --quantization awq
