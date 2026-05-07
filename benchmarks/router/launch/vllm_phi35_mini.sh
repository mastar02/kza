#!/usr/bin/env bash
# Phi-3.5-mini-instruct AWQ en :8213. Necesita HF download previo.
# Modelo recomendado: kaitchup/Phi-3.5-mini-instruct-AutoAWQ
# (verificar en HF; alternativa: cuantizarlo localmente con autoawq).
set -euo pipefail

MODEL="${MODEL:-/home/kza/kza/models/Phi-3.5-mini-instruct-AWQ}"
PORT=8213

if [[ ! -d "$MODEL" ]]; then
  echo "Modelo no encontrado en $MODEL"
  echo "Bajar con:"
  echo "  hf download kaitchup/Phi-3.5-mini-instruct-AutoAWQ \\"
  echo "    --local-dir /home/kza/kza/models/Phi-3.5-mini-instruct-AWQ"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1
exec /home/infra/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "phi-3.5-mini-awq" \
  --gpu-memory-utilization 0.45 \
  --max-model-len 4096 \
  --dtype half \
  --quantization awq \
  --trust-remote-code
