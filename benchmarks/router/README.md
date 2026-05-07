# Router benchmark

Mide accuracy + latencia del `FastRouter` de KZA contra varios modelos
candidatos para reemplazar el 7B AWQ actual en `:8100`.

## Modelos comparados

| id | runtime | puerto | notas |
|---|---|---|---|
| qwen2.5-0.5b-awq | vLLM | 8210 | curva extra-pequeña |
| qwen2.5-1.5b-awq | vLLM | 8211 | candidato chico |
| qwen2.5-3b-awq | vLLM | 8212 | candidato medio |
| phi-3.5-mini-awq | vLLM | 8213 | familia distinta |
| qwen2.5-7b-awq | vLLM | 8100 | **baseline producción** |
| qwen3-30b-a3b-q5 | ik_llama | 8200 | techo de calidad (reasoner) |

## Pasos

1. **Completar `golden_set.yaml`** con utterances reales (ver TODO en el archivo).
2. Bajar modelos faltantes en server:
   ```bash
   ssh kza
   hf download Qwen/Qwen2.5-3B-Instruct-AWQ \
     --local-dir /home/kza/kza/models/Qwen2.5-3B-Instruct-AWQ
   hf download kaitchup/Phi-3.5-mini-instruct-AutoAWQ \
     --local-dir /home/kza/kza/models/Phi-3.5-mini-instruct-AWQ
   ```
3. Levantar uno por uno (no en paralelo: GPU1 no aguanta varios) y correr:
   ```bash
   ssh kza 'bash benchmarks/router/launch/vllm_qwen_small.sh 0.5b' &
   sleep 30
   python benchmarks/router/runner.py --only qwen2.5-0.5b
   # bajar el server, repetir con 1.5b, 3b, phi-mini
   ```
4. Para el 7B prod y el 30B reasoner ya están corriendo, solo correr:
   ```bash
   python benchmarks/router/runner.py --only qwen2.5-7b
   python benchmarks/router/runner.py --only qwen3-30b
   ```
5. Generar reporte combinando los `results.json` de cada corrida.

## Métricas reportadas

- **accuracy** por tarea (`classify`, `reasoning`, `respond`)
- **well_formed_rate** — output respeta el formato (no inventa categorías)
- **ttft_ms** p50/p95/mean
- **total_ms** p50/p95/mean

## Notas operativas

- vLLM en :8100 es del user `infra` — no se toca.
- Cada modelo benchmark se levanta con `--gpu-memory-utilization` bajo para
  no chocar con el 7B prod en GPU1.
- El golden set y el SYSTEM_PROMPT_PREFIX son los mismos para todos los
  modelos — lo que se compara es la capacidad del modelo de seguir el prompt
  actual de KZA.
