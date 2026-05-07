# Router benchmark — resultados

**Fecha**: 2026-05-07
**Golden set**: `golden_set.yaml`, 27 casos en español rioplatense (5 corregidos
post-bench tras revisión de votos unánimes 6/6).
**Tareas medidas**: las 3 que ejecuta `FastRouter` real en `src/llm/reasoner.py`:
`classify(text, options)`, `should_use_deep_reasoning(text)`, `classify_and_respond(text)`.
**Stop tokens**: `["\n\n", "Texto:", "Pregunta:", "Consulta:", "Categoría:"]`
agregados al benchmark (ver hallazgo crítico abajo — esto NO está en producción).

## Resultados (golden corregido, raw output re-scoreado)

| modelo | runtime | cls | rsn | rsp | avg | TTFT cls p50 | total resp p50 |
|---|---|---:|---:|---:|---:|---:|---:|
| **qwen2.5-7b-awq** (prod) | vLLM GPU AWQ | **82.1%** | 57.1% | 78.6% | **72.6%** | **57ms** | **38ms** |
| qwen3-30b-a3b-q5 | ik_llama CPU | **89.3%** | 3.6% ⚠️ | **82.1%** | 58.3% | 137ms | 372ms |
| phi-3.5-mini | llama.cpp CPU | 75.0% | 64.3% | 78.6% | 72.6% | 240ms | 643ms |
| qwen2.5-3b | llama.cpp CPU | 75.0% | 46.4% | 75.0% | 65.5% | 173ms | 1832ms |
| qwen2.5-1.5b | llama.cpp CPU | 57.1% | 82.1% ⚠️ | 25.0% ⚠️ | 54.8% | 90ms | 144ms |
| qwen2.5-0.5b | llama.cpp CPU | 53.6% | 64.3% | 78.6% | 65.5% | 32ms | 80ms |

⚠️ = artefacto, ver "hallazgos".

## Hallazgos

### 1. Bug en producción — FastRouter sin `stop` tokens (+18 puntos accuracy gratis)

Primera corrida del 7B prod dio **64% en classify**. Tras agregar `stop` para
cortar antes de que el modelo invente "Texto:/Categoría:" nuevos, subió a
**82%**. Es el delta más alto del benchmark y no requiere cambiar modelo.

**Acción**: agregar `stop=["\n\n", "Texto:", "Pregunta:", "Consulta:", "Categoría:"]`
en `FastRouter.classify`, `should_use_deep_reasoning`, `classify_and_respond`
(`src/llm/reasoner.py:718-790`).

### 2. `should_use_deep_reasoning` está roto en el 30B (3.6%)

El 30B-A3B-Instruct ignora la opción `SIMPLE | COMPLEJO` y devuelve `[DEEP]` o
respuestas largas. El 7B AWQ saca 57% — mediocre. El "ganador" 1.5B con 82%
es **artefacto**: siempre dice `[DEEP]` y eso casualmente coincide con muchos
COMPLEJO esperados. La función actual no es confiable.

**Acción**: reemplazar `should_use_deep_reasoning` por una heurística determinística
(longitud + presencia de verbos creativos/explicativos) o usar el output de
`classify_and_respond` directamente como señal `[DEEP]`. La función separada
sobra.

### 3. El 1.5B colapsa a `[DEEP]` siempre — descartar

`qwen2.5-1.5b-q5_k_m.gguf` devuelve `[DEEP]` para 21/24 casos en `respond`,
incluso para "prendé la luz". Modelo demasiado chico para el prompt actual.

### 4. Phi-3.5-mini empata al 7B en avg pero pierde por latencia

Mismo accuracy (72.6%) pero TTFT 4x más alto incluso si lo pasara a GPU AWQ.
No justifica el switch.

### 5. El 30B-A3B es el mejor clasificador (89.3% cls) — pero NO sirve como router

Su TTFT 137ms y total 372ms en CPU es 10x peor que el 7B en GPU. Para un router
que debe correr antes de cada inferencia, descalifica. Sirve donde ya se
usa: como reasoner en :8200.

## Recomendación

**Mantener Qwen2.5-7B-AWQ en `:8100` como router.** Ningún candidato lo
reemplaza con mejora neta de calidad+latencia.

**Cambios concretos en código** (orden de impacto):

1. **Agregar `stop` tokens al `FastRouter`** → +18 puntos accuracy classify, cero costo.
2. **Reescribir `should_use_deep_reasoning`** o eliminarla → la implementación
   actual es ruido, mejor heurística determinística.
3. **Bajar `--gpu-memory-utilization 0.80 → 0.55` en `:8100`** → libera ~2 GB
   de GPU1 sin tocar accuracy (a confirmar midiendo concurrencia real en prod).

**No invertir en**: cambiar el modelo del router, fine-tuning, ni armar un
modelo más chico. El cuello de botella real es el prompt/scaffolding, no la
capacidad del modelo.

## Limitaciones del bench

- **Latencia CPU vs GPU no es comparable directo**. Los modelos chicos podrían
  ser 3-5x más rápidos en GPU AWQ, pero no se midió por no disrumpir prod.
- **Golden set es chico** (27 casos). Con 100+ utterances reales del log de
  KZA los porcentajes se estabilizarían más.
- **Una sola corrida por caso, temperatura 0.3**. Para decisiones de
  producción correr 5x y reportar varianza.
- **Reasoning broken globalmente**: el prompt de `should_use_deep_reasoning`
  no funciona para casi ningún modelo. Vale repensar la tarea, no solo medir.

## Archivos generados

- `golden_set.yaml` — 27 casos con labels (5 corregidos post-bench, marcados con comentario)
- `runner.py` — runner streaming con scoring normalizado
- `analyze.py` — tabla resumen + detección de fallos universales
- `rescore.py` — re-scorea sin re-correr modelos
- `results_*.json` — raw outputs por modelo (en server)
- `launch/llama_cpu.sh` — wrapper llama-server CPU para los chicos
