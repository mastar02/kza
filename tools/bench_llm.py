#!/usr/bin/env python3
"""Benchmark de LLMs locales para evaluar candidatos a reemplazar el 72B.

Mide tok/s realistas en el server KZA (Threadripper PRO 7965WX, 128GB DDR5,
~270 GB/s bandwidth). Reporta PP (prompt processing), TG (generation),
TTFT (time to first token), peak RAM y cantidad de tokens.

Output JSON para que sea fácil comparar modelos / quants:

    ssh kza 'cd ~/kza && .venv/bin/python tools/bench_llm.py \\
        --model ./models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \\
        --threads 24 --n-ctx 4096 --runs 3'

Uso típico — comparar 3 candidatos en una corrida:

    for m in models/*.gguf; do
      .venv/bin/python tools/bench_llm.py --model "$m" --runs 3 \\
        >> /tmp/bench_results.jsonl
    done
    jq -s '[.[] | {model, tg_tps_median, pp_tps_median, ram_gb}]' \\
      < /tmp/bench_results.jsonl

Diseñado para correr aislado del kza-voice.service: no toca ChromaDB, HA ni
GPUs. Un solo modelo en CPU. Pensado para ejecutarse cuando el resto del
sistema está quiescente para no contaminar la medición.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


# Prompt representativo de KZA: razonamiento conversacional en español, con
# system prompt + historia + pregunta. Deliberadamente "tamaño slow path".
SYSTEM_PROMPT = (
    "Sos KZA, un asistente de voz local que controla la casa de Gabriel. "
    "Hablás español rioplatense con voseo. Sos directo, breve y concreto. "
    "Cuando te preguntan algo, primero pensás un poco y después respondés "
    "en una o dos oraciones cortas. Si no sabés algo, lo decís sin inventar."
)
USER_PROMPT = (
    "Estoy en el escritorio. La luz está prendida pero el ambiente sigue "
    "frío y no entiendo por qué — el aire dice 24 grados y la termo del "
    "living también marca 24. ¿Puede ser que la diferencia sea por la "
    "ventana grande o hay algo más a chequear antes de subir el aire?"
)
GENERATION_TOKENS = 128


def _peak_ram_gb() -> float:
    """RAM RSS pico del proceso actual, en GB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss en Linux está en kilobytes; en macOS en bytes.
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024**3)
        return usage.ru_maxrss / (1024**2)
    except Exception:
        return 0.0


def _load_model(path: str, n_threads: int, n_ctx: int, n_batch: int):
    """Importa llama_cpp dinámicamente y carga el modelo."""
    from llama_cpp import Llama
    return Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=0,
        verbose=False,
        chat_format="chatml",
    )


def _format_chat(model, system: str, user: str) -> str:
    """Render del chat template. Si el modelo no tiene template, fallback simple."""
    try:
        # Algunos modelos exponen `apply_chat_template` vía tokenizer_
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    except Exception:
        return f"{system}\n\nUser: {user}\nAssistant:"


def _bench_run(model, prompt: str, max_tokens: int) -> dict[str, float]:
    """Una corrida de bench. Devuelve metrics individuales."""
    t_start = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    prompt_tokens = 0

    stream = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        completion_tokens += 1
        # Capturar prompt_tokens de la respuesta (algunos backends lo dan al final)
        usage = chunk.get("usage") or {}
        if usage.get("prompt_tokens"):
            prompt_tokens = usage["prompt_tokens"]

    t_end = time.perf_counter()

    if first_token_at is None:
        first_token_at = t_end  # no hubo tokens — TTFT = total

    ttft_ms = (first_token_at - t_start) * 1000
    pp_seconds = first_token_at - t_start
    tg_seconds = max(t_end - first_token_at, 1e-6)

    # Si no recibimos prompt_tokens del backend, tokenizamos directo
    if prompt_tokens == 0:
        try:
            prompt_tokens = len(model.tokenize(prompt.encode("utf-8")))
        except Exception:
            prompt_tokens = -1

    return {
        "ttft_ms": ttft_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "pp_tps": prompt_tokens / pp_seconds if prompt_tokens > 0 else 0.0,
        "tg_tps": completion_tokens / tg_seconds,
        "total_s": t_end - t_start,
    }


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path al GGUF")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 24)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3, help="Cantidad de corridas (mediana)")
    parser.add_argument("--max-tokens", type=int, default=GENERATION_TOKENS)
    parser.add_argument("--label", default=None, help="Tag opcional (ej: 'qwen3-14b-q4')")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    label = args.label or model_path.stem
    size_gb = model_path.stat().st_size / (1024**3)

    print(f"[bench] loading {label} ({size_gb:.1f}GB) "
          f"threads={args.threads} n_ctx={args.n_ctx}", file=sys.stderr)

    t0 = time.perf_counter()
    model = _load_model(str(model_path), args.threads, args.n_ctx, args.n_batch)
    load_seconds = time.perf_counter() - t0
    print(f"[bench] loaded in {load_seconds:.1f}s", file=sys.stderr)

    prompt = _format_chat(model, SYSTEM_PROMPT, USER_PROMPT)

    # Warmup — primera corrida descartada (CUDA-style cold start no aplica
    # en CPU pero hay efectos de cache/page-fault del mmap del modelo).
    print("[bench] warmup...", file=sys.stderr)
    _bench_run(model, prompt, max_tokens=16)

    runs: list[dict[str, float]] = []
    for i in range(args.runs):
        print(f"[bench] run {i+1}/{args.runs}...", file=sys.stderr)
        runs.append(_bench_run(model, prompt, max_tokens=args.max_tokens))

    result: dict[str, Any] = {
        "model": label,
        "model_path": str(model_path),
        "model_size_gb": round(size_gb, 2),
        "threads": args.threads,
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "load_seconds": round(load_seconds, 2),
        "runs": args.runs,
        "ttft_ms_median": round(_median([r["ttft_ms"] for r in runs]), 1),
        "pp_tps_median": round(_median([r["pp_tps"] for r in runs]), 1),
        "tg_tps_median": round(_median([r["tg_tps"] for r in runs]), 2),
        "total_s_median": round(_median([r["total_s"] for r in runs]), 2),
        "completion_tokens": runs[0]["completion_tokens"],
        "prompt_tokens": runs[0]["prompt_tokens"],
        "ram_gb_peak": round(_peak_ram_gb(), 2),
        "raw_runs": runs,
    }

    # Cleanup explícito antes de salir para reportar RAM real
    del model
    gc.collect()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"\n[bench] {label}: TG {result['tg_tps_median']} tok/s, "
        f"PP {result['pp_tps_median']} tok/s, "
        f"TTFT {result['ttft_ms_median']} ms, "
        f"RAM {result['ram_gb_peak']} GB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
