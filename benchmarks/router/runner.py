"""Benchmark del FastRouter de KZA contra varios modelos.

Para cada modelo en `MODELS`, mide:
  - Accuracy en classify / should_use_deep_reasoning / classify_and_respond
  - TTFT (time-to-first-token) y latencia total por request
  - Output bien formado (no inventa categorías)

Reutiliza el SYSTEM_PROMPT_PREFIX real del FastRouter para que el resultado
sea fiel a producción.

Uso:
    python benchmarks/router/runner.py \
        --golden benchmarks/router/golden_set.yaml \
        --out benchmarks/router/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import os

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.reasoner import FastRouter  # SYSTEM_PROMPT_PREFIX vive acá


# Endpoints. Pensado para correr EN el server (URLs 127.0.0.1).
# Las api_keys se leen de env (VLLM_API_KEY / LLAMA_API_KEY) o de los archivos
# /home/kza/secrets/{vllm,llama}-api-key si están disponibles.

def _load_key(env_var: str, file_path: str) -> str | None:
    val = os.environ.get(env_var)
    if val:
        return val.strip()
    p = Path(file_path)
    if p.exists():
        return p.read_text().strip()
    return None


_VLLM_KEY = _load_key("VLLM_API_KEY", "/home/kza/secrets/vllm-api-key")
_LLAMA_KEY = _load_key("LLAMA_API_KEY", "/home/kza/secrets/llama-api-key")


MODELS: list[dict] = [
    # Baseline producción (ya corriendo en GPU1, vLLM, AWQ INT4).
    {"id": "qwen2.5-7b-awq",   "url": "http://127.0.0.1:8100/v1",
     "served": "qwen2.5-7b-awq", "api_key": _VLLM_KEY},
    # Techo de calidad: reasoner 30B-A3B Q5_K_M ya corriendo en CPU (ik_llama.cpp).
    {"id": "qwen3-30b-a3b-q5", "url": "http://127.0.0.1:8200/v1",
     # served name real expuesto por ik_llama.cpp = path completo del .gguf
     "served": "/home/kza/kza/models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf",
     "api_key": _LLAMA_KEY},
    # Candidatos chicos: GGUF + llama.cpp en CPU (8 threads, no compite con prod).
    # Latencia es CPU-bound: comparable entre ellos pero no contra GPU AWQ
    # (factor 3-5x más lento esperado vs vLLM).
    {"id": "qwen2.5-0.5b-cpu", "url": "http://127.0.0.1:8210/v1", "served": "qwen2.5-0.5b"},
    {"id": "qwen2.5-1.5b-cpu", "url": "http://127.0.0.1:8211/v1", "served": "qwen2.5-1.5b"},
    {"id": "qwen2.5-3b-cpu",   "url": "http://127.0.0.1:8212/v1", "served": "qwen2.5-3b"},
    {"id": "phi-3.5-mini-cpu", "url": "http://127.0.0.1:8213/v1", "served": "phi-3.5-mini"},
]

SYSTEM = FastRouter.SYSTEM_PROMPT_PREFIX


@dataclass
class CallResult:
    case_id: str
    task: str           # classify | reasoning | respond
    raw_output: str
    correct: bool
    well_formed: bool
    ttft_ms: float
    total_ms: float


@dataclass
class ModelReport:
    model_id: str
    n_cases: int
    accuracy: dict[str, float] = field(default_factory=dict)
    well_formed_rate: dict[str, float] = field(default_factory=dict)
    ttft_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    total_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    raw: list[CallResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------- Prompts (idénticos al FastRouter) ----------------------

def prompt_classify(text: str, options: list[str]) -> str:
    return (
        f"{SYSTEM}Clasifica en: {', '.join(options)}\n"
        f"Texto: {text}\n"
        f"Categoría:"
    )


def prompt_reasoning(text: str) -> str:
    return (
        f"{SYSTEM}¿Requiere razonamiento complejo? Responde SIMPLE o COMPLEJO.\n"
        f"Pregunta: {text}\n"
        f"Respuesta:"
    )


def prompt_respond(text: str) -> str:
    return (
        f"{SYSTEM}Consulta: {text}\n\n"
        f"Si requiere razonamiento complejo responde [DEEP], si no responde directamente:"
    )


# ---------------------- Cliente streaming para medir TTFT ----------------------

async def call_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None = None,
) -> tuple[str, float, float]:
    """Devuelve (texto, ttft_ms, total_ms) usando streaming."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
        # Cortar antes de que el modelo invente nuevas vueltas del prompt.
        # FastRouter prod no setea stop — ese es un bug que vale arreglar
        # también en producción, pero para fairness del benchmark lo aplico
        # acá uniformemente a todos los modelos.
        "stop": ["\n\n", "Texto:", "Pregunta:", "Consulta:", "Categoría:"],
    }
    t0 = time.perf_counter()
    ttft_ms = -1.0
    chunks: list[str] = []
    async with client.stream("POST", f"{base_url}/completions", json=payload, headers=headers) as r:
        async for line in r.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            text = obj["choices"][0].get("text", "")
            if text and ttft_ms < 0:
                ttft_ms = (time.perf_counter() - t0) * 1000
            chunks.append(text)
    total_ms = (time.perf_counter() - t0) * 1000
    return "".join(chunks), ttft_ms, total_ms


# ---------------------- Scoring por tarea ----------------------

def _norm(s: str) -> str:
    """Lowercase + strip diacritics (música → musica) + collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", s)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_acc.lower().split())


def score_classify(raw: str, expected: str, options: list[str]) -> tuple[bool, bool]:
    # Tomar solo la primera línea no vacía (el resto es ruido del completion).
    first = next((ln for ln in raw.splitlines() if ln.strip()), "")
    norm = _norm(first)
    well_formed = any(_norm(opt) in norm for opt in options)
    correct = _norm(expected) in norm
    return correct, well_formed


def score_reasoning(raw: str, expected: str) -> tuple[bool, bool]:
    up = raw.strip().upper()
    has_simple = "SIMPLE" in up
    has_complejo = "COMPLEJO" in up
    well_formed = has_simple or has_complejo
    # Si aparece [DEEP] el modelo ignoró las opciones SIMPLE/COMPLEJO
    # (caso del 30B-A3B): no well_formed.
    if not well_formed:
        return False, False
    # Si dice ambos, gana el primero que aparece.
    if has_simple and has_complejo:
        winner = "SIMPLE" if up.find("SIMPLE") < up.find("COMPLEJO") else "COMPLEJO"
    else:
        winner = "SIMPLE" if has_simple else "COMPLEJO"
    return winner == expected, True


def score_respond(raw: str, expected_deep: bool) -> tuple[bool, bool]:
    is_deep = "[DEEP]" in raw[:32].upper()
    well_formed = True  # cualquier output es válido si no esperaba [DEEP]
    correct = (is_deep == expected_deep)
    return correct, well_formed


# ---------------------- Loop principal ----------------------

async def benchmark_model(model: dict, cases: list[dict]) -> ModelReport:
    rep = ModelReport(model_id=model["id"], n_cases=len(cases))
    api_key = model.get("api_key")

    async with httpx.AsyncClient(timeout=60.0) as client:
        for case in cases:
            for task_name, build_prompt, max_t, scorer_args in [
                ("classify", lambda c: prompt_classify(c["utterance"], c["classify_options"]),
                 20, lambda c, raw: score_classify(raw, c["classify_expected"], c["classify_options"])),
                ("reasoning", lambda c: prompt_reasoning(c["utterance"]),
                 10, lambda c, raw: score_reasoning(raw, c["reasoning_expected"])),
                ("respond", lambda c: prompt_respond(c["utterance"]),
                 80, lambda c, raw: score_respond(raw, c["respond_should_be_deep"])),
            ]:
                try:
                    out, ttft, total = await call_completion(
                        client, model["url"], model["served"], build_prompt(case), max_t, api_key,
                    )
                    correct, well_formed = scorer_args(case, out)
                    rep.raw.append(CallResult(
                        case_id=case["id"], task=task_name, raw_output=out[:200],
                        correct=correct, well_formed=well_formed,
                        ttft_ms=ttft, total_ms=total,
                    ))
                except Exception as e:
                    rep.errors.append(f"{case['id']}/{task_name}: {e!r}")

    # Agregar accuracy + percentiles por tarea
    for task in ("classify", "reasoning", "respond"):
        rows = [r for r in rep.raw if r.task == task]
        if not rows:
            continue
        rep.accuracy[task] = sum(r.correct for r in rows) / len(rows)
        rep.well_formed_rate[task] = sum(r.well_formed for r in rows) / len(rows)
        ttfts = [r.ttft_ms for r in rows if r.ttft_ms > 0]
        totals = [r.total_ms for r in rows]
        rep.ttft_ms[task] = _percentiles(ttfts)
        rep.total_ms[task] = _percentiles(totals)
    return rep


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0, "p95": 0, "mean": 0}
    s = sorted(values)
    return {
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
        "mean": statistics.mean(values),
    }


# ---------------------- CLI ----------------------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="benchmarks/router/golden_set.yaml")
    ap.add_argument("--out", default="benchmarks/router/results.json")
    ap.add_argument("--only", help="Filtrar por model_id (substring)")
    args = ap.parse_args()

    cases = yaml.safe_load(Path(args.golden).read_text())["cases"]
    models = [m for m in MODELS if not args.only or args.only in m["id"]]

    reports = []
    for m in models:
        # Skip endpoints que no responden (modelo no levantado todavía).
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                hdr = {"Authorization": f"Bearer {m['api_key']}"} if m.get("api_key") else {}
                await c.get(f"{m['url']}/models", headers=hdr)
        except Exception:
            print(f"==> {m['id']} SKIP (endpoint {m['url']} no responde)", flush=True)
            continue
        print(f"==> {m['id']} ({m['url']})", flush=True)
        rep = await benchmark_model(m, cases)
        reports.append(rep)
        print(f"    classify={rep.accuracy.get('classify', 0):.1%}  "
              f"reasoning={rep.accuracy.get('reasoning', 0):.1%}  "
              f"respond={rep.accuracy.get('respond', 0):.1%}  "
              f"ttft_p50_classify={rep.ttft_ms.get('classify', {}).get('p50', 0):.0f}ms")
        if rep.errors:
            print(f"    ERRORS: {len(rep.errors)} (primero: {rep.errors[0]})")

    Path(args.out).write_text(json.dumps([asdict(r) for r in reports], indent=2, default=str))
    print(f"\nResultados → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
