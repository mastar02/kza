#!/usr/bin/env python3
"""Scored eval of the climate/AC classifier against the live :8101.

This is NOT a pytest test. A probabilistic classifier is measured against a
threshold, not asserted for equality -- the assertEqual half lives in
tests/unit/nlu/test_climate_intent.py with a mocked router.

Run:
    .venv/bin/python3 benchmarks/router/climate_eval.py
    .venv/bin/python3 benchmarks/router/climate_eval.py --set benchmarks/router/climate_set.yaml

Requires an SSH tunnel to the fast-path llama-server and its bearer key in
the environment (:8101 returns HTTP 401 without it):

    ssh -f -N -L 8101:127.0.0.1:8101 kza
    export LLAMA_API_KEY="$(ssh kza 'cat /home/kza/secrets/llama-api-key')"

Go/no-go, all three must hold:
    - overall accuracy >= 90%          (rules baseline is 77.3%)
    - zero QUERY misread as ACTION     (unrequested physical action)
    - p95 <= 150ms                     (must stay under the classifier timeout)
"""

import argparse
import asyncio
import hashlib
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.reasoner import FastRouter  # noqa: E402
from src.nlu.climate_intent import (  # noqa: E402
    CLIMATE_PROMPT,
    ClimateIntent,
    ClimateIntentClassifier,
)

ACCURACY_FLOOR = 0.90
P95_CEILING_MS = 150.0

EXPECTED = {"ACCION": ClimateIntent.ACTION, "CONSULTA": ClimateIntent.QUERY}

# The brief's default ("qwen2.5-7b-instruct") is a friendly alias that does
# NOT exist on this endpoint. ik_llama.cpp's llama-server reports the full
# .gguf path it loaded as the model id, and the OpenAI client requires an
# exact string match against /v1/models -- verified 2026-08-04 against the
# live :8101 behind the SSH tunnel. Using the alias here would 400 on every
# single request.
DEFAULT_MODEL = "/home/kza/kza/models/Qwen2.5-7B-Instruct-Q4_K_M/Qwen2.5-7B-Instruct-Q4_K_M.gguf"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="benchmarks/router/climate_set.yaml")
    ap.add_argument("--base-url", default="http://127.0.0.1:8101/v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout-s", type=float, default=5.0,
                    help="generous here on purpose: we want to MEASURE the tail, "
                         "not have it swallowed by the production timeout")
    args = ap.parse_args()

    cases = yaml.safe_load(Path(args.set).read_text())["cases"]
    # api_key_env explicit: :8101 requires bearer auth (HTTP 401 without it),
    # and FastRouter's port-based heuristic already points 8101 at
    # LLAMA_API_KEY -- being explicit here just documents the requirement at
    # the call site instead of relying on the reader knowing the heuristic.
    router = FastRouter(base_url=args.base_url, model=args.model, api_key_env="LLAMA_API_KEY")
    clf = ClimateIntentClassifier(router, timeout_s=args.timeout_s)

    # Warm the prefix cache so the first case does not skew the tail.
    await clf.classify("calentar")

    latencies: list[float] = []
    correct = 0
    query_as_action: list[str] = []   # expensive error: unrequested action
    action_as_query: list[str] = []   # cheap error: says forecast, does nothing
    abstained: list[str] = []

    for case in cases:
        want = EXPECTED[case["expected"]]
        t0 = time.perf_counter()
        got = await clf.classify(case["utterance"])
        latencies.append((time.perf_counter() - t0) * 1000)

        if got == want:
            correct += 1
        elif got is None:
            abstained.append(case["id"])
        elif want is ClimateIntent.QUERY:
            query_as_action.append(case["id"])
        else:
            action_as_query.append(case["id"])

        mark = "OK  " if got == want else "MISS"
        label = got.value if got else "ABSTAIN"
        print(f"{mark} {case['id']:12} want={case['expected']:8} got={label:8} "
              f"{latencies[-1]:6.1f}ms  {case['utterance'][:52]}")

    n = len(cases)
    accuracy = correct / n
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p95 = ordered[min(int(n * 0.95), n - 1)]

    print("\n" + "=" * 72)
    print(f"casos                 {n}")
    print(f"accuracy              {correct}/{n} = {accuracy * 100:.1f}%   (piso {ACCURACY_FLOOR * 100:.0f}%)")
    print(f"consulta -> accion    {len(query_as_action)}  {query_as_action}   (error caro, piso 0)")
    print(f"accion -> consulta    {len(action_as_query)}  {action_as_query}   (error barato)")
    print(f"abstenciones          {len(abstained)}  {abstained}")
    print(f"latencia              p50 {p50:.0f}ms   p95 {p95:.0f}ms   (techo {P95_CEILING_MS:.0f}ms)")
    print(f"prompt fingerprint    {hashlib.sha256(CLIMATE_PROMPT.encode()).hexdigest()[:16]}")

    checks = {
        "accuracy": accuracy >= ACCURACY_FLOOR,
        "cero consulta->accion": not query_as_action,
        "p95": p95 <= P95_CEILING_MS,
    }
    for name, passed in checks.items():
        print(f"  [{'OK' if passed else 'NO'}] {name}")

    verdict = all(checks.values())
    print(f"\nVEREDICTO: {'GO' if verdict else 'NO-GO'}")
    print("=" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
