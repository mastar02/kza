"""Junta todos los results_*.json y genera un análisis legible."""

from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path("/home/kza/kza/benchmarks/router")


def load_all():
    seen: dict[str, dict] = {}
    for f in sorted(glob.glob(str(ROOT / "results_*.json"))):
        for r in json.load(open(f)):
            seen[r["model_id"]] = r  # último gana
    return list(seen.values())


def fmt(reps, key):
    return [(r["model_id"], r["accuracy"].get(key, 0)) for r in reps]


def main():
    reps = load_all()

    # Tabla resumen
    print(f"{'model':25s}  {'cls':>6s}  {'rsn':>6s}  {'rsp':>6s}  {'avg':>6s}  "
          f"{'ttft_cls':>9s}  {'ttft_rsp':>9s}  {'tot_rsp':>9s}")
    print("-" * 95)
    for r in reps:
        cls = r["accuracy"].get("classify", 0)
        rsn = r["accuracy"].get("reasoning", 0)
        rsp = r["accuracy"].get("respond", 0)
        avg = (cls + rsn + rsp) / 3
        ttft_cls = r["ttft_ms"].get("classify", {}).get("p50", 0)
        ttft_rsp = r["ttft_ms"].get("respond", {}).get("p50", 0)
        tot_rsp = r["total_ms"].get("respond", {}).get("p50", 0)
        print(f"{r['model_id']:25s}  {cls:6.1%}  {rsn:6.1%}  {rsp:6.1%}  {avg:6.1%}  "
              f"{ttft_cls:7.0f}ms  {ttft_rsp:7.0f}ms  {tot_rsp:7.0f}ms")

    # Outliers / casos donde TODOS fallan (señal de problema con golden o prompt)
    print("\n=== Casos donde TODOS los modelos fallan (señal de prompt/golden malo) ===")
    by_case_task: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for r in reps:
        for row in r["raw"]:
            by_case_task.setdefault((row["case_id"], row["task"]), []).append(
                (r["model_id"], row["correct"])
            )
    for (case, task), results in sorted(by_case_task.items()):
        if all(not ok for _, ok in results):
            print(f"  {case}/{task}: TODOS fallan ({len(results)} modelos)")

    print("\n=== Casos donde SOLO el 30B acierta y el resto falla (techo legítimo) ===")
    for (case, task), results in sorted(by_case_task.items()):
        rdict = dict(results)
        if rdict.get("qwen3-30b-a3b-q5") and not any(
            v for k, v in rdict.items() if k != "qwen3-30b-a3b-q5"
        ):
            print(f"  {case}/{task}")

    # Confiabilidad del modelo: ¿devuelve formato well_formed?
    print("\n=== well_formed_rate por tarea (¿el modelo devuelve algo parseable?) ===")
    print(f"{'model':25s}  {'cls_wf':>7s}  {'rsn_wf':>7s}")
    for r in reps:
        cls_wf = r["well_formed_rate"].get("classify", 0)
        rsn_wf = r["well_formed_rate"].get("reasoning", 0)
        print(f"{r['model_id']:25s}  {cls_wf:7.1%}  {rsn_wf:7.1%}")


if __name__ == "__main__":
    main()
