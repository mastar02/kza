"""Re-scorea los results existentes contra el golden corregido.

Evita re-correr modelos: usa los raw_output guardados.
"""
from __future__ import annotations

import glob
import json
import sys
import unicodedata
from pathlib import Path

import yaml

ROOT = Path("/home/kza/kza/benchmarks/router")


def _norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_acc.lower().split())


def score_classify(raw, expected, options):
    first = next((ln for ln in raw.splitlines() if ln.strip()), "")
    norm = _norm(first)
    return _norm(expected) in norm


def score_reasoning(raw, expected):
    up = raw.strip().upper()
    if "SIMPLE" in up and ("COMPLEJO" not in up or up.find("SIMPLE") < up.find("COMPLEJO")):
        return expected == "SIMPLE"
    if "COMPLEJO" in up:
        return expected == "COMPLEJO"
    return False


def score_respond(raw, expected_deep):
    is_deep = "[DEEP]" in raw[:32].upper()
    return is_deep == expected_deep


def main():
    cases = {c["id"]: c for c in yaml.safe_load((ROOT / "golden_set.yaml").read_text())["cases"]}
    seen: dict[str, dict] = {}
    for f in sorted(glob.glob(str(ROOT / "results_*.json"))):
        for r in json.load(open(f)):
            seen[r["model_id"]] = r
    reps = list(seen.values())

    results = []
    for r in reps:
        per_task = {"classify": [0, 0], "reasoning": [0, 0], "respond": [0, 0]}
        for row in r["raw"]:
            case = cases.get(row["case_id"])
            if not case:
                continue
            raw = row["raw_output"]
            if row["task"] == "classify":
                ok = score_classify(raw, case["classify_expected"], case["classify_options"])
            elif row["task"] == "reasoning":
                ok = score_reasoning(raw, case["reasoning_expected"])
            else:
                ok = score_respond(raw, case["respond_should_be_deep"])
            per_task[row["task"]][0] += int(ok)
            per_task[row["task"]][1] += 1
        results.append((r["model_id"], per_task, r["ttft_ms"], r["total_ms"]))

    print(f"{'model':25s}  {'cls':>6s}  {'rsn':>6s}  {'rsp':>6s}  {'avg':>6s}  "
          f"{'ttft_cls':>9s}  {'ttft_rsp':>9s}  {'tot_rsp':>9s}")
    print("-" * 95)
    for mid, pt, ttft, tot in results:
        cls = pt["classify"][0] / max(pt["classify"][1], 1)
        rsn = pt["reasoning"][0] / max(pt["reasoning"][1], 1)
        rsp = pt["respond"][0] / max(pt["respond"][1], 1)
        avg = (cls + rsn + rsp) / 3
        ttft_cls = ttft.get("classify", {}).get("p50", 0)
        ttft_rsp = ttft.get("respond", {}).get("p50", 0)
        tot_rsp = tot.get("respond", {}).get("p50", 0)
        print(f"{mid:25s}  {cls:6.1%}  {rsn:6.1%}  {rsp:6.1%}  {avg:6.1%}  "
              f"{ttft_cls:7.0f}ms  {ttft_rsp:7.0f}ms  {tot_rsp:7.0f}ms")


if __name__ == "__main__":
    main()
