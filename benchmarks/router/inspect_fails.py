"""Mostrar qué dijo cada modelo en los casos donde todos fallan."""
import glob
import json
from pathlib import Path

ROOT = Path("/home/kza/kza/benchmarks/router")
seen = {}
for f in sorted(glob.glob(str(ROOT / "results_*.json"))):
    for r in json.load(open(f)):
        seen[r["model_id"]] = r
reps = list(seen.values())

fail_cases = ["bdr-004", "bdr-006", "cmp-001", "cmp-003", "cnv-001", "prg-003"]
expected = {
    "bdr-004": ("hace frío", "conversacion"),
    "bdr-006": ("todo cerrado abajo", "pregunta"),
    "cmp-001": ("agendame una alarma para mañana...", "domotica"),
    "cmp-003": ("ayudame a redactar un mail...", "conversacion"),
    "cnv-001": ("explicame fotosíntesis", "conversacion"),
    "prg-003": ("está prendida la luz del cuarto", "pregunta"),
}
for case_id in fail_cases:
    utt, exp = expected[case_id]
    print(f"\n### {case_id}: '{utt}' (esperado: {exp})")
    for r in reps:
        for row in r["raw"]:
            if row["case_id"] == case_id and row["task"] == "classify":
                out = row["raw_output"][:80].replace("\n", " | ")
                mid = r["model_id"]
                print(f"  {mid:25s}: {out}")

# 1.5B respond outlier
print("\n=== 1.5B respond fails ===")
r = seen.get("qwen2.5-1.5b-cpu")
if r:
    for row in r["raw"]:
        if row["task"] == "respond" and not row["correct"]:
            out = row["raw_output"][:90].replace("\n", " | ")
            print(f"  {row['case_id']}: {out}")
