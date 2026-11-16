"""Print a compact view of a run_states results file."""
import json
import sys

for r in json.load(open(sys.argv[1])):
    print("=====", r["state"], json.dumps(r.get("residue", "")))
    if "harness_error" in r:
        print(r["harness_error"][-1500:])
        continue
    o = dict(r["outcome"])
    lt = o.pop("log_tail")
    print(json.dumps({k: o[k] for k in ("returncode", "error", "notify", "phases", "stop_seconds",
                                         "head_status", "receipt_sealed") if k in o}))
    if "-v" in sys.argv:
        print("\n".join(lt[-25:]))
    for d in r.get("diff", []):
        print("  DIFF", json.dumps(d)[:300])
    if "diff_error" in r:
        print("  DIFF_ERROR", r["diff_error"])
    for w in r["protected_writes"]:
        print("  W", w["phase"], w["db"].split("/")[-1], w["verb"], w["table"], w["count"], "|",
              " < ".join(w["stack"][:3]))
        for c in (w.get("caller") or []):
            print("      caller task=%s coro=%s | %s" % (c.get("task"), c.get("task_coro"),
                  " < ".join(c.get("frames", [])[:6])))
