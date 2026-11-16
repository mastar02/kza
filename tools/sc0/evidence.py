"""SC-0: assemble sc0-evidence.json from the measurement outputs (atomic write).

Inputs (all under --logs, the evidence log directory):
  states/<state>/result.json      run_states.py per-state result (latest run; projected_writes
                                  joins every table capture_state_authority can hash)
  inventory-classified.json       classify.py output
  compaction-server-prod-*.json   compaction_timing.py on the server
Usage: python tools/sc0/evidence.py --logs DIR --out sc0-evidence.json --commit SHA
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

ROOT_SLOT = "data/graph/"


def _state_summary(result: dict) -> dict:
    outcome = result["outcome"]
    changes = []
    for item in result.get("diff", []):
        name = item.get("table") or item.get("file")
        if "removed" in item:
            changes.append({"db": item["db"].replace(ROOT_SLOT, ""), "table": name,
                            "rows_removed": item["removed"], "rows_added": item["added"]})
        else:
            changes.append({"db": (item.get("db") or item.get("kind", "")).replace(ROOT_SLOT, ""),
                            "table": name, "changed": "projection value (ledger chain or new table)"})
    writers = collections.OrderedDict()
    for write in result.get("projected_writes", result.get("protected_writes", [])):
        site = write["stack"][0] if write["stack"] else None
        if site is None:
            callers = [c for c in (write.get("caller") or []) if c.get("frames")]
            site = callers[-1]["frames"][0] if callers else "?"
        key = (site, write["table"], write["phase"], write.get("projection", "full"))
        writers.setdefault(key, 0)
        writers[key] += write["count"]
    return {
        "residue": (result.get("residue") or {}).get("planted", []),
        "child_flags": (result.get("residue") or {}).get("child_flags", []),
        "receipt_sealed": outcome.get("receipt_sealed"), "head_after": outcome.get("head_status"),
        "exit": outcome.get("returncode"), "notify": outcome.get("notify"),
        "boot_to_ready_s_stubbed": float(outcome["phases"]["ready"]) if "ready" in outcome.get("phases", {}) else None,
        "stop_s": outcome.get("stop_seconds"), "settle_s": outcome.get("settle_seconds"),
        "protected_projection_changes": changes,
        "projected_write_statements": [{"site": k[0], "table": k[1], "phase": k[2], "projection": k[3],
                                        "statements": v} for k, v in writers.items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--narrative", required=True, help="JSON file with the hand-written sections")
    args = parser.parse_args()
    logs = Path(args.logs)
    states = {}
    for path in sorted((logs / "states").glob("*/result.json")):
        result = json.loads(path.read_text())
        states[result["state"]] = _state_summary(result)
    for path in sorted((logs / "states").glob("*/result.json")):
        if json.loads(path.read_text())["state"] != path.parent.name:
            raise SystemExit(f"state label mismatch: {path}")
    inventory = json.loads((logs / "inventory-classified.json").read_text())
    by_class = collections.Counter(row["class"] for row in inventory)
    by_projection = collections.Counter(row.get("projection", "full") for row in inventory)
    table = [{"site": f"{row['file']}:{row['line']}", "function": row["function"], "statement": row["verb"],
              "table": row["table"], "projection": row.get("projection", "full"), "call": row.get("call"),
              "class": row["class"], "reached_by": row["reached_by"],
              "observed_phases": row["observed_phases"], "observed_states": row["observed_states"]}
             for row in inventory]
    compaction = {path.stem: json.loads(path.read_text())
                  for path in sorted(logs.glob("compaction-*.json"))}
    narrative = json.loads(Path(args.narrative).read_text())
    evidence = {**narrative["head"],
                "inventory": {"method": narrative["inventory_method"], "counts_by_class": dict(by_class),
                              "counts_by_projection": dict(by_projection), "rows": table},
                "dynamic_diff": {"method": narrative["diff_method"], "states": states},
                **narrative["tail"],
                "compaction_raw": {name: {key: value for key, value in data.items()
                                          if key not in ("plan_view",)} for name, data in compaction.items()}}
    out = Path(args.out)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + "\n")
    os.replace(temporary, out)


if __name__ == "__main__":
    main()
