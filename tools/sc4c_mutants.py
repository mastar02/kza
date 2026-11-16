"""SC-4c mutants (sc-release-design v4 implementation_units SC-4c: 7): the policy allowlist's three canaries
(expiry, retired owner, orphan record), the named drop of home_assistant_routines, the unlisted-collection
refusal at plan and in the worker, and D5 for rebuilt stores. Restore-in-finally runner; a mutant is
killed only by a failing test (pytest exit 1).

Usage (server, a mutation clone): python tools/sc4c_mutants.py OUT.json [NAME ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
T = "tests/unit/tools/test_sc4c_chroma_closed_list.py::"
OC = "src/memory/owner_compaction.py"
C = "scripts/compaction.py"
ALLOWLIST = T + "test_the_policy_collection_keeps_only_facts_get_returns_with_a_live_owner"
E2E = T + "test_the_first_post_r1_transition_rebuilds_chroma_from_the_closed_list"

MUTANTS = [
    ("allowlist: expiry ignored (overdue fact kept)", OC,
     "                    or expires is not None and not (type(expires) in (int, float) and expires > now)\n",
     "", [ALLOWLIST, E2E]),
    ("allowlist: retired owner ignored", OC,
     "                    or owners.get(identity) in retired\n", "", [ALLOWLIST, E2E]),
    ("allowlist: orphan or stale record kept", OC,
     "    return (fact is not None and record[\"document\"] == fact[1]\n"
     "            and record[\"metadata\"] == {\"fact_policy\": 1, \"revision\": fact[0]})",
     "    return True", [ALLOWLIST, E2E]),
    ("home_assistant_routines copied instead of dropped", OC,
     "    if name in CHROMA_DROPPED:\n        return \"dropped\"\n    raise CompactionCloneError",
     "    if name in CHROMA_DROPPED:\n        return \"copied\"\n    raise CompactionCloneError", [E2E]),
    ("plan admits an unlisted collection", C,
     "    if name in CHROMA_DROPPED:\n        return \"dropped\"\n    return None",
     "    if name in CHROMA_DROPPED:\n        return \"dropped\"\n    return \"copied\"",
     [T + "test_a_legacy_user_memories_collection_refuses_the_plan_and_writes_nothing"]),
    ("worker admits an unlisted collection", OC,
     "        return \"dropped\"\n    raise CompactionCloneError(\"runtime_admission_compaction_unlisted_collection\")",
     "        return \"dropped\"\n    return \"copied\"",
     [T + "test_the_worker_refuses_an_unlisted_collection_the_plan_missed"]),
    ("D5 does not compare the rebuilt stores' pins", OC,
     "        _require(_store_pin(destination, store) == chroma_pins[store], \"payload_changed:chroma\")",
     "        pass", [T + "test_d5_refuses_a_rebuilt_store_changed_after_its_check"]),
]


def _wait_for_a_slot() -> None:
    """Shared server: start a pytest only while fewer than four of this user's run."""
    while True:
        count = subprocess.run(["pgrep", "-u", "yo", "-fc", "python.*-m pytest"], capture_output=True, text=True)
        if int(count.stdout.strip() or 0) < 4:
            return
        time.sleep(20)


def main() -> None:
    out = Path(sys.argv[1])
    wanted = set(sys.argv[2:])
    results = json.loads(out.read_text()) if out.exists() else []
    done = {item["mutant"] for item in results}
    for name, rel, old, new, tests in MUTANTS:
        if (wanted and name not in wanted) or name in done:
            continue
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        assert original.count(old) == 1, (name, original.count(old))
        _wait_for_a_slot()
        started = time.monotonic()
        try:
            path.write_text(original.replace(old, new), encoding="utf-8")
            run = subprocess.run(["nice", "-n", "10", PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                  "--timeout=600", *tests], cwd=ROOT, capture_output=True, text=True, timeout=2400)
        finally:
            path.write_text(original, encoding="utf-8")
        tail = run.stdout.strip().splitlines()[-1:] if run.stdout.strip() else []
        results.append({"mutant": name, "file": rel, "tests": tests, "killed": run.returncode == 1,
                        "exit": run.returncode, "seconds": round(time.monotonic() - started, 1), "tail": tail})
        print(name, "KILLED" if results[-1]["killed"] else "SURVIVED", tail, flush=True)
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out)
    print(sum(item["killed"] for item in results), "/", len(results))


if __name__ == "__main__":
    main()
