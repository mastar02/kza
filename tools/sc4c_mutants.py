"""SC-4c mutants (sc-release-design v4 implementation_units SC-4c: 7): the policy allowlist's three canaries
(expiry, retired owner, orphan record), the named drop of home_assistant_routines, the unlisted-collection
refusal at plan and in the worker, and D5 for rebuilt stores. Review round 1 adds REVIEW_1 (SC4C-RV-1:
death signal, guard, inherited lease, timeout; RV-2: the rebuild in the worker process; RV-3: the
afterimage and resume's evidence pins; RV-5: the exact names, the coded ledger error, SQLITE_TMPDIR, the
store census). SC4C_Q2B adds the named drop of the legacy 'user_memories' of data/memory_db (exact name,
that store only, never read, recorded by name). Restore-in-finally runner; a mutant is killed only by a
failing test (pytest exit 1). The household is home: every pytest runs on cores 24-47, niced.

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
     "    elif name in CHROMA_DROPPED or (store, name) in CHROMA_DROPPED_IN_STORE:\n        return \"dropped\"\n"
     "    raise CompactionCloneError",
     "    elif name in CHROMA_DROPPED:\n        return \"copied\"\n"
     "    elif (store, name) in CHROMA_DROPPED_IN_STORE:\n        return \"dropped\"\n"
     "    raise CompactionCloneError", [E2E]),
    ("plan admits an unlisted collection", C,
     "        return \"dropped\"\n    return None", "        return \"dropped\"\n    return \"copied\"",
     [T + "test_a_collection_like_the_legacy_one_refuses_the_plan_and_writes_nothing"]),
    ("worker admits an unlisted collection", OC,
     "        return \"dropped\"\n    raise CompactionCloneError(\"runtime_admission_compaction_unlisted_collection\")",
     "        return \"dropped\"\n    return \"copied\"",
     [T + "test_the_worker_refuses_an_unlisted_collection_the_plan_missed"]),
    ("D5 does not compare the rebuilt stores' pins", OC,
     "        _require(_store_pin(destination, store) == chroma_pins[store], \"payload_changed:chroma\")",
     "        pass", [T + "test_d5_refuses_a_rebuilt_store_changed_after_its_check"]),
]


# Review round 1. The in-worker rebuild (SC4C-RV-2's mutant, the pre-52e63d7 behaviour) runs the child's
# program with exec in this process; its parent check is satisfied as the real child's would be.
IN_PROCESS = """    import contextlib
    import io
    spec["parent"] = os.getppid()
    holder = SimpleNamespace(buffer=io.BytesIO(_json(spec)))
    saved, sys.stdin = sys.stdin, holder
    out = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out):
            exec(CHROMA_REBUILD_CHILD, {"__name__": "rebuild"})
    except SystemExit as stop:
        code = stop.code or 0
    finally:
        sys.stdin = saved
    completed = SimpleNamespace(returncode=code, stdout=out.getvalue().encode())
"""
CHILD_RUN = """    try:
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", CHROMA_REBUILD_CHILD], input=_json(spec),
                                   capture_output=True, env=environment, check=False, pass_fds=(lease,),
                                   timeout=CHROMA_REBUILD_TIMEOUT)
    except subprocess.TimeoutExpired:  # run() has killed and reaped the child
        raise CompactionCloneError("runtime_admission_compaction_chroma_rebuild_timeout") from None
"""
SIGKILL = T + "test_a_sigkilled_worker_takes_its_rebuild_child_with_it_and_abort_leaves_nothing"
OUTLIVES = T + "test_a_rebuild_child_that_outlives_its_worker_holds_the_lease_until_it_is_gone"
CLEAN = T + "test_each_store_is_written_by_a_clean_child_that_gets_only_k_s_records"
CLOSED = T + "test_the_closed_list_is_the_same_at_plan_and_in_the_worker"
REVIEW_1 = [
    ("RV-1 no death signal in the child", OC, '    "if sys.platform.startswith(\'linux\'):",',
     '    "if False:",', [SIGKILL]),
    ("RV-1 no parent guard in the child", OC,
     '    "    if os.getppid() != spec[\'parent\']:",\n    "        os._exit(5)",', '    "    return",',
     [OUTLIVES + "[guard]"]),
    ("RV-1 lease not inherited by the child", OC, "check=False, pass_fds=(lease,),", "check=False,",
     [OUTLIVES + "[lease_only]", CLEAN]),
    ("RV-1 no timeout on the child", OC, ",\n                                   timeout=CHROMA_REBUILD_TIMEOUT)", ")",
     [T + "test_a_rebuild_that_overruns_its_timeout_is_killed_and_refused"]),
    ("RV-2 rebuild in the worker process (no clean child)", OC, CHILD_RUN, IN_PROCESS, [CLEAN]),
    ("RV-3 child afterimage hash not compared", OC,
     '    _require(lines[-1:] == [hashlib.sha256(_json(_store_image(wanted))).hexdigest()], "payload_changed:chroma")',
     "    pass", [T + "test_the_child_s_afterimage_must_be_the_planned_one"]),
    ("RV-3 resume recomputes the pins instead of reading the evidence", OC,
     '    counts = verify_identity(plan, destination, receipt, chroma["pins"])',
     "    counts = verify_identity(plan, destination, receipt, {store: _store_pin(destination, store)\n"
     "                                                           for store in plan.chroma_stores})",
     [T + "test_resume_refuses_a_rebuilt_store_changed_after_the_evidence"]),
    ("RV-5 worker rebuilds a registered collection under any name", OC,
     "        if name.endswith(CHROMA_POLICY_SUFFIX) and len(name) > len(CHROMA_POLICY_SUFFIX):\n"
     "            return \"rebuilt\"\n    elif", "        return \"rebuilt\"\n    elif", [CLOSED]),
    ("RV-5 plan rebuilds a registered collection under any name", C,
     "        if name.endswith(CHROMA_POLICY_SUFFIX) and len(name) > len(CHROMA_POLICY_SUFFIX):\n"
     "            return \"rebuilt\"\n    elif", "        return \"rebuilt\"\n    elif", [CLOSED]),
    ("RV-5 worker copies any home_assistant_commands* name", OC, "    elif CHROMA_COMMAND_NAMES.fullmatch(name):",
     "    elif name.startswith(CHROMA_COMMAND_INDEX):", [CLOSED]),
    ("RV-5 deletion ledger error left uncoded", OC, "    except DeletionLedgerError as error:",
     "    except ZeroDivisionError as error:",
     [T + "test_a_deletion_ledger_the_allowlist_cannot_use_refuses_with_a_code"]),
    ("RV-5 child drops SQLITE_TMPDIR", OC, '"TMPDIR", "SQLITE_TMPDIR")', '"TMPDIR")',
     [T + "test_the_rebuild_child_gets_the_worker_s_environment_keys"]),
    ("RV-5 plan does not count other files in a store", C, '"other_files": _other_store_files(path.parent),',
     '"other_files": 0,', [T + "test_the_plan_counts_a_file_left_inside_a_store"]),
]
MUTANTS += REVIEW_1

# SC4C-Q2b (the user's answer of 2026-11-16): the legacy memory collection is a named drop.
LEGACY_E2E = T + "test_the_legacy_user_memories_collection_is_dropped_by_name_and_never_reaches_k"
LOOK_ALIKE = T + "test_a_collection_like_the_legacy_one_refuses_the_plan_and_writes_nothing"
IN_STORE = "(store, name) in CHROMA_DROPPED_IN_STORE"
SC4C_Q2B = [
    ("Q2b worker copies the legacy collection instead of dropping it", OC,
     "    elif name in CHROMA_DROPPED or (store, name) in CHROMA_DROPPED_IN_STORE:\n        return \"dropped\"\n"
     "    raise CompactionCloneError",
     "    elif name in CHROMA_DROPPED:\n        return \"dropped\"\n"
     "    elif (store, name) in CHROMA_DROPPED_IN_STORE:\n        return \"copied\"\n"
     "    raise CompactionCloneError", [LEGACY_E2E]),
    ("Q2b snapshot reads the legacy collection (classifier ignores the in-store drop)", OC,
     '    return "omit" if collection_kind(name, policy=registered, store=relative) == "dropped" else "export"',
     '    return "omit" if name in CHROMA_DROPPED else "export"', [LEGACY_E2E]),
    ("Q2b worker drops the legacy name in any store", OC, IN_STORE,
     "name in {dropped for _store, dropped in CHROMA_DROPPED_IN_STORE}", [CLOSED]),
    ("Q2b plan drops the legacy name in any store", C, IN_STORE,
     "name in {dropped for _store, dropped in CHROMA_DROPPED_IN_STORE}", [CLOSED, LOOK_ALIKE + "[other_store]"]),
    ("Q2b worker drops a look-alike (prefix)", OC, IN_STORE,
     "any(store == where and name.startswith(dropped) for where, dropped in CHROMA_DROPPED_IN_STORE)", [CLOSED]),
    ("Q2b plan drops a look-alike (prefix)", C, IN_STORE,
     "any(store == where and name.startswith(dropped) for where, dropped in CHROMA_DROPPED_IN_STORE)",
     [CLOSED, LOOK_ALIKE + "[look_alike]"]),
    ("Q2b plan drops a look-alike (case)", C, IN_STORE, "(store, name.lower()) in CHROMA_DROPPED_IN_STORE",
     [CLOSED, LOOK_ALIKE + "[case]"]),
    ("Q2b plan does not name the drop", C,
     '                if kind == "dropped":\n                    counts["dropped_collections"].append(name)\n', "",
     [LEGACY_E2E]),
    ("Q2b worker does not record the drop by name", OC,
     '            counts["dropped_collections"].append([item.store, item.collection])', "            pass",
     [LEGACY_E2E]),
]
MUTANTS += SC4C_Q2B


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
            run = subprocess.run(["taskset", "-c", "24-47", "nice", "-n", "10", PY, "-m", "pytest", "-q", "-x",
                                  "-p", "no:cacheprovider",
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
