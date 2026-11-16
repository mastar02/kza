"""SC-4b mutants (test_plan.mutant_rules): each refusal code, each W1 classifier boundary, the
transient-vs-drift mapping, the retire gates, the adopt refusal, the status view and the selection
split. Restore-in-finally runner; a mutant is killed only by a failing test (pytest exit 1).

Usage (server, a mutation clone): python tools/sc4b_mutants.py OUT.json [NAME ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
W = "tests/unit/tools/test_sc4b_w.py::"
EX = "tests/unit/tools/test_sc4b_release_exits.py::"
SC1B = "tests/unit/tools/test_sc1b_release_plan.py::"
C = "scripts/compaction.py"
LISTED = W + "test_each_listed_table_in_both_directions"
SS = EX + "test_a_tombstone_written_during_v_stops_safely_until_the_discard"
LOSS = EX + "test_a_loss_only_change_is_never_retired_and_abort_returns_to_h"
AT = EX + "test_transient_refusals_write_nothing_and_clear_with_their_cause"
W0 = W + "test_w0_maps_each_failure_reading_h"
W0_SQL = W + "test_w0_classifies_sqlite_errors_by_code_not_by_class"

MUTANTS = [
    # --- W1 classifier boundaries (8) ---
    ("W1: removal from W_LOSS is restrictive", C,
     "                restrictive |= change[\"added\"] > 0\n                loss |= change[\"removed\"] > 0",
     "                restrictive |= change[\"added\"] > 0\n                restrictive |= change[\"removed\"] > 0",
     [LISTED]),
    ("W1: addition to W_LOSS is loss", C,
     "                restrictive |= change[\"added\"] > 0\n                loss |= change[\"removed\"] > 0",
     "                loss |= change[\"added\"] > 0\n                loss |= change[\"removed\"] > 0", [LISTED]),
    ("W1: addition to W_GRANT is restrictive", C,
     "                restrictive |= change[\"removed\"] > 0\n                loss |= change[\"added\"] > 0",
     "                restrictive |= change[\"removed\"] > 0\n                restrictive |= change[\"added\"] > 0",
     [LISTED]),
    ("W1: removal from W_GRANT is loss", C,
     "                restrictive |= change[\"removed\"] > 0\n                loss |= change[\"added\"] > 0",
     "                loss |= change[\"removed\"] > 0\n                loss |= change[\"added\"] > 0", [LISTED]),
    ("W1: ledger strict prefix not recognised", C,
     "    return \"strict_prefix\" if prefix == current else \"other\"", "    return \"other\"",
     [W + "test_ledger_strict_prefix_is_loss_and_an_entry_h_lacks_is_restrictive"]),
    ("W1: ledger 'other' counted as loss", C,
     "        elif found[\"ledger\"] == \"other\":\n            restrictive = True",
     "        elif found[\"ledger\"] == \"other\":\n            loss = True",
     [W + "test_a_ledger_that_is_not_a_prefix_of_h_is_restrictive"]),
    ("W1: a table on neither list is ignored", C,
     "            elif change[\"added\"] or change[\"removed\"]:\n                restrictive = True",
     "            elif change[\"added\"] or change[\"removed\"]:\n                pass",
     [W + "test_any_other_difference_is_restrictive"]),
    ("W1: added rows not counted (a changed row reads as a removal)", C,
     "        found[\"tables\"][key] = {\"added\": _count(after, before), \"removed\": _count(before, after)}",
     "        found[\"tables\"][key] = {\"added\": 0, \"removed\": _count(before, after)}",
     [W + "test_any_other_difference_is_restrictive"]),
    # --- W1: databases, unclassifiable values, accounts, configuration, mode=rw ---
    ("W1: a missing or unreadable database may be loss_only", C,
     "        if found[\"status\"] != \"ok\":\n            restrictive = True",
     "        if found[\"status\"] != \"ok\":\n            pass",
     [W + "test_a_missing_database_holding_only_loss_tables_is_still_restrictive"]),
    ("W1: a missing database read as unreadable (lexists check dropped)", C,
     "    if not os.path.lexists(path):\n        found[\"status\"] = \"missing\"",
     "    if False:\n        found[\"status\"] = \"missing\"",
     [W + "test_a_deleted_database_is_restrictive_and_w1_never_creates_it"]),
    ("W1: no in-place journal recovery", C, "            _w1_recover(path)\n", "",
     [W + "test_a_kill_inside_a_write_transaction_is_recovered_in_place_and_equal"]),
    ("W1: an unclassifiable value ignored", C,
     "            report[\"unclassified\"][relative] = found[\"unclassified\"]\n            restrictive = True",
     "            report[\"unclassified\"][relative] = found[\"unclassified\"]",
     [W + "test_an_unclassifiable_value_counts_as_a_changed_row"]),
    ("W1: account differences ignored", C,
     "        restrictive |= status != \"ok\" or change[\"added\"] > 0 or change[\"removed\"] > 0",
     "        restrictive |= False", [W + "test_accounts_and_configuration_differences_are_restrictive"]),
    ("W1: configuration differences ignored", C,
     "        restrictive |= status != \"equal\"", "        restrictive |= False",
     [W + "test_accounts_and_configuration_differences_are_restrictive"]),
    ("W_GRANT without sessions", C, "W_GRANT = frozenset({\"credentials\", \"sessions\"})",
     "W_GRANT = frozenset({\"credentials\"})", [W + "test_the_two_lists_are_the_reviewed_constants", LISTED]),
    # --- W0: transient vs drift ---
    ("W0: a transient code mapped to drift", C,
     "    if isinstance(error, ValueError) and str(error) in W0_TRANSIENT_CODES:",
     "    if False and isinstance(error, ValueError) and str(error) in W0_TRANSIENT_CODES:", [W0]),
    ("W0: transient errno mapped to drift", C,
     "        if error.errno in W0_TRANSIENT_ERRNOS:", "        if False:", [W0]),
    ("W0: ENOENT transient", C, "                                 errno.EBUSY})",
     "                                 errno.EBUSY, errno.ENOENT})", [W0]),
    ("W0: SQLite classified by class", C,
     "        if _sqlite_primary(error) in W0_TRANSIENT_SQLITE:",
     "        if isinstance(error, sqlite3.OperationalError):", [W0_SQL]),
    ("W0: sdk_versions-only difference is drift", C,
     "        if changed == [\"sdk_versions\"]:  # the shared venv, not H's files (R4-PRIV-1)",
     "        if False:  # the shared venv, not H's files (R4-PRIV-1)", [AT]),
    ("W0: K' writers not checked", C,
     "        stage._assert_writers_closed(Path(intent[\"k_state_root\"]))\n    except ValueError:\n"
     "        _refuse(\"runtime_admission_writer_active\", {\"w0\": \"transient\", \"state\": \"k_prime\"})",
     "        pass\n    except ValueError:\n"
     "        _refuse(\"runtime_admission_writer_active\", {\"w0\": \"transient\", \"state\": \"k_prime\"})", [AT]),
    # --- abort with W ---
    ("abort: release mode uses Q6", C, "                if release:  # SC-4b: W replaces Q6",
     "                if False:  # SC-4b: W replaces Q6", [LOSS]),
    ("abort: restrictive proceeds (no safe stop)", C,
     "    if report[\"class\"] == W_RESTRICTIVE:\n        if discard_k_changes is None:",
     "    if False:\n        if discard_k_changes is None:", [SS]),
    ("abort: loss_only treated as a safe stop", C,
     "    if report[\"class\"] == W_RESTRICTIVE:\n        if discard_k_changes is None:",
     "    if report[\"class\"] != W_EQUAL:\n        if discard_k_changes is None:", [LOSS]),
    ("abort: discard digest not checked", C,
     "        if len(discard_k_changes) < 12 or not digest.startswith(discard_k_changes.lower()):",
     "        if False:", [SS]),
    ("abort: discard not recorded", C,
     "        record[\"discarded\"] = {\"w_sha256\": digest, \"counts\": _w_counts(report)}  # SC-D8 A",
     "        pass", [SS]),
    ("abort: a discard without --to-h", C,
     "        if discard_k_changes is not None and not to_h:", "        if False:", [SS]),
    ("abort: --discard-k-changes on a plain log", C,
     "        if discard_k_changes is not None and not release:", "        if False:",
     [EX + "test_a_discard_on_a_plain_log_is_refused"]),
    ("abort: V3 (sealed, unverified) needs 'verified'", C,
     "    if verified and head[\"status\"] == \"closed\" and not _release_head_is_v(root, log, head):",
     "    if verified and head[\"status\"] == \"closed\" and stage.compaction_verified(root, log, head) is None:",
     [EX + "test_a_run_without_the_acknowledgement_is_aborted_through_w"]),
    ("abort: the recheck uses Q6", C,
     "            if release:\n                record = _w_checks(", "            if False:\n                record = _w_checks(",
     [EX + "test_a_discard_interrupted_after_abort_started_finishes_through_the_w_recheck"]),
    # --- retire gates ---
    ("retire: release gates skipped", C,
     "            w_record = _release_retire_gates(root, log, head)  # before any other write (R4-OPS-10b)",
     "            w_record = None", [SS]),
    ("retire: K's whole authority not compared", C,
     "    if found[\"report\"][\"class\"] != W_EQUAL or not authority_equal:",
     "    if found[\"report\"][\"class\"] != W_EQUAL:",
     [EX + "test_a_protected_table_created_only_in_k_prime_is_never_retired"]),
    ("retire: resurrection scan skipped", C, "    if any(resurrected.values()):", "    if False:",
     [EX + "test_a_resurrected_fact_is_refused_at_retire_and_abort_returns_to_h"]),
    ("retire: sdk_versions-only K' difference not transient", C,
     "        if changed == [\"sdk_versions\"]:\n            _refuse(SDK_CHANGED, {\"bound\"",
     "        if False:\n            _refuse(SDK_CHANGED, {\"bound\"", [AT]),
    # --- adopt, status, gate, selection ---
    ("adopt under a pending release-mode log", "scripts/managed_launcher.py",
     "        if admission.compaction is not None and release_mode_log(admission.compaction):",
     "        if False:", [SS, EX + "test_v_killed_inside_a_write_transaction_is_aborted_losslessly"]),
    ("status: the armed run always 'consumed'", C,
     "    return \"consumed\" if _armed_run_consumed(root, log) else \"not consumed\"", "    return \"consumed\"",
     [EX + "test_serve_dying_between_verify_started_and_begin_run_leaves_the_run_unconsumed"]),
    ("CLI drops --discard-k-changes", C, "                                   discard_k_changes=getattr(args, \"discard_k_changes\", None))",
     "                                   discard_k_changes=None)",
     [W + "test_the_cli_passes_discard_k_changes_and_prints_the_w_report"]),
    ("release gate closed again", C, "RELEASE_EXITS_AVAILABLE = True", "RELEASE_EXITS_AVAILABLE = False",
     [SC1B + "test_first_post_r1_plan_admits_new_and_start_runs_k_from_new"]),
    ("selection split drops deleted_fact_revisions", "scripts/stage_runtime.py",
     "    if \"policy_index_deletions\" in names:\n        selected[\"deleted_fact_revisions\"]",
     "    if False:\n        selected[\"deleted_fact_revisions\"]",
     [W + "test_capture_digest_is_unchanged_across_the_selection_split"]),
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
