"""SC-4a mutants (test_plan.mutant_rules): one per skip entry and part, the acknowledgement, the
capture close, the flag, the records and the verified predicate. Restore-in-finally runner.

Usage (server, a mutation clone): python tools/sc4a_mutants.py OUT.json [ID ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
RUN = "tests/unit/tools/test_sc4a_verify_run.py::"
BOOT = "tests/integration/test_sc4a_verify_boot.py::"
STORES = "tests/unit/memory/test_sc4a_verify_stores.py::"
APP = "tests/unit/core/test_sc4a_application_verify.py::"
STRUCT = BOOT + "test_verify_mode_starts_no_listed_loop_and_a_normal_boot_starts_each"
BRANCH = BOOT + "test_verify_branch_reaches_no_ingress_or_house_effect_start"


def canary(name: str) -> str:
    return BOOT + f"test_residue_canary_leaves_k_prime_equal_to_k[{name}]"


MUTANTS = [
    # --- skip list (src) ---
    ("c-authority: commit instead of ROLLBACK", "src/dashboard/authority.py",
     "            if self._verify_only:\n                db.rollback()", "            if False:\n                db.rollback()",
     [BOOT + "test_the_authority_commits_nothing_in_verify_mode"]),
    ("c-authority: close-path rotate commits", "src/dashboard/authority.py",
     "            if self._verify_only:\n                db.rollback()",
     "            if self._verify_only and action != 'rotate':\n                db.rollback()",
     [BOOT + "test_the_authority_commits_nothing_in_verify_mode"]),
    ("b-retired-owner-prune runs", "src/main.py",
     "        if not verify_only:  # skip list b-retired-owner-prune", "        if True:  # skip list b-retired-owner-prune",
     [canary("retired_owner_facts")]),
    ("a-privacy-maintenance started", "src/memory/privacy_services.py",
     "        if self.maintenance is not None or self.verify_only:", "        if self.maintenance is not None:",
     [STORES + "test_privacy_services_verify_mode_runs_no_recover_and_no_maintenance"]),
    ("b-erasure-recover: erasures.recover runs", "src/memory/privacy_services.py",
     "                if not self.verify_only:  # skip list b-erasure-recover\n                    self.ledger.erasures.recover()",
     "                self.ledger.erasures.recover()\n                if not self.verify_only:  # skip list b-erasure-recover\n                    pass",
     [STORES + "test_privacy_services_verify_mode_runs_no_recover_and_no_maintenance"]),
    ("b-erasure-recover: index_deletions.recover runs", "src/memory/privacy_services.py",
     "                    self.ledger.erasures.recover()\n                    self.ledger.index_deletions.recover()",
     "                    self.ledger.erasures.recover()\n                self.ledger.index_deletions.recover()",
     [STORES + "test_privacy_services_verify_mode_runs_no_recover_and_no_maintenance"]),
    ("d-fact-owner-schema-prune runs", "src/memory/fact_ledger.py",
     "initialize_fact_owner_schema(db, prune=not self.verify_only)", "initialize_fact_owner_schema(db, prune=True)",
     [STORES + "test_fact_owner_prune_at_open_is_skipped_only_in_verify_mode"]),
    ("c-index-target-registration inserts", "src/memory/index_deletions.py",
     "        if getattr(self.ledger, \"verify_only\", False) is True:\n            return\n", "",
     [STORES + "test_register_target_inserts_nothing_in_verify_mode"]),
    ("d-recorders-initialize: active->unknown runs", "src/memory/capture_recorders.py",
     "                if not self.verify_only:  # skip list d-recorders-initialize\n",
     "                if True:  # skip list d-recorders-initialize\n",
     [STORES + "test_verify_capture_close_carries_every_row_unchanged_without_waiting"]),
    ("capture-close: verify drain deletes reserved rows", "src/memory/capture_recorders.py",
     "            # that no browser can send to a verification run; the rows stay as they were.\n            return",
     "            # that no browser can send to a verification run; the rows stay as they were.\n"
     "            await self._call(self._retire_reserved)\n            return",
     [STORES + "test_verify_capture_close_carries_every_row_unchanged_without_waiting"]),
    ("capture-close: verify drain waits for acknowledgements", "src/memory/capture_recorders.py",
     "            # that no browser can send to a verification run; the rows stay as they were.\n            return",
     "            # that no browser can send to a verification run; the rows stay as they were.\n"
     "            while self.pending_live():\n                await asyncio.sleep(0.025)\n            return",
     [STORES + "test_verify_capture_close_carries_every_row_unchanged_without_waiting"]),
    ("capture-close: a changed recorder row accepted", "src/memory/capture_recorders.py",
     "        return self._construction_rows is not None and rows == self._construction_rows",
     "        return self._construction_rows is not None",
     [STORES + "test_verify_capture_close_refuses_a_changed_recorder_row"]),
    ("capture-close: pre_close takes the normal pending_live path", "src/memory/runtime_closure.py",
     "        if self.verify_only:\n            # SC-4a capture_close", "        if False:\n            # SC-4a capture_close",
     [canary("recorders_crash_running")]),
    ("d-reminder-sending runs", "src/reminders/reminder_store.py",
     "                if not self._verify_only:\n                    db.execute(\"UPDATE reminder_home_actions",
     "                if True:\n                    db.execute(\"UPDATE reminder_home_actions",
     [STORES + "test_reminder_sending_action_is_left_as_is_only_in_verify_mode"]),
    ("d-diagnostic-repair: emitter retention start", "src/obs/diagnostic_erasure.py",
     "        if not verify_only:\n            try:\n                await emitter.start()",
     "        if True:\n            try:\n                await emitter.start()",
     [canary("diagnostic_quarantine_staged")]),
    ("d-diagnostic-repair: history loop", "src/obs/diagnostic_erasure.py",
     "    if not verify_only:\n        services.start()", "    if True:\n        services.start()", [STRUCT]),
    ("a-capture-retention loop", "src/ambient/erasure.py",
     "    if not verify_only:\n        driver.retention.start()", "    if True:\n        driver.retention.start()",
     [STRUCT]),
    ("a-voice-registry: boot prune", "src/users/voice_registry.py",
     "    if not verify_only:\n        await voices.acall(voices.ledger.prune)\n        voices.start()",
     "    await voices.acall(voices.ledger.prune)\n    if not verify_only:\n        voices.start()",
     [canary("expired_voice_group_request")]),
    ("a-voice-registry: maintain loop", "src/users/voice_registry.py",
     "    if not verify_only:\n        await voices.acall(voices.ledger.prune)\n        voices.start()",
     "    voices.start()\n    if not verify_only:\n        await voices.acall(voices.ledger.prune)",
     [STRUCT]),
    ("routine-recover: verify branch starts services", "src/application.py",
     "        stop.enter(StopPhase.PIPELINE)\n        await self.pipeline.prepare()",
     "        stop.enter(StopPhase.PIPELINE)\n        await self._start_services()\n        await self.pipeline.prepare()",
     [APP + "test_verify_branch_prepares_acknowledges_then_ready_and_starts_nothing", BRANCH]),
    ("a-wake-clip-maintenance", "src/wakeword/clip_store.py",
     "    if not verify_only:\n        store.start()", "    if True:\n        store.start()", [canary("wake_clip_fenced")]),
    ("a-other-loops: erasure coordinator (collector eraser)", "src/memory/index_erasure.py",
     "    if not verify_only:\n        driver.maintenance.start()\n        owner.start()",
     "    owner.start()\n    if not verify_only:\n        driver.maintenance.start()", [STRUCT]),
    ("a-other-loops: index maintenance", "src/memory/index_erasure.py",
     "    if not verify_only:\n        driver.maintenance.start()\n        owner.start()",
     "    driver.maintenance.start()\n    if not verify_only:\n        owner.start()", [STRUCT]),
    ("a-other-loops: live bus retention", "src/dashboard/live_erasure.py",
     "    if not verify_only:\n        bus.start()", "    if True:\n        bus.start()", [STRUCT]),
    ("a-other-loops: memory maintenance", "src/memory/services.py",
     "    if not verify_only:\n        services.start()", "    if True:\n        services.start()", [STRUCT]),
    ("a-other-loops: suggestion retention", "src/analytics/suggestion_erasure.py",
     "    if not verify_only:  # SC-4a skip list a-other-loops (suggestion retention)",
     "    if True:  # SC-4a skip list a-other-loops (suggestion retention)", [STRUCT]),
    ("a-other-loops: context retention", "src/orchestrator/context_erasure.py",
     "    if not verify_only:  # SC-4a skip list a-other-loops (context retention)",
     "    if True:  # SC-4a skip list a-other-loops (context retention)", [STRUCT]),
    ("d-boot-retention-clamps: event retention sweep", "src/analytics/erasure.py",
     "    if not verify_only:  # SC-4a skip list d-boot-retention-clamps (event retention sweep)",
     "    if True:  # SC-4a skip list d-boot-retention-clamps (event retention sweep)", [STRUCT]),
    ("d-boot-retention-clamps: event clamp", "src/analytics/event_logger.py",
     "            if not self.verify_only:\n                conn.execute(\"\"\"UPDATE events_policy_v1",
     "            if True:\n                conn.execute(\"\"\"UPDATE events_policy_v1",
     [canary("retention_shortened")]),
    ("d-boot-retention-clamps: suggestion clamp", "src/analytics/suggestion_store.py",
     "            if not self.verify_only:\n                clamp_suggestion_deadlines", "            if True:\n                clamp_suggestion_deadlines",
     [canary("retention_shortened")]),
    ("a-other-loops: event logger maintenance", "src/analytics/services.py",
     "    if retention is None and not verify_only:", "    if retention is None:",
     [STORES + "test_event_logger_maintenance_is_not_started_in_verify_mode"]),
    ("verify branch runs pipeline.run (router, features, audio)", "src/application.py",
     "        await self.pipeline.prepare()\n        logger.info(\"[Verify]",
     "        await self.pipeline.run()\n        logger.info(\"[Verify]",
     [APP + "test_verify_branch_prepares_acknowledges_then_ready_and_starts_nothing"]),
    ("verify branch starts the ambient path", "src/application.py",
     "        await self.pipeline.prepare()\n        logger.info(\"[Verify]",
     "        await self.pipeline.prepare()\n        if self.ambient_path is not None:\n"
     "            asyncio.create_task(self.ambient_path.start())\n        logger.info(\"[Verify]",
     [APP + "test_verify_branch_prepares_acknowledges_then_ready_and_starts_nothing", BRANCH]),
    # --- acknowledgement (R5-PRIV-2) ---
    ("child sends VERIFY outside the verify_only branch", "src/application.py",
     "        try:\n            if self.verify_only:\n                await self._run_verify(stop)",
     "        try:\n            if self.verify_run_id is not None and self.lifecycle is not None:\n"
     "                self.lifecycle.notifier.verify(self.verify_run_id)\n            if self.verify_only:\n"
     "                await self._run_verify(stop)",
     [APP + "test_a_normal_run_never_sends_verify_even_with_a_run_id"]),
    ("serve verifies without the acknowledgement", "scripts/managed_launcher.py",
     "            if not relay.verify_acknowledged:\n                # R5-PRIV-2",
     "            if False:\n                # R5-PRIV-2",
     [RUN + "test_ready_without_the_acknowledgement_is_stopped_at_once_and_never_verified[verify_ignored]"]),
    ("relay accepts VERIFY after READY", "scripts/managed_launcher.py",
     "if self._verify_run_id is not None and not self.ready and verify.group(1) == self._verify_run_id:",
     "if self._verify_run_id is not None and verify.group(1) == self._verify_run_id:",
     [RUN + "test_notify_relay_counts_verify_only_for_its_run_before_ready"]),
    ("relay accepts VERIFY of another run", "scripts/managed_launcher.py",
     "if self._verify_run_id is not None and not self.ready and verify.group(1) == self._verify_run_id:",
     "if self._verify_run_id is not None and not self.ready:",
     [RUN + "test_notify_relay_counts_verify_only_for_its_run_before_ready"]),
    # --- the flag ---
    ("flag added before the KZA_RUNTIME_* filter", "scripts/managed_launcher.py",
     "    child = {key: value for key, value in environ.items()\n             if key not in _CHILD_ENV_DROPPED and not key.startswith(\"KZA_RUNTIME_\")}\n"
     "    if release_verify is True:\n        child[RELEASE_VERIFY_ENV] = \"1\"",
     "    child = {key: value for key, value in environ.items()\n             if key not in _CHILD_ENV_DROPPED and not key.startswith(\"KZA_RUNTIME_\")}\n"
     "    if environ.get(RELEASE_VERIFY_ENV) == \"1\":\n        child[RELEASE_VERIFY_ENV] = \"1\"",
     [RUN + "test_child_environment_adds_the_flag_after_the_filter_only_when_asked"]),
    ("flag for any verify run (plain logs too)", "scripts/managed_launcher.py",
     "        if gate[\"kind\"] == \"verify\" and stage.release_mode_intent(gate[\"log\"]):",
     "        if gate[\"kind\"] is not None:", [RUN + "test_the_unit_cannot_inject_the_flag_and_h_runs_never_carry_it"]),
    ("child keeps the flag in its environment", "src/core/runtime_admission.py",
     "    raw = environ.pop(RELEASE_VERIFY_ENV, None)", "    raw = environ.get(RELEASE_VERIFY_ENV)",
     [RUN + "test_take_release_verify_flag_reads_once_and_needs_a_ticket"]),
    ("child accepts the flag without a ticket", "src/core/runtime_admission.py",
     "    if raw != \"1\" or environ.get(TICKET_ENV) is None:", "    if raw != \"1\":",
     [RUN + "test_take_release_verify_flag_reads_once_and_needs_a_ticket"]),
    # --- records and the verified predicate ---
    ("short run verified (no measured >= N)", "scripts/stage_runtime.py",
     "and math.isfinite(measured) and type(smoke_seconds) is int and measured >= smoke_seconds",
     "and math.isfinite(measured) and type(smoke_seconds) is int",
     [RUN + "test_release_run_verified_needs_both_records_and_the_monotonic_interval"]),
    ("verified without ready.json", "scripts/stage_runtime.py",
     "    if ready is None or smoke is None or ready[\"verify_acknowledged\"] is not True:",
     "    if smoke is None:", [RUN + "test_release_run_verified_needs_both_records_and_the_monotonic_interval"]),
    ("release log verified by the seal alone", "scripts/stage_runtime.py",
     "    if release_mode_intent(log) and not release_run_verified(",
     "    if False and not release_run_verified(",
     [RUN + "test_a_short_run_is_not_verified_and_serve_refuses_it_afterwards"]),
    ("smoke interval on the wall clock", "scripts/managed_launcher.py",
     "        measured = self.clock() - smoke[\"started\"]\n        if measured >= self.release_verify[\"smoke_seconds\"]:",
     "        measured = self.wall() - smoke[\"started\"]\n        if measured >= self.release_verify[\"smoke_seconds\"]:",
     [RUN + "test_a_wall_clock_step_backwards_during_v_keeps_a_good_run_verified"]),
    ("verify branch reached ignores ready.json", "scripts/compaction.py",
     "    return \"yes\" if ready is not None and ready[\"verify_acknowledged\"] is True else \"no\"",
     "    return \"yes\"", [RUN + "test_ready_without_the_acknowledgement_is_stopped_at_once_and_never_verified[verify_ignored]"]),
    ("SC4A-N1: plain retire of a verified release log", "scripts/compaction.py",
     "        if release_log(log):\n            _refuse(RELEASE_RETIRE_UNAVAILABLE)", "        if False:\n            _refuse(RELEASE_RETIRE_UNAVAILABLE)",
     [RUN + "test_release_verify_run_is_the_acknowledged_isolated_boot_and_serve_stops_it"]),
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
            run = subprocess.run(["nice", "-n", "10", PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--timeout=600", *tests],
                                 cwd=ROOT, capture_output=True, text=True, timeout=2400)
        finally:
            path.write_text(original, encoding="utf-8")
        tail = run.stdout.strip().splitlines()[-1:] if run.stdout.strip() else []
        results.append({"mutant": name, "file": rel, "tests": tests, "killed": run.returncode not in (0, 5),
                        "exit": run.returncode, "seconds": round(time.monotonic() - started, 1), "tail": tail})
        print(name, "KILLED" if results[-1]["killed"] else "SURVIVED", tail, flush=True)
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out)
    print(sum(item["killed"] for item in results), "/", len(results))


if __name__ == "__main__":
    main()
