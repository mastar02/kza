"""SC-0 (1): classify every static protected writer and join the dynamic observations.

Classes (reachability from process start until the sealed receipt, normal mode):
  boot_repair      one-shot write during construction or start-up before READY,
                   acting on residue (changes rows only when the residue exists)
  boot_init        idempotent initialization: writes only on a first boot or a
                   schema upgrade; a no-op statement on an initialized state
  every_txn        runs inside every DashboardAuthority transaction: boot
                   initialize (main.py:1982), the close-path final rotate
                   (authority.py:1053) and every user action
  recurring        a maintenance loop started during construction (runs before
                   and after READY until close)
  close_path       runs only on the stop path (exit stack / closure witness)
  ingress          needs audio, a browser stream or a house effect firing (V has none)
  user             only from a dashboard, voice or CLI request
  controller       compaction worker / adoption / operator script, never the app

Usage: python tools/sc0/classify.py --logs DIR [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.sc0.write_inventory import scan  # noqa: E402

# (file, function prefix) -> (class, how it is reached at 0bd9f98)
RULES: list[tuple[str, str, str, str]] = [
    ("src/ambient/control.py", "AmbientControl._load", "boot_init",
     "AmbientControl._initialize task spawned at build_ambient_control (main.py:1991); INSERT OR IGNORE of the default row"),
    ("src/ambient/control.py", "AmbientControl._write", "user", "ambient pause/resume (dashboard)"),
    ("src/ambient/control.py", "AmbientControl._restore_paused", "user", "failure path of a user resume"),
    ("src/ambient/store.py", "AmbientStore.reserve_audio_path", "ingress", "ambient archiving of captured audio"),
    ("src/ambient/store.py", "AmbientStore.set_audio_path", "ingress", "ambient archiving of captured audio"),
    ("src/ambient/store.py", "AmbientStore.discard_audio", "ingress", "ambient archiving of captured audio"),
    ("src/ambient/store.py", "AmbientStore.purge_expired", "recurring",
     "capture retention loop (build_capture_erasure -> retention.start, ambient/erasure.py:127; retention.py:239) and transcriber purge after ambient start (transcriber.py:641)"),
    ("src/ambient/store.py", "AmbientStore.erase_conversation", "recurring",
     "erasure driver step CaptureEraser.clean for a pending conversation erasure"),
    ("src/ambient/store.py", "AmbientStore._queue_audio_deletion", "recurring",
     "inside purge_expired / erase_conversation / discard_audio"),
    ("src/ambient/store.py", "AmbientStore._retry_audio_deletions", "recurring",
     "retention loop at construction and again after READY (observed: queued_audio_deletion)"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction:733", "every_txn",
     "expired-session purge of every transaction"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction:734", "every_txn",
     "expired-credential purge of every transaction"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction:741", "every_txn",
     "initialize/rotate when the legacy token digest or epoch changed"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction:743", "every_txn",
     "legacy_digest/legacy_epoch rewrite on initialize (boot) and rotate (close)"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction:789", "every_txn",
     "credentials of retired users, every transaction"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction:790", "every_txn",
     "orphan sessions, every transaction"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction:791", "every_txn",
     "clock_floor, every transaction: changes auth_meta on every boot and close"),
    ("src/dashboard/authority.py", "DashboardAuthority._transaction", "user",
     "issue/login/revise/retire_user/revoke/logout/credential_command branches"),
    ("src/dashboard/credential_commands.py", "", "user", "credential commands (operator CLI / dashboard)"),
    ("src/dashboard/credential_journal.py", "migrate_schema", "boot_init",
     "authority initialize (main.py:1982): first-boot metadata (:217) and schema_version '3' rewrite (:247, same value)"),
    ("src/dashboard/credential_journal.py", "insert_receipt", "user", "credential command receipt"),
    ("src/dashboard/live_receipts.py", "LiveReceiptStore._recover", "boot_init",
     "dynamic table: live_connections/live_resets only (not protected)"),
    ("src/dashboard/live_receipts.py", "", "ingress", "live browser stream (dashboard websocket)"),
    ("src/lists/list_store.py", "", "user", "list commands (voice/dashboard)"),
    ("src/memory/capture_control.py", "CaptureControl._load_policy", "boot_init",
     "capture_control._initialize task (build_capture_control, main.py:2004); INSERT only when the row is missing"),
    ("src/memory/capture_control.py", "CaptureControl._write_mode", "user", "mute/unmute"),
    ("src/memory/capture_control.py", "CaptureControl._force_muted", "user", "failed user transition restores mute"),
    ("src/memory/capture_recorders.py", "CaptureRecorders._initialize", "boot_repair",
     "active->unknown, recorders.initialize from capture_control._initialize (capture_control.py:233)"),
    ("src/memory/capture_recorders.py", "CaptureRecorders._retire_reserved", "close_path",
     "recorders.drain (capture_recorders.py:517): at close via _CaptureClosure.pre_close (runtime_closure.py:510) and, with the mode muted, already at boot via capture_control._restore -> _drain_recorders (capture_control.py:253-256, :386)"),
    ("src/memory/capture_recorders.py", "", "ingress", "browser recorder register/activate/retire/forget"),
    ("src/memory/deletion_ledger.py", "", "follows_caller",
     "append-only ledger rows of the deletion that calls it (observed: boot and tick retired-owner prune)"),
    ("src/memory/erasure_journal.py", "ErasureJournal.recover", "boot_repair",
     "privacy_services.initialize (privacy_services.py:77): tombstone backfill"),
    ("src/memory/erasure_journal.py", "ErasureJournal.ensure_operation", "user", "delete_conversation"),
    ("src/memory/fact_ledger.py", "initialize_fact_owner_schema", "boot_repair",
     "every FactLedger.initialize (fact_ledger.py:942) from privacy_services.initialize (privacy_services.py:75)"),
    ("src/memory/fact_ledger.py", "FactLedger._delete", "recurring",
     "via expire (privacy maintain loop), clear_user_facts <- prune_retired_owners (boot main.py:355 and tick privacy_services.py:164), and user deletes"),
    ("src/memory/fact_ledger.py", "FactLedger.prune_voice_attribution", "recurring",
     "privacy maintain loop (privacy_services.py:190), first tick during construction"),
    ("src/memory/fact_ledger.py", "FactLedger.clear_user_facts", "boot_repair",
     "prune_retired_owners at boot (main.py:355) and on the maintenance tick when deferred or hourly; also user retirement"),
    ("src/memory/fact_ledger.py", "FactLedger._count_not_mine", "ingress", "voice attribution of ambient proposals"),
    ("src/memory/fact_ledger.py", "", "user", "owner assignment, voice consent, delete_conversation, switch"),
    ("src/memory/legacy_quarantine.py", "", "controller", "stage_legacy_memory.py operator script"),
    ("src/memory/payload_derivation.py", "", "controller", "compaction worker (K materialization)"),
    ("src/memory/state_journal_derivation.py", "", "controller", "compaction worker (K materialization)"),
    ("src/obs/diagnostic_store.py", "DiagnosticStore._initialize", "boot_repair",
     "quarantine moved=0 -> 1 with rename; first metrics_emitter retention (metrics_emitter.py:222 -> diagnostic_store.py:427)"),
    ("src/obs/diagnostic_store.py", "DiagnosticStore._cleanup_staged", "boot_repair",
     "staged row delete + unlink at _initialize (:264) and on every _expire (:280, retention)"),
    ("src/obs/diagnostic_store.py", "DiagnosticStore._import", "boot_init", "first import of the JSONL catalog"),
    ("src/obs/diagnostic_store.py", "DiagnosticStore._publish", "recurring",
     "catalog prune/publish by the metrics retention (staged insert+delete in one pass)"),
    ("src/reminders/reminder_store.py", "ReminderStore.initialize", "boot_repair",
     "sending->uncertain at reminder_store.initialize (main.py:1082)"),
    ("src/reminders/reminder_store.py", "ReminderStore.claim_home_action", "ingress", "reminder firing (house effect)"),
    ("src/reminders/reminder_store.py", "ReminderStore.finish_home_action", "ingress", "reminder firing (house effect)"),
    ("src/reminders/reminder_store.py", "ReminderStore.mark_fired", "ingress", "reminder scheduler (start_reminders)"),
    ("src/reminders/reminder_store.py", "ReminderStore.cleanup_old", "user", "no caller in src at 0bd9f98"),
    ("src/reminders/reminder_store.py", "", "user", "reminder create/cancel/update"),
    ("src/routines/routine_store.py", "RoutineStore.recover_interrupted", "boot_repair",
     "RoutineScheduler.start (routine_scheduler.py:227) from Application._start_services (application.py:318), before READY"),
    ("src/routines/routine_store.py", "RoutineStore.save", "user", "routine save"),
    ("src/routines/routine_store.py", "RoutineStore.delete", "user", "routine delete"),
    ("src/routines/routine_store.py", "", "ingress", "routine runs (scheduler loop or manual run: house effect)"),
    ("src/users/voice_groups.py", "VoiceGroups.prune", "recurring",
     "VoiceLedger.prune (voice_ledger.py:561): boot prune in build_voice_registry (voice_registry.py:400) and VoiceRegistry.start maintain every 1 s (voice_registry.py:351-356)"),
    ("src/users/voice_groups.py", "", "user", "voice group correction"),
    ("src/users/voice_ledger.py", "VoiceLedger._delete", "recurring",
     "user profile delete (voice_ledger.py:544) and VoiceLedger.prune (:568) for a profile whose source conversation was erased"),
    ("src/users/voice_ledger.py", "", "user", "voice profile confirm"),
    ("src/users/voice_people.py", "", "user", "person links (dashboard) and user retirement"),
]


def rule_for(row: dict) -> tuple[str, str]:
    for file, prefix, kind, how in RULES:
        if row["file"] != file:
            continue
        name, _, line = prefix.partition(":")
        if name and not row["function"].startswith(name):
            continue
        if line and int(line) != row["line"]:
            continue
        return kind, how
    return "unclassified", ""


def observations(logs: Path) -> dict:
    seen: dict = {}
    for result in sorted(logs.glob("*/result.json")):
        data = json.loads(result.read_text())
        for write in data.get("protected_writes", []):
            site = write["stack"][0] if write["stack"] else None
            if site is None:
                # aiosqlite runs the statement on its own thread: use the submitter's frame.
                callers = [c for c in (write.get("caller") or []) if c.get("frames")]
                site = callers[-1]["frames"][0] if callers else None
            if site is None:
                continue
            file, line, function = site.split(":", 2)
            entry = seen.setdefault((file, int(line)), {"function": function, "states": set(), "phases": set()})
            entry["states"].add(data["state"])
            entry["phases"].add(write["phase"])
    return seen


def classify(logs: Path) -> list[dict]:
    seen = observations(logs)
    rows = []
    for row in scan(REPO):
        kind, how = rule_for(row)
        hits = [value for (file, line), value in seen.items()
                if file == row["file"] and line == row["line"]]
        rows.append({**row, "class": kind, "reached_by": how,
                     "observed_phases": sorted({p for hit in hits for p in hit["phases"]}),
                     "observed_states": sorted({s for hit in hits for s in hit["states"]})})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    rows = classify(Path(args.logs))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(f"{row['file']}:{row['line']}\t{row['verb']} {row['table']}\t{row['class']}\t"
              f"{','.join(row['observed_phases']) or '-'}")


if __name__ == "__main__":
    main()
