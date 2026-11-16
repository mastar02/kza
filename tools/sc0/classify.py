"""SC-0 (1): classify every static writer of a projected table and join the dynamic observations.

"Projected" is every table capture_state_authority can hash (write_inventory.
projected_tables: _PROTECTED_TABLES, the partially projected tables and the
schema markers). UNSCANNED lists the projection writers the SQL scan cannot
see (configuration and accounts files, the implicit sqlite_sequence row).

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
  upgrade          schema migration: runs only when H's schema is older than NEW's
                   (changes schema_contract_v1 and every row hash of the table)
  boot_config      config-driven boot clamp: rewrites rows only when NEW's
                   retention setting is shorter than the one the rows were written under
  follows_caller   runs inside the transaction of the statement that calls it

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
    # ---- partially projected tables (R1) ----
    ("src/ambient/store.py", "AmbientStore.add", "ingress", "ambient transcription of captured audio"),
    ("src/ambient/store.py", "AmbientStore.mark_distilled", "recurring",
     "ambient distiller loop (distiller.py:357/:394, ambient_path.start): writes only when H holds undistilled utterances; distilled!=0 rows are projected"),
    ("src/ambient/store.py", "AmbientStore._migrate", "upgrade",
     "ALTER pending_audio_deletions ADD conversation_id when H predates it"),
    ("src/ambient/store.py", "<constant _MIGRATIONS>", "upgrade",
     "utterances column migrations run by AmbientStore._migrate (store.py:224-227) when H lacks the column"),
    ("src/analytics/erasure.py", "EventEraser._clean", "recurring",
     "erasure driver step for events (pending conversation erasure)"),
    ("src/analytics/event_logger.py", "EventLogger._init_db", "boot_config",
     "every boot: EventLogger.initialize (analytics/services.py, before SuggestionStore at main.py:1058) clamps expires_at to analytics.retention_days; expires_at is not in event_text_deadline_fences (id, sources_json, text_expires_at), so no projected column changes"),
    ("src/analytics/event_logger.py", "EventLogger.log", "ingress", "event logging of house/voice actions"),
    ("src/analytics/event_logger.py", "EventLogger.cleanup_old_events", "recurring",
     "event logger maintenance loop (analytics/services.py:34 -> event_logger.py:778)"),
    ("src/analytics/event_storage.py", "sweep_events", "recurring",
     "event retention (analytics/erasure.py:32 EventRetention -> analytics/retention.py:92)"),
    ("src/analytics/suggestion_storage.py", "clamp_suggestion_deadlines", "boot_config",
     "every boot: SuggestionStore.__init__ (suggestion_store.py:106) from construct_suggestions (main.py:1058) clamps expires_at to memory.automatic_retention_days; non-pending/snoozed rows are hashed with every column and fences use expires_at"),
    ("src/analytics/suggestion_storage.py", "cleanup_suggestions", "recurring",
     "suggestion eraser step (analytics/suggestion_erasure.py:177)"),
    ("src/analytics/suggestion_store.py", "SuggestionStore.create", "ingress",
     "suggestion generation from observed patterns (suggestion_engine.py:138, request path)"),
    ("src/analytics/suggestion_store.py", "SuggestionStore.respond", "user", "accept/reject/snooze"),
    ("src/analytics/suggestion_sweep.py", "sweep_suggestions", "recurring",
     "suggestion retention (analytics/suggestion_retention.py:176, started by build_suggestion_erasure main.py:923)"),
    ("src/core/adoption_schema.py", "", "controller", "schema-only adoption (core/schema_only.py:159)"),
    ("src/memory/copy_cleanup.py", "", "controller", "compaction retained-copy cleanup"),
    ("src/memory/owner_compaction.py", "", "controller", "compaction worker (compaction_worker.py:55)"),
    ("src/memory/fact_ledger.py", "<constant FACT_OWNER_SCHEMA>", "boot_init",
     "schema marker INSERT inside FACT_OWNER_SCHEMA, run by initialize_fact_owner_schema only when the tables are absent (first boot or upgrade)"),
    ("src/memory/fact_ledger.py", "FactLedger.initialize:930", "upgrade",
     "ALTER fact_sources ADD dependencies when H predates it (fact_ledger.py:927-937)"),
    ("src/memory/fact_ledger.py", "FactLedger.initialize:935", "upgrade",
     "same migration: every automatic fact -> quarantined (revision+1) and queued for index deletion"),
    ("src/memory/fact_ledger.py", "FactLedger.store", "ingress", "fact extraction (conversation / ambient distiller)"),
    ("src/memory/fact_ledger.py", "FactLedger.confirm", "user", "fact confirmation"),
    ("src/memory/fact_ledger.py", "FactLedger.assign_owner", "user", "owner assignment"),
    ("src/memory/index_deletions.py", "IndexDeletionQueue.enqueue", "follows_caller",
     "inside every fact deletion/quarantine transaction (FactLedger._queue)"),
    ("src/memory/index_deletions.py", "IndexDeletionQueue.register_target", "boot_repair",
     "INSERT..SELECT of every deleted/quarantined fact without a row, at construction three times: index_deletions.recover (privacy_services.py:78 -> :94), register_policy_index_target from build_memory_services (services.py:118) and from build_erasure_services (index_erasure.py:152 -> :143), both through erasure_journal.register_target (:313 -> :327)"),
    ("src/memory/index_deletions.py", "IndexDeletionQueue.recover", "boot_repair",
     "privacy_services.py:78: state/claim reset only (deleted_fact_revisions hashes fact_id,MAX(revision): no projected change)"),
    ("src/memory/index_deletions.py", "", "recurring",
     "index maintenance claim/record/finish (memory/index_erasure.py:156-157): state columns only"),
    ("src/memory/voice_schema.py", "initialize_voice_people_schema", "boot_init", "schema marker on first boot"),
    ("src/memory/voice_schema.py", "initialize_voice_review_schema", "boot_init", "schema marker on first boot"),
    ("src/memory/voice_schema.py", "<constant VOICE_ATTRIBUTION_SCHEMA>", "boot_init",
     "schema marker INSERT inside VOICE_ATTRIBUTION_SCHEMA, run by initialize_voice_attribution_schema only when the objects are absent"),
    ("src/training/collector_erasure.py", "CollectorEraser._page", "recurring",
     "collector eraser (build_collector_erasure from build_wake_clip_store, clip_store.py:461-462): erasure step for a pending conversation erasure; also deletes expired or unavailable clips met on that page"),
    ("src/training/conversation_collector.py", "", "ingress", "conversation collection (voice turns)"),
    ("src/training/stt_correction_collector.py", "STTCorrectionCollector.cleanup_old", "user",
     "no caller in src at 0bd9f98"),
    ("src/training/stt_correction_collector.py", "", "ingress", "STT correction collection (voice)"),
    ("src/users/voice_groups.py", "VoiceGroups.initialize", "boot_init",
     "first open of a ledger without voice_group_schema (first boot, or upgrade: backfill of unknown, unprojected groups) + marker"),
    ("src/users/voice_groups.py", "VoiceGroups._seal", "ingress", "voice clustering"),
    ("src/users/voice_groups.py", "VoiceGroups._create", "ingress", "voice clustering (and the upgrade backfill)"),
    ("src/users/voice_groups.py", "VoiceGroups.add", "ingress", "voice clustering"),
    ("src/users/voice_groups.py", "VoiceGroups.correct", "user", "manual group correction"),
    ("src/users/voice_ledger.py", "VoiceLedger.put", "ingress", "voice fingerprint capture"),
    ("src/users/voice_ledger.py", "VoiceLedger.prune", "recurring",
     "VoiceLedger.prune: boot prune (voice_registry.py:400) and maintain loop (:401 -> :351-356)"),
    ("src/wakeword/clip_store.py", "WakeClipStore._sweep_native", "recurring",
     "wake clip store maintenance: build_wake_clip_store (main.py:925) -> store.start() (clip_store.py:463, :390-406) -> reconcile (:374-379) -> _sweep_native (:334): deletes unpublished, expired, stale-parent and source-unavailable clips during construction and then at each cycle deadline (min of now+60 s and the earliest clip deadline, :341/:368)"),
    ("src/wakeword/clip_store.py", "", "ingress", "wake clip writer reserve/publish/discard/rotate (wake_clip_writer.py:221, collectors)"),
    ("src/dashboard/live_receipts.py", "LiveReceiptStore._recover:109", "upgrade",
     "ALTER live_deliveries ADD owner_namespace when H predates it"),
    ("src/memory/capture_recorders.py", "CaptureRecorders._initialize:137", "upgrade",
     "ALTER capture_browser_recorders when H predates the kind column"),
    ("src/reminders/reminder_store.py", "reminder_schema", "upgrade", "ALTER reminders ADD scope when H predates it"),
]

# Projection writers the SQL scan cannot see (R5). Same row shape as the scan.
UNSCANNED: list[dict] = [
    {"file": "src/core/state_configuration.py", "line": 599, "function": "StateConfiguration._after_graph_close",
     "call": "file_write", "verb": "REPLACE FILE", "table": "config/settings.yaml", "projection": "configuration",
     "dynamic": False, "class": "close_path",
     "reached_by": "exit stack (state_configuration.py:145 push_async_exit) -> _commit (:455-467, os.replace :511): rewrites settings.yaml with reasoner.cloud.consent=false only when a consent withdrawal is pending (user-triggered; V has none, so not reached without traffic)"},
    {"file": "src/users/user_manager.py", "line": 246, "function": "UserManager._save",
     "call": "file_write", "verb": "REPLACE FILE", "table": "*_accounts_v1.json", "projection": "accounts",
     "dynamic": False, "class": "user",
     "reached_by": "UserManager._mutate (:233-243) from account edits only (os.replace :312); never at load"},
    {"file": "src/dashboard/live_receipts.py", "line": 290, "function": "LiveReceiptStore (live_deliveries INSERT)",
     "call": "implicit", "verb": "AUTOINCREMENT", "table": "sqlite_sequence", "projection": "partial",
     "dynamic": False, "class": "ingress",
     "reached_by": "sqlite_sequence name='live_deliveries' advances with each live delivery (dashboard websocket); no SQL statement names it"},
]


def rule_for(row: dict) -> tuple[str, str]:
    """The most specific matching rule: a line pin first, then the longest function prefix."""
    best, rank = ("unclassified", ""), None
    for file, prefix, kind, how in RULES:
        if row["file"] != file:
            continue
        name, _, line = prefix.partition(":")
        if name and not row["function"].startswith(name):
            continue
        if line and int(line) != row["line"]:
            continue
        score = (bool(line), len(name))
        if rank is None or score > rank:
            best, rank = (kind, how), score
    return best


def observations(logs: Path) -> dict:
    """(file, line, table) of the innermost src frame of every traced projected write."""
    seen: dict = {}
    for result in sorted(logs.glob("*/result.json")):
        data = json.loads(result.read_text())
        for write in data.get("projected_writes", data.get("protected_writes", [])):
            site = write["stack"][0] if write["stack"] else None
            if site is None:
                # aiosqlite runs the statement on its own thread: use the submitter's frame.
                callers = [c for c in (write.get("caller") or []) if c.get("frames")]
                site = callers[-1]["frames"][0] if callers else None
            if site is None:
                continue
            file, line, function = site.split(":", 2)
            entry = seen.setdefault((file, int(line), write["table"]),
                                    {"function": function, "states": set(), "phases": set()})
            entry["states"].add(data["state"])
            entry["phases"].add(write["phase"])
    return seen


def classify(logs: Path) -> list[dict]:
    seen = observations(logs)
    rows = []
    static = scan(REPO)
    # A DDL constant runs from the line of its executescript call, not its own
    # line: join an observation to the constant row of the same file and table.
    constants = {(row["file"], row["table"]): (row["file"], row["line"])
                 for row in static if row["call"] == "ddl_constant"}
    joined: dict = {}
    for (file, line, table), value in seen.items():
        if not any(row["file"] == file and row["line"] == line for row in static) \
                and (file, table) in constants:
            joined.setdefault(constants[(file, table)], []).append({**value, "via": f"{file}:{line}"})
        else:
            joined.setdefault((file, line), []).append(value)
    for row in static:
        kind, how = rule_for(row)
        hits = joined.get((row["file"], row["line"]), [])
        vias = sorted({hit["via"] for hit in hits if "via" in hit})
        rows.append({**row, "class": kind,
                     "reached_by": how + (f" (observed executing from {', '.join(vias)})" if vias else ""),
                     "observed_phases": sorted({p for hit in hits for p in hit["phases"]}),
                     "observed_states": sorted({s for hit in hits for s in hit["states"]})})
    known = {(row["file"], row["line"]) for row in rows}
    for (file, line), values in sorted(joined.items()):
        if (file, line) not in known:
            # A runtime writer the static scan missed: listed, never dropped.
            rows.append({"file": file, "line": line, "function": values[0]["function"], "call": "?",
                         "verb": "?", "table": "?", "projection": "?", "dynamic": True,
                         "class": "unclassified", "reached_by": "observed at runtime, not in the static scan",
                         "observed_phases": sorted({p for v in values for p in v["phases"]}),
                         "observed_states": sorted({s for v in values for s in v["states"]})})
    for row in UNSCANNED:
        rows.append({**row, "observed_phases": [], "observed_states": []})
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
        print(f"{row['file']}:{row['line']}\t{row['verb']} {row['table']}\t{row['projection']}\t{row['class']}\t"
              f"{','.join(row['observed_phases']) or '-'}")


if __name__ == "__main__":
    main()
