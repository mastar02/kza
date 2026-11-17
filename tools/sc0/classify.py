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
from dataclasses import dataclass
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.sc0.write_inventory import scan  # noqa: E402


@dataclass(frozen=True)
class Rule:
    """The class of every writer of one function with one of ``writes`` (SCR-P-1: no line pins).

    Attributes:
        file: Repo-relative source file.
        function: The writer's exact enclosing name as the scan reports it
            (``Class.method``, nested ``Class.method.inner``, ``<constant NAME>``).
        writes: ``"VERB table"`` keys this rule classifies (``"DELETE FROM sessions"``).
        kind: The reachability class (see the module docstring).
        how: How it is reached; the call sites cited are an explanation, never a match key.
        sql: A fragment of the statement text, only where one function writes the same
            key in two classes (DashboardAuthority._transaction); the fragment rule wins.
    """

    file: str
    function: str
    writes: tuple[str, ...]
    kind: str
    how: str
    sql: str = ""


# Every static writer of a projected table, keyed by (file, function, verb and table). A new writer,
# or one moved to another function, is 'unclassified' until a rule names it: there is no catch-all
# (tests/unit/tools/test_sc0_classify.py). Classes as reviewed at SC-0 (0bd9f98 / eec17927); the
# SC-R follow-up re-keyed the former line pins and classified ReminderStore.purge_notice_rows.
RULES: list[Rule] = [
    # src/ambient/control.py
    Rule("src/ambient/control.py", "AmbientControl._load",
         ("INSERT OR IGNORE INTO ambient_runtime_control",), "boot_init",
         "AmbientControl._initialize task spawned at build_ambient_control (main.py:1991); INSERT OR IGNORE of "
         "the default row"),
    Rule("src/ambient/control.py", "AmbientControl._write", ("UPDATE ambient_runtime_control",), "user",
         "ambient pause/resume (dashboard)"),
    Rule("src/ambient/control.py", "AmbientControl._restore_paused", ("UPDATE ambient_runtime_control",), "user",
         "failure path of a user resume"),
    # src/ambient/store.py
    Rule("src/ambient/store.py", "<constant _MIGRATIONS>", ("ALTER TABLE utterances",), "upgrade",
         "utterances column migrations run by AmbientStore._migrate (store.py:224-227) when H lacks the column"),
    Rule("src/ambient/store.py", "AmbientStore._migrate", ("ALTER TABLE pending_audio_deletions",), "upgrade",
         "ALTER pending_audio_deletions ADD conversation_id when H predates it"),
    Rule("src/ambient/store.py", "AmbientStore.add.insert", ("INSERT INTO utterances",), "ingress",
         "ambient transcription of captured audio"),
    Rule("src/ambient/store.py", "AmbientStore.reserve_audio_path.reserve",
         ("INSERT OR IGNORE INTO audio_write_intents",), "ingress",
         "ambient archiving of captured audio"),
    Rule("src/ambient/store.py", "AmbientStore.set_audio_path.attach",
         ("UPDATE utterances", "DELETE FROM audio_write_intents",), "ingress",
         "ambient archiving of captured audio"),
    Rule("src/ambient/store.py", "AmbientStore.mark_distilled.mark", ("UPDATE utterances",), "recurring",
         "ambient distiller loop (distiller.py:357/:394, ambient_path.start): writes only when H holds "
         "undistilled utterances; distilled!=0 rows are projected"),
    Rule("src/ambient/store.py", "AmbientStore.purge_expired.expire",
         ("DELETE FROM utterances", "DELETE FROM audio_write_intents",), "recurring",
         "capture retention loop (build_capture_erasure -> retention.start, ambient/erasure.py:127; "
         "retention.py:239) and transcriber purge after ambient start (transcriber.py:641)"),
    Rule("src/ambient/store.py", "AmbientStore.erase_conversation.erase",
         ("DELETE FROM audio_write_intents", "DELETE FROM utterances",), "recurring",
         "erasure driver step CaptureEraser.clean for a pending conversation erasure"),
    Rule("src/ambient/store.py", "AmbientStore._queue_audio_deletion",
         ("INSERT INTO pending_audio_deletions",), "recurring",
         "inside purge_expired / erase_conversation / discard_audio"),
    Rule("src/ambient/store.py", "AmbientStore.discard_audio.discard",
         ("DELETE FROM audio_write_intents",), "ingress",
         "ambient archiving of captured audio"),
    Rule("src/ambient/store.py", "AmbientStore._retry_audio_deletions.attempt.acknowledge",
         ("DELETE FROM pending_audio_deletions", "UPDATE pending_audio_deletions",), "recurring",
         "retention loop at construction and again after READY (observed: queued_audio_deletion)"),
    # src/analytics/erasure.py
    Rule("src/analytics/erasure.py", "EventEraser._clean", ("DELETE FROM events_policy_v1",), "recurring",
         "erasure driver step for events (pending conversation erasure)"),
    # src/analytics/event_logger.py
    Rule("src/analytics/event_logger.py", "EventLogger._init_db", ("UPDATE events_policy_v1",), "boot_config",
         "every boot: EventLogger.initialize (analytics/services.py, before SuggestionStore at main.py:1058) "
         "clamps expires_at to analytics.retention_days; expires_at is not in event_text_deadline_fences (id, "
         "sources_json, text_expires_at), so no projected column changes"),
    Rule("src/analytics/event_logger.py", "EventLogger.log", ("INSERT INTO events_policy_v1",), "ingress",
         "event logging of house/voice actions"),
    Rule("src/analytics/event_logger.py", "EventLogger.cleanup_old_events",
         ("DELETE FROM events_policy_v1", "UPDATE events_policy_v1",), "recurring",
         "event logger maintenance loop (analytics/services.py:34 -> event_logger.py:778)"),
    # src/analytics/event_storage.py
    Rule("src/analytics/event_storage.py", "sweep_events",
         ("DELETE FROM events_policy_v1", "UPDATE events_policy_v1",), "recurring",
         "event retention (analytics/erasure.py:32 EventRetention -> analytics/retention.py:92)"),
    # src/analytics/suggestion_storage.py
    Rule("src/analytics/suggestion_storage.py", "clamp_suggestion_deadlines",
         ("UPDATE suggestions_policy_v1",), "boot_config",
         "every boot: SuggestionStore.__init__ (suggestion_store.py:106) from construct_suggestions "
         "(main.py:1058) clamps expires_at to memory.automatic_retention_days; non-pending/snoozed rows are "
         "hashed with every column and fences use expires_at"),
    Rule("src/analytics/suggestion_storage.py", "cleanup_suggestions",
         ("DELETE FROM suggestions_policy_v1",), "recurring",
         "suggestion eraser step (analytics/suggestion_erasure.py:177)"),
    # src/analytics/suggestion_store.py
    Rule("src/analytics/suggestion_store.py", "SuggestionStore.create",
         ("INSERT INTO suggestions_policy_v1",), "ingress",
         "suggestion generation from observed patterns (suggestion_engine.py:138, request path)"),
    Rule("src/analytics/suggestion_store.py", "SuggestionStore.respond", ("UPDATE suggestions_policy_v1",), "user",
         "accept/reject/snooze"),
    # src/analytics/suggestion_sweep.py
    Rule("src/analytics/suggestion_sweep.py", "sweep_suggestions",
         ("DELETE FROM suggestions_policy_v1", "UPDATE suggestions_policy_v1",), "recurring",
         "suggestion retention (analytics/suggestion_retention.py:176, started by build_suggestion_erasure "
         "main.py:923)"),
    # src/core/adoption_schema.py
    Rule("src/core/adoption_schema.py", "requeue_residue", ("UPDATE policy_index_deletions",), "controller",
         "schema-only adoption (core/schema_only.py:159)"),
    # src/dashboard/authority.py
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction", ("DELETE FROM sessions",), "every_txn",
         "expired-session purge of every transaction", sql="expires_at<=?"),
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction", ("DELETE FROM credentials",), "every_txn",
         "expired-credential purge of every transaction", sql="expires_at<=?"),
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction", ("DELETE FROM sessions",), "every_txn",
         "initialize/rotate when the legacy token digest or epoch changed", sql="legacy_digest IS NOT NULL"),
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction",
         ("INSERT OR REPLACE INTO auth_meta",), "every_txn",
         "legacy_digest/legacy_epoch rewrite on initialize (boot) and rotate (close)", sql="VALUES (?,?)"),
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction",
         ("INSERT INTO credentials", "DELETE FROM sessions", "INSERT INTO sessions", "UPDATE credentials",
          "INSERT OR IGNORE INTO retired_users", "DELETE FROM credentials",), "user",
         "issue/login/revise/retire_user/revoke/logout/credential_command branches"),
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction", ("DELETE FROM credentials",), "every_txn",
         "credentials of retired users, every transaction", sql="IN (SELECT user_id FROM retired_users)"),
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction", ("DELETE FROM sessions",), "every_txn",
         "orphan sessions, every transaction", sql="NOT IN (SELECT id FROM credentials)"),
    Rule("src/dashboard/authority.py", "DashboardAuthority._transaction",
         ("INSERT OR REPLACE INTO auth_meta",), "every_txn",
         "clock_floor, every transaction: changes auth_meta on every boot and close (a credential command "
         "writes the same row again in its own transaction)", sql="'clock_floor'"),
    # src/dashboard/credential_commands.py
    Rule("src/dashboard/credential_commands.py", "NativeCredentialCommands._apply_sql",
         ("INSERT INTO credentials", "UPDATE credentials", "DELETE FROM sessions", "DELETE FROM credentials",), "user",
         "credential commands (operator CLI / dashboard)"),
    Rule("src/dashboard/credential_commands.py", "NativeCredentialCommands._completed_sql",
         ("UPDATE credential_attempts",), "user",
         "credential commands (operator CLI / dashboard)"),
    # src/dashboard/credential_journal.py
    Rule("src/dashboard/credential_journal.py", "migrate_schema",
         ("INSERT INTO auth_meta", "INSERT OR REPLACE INTO auth_meta",), "boot_init",
         "authority initialize (main.py:1982): first-boot metadata (:217) and schema_version '3' rewrite "
         "(:247, same value)"),
    Rule("src/dashboard/credential_journal.py", "insert_receipt", ("INSERT INTO credential_attempts",), "user",
         "credential command receipt"),
    # src/dashboard/live_receipts.py
    Rule("src/dashboard/live_receipts.py", "LiveReceiptStore._recover", ("ALTER TABLE live_deliveries",), "upgrade",
         "ALTER live_deliveries ADD owner_namespace when H predates it"),
    Rule("src/dashboard/live_receipts.py", "LiveReceiptStore._recover", ("DELETE FROM {?}",), "boot_init",
         "dynamic table: live_connections/live_resets only (not protected)"),
    Rule("src/dashboard/live_receipts.py", "LiveReceiptStore._reset_ack",
         ("DELETE FROM live_deliveries",), "ingress",
         "live browser stream (dashboard websocket)"),
    Rule("src/dashboard/live_receipts.py", "LiveReceiptStore._issue",
         ("INSERT INTO live_deliveries", "INSERT OR IGNORE INTO live_delivery_sources",
          "INSERT INTO live_delivery_lineage",), "ingress",
         "live browser stream (dashboard websocket)"),
    Rule("src/dashboard/live_receipts.py", "LiveReceiptStore._ack", ("DELETE FROM live_deliveries",), "ingress",
         "live browser stream (dashboard websocket)"),
    # src/lists/list_store.py
    Rule("src/lists/list_store.py", "ListStore.create_list.operation", ("INSERT INTO lists",), "user",
         "list commands (voice/dashboard)"),
    Rule("src/lists/list_store.py", "ListStore.add_item.operation", ("INSERT INTO list_items",), "user",
         "list commands (voice/dashboard)"),
    Rule("src/lists/list_store.py", "ListStore.remove_item.operation", ("DELETE FROM list_items",), "user",
         "list commands (voice/dashboard)"),
    Rule("src/lists/list_store.py", "ListStore.complete_item.operation", ("UPDATE list_items",), "user",
         "list commands (voice/dashboard)"),
    Rule("src/lists/list_store.py", "ListStore.clear_list.operation", ("DELETE FROM list_items",), "user",
         "list commands (voice/dashboard)"),
    Rule("src/lists/list_store.py", "ListStore.delete_list.operation",
         ("DELETE FROM list_items", "DELETE FROM lists",), "user",
         "list commands (voice/dashboard)"),
    # src/memory/capture_control.py
    Rule("src/memory/capture_control.py", "CaptureControl._load_policy",
         ("INSERT INTO capture_runtime_control",), "boot_init",
         "capture_control._initialize task (build_capture_control, main.py:2004); INSERT only when the row is "
         "missing"),
    Rule("src/memory/capture_control.py", "CaptureControl._write_mode", ("UPDATE capture_runtime_control",), "user",
         "mute/unmute"),
    Rule("src/memory/capture_control.py", "CaptureControl._force_muted",
         ("UPDATE capture_runtime_control",), "user",
         "failed user transition restores mute"),
    # src/memory/capture_recorders.py
    Rule("src/memory/capture_recorders.py", "CaptureRecorders._initialize",
         ("ALTER TABLE capture_browser_recorders",), "upgrade",
         "ALTER capture_browser_recorders when H predates the kind column"),
    Rule("src/memory/capture_recorders.py", "CaptureRecorders._initialize",
         ("UPDATE capture_browser_recorders",), "boot_repair",
         "active->unknown, recorders.initialize from capture_control._initialize"),
    Rule("src/memory/capture_recorders.py", "CaptureRecorders._register",
         ("INSERT INTO capture_browser_recorders",), "ingress",
         "browser recorder register/activate/retire/forget"),
    Rule("src/memory/capture_recorders.py", "CaptureRecorders._activate",
         ("UPDATE capture_browser_recorders",), "ingress",
         "browser recorder register/activate/retire/forget"),
    Rule("src/memory/capture_recorders.py", "CaptureRecorders._retire",
         ("DELETE FROM capture_browser_recorders",), "ingress",
         "browser recorder register/activate/retire/forget"),
    Rule("src/memory/capture_recorders.py", "CaptureRecorders._forget_transaction",
         ("DELETE FROM capture_browser_recorders", "INSERT INTO capture_browser_recorders",), "ingress",
         "browser recorder register/activate/retire/forget"),
    Rule("src/memory/capture_recorders.py", "CaptureRecorders._retire_reserved",
         ("DELETE FROM capture_browser_recorders",), "close_path",
         "recorders.drain (capture_recorders.py:517): at close via _CaptureClosure.pre_close "
         "(runtime_closure.py:510) and, with the mode muted, already at boot via capture_control._restore -> "
         "_drain_recorders (capture_control.py:253-256, :386)"),
    # src/memory/copy_cleanup.py
    Rule("src/memory/copy_cleanup.py", "_perform", ("DELETE FROM wake_clips_v1",), "controller",
         "compaction retained-copy cleanup"),
    # src/memory/deletion_ledger.py
    Rule("src/memory/deletion_ledger.py", "_write_once_salt",
         ("INSERT INTO deletion_ledger_salt_v1",), "follows_caller",
         "append-only ledger rows of the deletion that calls it (observed: boot and tick retired-owner prune)"),
    Rule("src/memory/deletion_ledger.py", "append_deletion", ("INSERT INTO deletion_ledger_v1",), "follows_caller",
         "append-only ledger rows of the deletion that calls it (observed: boot and tick retired-owner prune)"),
    # src/memory/erasure_journal.py
    Rule("src/memory/erasure_journal.py", "ErasureJournal.ensure_operation",
         ("INSERT INTO conversation_erasures",), "user",
         "delete_conversation"),
    Rule("src/memory/erasure_journal.py", "ErasureJournal.recover",
         ("INSERT INTO conversation_erasures",), "boot_repair",
         "privacy_services.initialize (privacy_services.py:77): tombstone backfill"),
    # src/memory/fact_ledger.py
    Rule("src/memory/fact_ledger.py", "<constant FACT_OWNER_SCHEMA>",
         ("INSERT INTO fact_owner_schema",), "boot_init",
         "schema marker INSERT inside FACT_OWNER_SCHEMA, run by initialize_fact_owner_schema only when the "
         "tables are absent (first boot or upgrade)"),
    Rule("src/memory/fact_ledger.py", "initialize_fact_owner_schema", ("DELETE FROM {?}",), "boot_repair",
         "every FactLedger.initialize (fact_ledger.py:942) from privacy_services.initialize "
         "(privacy_services.py:75)"),
    Rule("src/memory/fact_ledger.py", "FactLedger.initialize", ("ALTER TABLE fact_sources",), "upgrade",
         "ALTER fact_sources ADD dependencies when H predates it"),
    Rule("src/memory/fact_ledger.py", "FactLedger.initialize", ("UPDATE facts",), "upgrade",
         "same migration: every automatic fact -> quarantined (revision+1) and queued for index deletion"),
    Rule("src/memory/fact_ledger.py", "FactLedger.store",
         ("INSERT INTO facts", "INSERT INTO fact_sources", "INSERT INTO fact_dependencies",), "ingress",
         "fact extraction (conversation / ambient distiller)"),
    Rule("src/memory/fact_ledger.py", "FactLedger.confirm", ("UPDATE facts",), "user",
         "fact confirmation"),
    Rule("src/memory/fact_ledger.py", "FactLedger._delete",
         ("UPDATE facts", "DELETE FROM fact_sources", "DELETE FROM fact_owners",
          "DELETE FROM fact_owner_revisions",), "recurring",
         "via expire (privacy maintain loop), clear_user_facts <- prune_retired_owners (boot main.py:355 and "
         "tick privacy_services.py:164), and user deletes"),
    Rule("src/memory/fact_ledger.py", "FactLedger._write_owner",
         ("DELETE FROM fact_owners", "INSERT INTO fact_owners", "INSERT INTO fact_owner_revisions",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    Rule("src/memory/fact_ledger.py", "FactLedger.assign_owner", ("UPDATE facts",), "user",
         "owner assignment"),
    Rule("src/memory/fact_ledger.py", "FactLedger.set_voice_switch",
         ("INSERT INTO voice_attribution_control",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    Rule("src/memory/fact_ledger.py", "FactLedger._write_consent",
         ("INSERT INTO voice_attribution_consents",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    Rule("src/memory/fact_ledger.py", "FactLedger._attempt", ("INSERT INTO voice_verification_attempts",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    Rule("src/memory/fact_ledger.py", "FactLedger.reopen_voice_consent",
         ("DELETE FROM voice_verification_attempts",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    Rule("src/memory/fact_ledger.py", "FactLedger._count_not_mine",
         ("INSERT INTO voice_attribution_events", "INSERT INTO voice_attribution_suspensions",), "ingress",
         "voice attribution of ambient proposals"),
    Rule("src/memory/fact_ledger.py", "FactLedger.resume_voice_attribution",
         ("DELETE FROM voice_attribution_suspensions", "DELETE FROM voice_attribution_events",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    Rule("src/memory/fact_ledger.py", "FactLedger.prune_voice_attribution",
         ("DELETE FROM voice_verification_attempts", "DELETE FROM voice_attribution_events",), "recurring",
         "privacy maintain loop (privacy_services.py:190), first tick during construction"),
    Rule("src/memory/fact_ledger.py", "FactLedger._retire_voice_people",
         ("DELETE FROM voice_attribution_consents", "DELETE FROM voice_verification_attempts",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    Rule("src/memory/fact_ledger.py", "FactLedger.clear_user_facts",
         ("DELETE FROM voice_attribution_consents", "DELETE FROM voice_verification_attempts",), "boot_repair",
         "prune_retired_owners at boot (main.py:355) and on the maintenance tick when deferred or hourly; also "
         "user retirement"),
    Rule("src/memory/fact_ledger.py", "FactLedger.delete_conversation",
         ("INSERT OR IGNORE INTO conversation_tombstones",), "user",
         "owner assignment, voice consent, delete_conversation, switch"),
    # src/memory/index_deletions.py
    Rule("src/memory/index_deletions.py", "IndexDeletionQueue.enqueue",
         ("INSERT INTO policy_index_deletions",), "follows_caller",
         "inside every fact deletion/quarantine transaction (FactLedger._queue)"),
    Rule("src/memory/index_deletions.py", "IndexDeletionQueue.register_target",
         ("INSERT INTO policy_index_deletions",), "boot_repair",
         "INSERT..SELECT of every deleted/quarantined fact without a row, at construction three times: "
         "index_deletions.recover (privacy_services.py:78 -> :94), register_policy_index_target from "
         "build_memory_services (services.py:118) and from build_erasure_services (index_erasure.py:152 -> "
         ":143), both through erasure_journal.register_target (:313 -> :327)"),
    Rule("src/memory/index_deletions.py", "IndexDeletionQueue.recover",
         ("UPDATE policy_index_deletions",), "boot_repair",
         "privacy_services.py:78: state/claim reset only (deleted_fact_revisions hashes fact_id,MAX(revision): "
         "no projected change)"),
    Rule("src/memory/index_deletions.py", "IndexDeletionQueue.claim",
         ("UPDATE policy_index_deletions",), "recurring",
         "index maintenance claim/record/finish (memory/index_erasure.py:156-157): state columns only"),
    Rule("src/memory/index_deletions.py", "IndexDeletionQueue.record_completed",
         ("UPDATE policy_index_deletions",), "recurring",
         "index maintenance claim/record/finish (memory/index_erasure.py:156-157): state columns only"),
    Rule("src/memory/index_deletions.py", "IndexDeletionQueue.finish",
         ("UPDATE policy_index_deletions",), "recurring",
         "index maintenance claim/record/finish (memory/index_erasure.py:156-157): state columns only"),
    # src/memory/legacy_quarantine.py
    Rule("src/memory/legacy_quarantine.py", "_import",
         ("INSERT INTO legacy_quarantine_batches_v1", "INSERT INTO legacy_quarantine_records_v1",), "controller",
         "stage_legacy_memory.py operator script"),
    # src/memory/owner_compaction.py
    Rule("src/memory/owner_compaction.py", "_relocate_columns",
         ("UPDATE suggestions_policy_v1", "UPDATE utterances", "UPDATE {?}",), "controller",
         "compaction worker (compaction_worker.py:55)"),
    # src/memory/payload_derivation.py
    Rule("src/memory/payload_derivation.py", "_replace_payload_rows",
         ("DELETE FROM {?}", "INSERT INTO {?}", "UPDATE {?}",), "controller",
         "compaction worker (K materialization)"),
    # src/memory/state_journal_derivation.py
    Rule("src/memory/state_journal_derivation.py", "_replace_current",
         ("DELETE FROM {?}", "UPDATE wake_clips_v1",), "controller",
         "compaction worker (K materialization)"),
    # src/memory/voice_schema.py
    Rule("src/memory/voice_schema.py", "initialize_voice_people_schema",
         ("INSERT INTO voice_person_schema",), "boot_init",
         "schema marker on first boot"),
    Rule("src/memory/voice_schema.py", "initialize_voice_review_schema",
         ("INSERT INTO voice_capture_review_schema",), "boot_init",
         "schema marker on first boot"),
    Rule("src/memory/voice_schema.py", "<constant VOICE_ATTRIBUTION_SCHEMA>",
         ("INSERT INTO voice_attribution_schema",), "boot_init",
         "schema marker INSERT inside VOICE_ATTRIBUTION_SCHEMA, run by initialize_voice_attribution_schema "
         "only when the objects are absent"),
    # src/obs/diagnostic_store.py
    Rule("src/obs/diagnostic_store.py", "DiagnosticStore._import",
         ("INSERT INTO quarantines", "UPDATE quarantines",), "boot_init",
         "first import of the JSONL catalog"),
    Rule("src/obs/diagnostic_store.py", "DiagnosticStore._initialize", ("UPDATE quarantines",), "boot_repair",
         "quarantine moved=0 -> 1 with rename; first metrics_emitter retention (metrics_emitter.py:222 -> "
         "diagnostic_store.py:427)"),
    Rule("src/obs/diagnostic_store.py", "DiagnosticStore._cleanup_staged", ("DELETE FROM staged",), "boot_repair",
         "staged row delete + unlink at _initialize (:264) and on every _expire (:280, retention)"),
    Rule("src/obs/diagnostic_store.py", "DiagnosticStore._publish",
         ("INSERT INTO staged", "DELETE FROM staged",), "recurring",
         "catalog prune/publish by the metrics retention (staged insert+delete in one pass)"),
    # src/reminders/reminder_store.py
    Rule("src/reminders/reminder_store.py", "reminder_schema", ("ALTER TABLE reminders",), "upgrade",
         "ALTER reminders ADD scope when H predates it"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.initialize.schema",
         ("UPDATE reminder_home_actions",), "boot_repair",
         "sending->uncertain at reminder_store.initialize (main.py:1082)"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.claim_home_action.operation",
         ("INSERT OR IGNORE INTO reminder_home_actions",), "ingress",
         "reminder firing (house effect)"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.finish_home_action.operation",
         ("UPDATE reminder_home_actions",), "ingress",
         "reminder firing (house effect)"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.purge_notice_rows.operation",
         ("DELETE FROM reminder_home_actions",), "recurring",
         "notice retention loop: ReminderScheduler.start -> _cleanup_notices every NOTICE_CLEANUP_INTERVAL_S "
         "(48 h retention, T034-U2), started by start_reminders (application.py _start_services; skipped by "
         "Application._run_verify); also a web notice dismissal (user)"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.create.operation", ("INSERT INTO reminders",), "user",
         "reminder create/cancel/update"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.mark_fired.operation", ("UPDATE reminders",), "ingress",
         "reminder scheduler (start_reminders)"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.cancel.operation", ("UPDATE reminders",), "user",
         "reminder create/cancel/update"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.update_trigger.operation", ("UPDATE reminders",), "user",
         "reminder create/cancel/update"),
    Rule("src/reminders/reminder_store.py", "ReminderStore.cleanup_old.operation",
         ("DELETE FROM reminders",), "user",
         "no caller in src at 0bd9f98"),
    # src/routines/routine_store.py
    Rule("src/routines/routine_store.py", "RoutineStore.save.operation", ("INSERT INTO routines",), "user",
         "routine save"),
    Rule("src/routines/routine_store.py", "RoutineStore.delete.operation", ("DELETE FROM routines",), "user",
         "routine delete"),
    Rule("src/routines/routine_store.py", "RoutineStore.claim_run.operation",
         ("INSERT INTO routine_runs", "INSERT INTO routine_run_actions",), "ingress",
         "routine runs (scheduler loop or manual run: house effect)"),
    Rule("src/routines/routine_store.py", "RoutineStore.mark_action_sending.operation",
         ("UPDATE routine_run_actions",), "ingress",
         "routine runs (scheduler loop or manual run: house effect)"),
    Rule("src/routines/routine_store.py", "RoutineStore.record_action.operation",
         ("UPDATE routine_run_actions",), "ingress",
         "routine runs (scheduler loop or manual run: house effect)"),
    Rule("src/routines/routine_store.py", "RoutineStore.finish_run.operation",
         ("UPDATE routine_run_actions", "UPDATE routine_runs",), "ingress",
         "routine runs (scheduler loop or manual run: house effect)"),
    Rule("src/routines/routine_store.py", "RoutineStore.recover_interrupted.operation",
         ("UPDATE routine_run_actions", "UPDATE routine_runs",), "boot_repair",
         "RoutineScheduler.start (routine_scheduler.py:227) from Application._start_services "
         "(application.py:318), before READY"),
    # src/training/collector_erasure.py
    Rule("src/training/collector_erasure.py", "CollectorEraser._page", ("DELETE FROM wake_clips_v1",), "recurring",
         "collector eraser (build_collector_erasure from build_wake_clip_store, clip_store.py:461-462): "
         "erasure step for a pending conversation erasure; also deletes expired or unavailable clips met on "
         "that page"),
    # src/training/conversation_collector.py
    Rule("src/training/conversation_collector.py", "ConversationCollector._commit",
         ("UPDATE wake_clips_v1",), "ingress",
         "conversation collection (voice turns)"),
    Rule("src/training/conversation_collector.py", "ConversationCollector._run_job",
         ("INSERT INTO conversation_turns_v1",), "ingress",
         "conversation collection (voice turns)"),
    # src/training/stt_correction_collector.py
    Rule("src/training/stt_correction_collector.py", "STTCorrectionCollector._save",
         ("INSERT INTO stt_corrections_v1",), "ingress",
         "STT correction collection (voice)"),
    Rule("src/training/stt_correction_collector.py", "STTCorrectionCollector._mark",
         ("UPDATE stt_corrections_v1",), "ingress",
         "STT correction collection (voice)"),
    Rule("src/training/stt_correction_collector.py", "STTCorrectionCollector.cleanup_old.shorten",
         ("UPDATE wake_clips_v1",), "user",
         "no caller in src at 0bd9f98"),
    # src/users/voice_groups.py
    Rule("src/users/voice_groups.py", "VoiceGroups._seal",
         ("INSERT OR REPLACE INTO voice_group_lineage",), "ingress",
         "voice clustering"),
    Rule("src/users/voice_groups.py", "VoiceGroups._create",
         ("INSERT INTO voice_groups", "INSERT INTO voice_group_members",), "ingress",
         "voice clustering (and the upgrade backfill)"),
    Rule("src/users/voice_groups.py", "VoiceGroups.initialize", ("INSERT INTO voice_group_schema",), "boot_init",
         "first open of a ledger without voice_group_schema (first boot, or upgrade: backfill of unknown, "
         "unprojected groups) + marker"),
    Rule("src/users/voice_groups.py", "VoiceGroups.add",
         ("INSERT INTO voice_group_members", "UPDATE voice_groups",), "ingress",
         "voice clustering"),
    Rule("src/users/voice_groups.py", "VoiceGroups.correct",
         ("DELETE FROM voice_group_members", "DELETE FROM voice_groups", "UPDATE voice_groups",
          "INSERT INTO voice_group_members", "INSERT INTO voice_group_requests",
          "INSERT INTO voice_group_request_lineage",), "user",
         "manual group correction"),
    Rule("src/users/voice_groups.py", "VoiceGroups.prune",
         ("DELETE FROM voice_groups", "DELETE FROM voice_group_requests",), "recurring",
         "VoiceLedger.prune (voice_ledger.py:561): boot prune in build_voice_registry (voice_registry.py:400) "
         "and VoiceRegistry.start maintain every 1 s (voice_registry.py:351-356)"),
    # src/users/voice_ledger.py
    Rule("src/users/voice_ledger.py", "VoiceLedger.put", ("INSERT INTO voice_fingerprints",), "ingress",
         "voice fingerprint capture"),
    Rule("src/users/voice_ledger.py", "VoiceLedger.confirm",
         ("INSERT INTO voice_profiles", "INSERT INTO voice_profile_sources",
          "INSERT INTO voice_review_requests",), "user",
         "voice profile confirm"),
    Rule("src/users/voice_ledger.py", "VoiceLedger._delete",
         ("UPDATE voice_profiles", "UPDATE voice_fingerprints", "UPDATE voice_review_requests",), "recurring",
         "user profile delete (voice_ledger.py:544) and VoiceLedger.prune (:568) for a profile whose source "
         "conversation was erased"),
    Rule("src/users/voice_ledger.py", "VoiceLedger.prune", ("DELETE FROM voice_fingerprints",), "recurring",
         "VoiceLedger.prune: boot prune (voice_registry.py:400) and maintain loop (:401 -> :351-356)"),
    # src/users/voice_people.py
    Rule("src/users/voice_people.py", "VoicePersonLinks.set_person", ("INSERT INTO voice_profile_people",), "user",
         "person links (dashboard) and user retirement"),
    Rule("src/users/voice_people.py", "VoicePersonLinks.clear_user_links", ("UPDATE voice_profile_people",), "user",
         "person links (dashboard) and user retirement"),
    Rule("src/users/voice_people.py", "VoicePersonLinks.unlink_profile", ("UPDATE voice_profile_people",), "user",
         "person links (dashboard) and user retirement"),
    # src/wakeword/clip_store.py
    Rule("src/wakeword/clip_store.py", "WakeClipStore.reserve", ("INSERT INTO wake_clips_v1",), "ingress",
         "wake clip writer reserve/publish/discard/rotate (wake_clip_writer.py:221, collectors)"),
    Rule("src/wakeword/clip_store.py", "WakeClipStore.publish", ("UPDATE wake_clips_v1",), "ingress",
         "wake clip writer reserve/publish/discard/rotate (wake_clip_writer.py:221, collectors)"),
    Rule("src/wakeword/clip_store.py", "WakeClipStore.discard", ("DELETE FROM wake_clips_v1",), "ingress",
         "wake clip writer reserve/publish/discard/rotate (wake_clip_writer.py:221, collectors)"),
    Rule("src/wakeword/clip_store.py", "WakeClipStore.rotate", ("DELETE FROM wake_clips_v1",), "ingress",
         "wake clip writer reserve/publish/discard/rotate (wake_clip_writer.py:221, collectors)"),
    Rule("src/wakeword/clip_store.py", "WakeClipStore._sweep_native", ("DELETE FROM wake_clips_v1",), "recurring",
         "wake clip store maintenance: build_wake_clip_store (main.py:925) -> store.start() "
         "(clip_store.py:463, :390-406) -> reconcile (:374-379) -> _sweep_native (:334): deletes unpublished, "
         "expired, stale-parent and source-unavailable clips during construction and then at each cycle "
         "deadline (min of now+60 s and the earliest clip deadline, :341/:368)"),
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


def matching_rules(row: dict) -> list[Rule]:
    """Every rule naming the row's file, exact function and 'VERB table' key (and its sql fragment)."""
    key = row["verb"] + " " + row["table"]
    return [rule for rule in RULES
            if rule.file == row["file"] and rule.function == row["function"] and key in rule.writes
            and rule.sql in row.get("sql", "")]


def rule_for(row: dict) -> tuple[str, str]:
    """The row's class: the one rule with an sql fragment, else the one keyed rule.

    No matching rule is 'unclassified' and two rules of equal precedence are
    'ambiguous': both fail tests/unit/tools/test_sc0_classify.py.
    """
    found = matching_rules(row)
    specific = [rule for rule in found if rule.sql]
    candidates = specific or found
    if not candidates:
        return "unclassified", ""
    if len(candidates) > 1:
        return "ambiguous", "; ".join(rule.how for rule in candidates)
    return candidates[0].kind, candidates[0].how


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
