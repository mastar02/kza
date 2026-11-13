"""R1's side of the adoption (U3b-0e2): schema-only migration, ledger seed, residue, targets.

Everything here runs only inside the adoption's schema-only child
(:mod:`src.core.schema_only`, '-I -B', no ticket, no SDK, no network) on an
adoption copy that :func:`src.core.schema_only.state_decision` accepted. It
never opens Chroma and never reads secrets. Each phase prints a content-free
JSON summary; ids stay in memory.

Phases (adoption design v3 sections 3 and 4):

* ``schema`` (S1): the R1 open-time DDL of every carried store whose b0866e8
  schema differs from R1's contract: the privacy ledger (FactLedger.initialize
  plus the live-receipt owner migration) and the reminders store. No other
  store changes between b0866e8 and R1 (the build-time diff test pins this).
* ``seed``: :func:`src.memory.deletion_ledger.seed_adoption_baseline` on the
  empty, saltless ledger of each authority, from the deletion marks the copy
  already holds.
* ``residue`` (P3=A): re-queue what the per-store resurrection check counts:
  facts through policy_index_deletions, conversations by reopening the
  governing step (an operation is created through the journal API when the
  tombstone has none; conversation_erasures.conversation_id is UNIQUE, 0d4a
  review F8). A voice residue, an unmarkable conversation occurrence (no
  governing target) or any residue left afterwards refuses.
* ``targets``: the erasure-target set R1 registers at its first start for the
  copy's configuration, with descriptors resolved under the final state root.
"""
from __future__ import annotations

from contextlib import closing
import json
import logging
import os
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

RESIDUE_REASON = "interrupted_by_restart"  # ErasureError.RESTARTED: restored data may exist behind a receipt
DEFAULT_PRIVACY = "./data/memory_privacy.db"
DEFAULT_REMINDERS = "./data/reminders.db"
DEFAULT_AUTHORITY = "./data/dashboard_auth.db"


class AdoptionSchemaError(RuntimeError):
    """A content-free refusal of one adoption phase."""


def settings(state: Path) -> dict:
    """The adoption copy's configuration (already transformed by the controller)."""
    import yaml

    path = Path(state) / "config" / "settings.yaml"
    if path.is_symlink() or not path.is_file():
        raise AdoptionSchemaError("adoption_schema_settings_missing")
    value = yaml.safe_load(path.read_text())
    if type(value) is not dict:
        raise AdoptionSchemaError("adoption_schema_settings_invalid")
    return value


def state_path(state: Path, value, default: str) -> Path:
    """A configured store path, resolved inside the copy; anything outside refuses."""
    raw = os.fspath(value if value not in (None, "") else default)
    path = Path(raw)
    if path.is_absolute():
        raise AdoptionSchemaError("adoption_schema_path_absolute")
    resolved = Path(os.path.normpath(Path(state) / path))
    if not resolved.is_relative_to(Path(state)) or resolved == Path(state):
        raise AdoptionSchemaError("adoption_schema_path_foreign")
    return resolved


def _privacy_path(state: Path, config: dict) -> Path:
    return state_path(state, (config.get("memory") or {}).get("privacy_db_path"), DEFAULT_PRIVACY)


def _authority_path(state: Path, config: dict) -> Path:
    auth = ((config.get("dashboard") or {}).get("auth") or {})
    return state_path(state, auth.get("db_path"), DEFAULT_AUTHORITY)


def initialize_privacy_schema(state: Path) -> None:
    """FactLedger.initialize (voice, owner, attribution and ledger groups) and the live-receipt migration."""
    from src.dashboard.live_receipts import LiveReceiptStore
    from src.memory.fact_ledger import FactLedger

    path = _privacy_path(state, settings(state))
    if not path.is_file():
        return
    ledger = FactLedger(str(path))
    ledger.initialize()
    try:
        with ledger._transaction() as db:
            LiveReceiptStore(ledger)._recover(db)
    finally:
        ledger.close()


def initialize_reminder_schema(state: Path) -> None:
    """The reminders DDL (adds reminders.scope) without the store's send recovery."""
    from src.reminders.reminder_store import reminder_schema

    path = state_path(state, (settings(state).get("reminders") or {}).get("db_path"), DEFAULT_REMINDERS)
    if not path.is_file():
        return
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            reminder_schema(db)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise


SCHEMA_INITIALIZERS = (initialize_privacy_schema, initialize_reminder_schema)


def seed_entries(db: sqlite3.Connection, adoption_time: float) -> list:
    """The (kind, subject, at) marks one authority already holds (design v3 section 4).

    conversation_tombstones (at = deleted_at), facts in state 'deleted' (at =
    the adoption time: no deletion time exists, review F18), voice_profiles
    with deleted_at (at = deleted_at) and retired_users (at = retired_at).
    """
    from src.memory.deletion_ledger import DeletionKind

    names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    entries = []
    if "conversation_tombstones" in names:
        entries += [(DeletionKind.CONVERSATION_ERASED, row[0], row[1]) for row in db.execute(
            "SELECT conversation_id, deleted_at FROM conversation_tombstones")]
    if "facts" in names:
        entries += [(DeletionKind.FACT_DELETED, row[0], adoption_time) for row in db.execute(
            "SELECT id FROM facts WHERE state='deleted'")]
    if "voice_profiles" in names:
        entries += [(DeletionKind.VOICE_PROFILE_DELETED, row[0], row[1]) for row in db.execute(
            "SELECT id, deleted_at FROM voice_profiles WHERE deleted_at IS NOT NULL")]
    if "retired_users" in names:
        entries += [(DeletionKind.USER_RETIRED, row[0], row[1]) for row in db.execute(
            "SELECT user_id, retired_at FROM retired_users")]
    return entries


def seed_ledgers(state: Path, adoption_time: float) -> dict:
    """Seed the privacy ledger and, when present, the dashboard authority's ledger."""
    from src.memory.deletion_ledger import LEDGER_TABLE, seed_adoption_baseline

    config = settings(state)
    counts: dict[str, dict[str, int]] = {}
    for path in (_privacy_path(state, config), _authority_path(state, config)):
        if not path.is_file():
            continue
        with closing(sqlite3.connect(path, isolation_level=None)) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if LEDGER_TABLE not in names:
                raise AdoptionSchemaError("adoption_seed_ledger_missing")
            db.execute("BEGIN IMMEDIATE")
            try:
                entries = seed_entries(db, adoption_time)
                seed_adoption_baseline(db, entries)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        per_kind: dict[str, int] = {}
        for kind, _subject, _at in entries:
            per_kind[kind.value] = per_kind.get(kind.value, 0) + 1
        counts[path.relative_to(state).as_posix()] = per_kind
    return counts


class _Actor:
    def __init__(self, subject: str):
        self.subject = subject


class _Clock:
    """The journal API's ledger surface: only the clock (ensure_operation, enqueue)."""

    def __init__(self, at: float):
        self._at = at

    def clock(self) -> float:
        return self._at


def _residue_scan(state: Path, state_root: Path) -> dict:
    from scripts import managed_launcher

    return managed_launcher.resurrection_residue(state, descriptor_root=state_root)


def requeue_residue(state: Path, state_root: Path, adoption_time: float, plan_sha256: str) -> dict:
    """P3=A on the copy: re-queue every counted residue or refuse (design v3 section 4).

    Runs after the seed (the ledger names what was erased) and after the
    relocation (descriptors name ``state_root``, 0d4a review F8).

    Returns:
        Content-free counts of what was re-queued.

    Raises:
        AdoptionSchemaError: ``runtime_admission_adoption_residue:<family>``.
    """
    from src.memory.erasure_journal import ErasureJournal
    from src.memory.index_deletions import IndexCleanupError, IndexDeletionQueue

    residue = _residue_scan(state, state_root)
    if residue["voice_profile"]:
        raise AdoptionSchemaError("runtime_admission_adoption_residue:voice_profile")
    if residue["voice_consent"]:
        raise AdoptionSchemaError("runtime_admission_adoption_residue:voice_consent")
    for occurrences in residue["conversation"].values():
        if any(not governing for _participant, governing in occurrences):
            raise AdoptionSchemaError("runtime_admission_adoption_residue:conversation")
    report = {"conversation_operations_created": 0, "conversation_steps_reopened": 0, "facts_requeued": 0}
    if not residue["conversation"] and not residue["fact"]:
        return report
    path = _privacy_path(state, settings(state))
    clock = _Clock(adoption_time)
    journal, queue = ErasureJournal(clock), IndexDeletionQueue(clock)
    actor = _Actor("adoption:" + plan_sha256)
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("BEGIN IMMEDIATE")
        try:
            for conversation in sorted(residue["conversation"]):
                row = db.execute("SELECT id FROM conversation_erasures WHERE conversation_id=?",
                                 (conversation,)).fetchone()
                if row is None:
                    if db.execute("SELECT 1 FROM conversation_tombstones WHERE conversation_id=?",
                                  (conversation,)).fetchone() is None:
                        raise AdoptionSchemaError("runtime_admission_adoption_residue:conversation")
                    journal.ensure_operation(db, conversation, actor)  # pending steps on every target
                    report["conversation_operations_created"] += 1
                    continue
                operation = row[0]
                governing = sorted(set().union(*(targets for _participant, targets in residue["conversation"][conversation])))
                for target in governing:
                    report["conversation_steps_reopened"] += db.execute(
                        "UPDATE erasure_steps SET state='pending', claim_token=NULL, cursor=NULL, next_attempt_at=0,"
                        " last_error=? WHERE operation_id=? AND target_id=? AND state!='pending'",
                        (RESIDUE_REASON, operation, target)).rowcount
            for fact in sorted(residue["fact"]):
                row = db.execute("SELECT revision FROM facts WHERE id=?", (fact,)).fetchone()
                if row is None:
                    raise AdoptionSchemaError("runtime_admission_adoption_residue:fact")
                queue.enqueue(db, fact, row[0])
                db.execute("UPDATE policy_index_deletions SET state='pending', claim_token=NULL, next_attempt_at=0,"
                           " last_error=? WHERE fact_id=? AND state!='pending'",
                           (IndexCleanupError.RECHECK.value, fact))
                report["facts_requeued"] += 1
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    after = _residue_scan(state, state_root)
    if after["conversation"] or after["fact"] or after["voice_profile"] or after["voice_consent"]:
        raise AdoptionSchemaError("runtime_admission_adoption_residue:remaining")
    return report


def first_start_targets(config: dict, state_root: Path, privacy_db: Path | None = None) -> list[dict]:
    """The erasure targets R1 registers at its first start (adoption design v2 2d post-condition).

    Mirrors the registrations src.main.build_application makes for
    ``config`` (an integration test compares the two), with every relative
    path resolved as the managed child resolves it: under ``state_root``
    (the runtime's config/data links point there). ``privacy_db`` supplies
    the data-registered collector targets: one quarantine guard per legacy
    quarantine batch. Returns sorted ``{participant, version, descriptor}``.
    """
    from src.analytics.event_storage import event_descriptor
    from src.analytics.suggestion_storage import RUNTIME_TARGET, suggestion_descriptor
    from src.core.paths import TRAINING_OUTPUT_LORA, TRAINING_OUTPUT_NIGHTLY, TRAINING_OUTPUT_WHISPER
    from src.dashboard.live_erasure import LIVE_TARGET, REST_TARGET, VOICE_REVIEW_TARGET
    from src.memory.producer_erasure import PRODUCER_TARGET
    from src.obs.diagnostic_erasure import diagnostic_descriptor
    from src.orchestrator.context_storage import snapshot_targets
    from src.users.voice_registry import VOICE_TARGET

    root = Path(state_root)

    def at(value) -> str:
        path = Path(os.fspath(value))
        return str((path if path.is_absolute() else root / path).resolve())

    memory = config.get("memory", {}) or {}
    ambient = config.get("ambient", {}) or {}
    analytics = config.get("analytics", {}) or {}
    training = config.get("training", {}) or {}
    nightly = training.get("nightly", {}) or {}
    clips = ((config.get("rooms", {}) or {}).get("wake_word", {}) or {}).get("capture_clips", {}) or {}
    contexts = (((config.get("orchestrator", {}) or {}).get("context", {}) or {}).get("persistence", {}) or {})
    targets = [
        ("semantic_index", 1, {"path": at(memory.get("chroma_path", "./data/memory_db")),
                               "collection": memory.get("long_term_collection", "user_memories") + "_policy_v1"}),
        ("voices", 1, dict(VOICE_TARGET)), ("voices", 2, {"kind": "unresolved_voice_inventory"}),
        ("diagnostics", 2, diagnostic_descriptor(at((config.get("observability", {}) or {}).get(
            "metrics_path", "/home/kza/logs/kza-metrics.jsonl")))),
        ("diagnostics", 1, {"kind": "unresolved_diagnostic_inventory"}),
        ("captures", 1, {"db_path": at(ambient.get("db_path", "./data/ambient.db")),
                         "audio_dir": at((ambient.get("keep_audio", {}) or {}).get("dir", "./data/ambient_audio"))}),
        ("events", 1, event_descriptor(at(analytics.get("events_db", "./data/events.db")))),
        ("suggestions", 1, suggestion_descriptor(at(analytics.get("suggestions_path", "./data/suggestions.json")))),
        ("suggestions", 1, dict(RUNTIME_TARGET)),
        *(("contexts", 1, descriptor) for descriptor in snapshot_targets(at(contexts.get("base_path", "data/contexts")))),
        ("contexts", 1, {"kind": "runtime", "name": "conversation_buffers"}),
        ("collectors", 1, {"kind": "wake_clips", "root": at(clips.get("dir", "data/wakeword_training/captured"))}),
        ("collectors", 1, {"kind": "stt_corrections", "root": at(
            (training.get("stt_corrections", {}) or {}).get("data_dir", "data/stt_corrections"))}),
        ("collectors", 1, {"kind": "conversations", "root": at(training.get("conversations_dir", "data/conversations"))}),
        ("collectors", 1, {"kind": "conversation_exports", "root": at(
            training.get("exports_dir", "data/conversation_exports"))}),
        *(("collectors", 1, {"kind": "legacy_training", "root": at(value)}) for value in (
            (training.get("habits", {}) or {}).get("data_dir", "data/habit_training"),
            nightly.get("data_dir", "data/nightly_training"),
            (training.get("lora", {}) or {}).get("output_dir", TRAINING_OUTPUT_LORA),
            nightly.get("output_dir", TRAINING_OUTPUT_NIGHTLY),
            (nightly.get("whisper_finetune", {}) or {}).get("output_dir", TRAINING_OUTPUT_WHISPER))),
        ("streams", 1, dict(LIVE_TARGET)), ("streams", 3, dict(REST_TARGET)),
        ("streams", 4, dict(VOICE_REVIEW_TARGET)), ("streams", 2, {"kind": "unresolved_stream_inventory"}),
        ("producers", 1, dict(PRODUCER_TARGET)),
    ]
    if privacy_db is not None and Path(privacy_db).is_file():
        with closing(sqlite3.connect(Path(privacy_db).as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='legacy_quarantine_batches_v1'").fetchone():
                targets += [("collectors", 1, {"kind": "quarantine_locator_coverage", "batch_id": row[0]})
                            for row in db.execute("SELECT id FROM legacy_quarantine_batches_v1")]
    unique = {json.dumps([participant, version, descriptor], sort_keys=True): {
        "participant": participant, "version": version, "descriptor": descriptor}
        for participant, version, descriptor in targets}
    return [unique[key] for key in sorted(unique)]
