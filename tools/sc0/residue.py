"""SC-0 (2) residue states: rows injected into a copy of the sealed template's data/.

Each injector gets the copied ``data`` directory and ``final`` (the absolute
graph root the state will have once the slot is prepared: every registered
path is absolute, so the staging path is fixed). Rows are written with the
real ledger APIs where a valid row needs them (facts, owners, erasure
operations) and with SQL where the residue is exactly a row an interrupted
or older process leaves. All content is synthetic. Each returns a small
content-free description of what it planted.

The design's list (sc-release-design-v4 SC-0 scope) plus states the
inventory and the review added: routine run interrupted, owned fact past its
expiry, deferred retired-owner prune, expired voice-group request, fenced
wake clips (R2), deleted fact without an index row (R3), shortened retention
settings (R4, with config overrides) and two 75 s settles (R6).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
from uuid import uuid4

from scripts.stage_runtime import _PROTECTED_TABLES as PROTECTED

PERSON = "user_" + "a" * 32  # retired owner
RETIRED_LOGIN = "user_" + "c" * 32  # retired user holding a credential and a session
DAY = 86400.0


def _graph(data: Path) -> Path:
    return data / "graph"


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    return db


def _ledger(data: Path):
    from src.memory.fact_ledger import FactLedger
    ledger = FactLedger(str(_graph(data) / "privacy.db"), clock=time.time)
    ledger.initialize()
    return ledger


def _source(*, expires_in: float = DAY):
    from src.memory.fact_ledger import MemoryOrigin, MemorySource
    now = time.time()
    return MemorySource(str(uuid4()), str(uuid4()), now - 1, now + expires_in, MemoryOrigin.VOICE, ())


def _owned_fact(ledger, person: str, content: str, *, expires_in: float = DAY):
    from src.memory.fact_ledger import MemoryAssigner
    fact = ledger.store(content, "fact", .9, (_source(expires_in=expires_in),))
    ledger.assign_owner(fact.id, person, fact.revision, 0, MemoryAssigner("synthetic-operator"),
                        retired=lambda ids: False, active=lambda person_id: True)
    return fact


def _floor(data: Path) -> float:
    """The authority's persisted clock floor: history rows must not be later (history_clock check)."""
    with _connect(_graph(data) / "dashboard-auth.db") as db:
        row = db.execute("SELECT value FROM auth_meta WHERE key='clock_floor'").fetchone()
    return float(row[0]) if row is not None else time.time()


def _retire(data: Path, user_id: str) -> None:
    from src.memory.deletion_ledger import DeletionKind, append_deletion
    with _connect(_graph(data) / "dashboard-auth.db") as db:
        now = min(time.time(), _floor(data)) - 60
        db.execute("INSERT OR IGNORE INTO retired_users VALUES (?,?)", (user_id, now))
        append_deletion(db, DeletionKind.USER_RETIRED, user_id, now)


def _token_hash() -> str:
    token = secrets.token_urlsafe(32)[:43]
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def clean(data, final):
    return {"planted": []}


def retired_owner_facts(data, final):
    """A durably retired account still owns facts (crash between retire and clear_user_facts)."""
    ledger = _ledger(data)
    try:
        facts = [_owned_fact(ledger, PERSON, f"Synthetic fact {n} of a retired owner").id for n in range(2)]
    finally:
        ledger.close()
    _retire(data, PERSON)
    return {"planted": ["retired_users+deletion_ledger(dashboard-auth)", "2 facts with fact_owners of that user"]}


def retired_owner_facts_deferred(data, final):
    """Same rows; the child answers the authority probe None until READY (prune deferred)."""
    result = retired_owner_facts(data, final)
    result["child_flags"] = ["--probe-none-until-ready"]
    return result


def tombstone_without_operation(data, final):
    """An older release's tombstone with no erasure operation (erasures.recover backfills)."""
    with _connect(_graph(data) / "privacy.db") as db:
        db.execute("INSERT INTO conversation_tombstones VALUES (?,?,?)",
                   (str(uuid4()), time.time() - 3600, "synthetic-reviewer"))
    return {"planted": ["conversation_tombstones row, no conversation_erasures"]}


def retired_user_credential_session(data, final):
    """A credential and a session of a retired user (purged by every authority transaction)."""
    _retire(data, RETIRED_LOGIN)
    now = min(time.time(), _floor(data)) - 1
    credential = uuid4().hex
    with _connect(_graph(data) / "dashboard-auth.db") as db:
        db.execute("INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,?)",
                   (credential, secrets.token_hex(32), "synthetic-home", "home", RETIRED_LOGIN,
                    now - 3600, now + 7 * DAY, 1, "[]"))
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                   (secrets.token_hex(32), credential, None, now - 600, now + DAY, 1))
    return {"planted": ["retired_users", "credentials(user retired)", "sessions(of it)"]}


def pending_erasure_step(data, final):
    """A conversation erasure requested through the ledger, its steps still pending."""
    from src.memory.fact_ledger import MemoryReviewer
    ledger = _ledger(data)
    try:
        ledger.delete_conversation(str(uuid4()), MemoryReviewer("synthetic-reviewer"))
    finally:
        ledger.close()
    with _connect(_graph(data) / "privacy.db") as db:
        steps = db.execute("SELECT COUNT(*) FROM erasure_steps WHERE state='pending'").fetchone()[0]
    return {"planted": ["conversation_tombstones+conversation_erasures", f"{steps} pending erasure_steps"]}


def queued_audio_deletion(data, final):
    """A durable audio deletion still queued (retention retries it)."""
    audio = _graph(data) / "audio"
    audio.mkdir(exist_ok=True)
    name = f"synthetic-{uuid4().hex}.wav"
    (audio / name).write_bytes(b"RIFF0000WAVE")
    with _connect(_graph(data) / "ambient.db") as db:
        db.execute("INSERT INTO pending_audio_deletions(path, requested_at, conversation_id) VALUES (?,?,NULL)",
                   (str(final / "audio" / name), time.time() - 600))
    return {"planted": ["pending_audio_deletions row + its file"]}


def expired_sessions_credentials(data, final):
    """An expired credential and an expired session (purged by every authority transaction)."""
    now = min(time.time(), _floor(data)) - 1
    credential = uuid4().hex
    with _connect(_graph(data) / "dashboard-auth.db") as db:
        db.execute("INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,?)",
                   (credential, secrets.token_hex(32), "synthetic-operator", "operator", None,
                    now - 3 * DAY, now - DAY, 1, "[]"))
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                   (secrets.token_hex(32), credential, None, now - 2 * DAY, now - DAY, 1))
    return {"planted": ["expired credentials row", "expired sessions row"]}


def _recorders(data, states: list[str], mode: str) -> dict:
    with _connect(_graph(data) / "privacy.db") as db:
        for state in states:
            db.execute("INSERT INTO capture_browser_recorders VALUES (?,?,'browser')", (_token_hash(), state))
        revision = db.execute("SELECT revision FROM capture_runtime_control WHERE id=1").fetchone()[0]
        db.execute("UPDATE capture_runtime_control SET mode=?, revision=? WHERE id=1",
                   (mode, revision + (mode != "running")))
    return {"planted": [f"capture_browser_recorders: {','.join(states)}", f"capture mode {mode}"]}


def recorders_clean_stop_running(data, final):
    """Clean-stop residue: an active and a reserved recorder, capture running."""
    return _recorders(data, ["active", "reserved"], "running")


def recorders_clean_stop_muted(data, final):
    """Clean-stop residue: an active and a reserved recorder, capture muted."""
    return _recorders(data, ["active", "reserved"], "muted")


def recorders_crash_running(data, final):
    """Crash residue: reserved, active and unknown (carried) recorders, capture running."""
    return _recorders(data, ["reserved", "active", "unknown"], "running")


def recorders_crash_muted(data, final):
    """Crash residue: reserved, active and unknown (carried) recorders, capture muted."""
    return _recorders(data, ["reserved", "active", "unknown"], "muted")


def recorder_orphan_unknown(data, final):
    """R5V-1 pre-existing issue: one orphaned unknown owner only (normal-mode receipt?)."""
    return _recorders(data, ["unknown"], "running")


def reminder_action_sending(data, final):
    """A reminder home action left in 'sending' (boot turns it uncertain)."""
    body = json.dumps({"domain": "light", "entity_id": "light.living", "service": "turn_on"},
                      sort_keys=True, separators=(",", ":"))
    with _connect(_graph(data) / "reminders.db") as db:
        db.execute("INSERT INTO reminder_home_actions (request_id, body, status) VALUES (?,?,'sending')",
                   (str(uuid4()), body))
    return {"planted": ["reminder_home_actions status=sending"]}


def diagnostic_quarantine_staged(data, final):
    """Diagnostic store crash residue: a quarantine intent with moved=0 and a staged file."""
    graph = _graph(data)
    source = "metrics_policy_v1.jsonl.1"
    if not (graph / source).exists():
        (graph / source).write_text("")
    staged = f".diag-{uuid4().hex}.tmp"
    (graph / staged).write_text("{}")
    with _connect(graph / "metrics_policy_v1.jsonl.sqlite3") as db:
        db.execute("INSERT INTO quarantines VALUES (?,?,0)", (source, f"{source}.quarantine-{uuid4()}"))
        db.execute("INSERT INTO staged VALUES (?)", (staged,))
    return {"planted": ["quarantines moved=0 (rotated file present)", "staged row + file"]}


def deleted_fact_owner_rows(data, final):
    """fact_owners and fact_owner_revisions rows whose fact is DELETED (pruned at schema open)."""
    ledger = _ledger(data)
    try:
        fact = _owned_fact(ledger, "user_" + "b" * 32, "Synthetic owned fact later deleted")
    finally:
        ledger.close()
    with _connect(_graph(data) / "privacy.db") as db:
        db.execute("UPDATE facts SET state='deleted', content=NULL WHERE id=?", (fact.id,))
    return {"planted": ["facts state=deleted with fact_owners+fact_owner_revisions rows"]}


def routine_run_interrupted(data, final):
    """A routine run left 'running' with a sending and a pending action (boot marks interrupted)."""
    run_id = uuid4().hex
    with _connect(_graph(data) / "routines.db") as db:
        db.execute("INSERT INTO routine_runs (run_id, routine_id, occurrence_key, actor, source, status, reason, "
                   "snapshot_json, scheduled_for) VALUES (?,?,?,?,?,?,?,?,?)",
                   (run_id, "synthetic-routine", "manual:" + uuid4().hex, "synthetic-operator", "manual",
                    "running", None, "{}", None))
        db.executemany("INSERT INTO routine_run_actions (run_id, action_index, entity_id, status) VALUES (?,?,?,?)",
                       [(run_id, 0, "light.living", "sending"), (run_id, 1, "light.kitchen", "pending")])
    return {"planted": ["routine_runs running", "routine_run_actions sending+pending"]}


def owned_fact_expired(data, final):
    """An owned automatic fact whose expiry passed (owner assignment normally stops expiry)."""
    ledger = _ledger(data)
    try:
        fact = _owned_fact(ledger, "user_" + "d" * 32, "Synthetic owned fact past expiry")
    finally:
        ledger.close()
    with _connect(_graph(data) / "privacy.db") as db:
        state = db.execute("SELECT state, expires_at FROM facts WHERE id=?", (fact.id,)).fetchone()
        db.execute("UPDATE facts SET expires_at=? WHERE id=?", (time.time() - 60, fact.id))
    return {"planted": [f"owned fact state={state['state']} expires_at set in the past"]}


def expired_voice_group_request(data, final):
    """A manual voice-group request past its expiry (the voice registry's prune deletes it)."""
    from src.memory.fact_ledger import MemoryOrigin, MemorySource
    now = time.time()
    source = MemorySource(str(uuid4()), str(uuid4()), now - DAY, now - 600, MemoryOrigin.VOICE, ())
    with _connect(_graph(data) / "privacy.db") as db:
        db.execute("INSERT INTO voice_group_requests VALUES (?,?,?,?,?)",
                   ("synthetic-operator", str(uuid4()), secrets.token_hex(32), now - 600,
                    json.dumps([source.to_dict()])))
    return {"planted": ["voice_group_requests row past expiry"]}


def _wake_target(db) -> str:
    return db.execute("SELECT id FROM erasure_targets WHERE participant='collectors' "
                      "AND descriptor LIKE '%\"wake_clips\"%'").fetchone()[0]


def _clip(db, target: str, *, published: int, expires_at: float, sources) -> None:
    now = time.time()
    db.execute("INSERT INTO wake_clips_v1 (id,target_id,bucket,room_id,score,sources_json,captured_at,"
               "expires_at,published) VALUES (?,?,?,?,?,?,?,?,?)",
               (str(uuid4()), target, "positive", "living", .9,
                json.dumps([source.to_dict() for source in sources], separators=(",", ":")),
                now - 120, expires_at, published))


def wake_clip_fenced(data, final):
    """R2: a published clip past its own TTL with a live source and an unpublished live clip.

    Both are wake_deadline_fences rows (expires_at below the source deadline);
    production leaves them routinely (expiring clips, a crash before publish).
    """
    source = _source(expires_in=DAY)
    with _connect(_graph(data) / "privacy.db") as db:
        target = _wake_target(db)
        _clip(db, target, published=1, expires_at=time.time() - 60, sources=(source,))
        _clip(db, target, published=0, expires_at=time.time() + 3600, sources=(source,))
    return {"planted": ["wake_clips_v1 published, own TTL passed, source live (fence)",
                        "wake_clips_v1 unpublished, live (fence)"]}


def wake_clip_expiring_late(data, final):
    """R6: a published fenced clip whose TTL passes ~50 s after planting (after READY).

    The construction sweep leaves it; the store's next cycle (deadline = the
    earliest clip expiry, at most now+60 s) deletes it. Settle 75 s so that a
    loop first firing between 15 s and the planned N (<= 60 s) is observed.
    """
    source = _source(expires_in=DAY)
    with _connect(_graph(data) / "privacy.db") as db:
        _clip(db, _wake_target(db), published=1, expires_at=time.time() + 50, sources=(source,))
    return {"planted": ["wake_clips_v1 published, TTL ~50 s after planting, source live (fence)"],
            "settle": 75.0}


def clean_long_settle(data, final):
    """R6: the clean state held 75 s after READY (largest planned N plus one 60 s cycle)."""
    return {"planted": [], "settle": 75.0}


def deleted_fact_no_index_row(data, final):
    """R3: a deleted fact without a policy_index_deletions row.

    Left by a fact deleted while no semantic_index target was registered (an
    older state); register_target's INSERT..SELECT backfills it at construction.
    """
    ledger = _ledger(data)
    try:
        fact = ledger.store("Synthetic fact deleted early", "fact", .9, (_source(),))
    finally:
        ledger.close()
    with _connect(_graph(data) / "privacy.db") as db:
        db.execute("UPDATE facts SET content=NULL, state='deleted', expires_at=NULL, revision=revision+1 "
                   "WHERE id=?", (fact.id,))
        db.execute("DELETE FROM fact_sources WHERE fact_id=?", (fact.id,))
        db.execute("DELETE FROM policy_index_deletions WHERE fact_id=?", (fact.id,))
    return {"planted": ["facts state=deleted, no policy_index_deletions row"]}


def retention_shortened(data, final):
    """R4: rows written under 30/90-day retention, NEW configured with 1/7 days.

    An accepted suggestion (projected with every column, and a deadline
    fence) whose expires_at is observed_at + 40 h, inside its 47 h source,
    and an event whose expires_at is timestamp + 80 days and whose
    text_expires_at is a text fence (1 h before its source). NEW runs with
    memory.automatic_retention_days=1 and analytics.retention_days=7, so the
    boot clamps
    (suggestion_storage.py:158 via suggestion_store.py:106, event_logger.py:211)
    have rows to rewrite.
    """
    from src.analytics.event_evidence import EventDependency
    now = time.time()
    observed = now - 3600
    source = _source(expires_in=47 * 3600)
    evidence = EventDependency(secrets.token_hex(32), 1, secrets.token_hex(32), now + 25 * DAY)
    graph = _graph(data)
    with _connect(graph / "suggestions.json.policy-v1.sqlite3") as db:
        db.execute("INSERT INTO suggestions_policy_v1 (id,pattern_key,created_at,observed_at,expires_at,status,"
                   "snooze_until,pattern_json,evidence_json,sources_json,message) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   ("sug_" + uuid4().hex, "synthetic-pattern", observed, observed, observed + 40 * 3600,
                    "accepted", None, "{}", json.dumps([evidence.to_dict()]),
                    json.dumps([source.to_dict()]), "Sugerencia sintética"))
    with _connect(graph / "events.db") as db:
        db.execute("INSERT INTO events_policy_v1 (timestamp,event_type,entity_id,action,context_json,hour,minute,"
                   "weekday,is_weekend,time_of_day,sources_json,metrics_json,expires_at,text_expires_at) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (now - 3600, "device_action", "light.living", "turn_on", "{}", 12, 0, 1, 0, "afternoon",
                    json.dumps([source.to_dict()]), "{}", now + 80 * DAY, source.expires_at - 3600))
    return {"planted": ["suggestions_policy_v1 accepted, expires_at observed+40 h (fence)",
                        "events_policy_v1 expires_at timestamp+80 d, text_expires_at 1 h before its source (text fence)"],
            "config": {"memory.automatic_retention_days": 1, "analytics.retention_days": 7}}


def no_default_control_row(data, final):
    """CR-7 extra (SC-4a): H without the default ambient control row (no_op_exemption backstop).

    AmbientControl._load inserts it when absent, in V as at any boot: K' gains
    a row, so retire's strict equality refuses (abort with W or SS).
    """
    with _connect(_graph(data) / "privacy.db") as db:
        db.execute("DELETE FROM ambient_runtime_control")
    return {"planted": ["ambient_runtime_control row removed"]}


STATES = {
    "clean": clean,
    "retired_owner_facts": retired_owner_facts,
    "retired_owner_facts_deferred": retired_owner_facts_deferred,
    "tombstone_without_operation": tombstone_without_operation,
    "retired_user_credential_session": retired_user_credential_session,
    "pending_erasure_step": pending_erasure_step,
    "queued_audio_deletion": queued_audio_deletion,
    "expired_sessions_credentials": expired_sessions_credentials,
    "recorders_clean_stop_running": recorders_clean_stop_running,
    "recorders_clean_stop_muted": recorders_clean_stop_muted,
    "recorders_crash_running": recorders_crash_running,
    "recorders_crash_muted": recorders_crash_muted,
    "recorder_orphan_unknown": recorder_orphan_unknown,
    "reminder_action_sending": reminder_action_sending,
    "diagnostic_quarantine_staged": diagnostic_quarantine_staged,
    "deleted_fact_owner_rows": deleted_fact_owner_rows,
    "routine_run_interrupted": routine_run_interrupted,
    "owned_fact_expired": owned_fact_expired,
    "expired_voice_group_request": expired_voice_group_request,
    "wake_clip_fenced": wake_clip_fenced,
    "deleted_fact_no_index_row": deleted_fact_no_index_row,
    "retention_shortened": retention_shortened,
    "wake_clip_expiring_late": wake_clip_expiring_late,
    "clean_long_settle": clean_long_settle,
    "no_default_control_row": no_default_control_row,
}
