"""U3b-0d4a: conversation resurrection per store, on stores written by the real code.

The fixture (tests/mocks/conversation_resurrection) erases one of two seeded
conversations through the real coordinator with every target registered,
including the three cleaner-less literals, whose steps stay pending forever.
Before 0d4a any pending step excluded the conversation, so none of these
restores was ever counted (review F5).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3

import pytest

from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks import conversation_resurrection as fixture

launcher.bind_stage(stage)

LITERALS = {("diagnostics", 1), ("streams", 2), ("voices", 2)}
NONE = dict.fromkeys(launcher._RESURRECTION_KINDS, 0)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("erased-conversation")
    erased = asyncio.run(fixture.build_erased_state(base / "state", base / "before"))
    shutil.copytree(erased.state, base / "pristine", symlinks=True)
    return erased, base / "pristine"


@pytest.fixture
def erased(built):
    """The erased state, reset at its own path before each test."""
    erased, pristine = built
    return fixture.reset(erased, pristine)


@pytest.fixture
def state(erased):
    return erased


def _conversation(state) -> int:
    return launcher.resurrection_counts(state.state)["conversation"]


def _participant(state, participant: str) -> int:
    return launcher.resurrection_detail(state.state)["conversation"][participant]


def _step_states(state, participant: str) -> set[str]:
    return {step["state"] for step in fixture.steps(state) if step["participant"] == participant}


def _ambient_only(state, keep: str) -> None:
    """Restore the older ambient.db, then keep only one of its two conversation sources."""
    fixture.restore_file(state, "data/ambient.db")
    with sqlite3.connect(state.state / "data" / "ambient.db") as db:
        if keep == "utterances":
            db.execute("DELETE FROM capture_conversations WHERE id=?", (state.subject,))
        else:
            db.execute("DELETE FROM utterances WHERE conversation_id=?", (state.subject,))


RESTORES = {
    "utterances": ("captures", lambda s, erased: _ambient_only(s, "utterances")),
    "capture_conversations": ("captures", lambda s, erased: _ambient_only(s, "capture_conversations")),
    "events_policy_v1": ("events", lambda s, erased: fixture.restore_file(s, "data/events.db")),
    "wake_clips_v1": ("collectors", lambda s, erased: fixture.restore_clip_rows(s, tables=("wake_clips_v1",))),
    "contexts": ("contexts", lambda s, erased: fixture.restore_context_files(s)),
    "chroma": ("semantic_index", lambda s, erased: fixture.restore_index_with_conversation_metadata(s)),
    # Review F1: only the collector JSON/JSONL files under policy_v1, without their wake_clips_v1 rows.
    "collector_files": ("collectors", lambda s, erased: fixture.restore_collector_files(s)),
}


def _restore_for_cleaner(state, source: str) -> None:
    """The cleaner needs the file's row: an orphan collector file keeps the step pending by design."""
    if source == "collector_files":
        fixture.restore_clip_rows(state)
        fixture.restore_collector_files(state, suffixes=(".json", ".jsonl", ".wav"))
    else:
        RESTORES[source][1](state, state)


# --- the production-shaped fixture itself ---------------------------------------------------------

def test_the_erased_state_has_the_literal_blockers_pending_and_counts_nothing(erased):
    pending = {(step["participant"], step["version"]) for step in fixture.steps(erased)
               if step["state"] != "complete"}
    assert LITERALS <= pending  # the three cleaner-less literal targets (F5)
    assert pending - LITERALS == {("producers", 0)}  # 'unregistered' placeholder: no target
    assert launcher.resurrection_counts(erased.state) == NONE
    assert all(count == 0 for count in launcher.resurrection_detail(erased.state)["conversation"].values())


# --- each map entry: restored -> counted; its own step pending -> exempt; its cleaner -> 0 -------

@pytest.mark.parametrize("source", sorted(RESTORES))
def test_a_restored_store_counts_although_cleaner_less_steps_are_pending(erased, state, source):
    participant, restore = RESTORES[source]
    restore(state, erased)
    assert _conversation(state) == 1 and _participant(state, participant) == 1
    others = {name: count for name, count in launcher.resurrection_detail(state.state)["conversation"].items()
              if name != participant}
    assert set(others.values()) == {0}  # the occurrence is attributed to its own store only


@pytest.mark.parametrize("source", sorted(RESTORES))
def test_the_stores_own_pending_step_exempts_it(erased, state, source):
    participant, restore = RESTORES[source]
    restore(state, erased)
    if participant == "semantic_index":
        # ErasureJournal.recover does not reopen semantic_index steps; the index
        # deletion journal does at graph start. Mark the step as that start would.
        assert fixture.set_step(state, participant, "pending") == 1
    else:
        fixture.restart_journal(state)  # the real leased restart: local steps back to pending
    assert _step_states(state, participant) == {"pending"}
    assert launcher.resurrection_counts(state.state) == NONE | (
        {"fact": 1} if participant == "semantic_index" else {})  # the restored embedding is also a fact id
    exempt = launcher.resurrection_detail(state.state)["conversation_exempt"]  # review F7
    assert exempt[participant] == 1 and sum(exempt.values()) == 1


@pytest.mark.parametrize("source", sorted(RESTORES))
def test_the_entrys_cleaner_removes_the_occurrence_and_completes_its_step(erased, state, source):
    participant, _ = RESTORES[source]
    _restore_for_cleaner(state, source)
    assert _participant(state, participant) == 1
    asyncio.run(fixture.run_cleaners(state))  # real graph: leased recovery, then every cleaner, settled
    assert _step_states(state, participant) == {"complete"}
    assert launcher.resurrection_counts(state.state) == NONE
    assert set(launcher.resurrection_detail(state.state)["conversation_exempt"].values()) == {0}


# --- the acceptance cases of 0d4a -----------------------------------------------------------------

@pytest.mark.parametrize("store", ["data/events.db", "data/ambient.db", "clip store"])
def test_an_older_store_restored_after_a_completed_erasure_is_counted(erased, state, store):
    if store == "clip store":
        fixture.restore_clip_rows(state)
    else:
        fixture.restore_file(state, store)
    assert _conversation(state) >= 1
    fixture.restart_journal(state)  # its own step pending again: the next leased run cleans it
    assert _conversation(state) == 0


def test_a_tombstone_without_an_erasure_operation_is_counted(erased, state):
    """A seeded (legacy) tombstone: ledger and tombstone rows, no conversation_erasures operation."""
    fixture.restore_file(state, "data/events.db")
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        db.execute("DELETE FROM erasure_steps WHERE operation_id=?", (state.operation,))
        db.execute("DELETE FROM conversation_erasures WHERE id=?", (state.operation,))
        assert db.execute("SELECT COUNT(*) FROM conversation_tombstones WHERE conversation_id=?",
                          (state.subject,)).fetchone()[0] == 1
    assert _conversation(state) == 1
    fixture.restart_journal(state)  # recover() creates the operation with pending steps (N5 re-queue)
    assert _conversation(state) == 0


def test_a_turn_follows_its_parent_clips_step_and_an_orphan_turn_counts(erased, state):
    fixture.restore_clip_rows(state)
    assert _participant(state, "collectors") == 1
    fixture.restart_journal(state)  # the parent clips' collector steps are pending: the cascade will remove the turn
    assert _conversation(state) == 0
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        db.execute("PRAGMA foreign_keys=OFF")  # orphan turns: no parent clip left to cascade from
        parents = db.execute("SELECT COUNT(*) FROM wake_clips_v1 w JOIN conversation_turns_v1 t ON t.id=w.id"
                             " WHERE w.sources_json LIKE ?", ("%" + state.subject + "%",)).fetchone()[0]
        assert parents >= 1
        db.execute("DELETE FROM wake_clips_v1 WHERE sources_json LIKE ?", ("%" + state.subject + "%",))
        assert db.execute("SELECT COUNT(*) FROM conversation_turns_v1 WHERE input_sources_json LIKE ?",
                          ("%" + state.subject + "%",)).fetchone()[0] >= 1
    assert _conversation(state) == 1 and _participant(state, "collectors") == 1


def test_contexts_use_the_longest_registered_directory(erased, state):
    """Only the longest registered directory governs: an ancestor target's step neither exempts nor counts."""
    written = fixture.restore_context_files(state)
    assert all("policy_v3" in path.parts for path in written)
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        ancestor = _register(db, "contexts", {"format": 2, "kind": "snapshots",
                                              "path": str(erased.state / "data" / "contexts")})
        db.execute("INSERT INTO erasure_steps(operation_id, participant, target_id, version, state)"
                   " VALUES (?,?,?,?,?)", (state.operation, "contexts", ancestor, 1, "pending"))
    assert _participant(state, "contexts") == 1  # policy_v3's own step is complete
    fixture.restart_journal(state)  # every contexts step pending, policy_v3's included
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        db.execute("UPDATE erasure_steps SET state='complete' WHERE target_id=?", (ancestor,))
    assert _participant(state, "contexts") == 0  # governed by policy_v3 alone; the ancestor is irrelevant


def _register(db, participant: str, descriptor: dict, version: int = 1) -> str:
    """The exact erasure_targets row ErasureJournal._register_target writes."""
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = hashlib.sha256(json.dumps([participant, version, encoded]).encode()).hexdigest()
    db.execute("INSERT INTO erasure_targets VALUES (?,?,?,?)", (key, participant, version, encoded))
    return key


def _pending_step(db, operation: str, participant: str, target: str, version: int = 1) -> None:
    db.execute("INSERT INTO erasure_steps(operation_id, participant, target_id, version, state)"
               " VALUES (?,?,?,?,'pending')", (operation, participant, target, version))


def test_only_a_version_1_target_governs(erased, state):
    """An events target of another version, even with a pending step, never exempts events_policy_v1."""
    fixture.restore_file(state, "data/events.db")
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        (identifier, encoded), = db.execute("SELECT id, descriptor FROM erasure_targets WHERE participant='events'")
        db.execute("DELETE FROM erasure_steps WHERE target_id=?", (identifier,))
        db.execute("DELETE FROM erasure_targets WHERE id=?", (identifier,))
        other = _register(db, "events", json.loads(encoded), version=2)
        _pending_step(db, state.operation, "events", other, version=2)
    assert _participant(state, "events") == 1


@pytest.mark.parametrize("second_pending", [False, True])
def test_every_governing_capture_target_must_be_pending(erased, state, second_pending):
    """Two captures targets for one database (keep_audio.dir changed): exempt only if both steps are pending."""
    fixture.restore_file(state, "data/ambient.db")
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        (encoded,), = db.execute("SELECT descriptor FROM erasure_targets WHERE participant='captures'")
        moved = dict(json.loads(encoded), audio_dir=str(state.state / "data" / "audio-moved"))
        _pending_step(db, state.operation, "captures", _register(db, "captures", moved))
    assert _participant(state, "captures") == 1  # the original target's step is complete
    if second_pending:
        fixture.restart_journal(state)  # both captures steps pending
        assert _participant(state, "captures") == 0


@pytest.mark.parametrize("pending", [False, True])
def test_a_shadow_copy_with_descriptor_root_gives_the_in_place_counts(erased, state, tmp_path, pending):
    for restore in ("events_policy_v1", "utterances", "contexts", "chroma"):
        RESTORES[restore][1](state, erased)
    if pending:
        fixture.restart_journal(state)
        fixture.set_step(state, "semantic_index", "pending")
    in_place = launcher.resurrection_detail(state.state)
    assert launcher.resurrection_counts(state.state)["conversation"] == (0 if pending else 1)
    shadow = launcher._shadow_state(state.state, tmp_path / "shadow")
    assert launcher.resurrection_detail(shadow, descriptor_root=state.state) == in_place
    assert launcher.resurrection_counts(shadow, descriptor_root=state.state) == launcher.resurrection_counts(
        state.state)
    if pending:  # without descriptor_root no descriptor is under the shadow: nothing governs, all counts
        assert launcher.resurrection_counts(shadow)["conversation"] == 1


def test_two_erasure_journals_are_ambiguous(erased, state):
    shutil.copy2(state.state / "data" / "privacy.db", state.state / "data" / "privacy-copy.db")
    with pytest.raises(ValueError, match="runtime_admission_recovery_erasure_journal_ambiguous"):
        launcher.resurrection_counts(state.state)


def test_fact_voice_and_consent_counts_are_unchanged_by_conversation_restores(erased, state):
    for _, restore in RESTORES.values():
        restore(state, erased)
    counts = launcher.resurrection_counts(state.state)
    assert counts["conversation"] == 1
    assert {family: counts[family] for family in ("fact", "voice_profile", "voice_consent")} == {
        "fact": 1, "voice_profile": 0, "voice_consent": 0}  # the restored embedding is the erased fact id


# --- review F2: branches the first round left unpinned ---------------------------------------------

def _target_id(state, participant: str, **match) -> str:
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        rows = [(identifier, json.loads(encoded)) for identifier, encoded in db.execute(
            "SELECT id, descriptor FROM erasure_targets WHERE participant=? AND version=1", (participant,))]
    found = [identifier for identifier, descriptor in rows
             if all(descriptor.get(key) == value for key, value in match.items())]
    assert len(found) == 1
    return found[0]


def _set_target_step(state, target: str, value: str) -> None:
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        assert db.execute("UPDATE erasure_steps SET state=? WHERE operation_id=? AND target_id=?",
                          (value, state.operation, target)).rowcount == 1


def _recovery_checks(state, checked):
    """R-b's own check sequence (minimal binding: nothing bound), as serve and adopt --dry-run run it."""
    report = {"failures": [], "regressions": [], "resurrected": {}, "ledger_bound": False, "databases": 0}
    return launcher._recovery_checks(state.state, checked, {"authority": {"projection": {"tables": {}}}},
                                     report, lambda name: None)


@pytest.mark.parametrize("step_state", ["running", "blocked"])
def test_a_running_or_blocked_own_step_is_unfinished(erased, state, step_state):
    """A crash in mid-step leaves 'running' (R-b runs before recover() turns it into 'pending')."""
    fixture.restore_file(state, "data/events.db")
    assert fixture.set_step(state, "events", step_state) == 1
    assert _conversation(state) == 0
    assert launcher.resurrection_detail(state.state)["conversation_exempt"]["events"] == 1


def test_a_clip_row_naming_a_non_collector_target_is_not_governed(erased, state):
    fixture.restore_clip_rows(state, tables=("wake_clips_v1",))
    fixture.restart_journal(state)  # every local step pending, the events step included
    events = _target_id(state, "events")
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        assert db.execute("UPDATE wake_clips_v1 SET target_id=? WHERE sources_json LIKE ?",
                          (events, "%" + state.subject + "%")).rowcount >= 1
    assert _participant(state, "collectors") == 1


def test_the_chroma_collection_name_must_match(erased, state):
    fixture.restore_index_with_conversation_metadata(state)
    own = _target_id(state, "semantic_index")
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        (encoded,), = db.execute("SELECT descriptor FROM erasure_targets WHERE id=?", (own,))
        other = _register(db, "semantic_index", dict(json.loads(encoded), collection="other_policy_v1"))
        db.execute("INSERT INTO erasure_steps(operation_id, participant, target_id, version, state)"
                   " VALUES (?,?,?,1,'complete')", (state.operation, "semantic_index", other))
    _set_target_step(state, own, "pending")
    assert _participant(state, "semantic_index") == 0  # the other collection's complete step is irrelevant


def test_the_events_table_must_be_events_policy_v1(erased, state):
    fixture.restore_file(state, "data/events.db")
    own = _target_id(state, "events")
    with sqlite3.connect(state.state / "data" / "privacy.db") as db:
        (encoded,), = db.execute("SELECT descriptor FROM erasure_targets WHERE id=?", (own,))
        other = _register(db, "events", dict(json.loads(encoded), table="suggestions_policy_v1"))
        db.execute("INSERT INTO erasure_steps(operation_id, participant, target_id, version, state)"
                   " VALUES (?,?,?,1,'complete')", (state.operation, "events", other))
    _set_target_step(state, own, "pending")
    assert _participant(state, "events") == 0


def test_a_journal_fragment_in_a_second_file_is_ambiguous(erased, state):
    with sqlite3.connect(state.state / "data" / "stray.db") as db:
        db.execute("CREATE TABLE erasure_steps (operation_id TEXT)")
    with pytest.raises(ValueError, match="runtime_admission_recovery_erasure_journal_ambiguous"):
        launcher.resurrection_counts(state.state)


def test_a_symlinked_descriptor_root_still_governs(erased, state, tmp_path):
    """Descriptors are resolved paths; a binding state_root reached through a link must still match."""
    fixture.restore_file(state, "data/events.db")
    fixture.restart_journal(state)
    link = tmp_path / "state-link"
    link.symlink_to(state.state)
    shadow = launcher._shadow_state(state.state, tmp_path / "shadow")
    assert launcher.resurrection_counts(shadow, descriptor_root=link)["conversation"] == 0


def test_recovery_checks_relativize_against_the_state_not_the_shadow(erased, state, tmp_path):
    fixture.restore_file(state, "data/events.db")
    fixture.restart_journal(state)
    shadow = launcher._shadow_state(state.state, tmp_path / "shadow")
    report = _recovery_checks(state, shadow)
    assert report["failures"] == [] and report["resurrected"]["conversation"] == 0
    assert report["resurrection_detail"]["conversation_exempt"]["events"] == 1


@pytest.mark.parametrize("error", [sqlite3.OperationalError("synthetic"), RecursionError()])
def test_an_unreadable_scan_is_a_refusal_not_an_exception(erased, state, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(launcher, "_resurrection_scan", fail)
    assert _recovery_checks(state, state.state)["failures"] == ["runtime_admission_recovery_resurrection_unreadable"]


# --- review F5: scanned JSON that does not parse refuses ------------------------------------------

@pytest.mark.parametrize("payload", ["[" * 200000, "{not json", "7"])
def test_unparsable_or_deep_json_in_a_scanned_column_refuses(erased, state, payload):
    with sqlite3.connect(state.state / "data" / "events.db") as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(events_policy_v1)")]
        row = db.execute("SELECT * FROM events_policy_v1 LIMIT 1").fetchone()
        values = dict(zip(columns, row), id=None, sources_json=payload)
        db.execute(f"INSERT INTO events_policy_v1 ({','.join(values)}) VALUES ({','.join('?' * len(values))})",
                   tuple(values.values()))
    if payload == "7":  # valid JSON without conversation ids: nothing to refuse
        assert _conversation(state) == 0
        return
    with pytest.raises(ValueError, match="runtime_admission_recovery_resurrection_unreadable"):
        launcher.resurrection_counts(state.state)
    failures = _recovery_checks(state, state.state)["failures"]  # never an exception out of R-b
    if payload.startswith("["):
        assert failures == ["runtime_admission_recovery_resurrection_unreadable"]
    else:  # the authority capture, which runs first, already refuses invalid JSON with its own message
        assert len(failures) == 1


def test_a_corrupt_collector_jsonl_line_refuses(erased, state):
    path = next((state.state / "data" / "exports" / "policy_v1").glob("*.jsonl"))
    path.write_bytes(path.read_bytes() + b"[" * 200000 + b"\n")
    with pytest.raises(ValueError, match="runtime_admission_recovery_resurrection_unreadable"):
        launcher.resurrection_counts(state.state)


# --- review F1: every file the real code writes is mapped or classified ----------------------------

def test_every_file_kind_the_real_code_writes_is_mapped_or_classified(erased):
    mapped = {entry.source for entry in launcher.CONVERSATION_SOURCES}
    classified = set(launcher.CONVERSATION_NOT_SCANNED)
    kinds = {}
    for directory in (erased.before, erased.state):  # before the erasure every store holds both cohorts
        for path in directory.rglob("*"):
            if path.is_file():
                kinds.setdefault(launcher.conversation_file_kind(path.relative_to(directory)), path.name)
    unknown = {kind: name for kind, name in kinds.items()
               if kind != "sqlite" and kind not in mapped and kind not in classified}
    assert unknown == {}
    assert {"collectors:policy_v1/*.json", "collectors:policy_v1/*.jsonl", "contexts:*.json",
            "file:policy_v1/*.wav", "file:policy_v1/*.flac"} <= set(kinds)


# --- second-round mutants (F1 files, F3 cache) ------------------------------------------------------

def test_a_multi_record_export_jsonl_is_read_line_by_line(erased, state):
    """Export files hold one JSON record per line; a later line can carry the erased conversation."""
    restored = fixture.restore_collector_files(state, suffixes=(".jsonl",))
    foreign = [path for path in (state.state / "data" / "exports" / "policy_v1").glob("*.jsonl")
               if path not in restored]
    assert foreign
    for path in restored:  # prepend a surviving record so the subject's record is the second line
        path.write_bytes(foreign[0].read_bytes().rstrip(b"\n") + b"\n" + path.read_bytes())
    assert _participant(state, "collectors") == 1


def test_a_shadow_copy_carries_the_export_jsonl(erased, state, tmp_path):
    fixture.restore_collector_files(state, suffixes=(".jsonl",))
    assert _participant(state, "collectors") == 1
    shadow = launcher._shadow_state(state.state, tmp_path / "shadow")
    assert launcher.resurrection_detail(shadow, descriptor_root=state.state) == launcher.resurrection_detail(
        state.state)


def test_each_locator_of_one_source_has_its_own_governing_target(erased, state):
    """events.db (its step pending: exempt) and an unregistered copy beside it (counted)."""
    fixture.restore_file(state, "data/events.db")
    shutil.copy2(state.state / "data" / "events.db", state.state / "data" / "events-restored.db")
    fixture.restart_journal(state)
    detail = launcher.resurrection_detail(state.state)
    assert detail["conversation"]["events"] == 1 and detail["conversation_exempt"]["events"] == 1
