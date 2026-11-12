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
}


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


@pytest.mark.parametrize("source", sorted(RESTORES))
def test_the_entrys_cleaner_removes_the_occurrence_and_completes_its_step(erased, state, source):
    participant, restore = RESTORES[source]
    restore(state, erased)
    assert _participant(state, participant) == 1
    asyncio.run(fixture.run_cleaners(state))  # real graph: leased recovery, then every cleaner, settled
    assert _step_states(state, participant) == {"complete"}
    assert launcher.resurrection_counts(state.state) == NONE


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
