"""Real journal transformations without fabricated controller closure ACKs.

Transformation tests run below admission. The public missing-lease test covers
that boundary; managed Graph integration must supply the actual close receipt.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.memory import state_journal_derivation as derivation
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from src.wakeword.clip_store import _SCHEMA as CLIP_SCHEMA
from tests.mocks.memory_sources import memory_source


@pytest.fixture
def journal(tmp_path):
    ledger = FactLedger(str(tmp_path / "privacy.db"))
    ledger.initialize()
    collector = ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,
        {"kind": "wake_clips", "root": str(tmp_path / "old" / "wake")})
    index = ledger.erasures.register_target(ErasureParticipant.SEMANTIC_INDEX, 1,
        {"path": str(tmp_path / "old" / "chroma"), "collection": "unit_policy_v1"})
    producer = ledger.erasures.register_target(ErasureParticipant.PRODUCERS, 1,
        {"kind": "runtime_producers", "protocol": 1})
    events = ledger.erasures.register_target(ErasureParticipant.EVENTS, 1,
        {"path": str(tmp_path / "old" / "events.db"), "table": "events_policy_v1"})
    subject, foreign = memory_source(), memory_source()
    deleted = ledger.store("Synthetic sourced automatic fact", "fact", .9, (subject,))
    proposal = ledger.store("Synthetic reviewed manual fact", "fact", .9, (foreign,), proposed=True)
    confirmed = ledger.confirm(proposal.id, MemoryReviewer("synthetic-reviewer"))
    ledger.delete_conversation(subject.conversation_id, MemoryReviewer("synthetic-reviewer"))
    ledger._db.executescript(CLIP_SCHEMA)
    clip_id = str(uuid4())
    with ledger._transaction() as db:
        db.execute("INSERT INTO wake_clips_v1(id,target_id,bucket,room_id,score,sources_json,"
            "captured_at,expires_at,published) VALUES (?,?,?,?,?,?,?,?,?)",
            (clip_id, collector, "accepted", "unit-room", .9,
             json.dumps([subject.to_dict()]), subject.captured_at, subject.expires_at, 0))
        # Historical fixture metadata, never a callback claiming native release.
        db.execute("UPDATE erasure_steps SET state='complete',cursor=NULL WHERE target_id=?", (producer,))
        db.execute("UPDATE erasure_steps SET state='running',claim_token='historical-token',"
                   "cursor='historical-cursor',attempts=3 WHERE target_id=?", (collector,))
    try:
        yield SimpleNamespace(ledger=ledger, collector=collector, index=index, producer=producer, events=events,
            clip_id=clip_id, subject=subject, deleted=deleted, confirmed=confirmed)
    finally:
        ledger.close()


def _targets(journal):
    return [dict(target, participant=participant.value) for participant in ErasureParticipant
            for target in journal.ledger.erasures.targets(participant)]


def _local(journal, tmp_path):
    state = (tmp_path / "derived").resolve()
    originals = _targets(journal)
    relocations = {}
    for target in originals:
        descriptor = dict(target["descriptor"])
        for name in ("root", "path"):
            if name in descriptor:
                descriptor[name] = str(state / Path(descriptor[name]).name)
        relocations[target["id"]] = descriptor
    mapping, records = derivation._relocations(stage, state, [{"targets": originals}], relocations)
    return state, mapping, records


def test_real_registration_reopens_proof_and_preserves_authority(journal, tmp_path):
    ledger = journal.ledger
    before = derivation._projection(ledger._db, immutable=True)
    operations = [tuple(row) for row in ledger._db.execute("SELECT * FROM conversation_erasures")]
    _, mapping, records = _local(journal, tmp_path)
    counts = derivation._replace_current(ledger, records, mapping)
    assert counts["steps"] >= 10 and counts["index_cleanup"] == 1
    assert derivation._projection(ledger._db, immutable=True) == before
    assert [tuple(row) for row in ledger._db.execute("SELECT * FROM conversation_erasures")] == operations
    assert ledger.conversation_committed_revoked(journal.subject.conversation_id)
    assert ledger.get(journal.confirmed.id).revision == journal.confirmed.revision
    assert ledger.get(journal.confirmed.id).confirmed_by == "synthetic-reviewer"
    targets = {row[0] for row in ledger._db.execute("SELECT id FROM erasure_targets")}
    assert targets == {target["id"] for target in records}
    assert journal.collector not in targets and journal.index not in targets
    assert journal.producer in targets  # Stable protocol, registered afresh.
    assert ledger._db.execute("SELECT target_id FROM wake_clips_v1 WHERE id=?",
                              (journal.clip_id,)).fetchone()[0] == mapping[journal.collector]
    for row in ledger._db.execute("SELECT * FROM erasure_steps"):
        assert row["state"] == "pending" and row["attempts"] == 0
        assert row["claim_token"] is row["cursor"] is None
    for row in ledger._db.execute("SELECT * FROM policy_index_deletions"):
        assert row["state"] == "pending" and row["attempts"] == 0 and row["claim_token"] is None
    jobs = {row["fact_id"]: dict(row) for row in ledger._db.execute("SELECT * FROM index_jobs")}
    assert jobs[journal.deleted.id]["operation"] == "delete"
    assert jobs[journal.confirmed.id]["operation"] == "upsert"
    assert all(row["attempts"] == 0 and row["last_error"] is None for row in jobs.values())
    assert not ledger._db.execute("PRAGMA foreign_key_check").fetchall()


def test_missing_clip_relocation_rolls_back_the_whole_transaction(journal, tmp_path):
    before = derivation._projection(journal.ledger._db)
    _, mapping, records = _local(journal, tmp_path)
    del mapping[journal.collector]
    with pytest.raises(ValueError, match="^state_journal_payload_target_missing$"):
        derivation._replace_current(journal.ledger, records, mapping)
    assert derivation._projection(journal.ledger._db) == before
    assert not journal.ledger._db.execute("PRAGMA foreign_key_check").fetchall()


def test_blocked_preparation_keeps_real_sidecar_non_admissible(journal, tmp_path):
    state, mapping, records = _local(journal, tmp_path)
    state.mkdir(mode=0o700)
    derivation._replace_current(journal.ledger, records, mapping)
    destination = sqlite3.connect(state / "privacy.db")
    try:
        journal.ledger._db.backup(destination)
    finally:
        destination.close()
    policy = {"schema_version": 1, "source_bindings": [], "owned_targets": [],
        "delegated_targets": [], "historical_targets": []}
    registered = derivation._publish_ownership(stage, tmp_path, state, policy, records,
                                               keep_blocked=True)
    path = state / ".kza-target-ownership.json"
    assert json.loads(path.read_bytes()) == policy
    assert registered and set(registered) == set(stage._target_inventory(state))
    assert json.loads(path.read_bytes())["owned_targets"] == []
    # This is an actual inventory check, not an admission/SDK-close receipt.
    binding = {"state_root": str(state),
        "ownership_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="runtime_admission_active_inventory_mismatch"):
        stage.validate_target_ownership(tmp_path, binding)


def test_foreign_target_cannot_be_relabelled_as_derived_local(journal, tmp_path):
    state, _, _ = _local(journal, tmp_path)
    targets = _targets(journal)
    descriptors = {target["id"]: dict(target["descriptor"]) for target in targets}
    with pytest.raises(ValueError, match="^state_journal_foreign_relocation$"):
        derivation._relocations(stage, state, [{"targets": targets}], descriptors)


def test_extra_unproved_old_target_id_is_rejected(journal, tmp_path):
    state, _, _ = _local(journal, tmp_path)
    targets = _targets(journal)
    descriptors = {target["id"]: dict(target["descriptor"]) for target in targets}
    descriptors["0" * 64] = {"kind": "runtime_producers", "protocol": 1}
    with pytest.raises(ValueError, match="^state_journal_relocation_inventory$"):
        derivation._relocations(stage, state, [{"targets": targets}], descriptors)


def test_event_dependency_hashes_are_distinct_and_revision_is_not_rewritten(journal, tmp_path):
    from src.analytics.event_evidence import event_target_id, project_pattern_event

    targets = _targets(journal)
    _, mapping, records = _local(journal, tmp_path)
    journal_ids, evidence_ids = derivation._event_maps([{"targets": targets}], mapping, records)
    old = next(target for target in targets if target["id"] == journal.events)
    new = next(target for target in records if target["id"] == mapping[journal.events])
    old_evidence = event_target_id(old["descriptor"])
    new_evidence = event_target_id(new["descriptor"])
    assert journal_ids == {journal.events: mapping[journal.events]}
    assert evidence_ids == {old_evidence: new_evidence}
    assert old_evidence != journal.events and new_evidence != mapping[journal.events]
    source = memory_source()
    row = {"id": 1, "timestamp": source.captured_at, "event_type": "action",
        "entity_id": "light.synthetic", "action": "turn_on", "user_id": None,
        "expires_at": source.expires_at, "sources_json": json.dumps([source.to_dict()])}
    before = project_pattern_event(row, old_evidence).dependency
    after = project_pattern_event(row, new_evidence).dependency
    assert before.revision == after.revision
    assert before.event_id == after.event_id and before.expires_at == after.expires_at
    assert before.target_id != after.target_id


def test_whole_archive_retains_original_claims_and_descriptors(journal, tmp_path):
    database = {"schema": [], "tables": {name: {"columns": [], "primary_key": [],
        "rows": [dict(row) for row in journal.ledger._db.execute("SELECT * FROM " + name)]}
        for name in derivation._JOURNAL}}
    targets = _targets(journal)
    snapshot = SimpleNamespace(snapshot_root=tmp_path / "closed-image", source_binding_id="a" * 64)
    source = {"snapshot": snapshot, "database": database, "manifest": {"unit_serializer_input": True},
        "binding": {"state_root": str(tmp_path / "old-state"), "state_identity": {"device": 1, "inode": 2}},
        "closure": None, "targets": targets, "policy": None}
    archive = derivation._archive(source, "data/graph/privacy.db")
    assert archive["targets"] == targets
    assert archive["journal"]["tables"] == database["tables"]
    assert any(row["state"] == "complete" for row in archive["journal"]["tables"]["erasure_steps"]["rows"])
    assert any(row["claim_token"] == "historical-token"
        for row in archive["journal"]["tables"]["erasure_steps"]["rows"])
    assert archive["current_execution_grant"] is False
    assert archive["known_copy_obligations"][1]["status"] == "known_retained_nonserving_copy"
    assert archive["source_closure"] is None  # No forged controller receipt.


def test_public_qualification_requires_controller_lease_before_files(tmp_path):
    controller = stage.AdmissionController(tmp_path / "not-created")
    with pytest.raises(ValueError, match="runtime_admission_controller_not_entered"):
        derivation.qualify_and_register_derived_journal(controller, None,
            tmp_path / "destination", privacy_relative="data/graph/privacy.db", target_relocations={})
    assert not (tmp_path / "not-created").exists()
    assert not (tmp_path / "destination").exists()


def test_receipt_target_mapping_is_immutable(tmp_path):
    receipt = derivation.JournalDerivationReceipt("a" * 64, tmp_path, "data/privacy.db",
        "b" * 64, tmp_path / "history.json", (), {"old": "new"}, ("new",), {"steps": 10})
    with pytest.raises(TypeError):
        receipt.target_id_map["old"] = "foreign"
    with pytest.raises(TypeError):
        receipt.pending_counts["steps"] = 0
    with pytest.raises(FrozenInstanceError):
        receipt.history_sha256 = "c" * 64


def _cut_case(journal, tmp_path):
    """Pure proof metadata over real ledger rows, not an admitted snapshot."""
    fact = journal.ledger.store("Synthetic latest-only ordinary fact", "fact", .8, (memory_source(),))
    db = journal.ledger._db
    tables = {}
    for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        columns = [{"name": row[1], "type": row[2], "notnull": row[3],
                    "default": row[4], "pk": row[5]}
                   for row in db.execute('PRAGMA table_info("' + name + '")')]
        tables[name] = {"columns": columns,
            "primary_key": [row["name"] for row in sorted(columns, key=lambda row: row["pk"]) if row["pk"]],
            "rows": [dict(row) for row in db.execute('SELECT * FROM "' + name + '"')]}
    latest = SimpleNamespace(source_binding_id="b" * 64,
        manifest={"databases": {"data/graph/privacy.db": {"tables": tables}}, "files": {}})
    ordinary = SimpleNamespace(source_binding_id="a" * 64, manifest={"codec_only": True})
    state = tmp_path / "cut-destination"
    state.mkdir(mode=0o700)
    row = next(row for row in tables["facts"]["rows"] if row["id"] == fact.id)
    def digest(value):
        return hashlib.sha256(derivation._json(value)).hexdigest()
    proof = {"schema_version": 1, "kind": "ordinary_payload_cut",
        "ordinary_binding_id": ordinary.source_binding_id, "latest_binding_id": latest.source_binding_id,
        "ordinary_manifest_sha256": digest(ordinary.manifest),
        "latest_manifest_sha256": digest(latest.manifest), "plan_sha256": "c" * 64,
        "destination_root": str(state), "destination_identity": stage._identity(state),
        "removed_rows": [{"database": "data/graph/privacy.db", "table": "facts",
            "primary_key": ["id"], "identity": [row["id"]], "row_sha256": digest(row),
            "family": "semantic_index"}], "removed_files": []}
    return proof, [{"snapshot": latest}, {"snapshot": ordinary}], state, tables


def test_cut_preparation_is_bound_to_exact_sources_identity_and_rows(journal, tmp_path):
    proof, sources, state, _ = _cut_case(journal, tmp_path)
    sealed = derivation._payload_cut_proof(stage, proof, sources, state)
    assert sealed == proof and sealed is not proof
    proof["removed_rows"][0]["row_sha256"] = "0" * 64
    assert sealed["removed_rows"][0]["row_sha256"] != "0" * 64
    assert "cleanup_complete" not in sealed
    assert not journal.ledger.conversation_committed_revoked(
        next(row["conversation_id"] for row in journal.ledger._db.execute(
            "SELECT conversation_id FROM fact_sources WHERE fact_id=?",
            (sealed["removed_rows"][0]["identity"][0],))))


@pytest.mark.parametrize("change", ["row_hash", "latest_binding", "destination_inode", "manual_row", "extra_ack"])
def test_cut_preparation_rejects_unbound_or_protected_evidence(journal, tmp_path, change):
    proof, sources, state, tables = _cut_case(journal, tmp_path)
    if change == "row_hash":
        proof["removed_rows"][0]["row_sha256"] = "0" * 64
    elif change == "latest_binding":
        proof["latest_binding_id"] = "f" * 64
    elif change == "destination_inode":
        proof["destination_identity"]["inode"] += 1
    elif change == "manual_row":
        row = next(row for row in tables["facts"]["rows"] if row["id"] == journal.confirmed.id)
        proof["removed_rows"][0]["identity"] = [row["id"]]
        proof["removed_rows"][0]["row_sha256"] = hashlib.sha256(derivation._json(row)).hexdigest()
    else:
        proof["cleanup_complete"] = True
    with pytest.raises(ValueError):
        derivation._payload_cut_proof(stage, proof, sources, state)


def test_owner_and_revision_rows_survive_journal_replacement_byte_identical(journal, tmp_path):
    """U3b W8: owner tables are covered by _replace_current's immutable projection."""
    from src.memory.fact_ledger import MemoryAssigner, MemoryOwnerActor
    ledger = journal.ledger
    operator = MemoryAssigner("synthetic-operator")
    allow = {"retired": lambda ids: False, "active": lambda person: True}
    owned = ledger.store("Synthetic fact to own", "fact", .9, (memory_source(),))
    returned = ledger.store("Synthetic fact to return", "fact", .9, (memory_source(),))
    ledger.assign_owner(journal.confirmed.id, "user_one", journal.confirmed.revision, 0, operator, **allow)
    ledger.assign_owner(owned.id, "user_two", owned.revision, 0, operator, **allow)
    ledger.assign_owner(returned.id, "user_two", returned.revision, 0, operator, **allow)
    assert ledger.return_owned(returned.id, "user_two", MemoryOwnerActor("synthetic-home", "user_two"))

    def rows():  # Values and their SQLite storage types, row by row.
        return {table: [(tuple(row), tuple(type(value).__name__ for value in row))
                        for row in ledger._db.execute("SELECT * FROM " + table + " ORDER BY 1")]
                for table in ("fact_owners", "fact_owner_revisions", "fact_owner_schema")}

    before = rows()
    assert len(before["fact_owners"]) == 2 and len(before["fact_owner_revisions"]) == 3
    _, mapping, records = _local(journal, tmp_path)
    derivation._replace_current(ledger, records, mapping)
    assert rows() == before
    assert not ledger._db.execute("PRAGMA foreign_key_check").fetchall()


def test_journal_replacement_refuses_an_owner_row_change(journal, tmp_path, monkeypatch):
    """The immutable projection includes the owner tables: a change inside the transaction aborts it."""
    ledger = journal.ledger
    ledger._db.execute("INSERT INTO fact_owner_revisions VALUES (?,1,'operator_unassigned',0,1)",
                       (journal.confirmed.id,))
    before = derivation._projection(ledger._db)
    original = ledger._queue

    def tamper(db, *args):
        db.execute("UPDATE fact_owner_revisions SET last_change='owner_returned'")
        return original(db, *args)

    monkeypatch.setattr(ledger, "_queue", tamper)
    _, mapping, records = _local(journal, tmp_path)
    with pytest.raises(ValueError, match="protected_records_changed"):
        derivation._replace_current(ledger, records, mapping)
    monkeypatch.undo()
    assert derivation._projection(ledger._db) == before


def test_voice_attribution_rows_survive_journal_replacement_with_an_advanced_clock(journal, tmp_path):
    """U3b-4a-0 (R1): aged attempts and events are protected rows; replacement never prunes them."""
    import time
    from tests.unit.memory.test_voice_attribution_schema import seed_voice_attribution_rows
    ledger = journal.ledger
    seed_voice_attribution_rows(ledger._db, time.time())
    tables = ("voice_attribution_consents", "voice_verification_attempts", "voice_attribution_events",
              "voice_attribution_suspensions", "voice_attribution_control", "voice_attribution_schema")

    def rows():  # Values and their SQLite storage types, row by row.
        return {table: [(tuple(row), tuple(type(value).__name__ for value in row))
                        for row in ledger._db.execute("SELECT * FROM " + table + " ORDER BY 1")]
                for table in tables}

    before = rows()
    assert all(before[table] for table in tables)
    ledger.clock = lambda: time.time() + 400 * 86400  # past every attempt and event window
    _, mapping, records = _local(journal, tmp_path)
    derivation._replace_current(ledger, records, mapping)
    assert rows() == before
    assert not ledger._db.execute("PRAGMA foreign_key_check").fetchall()
