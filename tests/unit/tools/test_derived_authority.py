"""Pure equivalence checks; these fixtures issue no serving/closure receipts."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from scripts import stage_runtime as stage
from src.analytics.event_evidence import event_target_id, project_pattern_event


def _ddl(filename, table):
    tree = ast.parse((Path(__file__).resolve().parents[3] / filename).read_text())
    import re
    pattern = re.compile(r"CREATE TABLE(?: IF NOT EXISTS)?\s+" + table + r"\s*\(", re.I)
    return next(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                and type(node.value) is str and pattern.search(node.value))


def _registration(descriptor, participant="events", version=1):
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(json.dumps([participant, version, encoded]).encode()).hexdigest()
    return {"id": identity, "participant": participant, "version": version, "descriptor": descriptor}


def _binding(state, slot):
    _, schemas, _ = stage._state_layout(state)
    return {"state_root": str(state), "slot": slot, "schemas": schemas,
            "authority": stage.capture_state_authority(state)}


@pytest.fixture
def relocation(tmp_path):
    source, candidate = tmp_path / "N", tmp_path / "R"
    (source / "data").mkdir(parents=True)
    descriptor = {"path": str(source / "data/events.db"), "table": "events_policy_v1"}
    registration = _registration(descriptor)
    sources = json.dumps([{"capture_id": "11111111-1111-4111-8111-111111111111",
        "conversation_id": "22222222-2222-4222-8222-222222222222", "captured_at": 100.0,
        "expires_at": 500.0, "origin": "interaction"}])
    with sqlite3.connect(source / "data/events.db") as db:
        db.executescript(_ddl("src/analytics/event_logger.py", "events_policy_v1"))
        db.execute("INSERT INTO events_policy_v1(id,timestamp,event_type,entity_id,action,user_id,"
            "sources_json,metrics_json,expires_at,text_expires_at) "
            "VALUES(1,100,'state_change','light.x','on','u',?,'{}',500,500)", (sources,))
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM events_policy_v1").fetchone()
        evidence = project_pattern_event(row, event_target_id(descriptor)).dependency.to_dict()
    with sqlite3.connect(source / "data/privacy.db") as db:
        db.execute("CREATE TABLE erasure_targets (id TEXT PRIMARY KEY, participant TEXT NOT NULL, "
                   "version INTEGER NOT NULL, descriptor TEXT NOT NULL)")
        db.execute("INSERT INTO erasure_targets VALUES(?,?,?,?)", (registration["id"], "events", 1,
                   json.dumps(descriptor, sort_keys=True, separators=(",", ":"))))
    with sqlite3.connect(source / "data/suggestions.sqlite3") as db:
        db.executescript(_ddl("src/analytics/suggestion_store.py", "suggestions_policy_v1"))
        db.execute("INSERT INTO suggestions_policy_v1 VALUES(1,?,'pattern',101,100,400,'accepted',"
            "NULL,'{}',?,?,'manual proposal')", ("sug_" + "a" * 32, json.dumps([evidence]), sources))
    shutil.copytree(source, candidate)
    destination_descriptor = {"path": str(candidate / "data/events.db"), "table": "events_policy_v1"}
    destination_registration = _registration(destination_descriptor)
    with sqlite3.connect(candidate / "data/privacy.db") as db:
        db.execute("DELETE FROM erasure_targets")
        db.execute("INSERT INTO erasure_targets VALUES(?,?,?,?)", (destination_registration["id"], "events", 1,
                   json.dumps(destination_descriptor, sort_keys=True, separators=(",", ":"))))
    destination_evidence = {**evidence, "target_id": event_target_id(destination_descriptor)}
    with sqlite3.connect(candidate / "data/suggestions.sqlite3") as db:
        db.execute("UPDATE suggestions_policy_v1 SET evidence_json=?", (json.dumps([destination_evidence],
                   sort_keys=True, separators=(",", ":")),))
    return {"root": tmp_path, "source": source, "candidate": candidate, "old": registration,
        "new": destination_registration, "evidence": evidence, "after_evidence": destination_evidence,
        "mapping": {registration["id"]: destination_registration["id"]}}


def _equivalent(value):
    return stage._derived_authority_equivalent(value["root"], _binding(value["source"], "N"),
                                               _binding(value["candidate"], "R"), value["mapping"])


def test_typed_event_locator_change_preserves_every_other_protected_field(relocation):
    value = relocation
    before, after = _binding(value["source"], "N"), _binding(value["candidate"], "R")
    assert before["authority"]["digest"] != after["authority"]["digest"]
    assert value["old"]["id"] != value["evidence"]["target_id"]
    assert value["evidence"]["revision"] == value["after_evidence"]["revision"]
    original = deepcopy(before)
    assert stage._derived_authority_equivalent(value["root"], before, after, value["mapping"])
    assert before == original


@pytest.mark.parametrize("assignment", ["status='rejected'", "message='changed'", "expires_at=399",
    "observed_at=99", "pattern_key='changed'", "snooze_until=350", "row_id=2"])
def test_manual_changes_are_never_locator_authorizations(relocation, assignment):
    with sqlite3.connect(relocation["candidate"] / "data/suggestions.sqlite3") as db:
        db.execute("UPDATE suggestions_policy_v1 SET " + assignment)
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        _equivalent(relocation)


@pytest.mark.parametrize("field,value", [("revision", "b" * 64), ("event_id", 2),
    ("expires_at", 499.0), ("target_id", "c" * 64), ("unexpected", "extra")])
def test_evidence_keys_revision_deadline_and_identity_remain_exact(relocation, field, value):
    evidence = {**relocation["after_evidence"], field: value}
    with sqlite3.connect(relocation["candidate"] / "data/suggestions.sqlite3") as db:
        db.execute("UPDATE suggestions_policy_v1 SET evidence_json=?", (json.dumps([evidence]),))
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        _equivalent(relocation)


@pytest.mark.parametrize("statement", ["DELETE FROM events_policy_v1",
    "UPDATE events_policy_v1 SET action='off'", "UPDATE events_policy_v1 SET expires_at=499",
    "UPDATE events_policy_v1 SET trigger_phrase='different raw parent payload'"])
def test_referenced_event_parent_must_be_present_and_exact(relocation, statement):
    with sqlite3.connect(relocation["candidate"] / "data/events.db") as db:
        db.execute(statement)
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        _equivalent(relocation)


def test_other_authority_differences_are_not_globally_ignored(relocation):
    for state in (relocation["source"], relocation["candidate"]):
        (state / "config").mkdir()
        (state / "config/settings.yaml").write_text("reasoner:\n  cloud:\n    consent: false\n")
    (relocation["candidate"] / "config/settings.yaml").write_text("reasoner:\n  cloud:\n    consent: true\n")
    assert not _equivalent(relocation)


@pytest.mark.parametrize("mapping", [{}, {"missing": "missing"}])
def test_unregistered_target_mapping_is_rejected(relocation, mapping):
    relocation["mapping"] = mapping
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        _equivalent(relocation)


def test_changed_event_schema_is_not_a_relocation(relocation):
    with sqlite3.connect(relocation["candidate"] / "data/events.db") as db:
        db.execute("CREATE INDEX unexpected_index ON events_policy_v1(action)")
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        _equivalent(relocation)


def test_registered_different_participant_is_not_an_event_capability(relocation):
    descriptor = {"path": str(relocation["candidate"] / "data/events.db"), "table": "suggestions_policy_v1"}
    record = _registration(descriptor, participant="suggestions")
    with sqlite3.connect(relocation["candidate"] / "data/privacy.db") as db:
        db.execute("DELETE FROM erasure_targets")
        db.execute("INSERT INTO erasure_targets VALUES(?,?,?,?)", (record["id"], "suggestions", 1,
                   json.dumps(descriptor, sort_keys=True, separators=(",", ":"))))
    relocation["mapping"] = {relocation["old"]["id"]: record["id"]}
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        _equivalent(relocation)


def test_local_but_different_relative_event_path_is_rejected(relocation):
    destination = relocation["candidate"] / "data/other-events.db"
    shutil.copyfile(relocation["candidate"] / "data/events.db", destination)
    descriptor = {"path": str(destination), "table": "events_policy_v1"}
    record = _registration(descriptor)
    with sqlite3.connect(relocation["candidate"] / "data/privacy.db") as db:
        db.execute("DELETE FROM erasure_targets")
        db.execute("INSERT INTO erasure_targets VALUES(?,?,?,?)", (record["id"], "events", 1,
                   json.dumps(descriptor, sort_keys=True, separators=(",", ":"))))
    relocation["mapping"] = {relocation["old"]["id"]: record["id"]}
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        _equivalent(relocation)
