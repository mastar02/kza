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


@pytest.fixture
def manual_pair(tmp_path):
    """Real typed rows/files, with no admission receipt or simulated close."""
    source, candidate = tmp_path / "manual-N", tmp_path / "manual-R"
    (source / "data").mkdir(parents=True)
    ids = [f"00000000-0000-4000-8000-{number:012d}" for number in range(1, 5)]
    roots = {kind: source / "data" / kind for kind in ("conversations", "conversation_exports")}
    targets = {kind: _registration({"kind": kind, "root": str(root)}, "collectors")
               for kind, root in roots.items()}
    sources = json.dumps([{"capture_id": "11111111-1111-4111-8111-111111111111",
        "conversation_id": "22222222-2222-4222-8222-222222222222", "captured_at": 100.0,
        "expires_at": 200.0, "origin": "interaction"}])
    with sqlite3.connect(source / "data/clips.db") as db:
        db.execute("CREATE TABLE erasure_targets (id TEXT PRIMARY KEY, participant TEXT NOT NULL, "
                   "version INTEGER NOT NULL, descriptor TEXT NOT NULL)")
        for name in ("wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
                     "conversation_export_parents_v1"):
            db.executescript(_ddl("src/wakeword/clip_store.py", name))
        for target in targets.values():
            db.execute("INSERT INTO erasure_targets VALUES(?,?,?,?)", (target["id"], "collectors", 1,
                json.dumps(target["descriptor"], sort_keys=True, separators=(",", ":"))))
        for index, identity in enumerate(ids):
            kind = "conversation_exports" if index == 2 else "conversations"
            bucket = "conversation_export" if index == 2 else "conversation_turn"
            db.execute("INSERT INTO wake_clips_v1 VALUES(?,?,?,?,?,?,?,?,?,1)",
                (index + 1, identity, targets[kind]["id"], bucket, "room", 0.5, sources, 100, 200))
            if index != 2:
                db.execute("INSERT INTO conversation_turns_v1 VALUES(?,?,?,?,?)",
                    (identity, "group", sources, "corrected" if index == 0 else "unmarked", "digest"))
            root = roots[kind] / "policy_v1"
            root.mkdir(parents=True, exist_ok=True)
            (root / (identity + (".jsonl" if index == 2 else ".json"))).write_bytes(
                b"private fixture content " + identity.encode())
        for parent in ids[:2]:
            db.execute("INSERT INTO conversation_export_parents_v1 VALUES(?,?)", (ids[2], parent))
    shutil.copytree(source, candidate)
    mapping = {}
    with sqlite3.connect(candidate / "data/clips.db") as db:
        db.execute("DELETE FROM erasure_targets")
        for kind, target in targets.items():
            new = _registration({"kind": kind, "root": str(candidate / "data" / kind)}, "collectors")
            mapping[target["id"]] = new["id"]
            db.execute("INSERT INTO erasure_targets VALUES(?,?,?,?)", (new["id"], "collectors", 1,
                json.dumps(new["descriptor"], sort_keys=True, separators=(",", ":"))))
            db.execute("UPDATE wake_clips_v1 SET target_id=? WHERE target_id=?", (new["id"], target["id"]))
    return {"root": tmp_path, "source": source, "candidate": candidate, "ids": ids, "mapping": mapping}


def _manual_binding(state, name):
    return dict(_binding(state, name), state_identity=stage._identity(state), ownership_sha256=None,
        schema_version=1, runtime_root=str(state.parent / "runtime"), release_sha256="0" * 64,
        code_commit="1" * 40, python_magic_hex="00000000", store_roots={".": stage._identity(state)},
        sdk_versions={})


def _manual_captures(value):
    return tuple(stage._manual_capture(value["root"], _manual_binding(value[key], key))
                 for key in ("source", "candidate"))


def test_accredited_manual_relocation_preserves_full_export_component(manual_pair):
    before, after = _manual_captures(manual_pair)
    assert stage._manual_mapping(before, after) == manual_pair["mapping"]
    stage._require_manual_equivalent(before, after)
    selected = after["stores"]["data/clips.db"]
    assert set(selected["clips"]) == set(manual_pair["ids"][:3])
    assert len(selected["edges"]) == 2
    assert "private fixture content" not in json.dumps(after)


@pytest.mark.parametrize("statement", [
    "DELETE FROM wake_clips_v1 WHERE bucket='conversation_export'",
    "UPDATE wake_clips_v1 SET score=0.9 WHERE bucket='conversation_export'",
    "UPDATE wake_clips_v1 SET published=0 WHERE bucket='conversation_export'",
    "DELETE FROM conversation_export_parents_v1 WHERE parent_id LIKE '%000001'",
    "UPDATE conversation_turns_v1 SET digest='changed' WHERE id LIKE '%000002'",
    "UPDATE wake_clips_v1 SET room_id='changed' WHERE id LIKE '%000002'",
    "DELETE FROM conversation_turns_v1 WHERE id LIKE '%000002'",
], ids=["export-removed", "export-score", "export-publication", "manual-parent-edge",
        "unmarked-parent-digest", "unmarked-parent-room", "unmarked-parent-row"])
def test_v1_equal_does_not_hide_export_or_parent_changes(manual_pair, statement):
    value = manual_pair
    before = stage.capture_state_authority(value["candidate"])
    with sqlite3.connect(value["candidate"] / "data/clips.db") as db:
        db.execute(statement)
    assert stage.capture_state_authority(value["candidate"]) == before
    with pytest.raises(ValueError):
        stage._require_manual_equivalent(*_manual_captures(value))


@pytest.mark.parametrize("record,operation", [(2, "change"), (2, "remove"), (1, "change")],
                         ids=["export-bytes", "export-absence", "parent-bytes"])
def test_database_equality_does_not_hide_manual_file_changes(manual_pair, record, operation):
    value = manual_pair
    before = stage.capture_state_authority(value["candidate"])
    kind, suffix = ("conversation_exports", ".jsonl") if record == 2 else ("conversations", ".json")
    file = value["candidate"] / "data" / kind / "policy_v1" / (value["ids"][record] + suffix)
    if operation == "change":
        file.write_bytes(b"different private bytes")
    else:
        file.unlink()
    assert stage.capture_state_authority(value["candidate"]) == before
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        stage._require_manual_equivalent(*_manual_captures(value))


def test_unrelated_ordinary_collector_bytes_do_not_enter_manual_seal(manual_pair):
    before = _manual_captures(manual_pair)[1]
    file = manual_pair["candidate"] / "data/conversations/policy_v1" / (manual_pair["ids"][3] + ".json")
    file.write_bytes(b"ordinary payload may change")
    after = _manual_captures(manual_pair)[1]
    stage._manual_module().verify_manual_components(before, after)


def test_event_capability_cannot_accredit_a_foreign_collector_owner(manual_pair):
    value = manual_pair
    foreign = _registration({"kind": "conversation_exports", "root": str(value["source"] / "data/conversation_exports")},
                            "collectors")
    with sqlite3.connect(value["candidate"] / "data/clips.db") as db:
        old = db.execute("SELECT target_id FROM wake_clips_v1 WHERE bucket='conversation_export'").fetchone()[0]
        db.execute("UPDATE wake_clips_v1 SET target_id=? WHERE target_id=?", (foreign["id"], old))
        db.execute("UPDATE erasure_targets SET id=?,descriptor=? WHERE id=?", (foreign["id"],
            json.dumps(foreign["descriptor"], sort_keys=True, separators=(",", ":")), old))
    with pytest.raises(ValueError, match="runtime_admission_foreign_target"):
        _manual_captures(value)


def test_existing_head_without_manual_companion_is_not_retroactively_sealed(manual_pair):
    value = manual_pair
    binding = _manual_binding(value["source"], "source")
    identity = stage._store_binding(value["root"], binding)
    head = {"schema_version": 1, "generation": 7, "binding_id": identity, "status": "closed", "run_id": None}
    stage._atomic_record(value["root"] / "admission/head.json", head)
    with pytest.raises(ValueError, match="runtime_admission_manual_legacy_unsealed"):
        stage._verify_manual_head(value["root"], head, binding)
    assert not (value["root"] / "admission/manual-components").exists()


def test_closed_cas_cannot_publish_a_prepared_binding_without_companion(manual_pair):
    value = manual_pair
    identity = stage._store_binding(value["root"], _manual_binding(value["source"], "source"))
    with pytest.raises(ValueError, match="runtime_admission_manual_legacy_unsealed"):
        stage._cas_head(value["root"], None, identity, "closed")
    assert stage.read_admission_head(value["root"]) is None


@pytest.mark.parametrize("existing_head", [False, True], ids=["nonempty-v1", "old-head-empty-now"])
def test_mutable_empty_capture_cannot_bootstrap_old_or_nonempty_v1(manual_pair, monkeypatch, existing_head):
    value = manual_pair
    binding = _manual_binding(value["source"], "source")
    monkeypatch.setattr(stage, "_binding_snapshot", lambda root, name: binding)
    monkeypatch.setattr(stage, "_manual_capture", lambda root, current: {
        "schema_version": 1, "state_root": binding["state_root"], "state_identity": binding["state_identity"], "stores": {}})
    expected = None
    if existing_head:
        binding["authority"]["projection"]["tables"] = {}
        expected = {"schema_version": 1, "generation": 2, "binding_id": stage._digest(binding),
                    "status": "closed", "run_id": None}
    with pytest.raises(ValueError, match="runtime_admission_manual_legacy_unsealed"):
        stage._prepare_manual_head(value["root"], expected, binding, previous=None)
    assert stage.read_admission_head(value["root"]) is None


def test_companion_write_failure_cannot_publish_closed_head(tmp_path, monkeypatch):
    state = tmp_path / "empty-state"
    state.mkdir()
    binding = _manual_binding(state, "empty")
    stage._store_binding(tmp_path, binding)
    monkeypatch.setattr(stage, "_binding_snapshot", lambda root, name: binding)
    original = stage._atomic_record

    def fail_companion(path, value, **kwargs):
        if "manual-components" in path.parts:
            raise OSError("synthetic-companion-write-failure")
        return original(path, value, **kwargs)

    monkeypatch.setattr(stage, "_atomic_record", fail_companion)
    with pytest.raises(OSError, match="synthetic-companion-write-failure"):
        stage._prepare_manual_head(tmp_path, None, binding, previous=None)
    assert stage.read_admission_head(tmp_path) is None


def test_native_close_proof_is_required_before_a_closure_seal(tmp_path):
    with pytest.raises(ValueError, match="runtime_admission_manual_close_proof"):
        stage._manual_receipt_proof(tmp_path, {"schema_version": 1, "status": "closed"})
    assert not (tmp_path / "admission").exists()


def test_source_file_change_cannot_be_adopted_by_candidate_comparison(manual_pair, monkeypatch):
    """Isolate the source-projection guard; this fixture issues no close ACK."""
    value = manual_pair
    binding = _manual_binding(value["source"], "source")
    capture = stage._manual_capture(value["root"], binding)
    head = {"binding_id": stage._digest(binding)}
    monkeypatch.setattr(stage, "_read_manual_head", lambda root, current: {"capture": capture})
    file = value["source"] / "data/conversation_exports/policy_v1" / (value["ids"][2] + ".jsonl")
    file.write_bytes(b"changed after the attested projection")
    assert _manual_binding(value["source"], "source") == binding
    with pytest.raises(ValueError, match="privacy_authority_stale"):
        stage._verify_manual_head(value["root"], head, binding)


def test_bootstrap_companion_mismatch_cannot_advance_head(tmp_path, monkeypatch):
    state = tmp_path / "empty-state"
    state.mkdir()
    binding = _manual_binding(state, "empty")
    identity = stage._store_binding(tmp_path, binding)
    monkeypatch.setattr(stage, "_binding_snapshot", lambda root, name: binding)
    record = stage._prepare_manual_head(tmp_path, None, binding, previous=None)
    assert stage.read_admission_head(tmp_path) is None  # Preparation confers no selection.
    path = stage._manual_head_path(tmp_path, record["head"])
    record["head"]["generation"] = 9
    stage._atomic_record(path, record)
    with pytest.raises(ValueError, match="runtime_admission_manual_companion_schema"):
        stage._cas_head(tmp_path, None, identity, "closed")
    assert stage.read_admission_head(tmp_path) is None


def test_bridge_ticket_and_receipt_paths_do_not_load_manual_contract():
    tree = ast.parse(Path(stage.__file__).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    pending, seen = ["validate_admission_ticket", "write_admission_receipt"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        assert not name.startswith("_manual") and name != "_verify_manual_head"
        pending.extend(node.func.id for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id in functions)


@pytest.mark.parametrize("field", ["schema_version", "head.schema_version", "head.generation"])
def test_boolean_companion_versions_and_head_are_not_integer_versions(tmp_path, monkeypatch, field):
    state = tmp_path / "empty-state"
    state.mkdir()
    binding = _manual_binding(state, "empty")
    identity = stage._store_binding(tmp_path, binding)
    monkeypatch.setattr(stage, "_binding_snapshot", lambda root, name: binding)
    record = stage._prepare_manual_head(tmp_path, None, binding, previous=None)
    head = stage._cas_head(tmp_path, None, identity, "closed")  # Empty bootstrap, no native-close claim.
    path = stage._manual_head_path(tmp_path, head)
    if field.startswith("head."):
        record["head"][field.split(".")[1]] = True
    else:
        record[field] = True
    stage._atomic_record(path, record)
    with pytest.raises(ValueError, match="runtime_admission_manual_companion_schema"):
        stage._verify_manual_head(tmp_path, head, binding)
    assert stage.read_admission_head(tmp_path) == head


@pytest.mark.parametrize("field", ["schema_version", "generation"])
def test_boolean_current_head_is_rejected_by_new_companion_parser(tmp_path, field):
    head = {"schema_version": 1, "generation": 1, "binding_id": "a" * 64, "status": "closed", "run_id": None}
    head[field] = True
    with pytest.raises(ValueError, match="runtime_admission_manual_companion_schema"):
        stage._read_manual_head(tmp_path, head)


def test_boolean_seal_version_is_rejected_before_any_close_evidence(tmp_path):
    seal = {"schema_version": True, "binding_id": "a" * 64, "closure_proof": {}, "closed_head": {}, "capture": {}}
    identity = stage._digest(seal)
    stage._atomic_record(tmp_path / "admission/manual-components/seals" / (identity + ".json"), seal)
    with pytest.raises(ValueError, match="runtime_admission_manual_seal_schema"):
        stage._read_manual_seal(tmp_path, identity)


@pytest.mark.parametrize("field", ["schema_version", "running_head.schema_version"])
def test_boolean_close_proof_version_is_rejected_before_receipt_lookup(tmp_path, field):
    proof = {"schema_version": 1, "binding_id": "a" * 64,
        "running_head": {"schema_version": 1, "generation": 2, "binding_id": "a" * 64,
                         "status": "running", "run_id": "b" * 32},
        "child_pid": 12345, "receipt_sha256": "c" * 64}
    if field.startswith("running_head."):
        proof["running_head"]["schema_version"] = True
    else:
        proof[field] = True
    with pytest.raises(ValueError, match="runtime_admission_manual_close_proof"):
        stage._manual_receipt_proof(tmp_path, proof)
    assert not (tmp_path / "admission").exists()


@pytest.mark.parametrize("field", ["schema_version", "resources.graph_closed"])
def test_malformed_receipt_boolean_integer_coercion_never_authorizes_close(tmp_path, field):
    """Only rejected malformed receipts are manufactured; no serving ACK is used."""
    proof = {"schema_version": 1, "binding_id": "a" * 64,
        "running_head": {"schema_version": 1, "generation": 2, "binding_id": "a" * 64,
                         "status": "running", "run_id": "b" * 32},
        "child_pid": 12345, "receipt_sha256": None}
    receipt = {"schema_version": 1, "binding_id": "a" * 64, "generation": 2, "run_id": "b" * 32,
               "child_pid": 12345, "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True)}
    if field.startswith("resources."):
        receipt["resources"]["graph_closed"] = 1
    else:
        receipt[field] = True
    path = tmp_path / "admission/receipts" / ("b" * 32 + ".json")
    stage._atomic_record(path, receipt)
    proof["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="runtime_admission_manual_close_proof"):
        stage._manual_receipt_proof(tmp_path, proof)
    assert stage.read_admission_head(tmp_path) is None
