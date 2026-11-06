"""Payload protocol tests, not synthetic Graph/SDK retirement evidence.

The small SQLite images below exercise immutable-image planning below admission.
Only managed integration can supply a real closed controller witness.
"""
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from src.memory import payload_derivation as payload
from src.memory import state_snapshot as snapshots
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from tests.mocks.memory_sources import memory_source


NOW = 1000.0
PRIVACY = "data/graph/privacy.db"
DIAGNOSTICS = "data/graph/metrics.jsonl.sqlite3"
EXTRA_SCHEMA = """
CREATE TABLE utterances(id TEXT PRIMARY KEY,capture_id TEXT,conversation_id TEXT,
 expires_at REAL,distilled INTEGER,audio_path TEXT,text_expires_at REAL,text_json TEXT);
CREATE TABLE capture_conversations(id TEXT PRIMARY KEY,started REAL);
CREATE TABLE wake_clips_v1(sequence INTEGER PRIMARY KEY,id TEXT UNIQUE,target_id TEXT,
 bucket TEXT,room_id TEXT,score REAL,sources_json TEXT,captured_at REAL,expires_at REAL,published INTEGER);
CREATE TABLE stt_corrections_v1(id TEXT PRIMARY KEY,corrected_text TEXT,user_id TEXT,used INTEGER);
CREATE TABLE conversation_turns_v1(id TEXT PRIMARY KEY,quality TEXT);
CREATE TABLE conversation_export_parents_v1(export_id TEXT,parent_id TEXT,PRIMARY KEY(export_id,parent_id));
CREATE TABLE events_policy_v1(id INTEGER PRIMARY KEY,sources_json TEXT,expires_at REAL);
CREATE TABLE suggestions_policy_v1(id TEXT PRIMARY KEY,status TEXT,snooze_until REAL,
 sources_json TEXT,expires_at REAL,evidence_json TEXT,pattern_json TEXT);
CREATE TABLE voice_fingerprints(id TEXT PRIMARY KEY,sources_json TEXT,expires_at REAL,
 suggested_profile TEXT,suggested_revision INTEGER,similarity REAL);
CREATE TABLE voice_fingerprint_sources(fingerprint_id TEXT,capture_id TEXT,PRIMARY KEY(fingerprint_id,capture_id));
CREATE TABLE voice_capture_reviews(fingerprint_id TEXT PRIMARY KEY,state TEXT);
CREATE TABLE voice_groups(id TEXT PRIMARY KEY,state TEXT,reason TEXT);
CREATE TABLE voice_group_members(group_id TEXT,fingerprint_id TEXT,PRIMARY KEY(group_id,fingerprint_id));
CREATE TABLE voice_group_lineage(group_id TEXT,conversation_id TEXT,PRIMARY KEY(group_id,conversation_id));
CREATE TABLE live_deliveries(id TEXT PRIMARY KEY);
CREATE TABLE live_delivery_sources(id TEXT PRIMARY KEY);
CREATE TABLE live_delivery_lineage(id TEXT PRIMARY KEY);
"""


def _image(tmp_path, name, *, statements=(), seed=None):
    source = tmp_path / (name + "-source")
    (source / "data/graph").mkdir(parents=True)
    ledger = FactLedger(str(source / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    if seed is not None:
        seed(ledger)
    ledger.close()
    with closing(sqlite3.connect(source / PRIVACY)) as db, db:
        db.executescript(EXTRA_SCHEMA.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS "))
        for sql, values in statements:
            db.execute(sql, values)
    with closing(sqlite3.connect(source / DIAGNOSTICS)) as db, db:
        db.executescript("CREATE TABLE metrics(id TEXT PRIMARY KEY,born REAL,expires REAL,payload TEXT,"
            "partition INTEGER,bytes INTEGER);CREATE TABLE lineage(record_id TEXT,conversation_id TEXT,"
            "PRIMARY KEY(record_id,conversation_id));CREATE TABLE state(key TEXT PRIMARY KEY,value INTEGER);")
    root = tmp_path / (name + "-image")
    databases, files, schemas, contracts = {}, {}, {}, {}
    for relative in (PRIVACY, DIAGNOSTICS):
        copied = root / "state" / relative
        snapshots._backup(source / relative, copied)
        database = snapshots._database(copied, root)
        databases[relative] = database
        schemas[relative] = payload._digest([[row[key] for key in ("type", "name", "tbl_name", "sql")]
            for row in database["schema"]])
        contracts[relative] = {"schema_contract_v1": {"columns": {
            table: {column["name"]: column["type"].upper() for column in data["columns"]}
            for table, data in database["tables"].items() if not table.startswith("sqlite_")},
            "application_id": database["application_id"], "user_version": database["user_version"]}}
    for relative, record in snapshots._inventory(root).items():
        files[relative] = dict(record, kind="sqlite_backup")
    info = source.stat()
    binding = {"state_root": str(source), "state_identity": {"device": info.st_dev, "inode": info.st_ino},
        "schemas": schemas, "authority": {"projection": {"tables": contracts}}}
    identity = payload._digest(binding)
    manifest = {"schema_version": 1, "source_binding_id": identity, "source_binding": binding,
        "files": files, "databases": databases, "index_exports": [{"source_path": str(source / "data/graph/chroma"),
            "relative_path": "data/graph/chroma", "collection": "unit_policy_v1", "present": False,
            "metadata": None, "sdk_version": None, "fields": ["ids", "documents", "metadatas", "embeddings"],
            "records": []}]}
    (root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    return snapshots.ClosedStateSnapshot(root, identity, binding, manifest)


def _layout(destination):
    roots = (("ambient_capture", "data/graph/audio"), ("collectors", "data/graph/clips"),
        ("contexts", "data/graph/contexts"), ("diagnostics", "data/graph/metrics.jsonl"))
    return payload.DerivationLayout(destination, PRIVACY,
        tuple(payload.DatabaseRelocation(path, path, path) for path in (PRIVACY, DIAGNOSTICS)),
        tuple(payload.FileRootRelocation(family, path, path, path) for family, path in roots),
        (payload.IndexRelocation("data/graph/chroma", "data/graph/chroma", "data/graph/chroma", "unit_policy_v1"),), {})


def test_public_planner_is_read_only_and_deterministic_without_sdk(tmp_path, monkeypatch):
    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    monkeypatch.setattr(payload.importlib.metadata, "version", lambda _name: pytest.fail("SDK lookup during plan"))
    first = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    second = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    assert first.plan_sha256 == second.plan_sha256
    assert first.coverage == tuple(sorted(payload.FAMILIES))
    assert not destination.exists()
    assert snapshots.verify_snapshot(ordinary) and snapshots.verify_snapshot(latest)


@pytest.mark.parametrize("mutation", ["blob", "extra_file", "binding", "layout_escape"])
def test_public_preflight_rejects_changed_inputs_before_destination(tmp_path, mutation):
    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    layout = _layout(destination)
    if mutation == "blob":
        (ordinary.snapshot_root / "state" / PRIVACY).write_bytes(b"not the pinned database")
    elif mutation == "extra_file":
        (latest.snapshot_root / "state/unlisted").write_bytes(b"unexpected")
    elif mutation == "binding":
        latest.source_binding["state_root"] = "/foreign"
    else:
        layout = replace(layout, privacy_relative="../privacy.db")
    with pytest.raises(ValueError):
        payload.plan_payload_cut(ordinary, latest, now=NOW, layout=layout)
    assert not destination.exists()


def _select(ordinary, latest):
    old = ordinary.manifest["databases"][PRIVACY]
    new = latest.manifest["databases"][PRIVACY]
    policy = payload._Policy(ordinary, latest, old, new, NOW)
    policy.database = PRIVACY
    return payload._select_database(old, new, policy), policy


def test_all_stt_corrections_and_feedback_successors_keep_latest_closure(tmp_path):
    source = memory_source(now=NOW-1)
    encoded = json.dumps([source.to_dict()])
    def clips(quality, corrected, used):
        return (("INSERT INTO wake_clips_v1 VALUES(1,'stt','target','stt_correction','room',.9,?,?,?,0)",
                    (encoded, source.captured_at, source.expires_at)),
            ("INSERT INTO stt_corrections_v1 VALUES('stt',?,'original-user',?)", (corrected, used)),
            ("INSERT INTO wake_clips_v1 VALUES(2,'turn','target','conversation','room',.9,?,?,?,0)",
                    (encoded, source.captured_at, source.expires_at)),
            ("INSERT INTO conversation_turns_v1 VALUES('turn',?)", (quality,)))
    ordinary = _image(tmp_path, "O", statements=clips("unmarked", "original correction", 0))
    latest = _image(tmp_path, "N", statements=clips("good", "latest correction", 1))
    selected, _ = _select(ordinary, latest)
    assert {image.values["id"] for image in selected["wake_clips_v1"].values()} == {"stt", "turn"}
    assert all(image.source == "latest" for image in selected["wake_clips_v1"].values())
    assert next(iter(selected["stt_corrections_v1"].values())).values["corrected_text"] == "latest correction"
    assert next(iter(selected["stt_corrections_v1"].values())).values["used"] == 1
    assert next(iter(selected["conversation_turns_v1"].values())).values["quality"] == "good"


def test_ordinary_missing_row_needs_qualified_cut_not_just_same_source():
    table = {"columns": [{"name": "id"}], "primary_key": ["id"], "rows": [{"id": "N-only"}]}
    policy = SimpleNamespace(absent=lambda _table, _row: (_ for _ in ()).throw(
        payload.PayloadDerivationError("payload_ordinary_absence_ambiguous")))
    with pytest.raises(ValueError, match="payload_ordinary_absence_ambiguous"):
        payload._ordinary(table, dict(table, rows=[]), lambda row: True, now=NOW, policy=policy, table="events")


def test_shortened_latest_deadlines_are_never_renewed():
    source = memory_source(now=NOW-1)
    row = {"sources_json": json.dumps([source.to_dict()]), "expires_at": source.expires_at,
        "text_expires_at": source.expires_at, "text_json": "ordinary transcript"}
    latest = dict(row, expires_at=NOW+20, text_expires_at=NOW-1)
    cut = payload._blend(row, latest, NOW)
    assert cut["expires_at"] == NOW+20
    assert cut["text_expires_at"] == NOW-1 and cut["text_json"] is None
    assert json.loads(cut["sources_json"]) == [source.to_dict()]


def test_real_tombstone_blocks_raw_capture_and_retains_deleted_fact(tmp_path):
    source = memory_source(now=NOW-1)
    def seed(ledger):
        ledger.store("Synthetic ordinary fact", "fact", .8, (source,))
    def revoke(ledger):
        seed(ledger)
        ledger.delete_conversation(source.conversation_id, MemoryReviewer("synthetic-reviewer"))
    capture = (("INSERT INTO utterances VALUES('capture',?,?,?,0,NULL,?,?)",
        (source.capture_id, source.conversation_id, source.expires_at, source.expires_at, "historical raw text")),
        ("INSERT INTO capture_conversations VALUES(?,?)", (source.conversation_id, source.captured_at)))
    ordinary = _image(tmp_path, "O", statements=capture, seed=seed)
    latest = _image(tmp_path, "N", statements=capture, seed=revoke)
    selected, policy = _select(ordinary, latest)
    assert not selected["utterances"]
    deleted, = (image.values for image in selected["facts"].values())
    assert deleted["state"] == "deleted" and deleted["content"] is None
    assert source.conversation_id in policy.revoked


def test_same_capture_id_with_changed_scope_is_not_relabelled(tmp_path):
    source = memory_source(now=NOW-1)
    original = json.dumps([source.to_dict()])
    altered = source.to_dict()
    altered["expires_at"] -= 1
    def event(encoded):
        return (("INSERT INTO events_policy_v1 VALUES(1,?,?)", (encoded, NOW+50)),)
    ordinary = _image(tmp_path, "O", statements=event(original))
    latest = _image(tmp_path, "N", statements=event(json.dumps([altered])))
    with pytest.raises(ValueError, match="payload_source_identity_changed"):
        _select(ordinary, latest)


def test_sql_updates_do_not_replace_rows_or_cascade_manual_children(tmp_path):
    path = tmp_path / "rows.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript("CREATE TABLE events(id INTEGER PRIMARY KEY,body TEXT);"
            "CREATE TABLE decisions(id INTEGER PRIMARY KEY,event_id INTEGER REFERENCES events(id) ON DELETE CASCADE);"
            "INSERT INTO events VALUES(1,'N'),(2,'N-only');INSERT INTO decisions VALUES(7,1);")
        cut = payload.TableCut("rows.db", "events", ("id",), (), ())
        payload._replace_payload_rows(db, cut, [{"id": 1, "body": "N"}, {"id": 2, "body": "N-only"}],
            [{"id": 1, "body": "O"}])
        assert list(db.execute("SELECT * FROM decisions")) == [(7, 1)]
        assert list(db.execute("SELECT * FROM events")) == [(1, "O")]


def test_materializer_requires_real_closed_binding_before_clone(tmp_path):
    from scripts.stage_runtime import AdmissionController
    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    with AdmissionController(tmp_path / "stage") as controller:
        with pytest.raises((ValueError, FileNotFoundError)):
            payload.materialize_derived_state(plan, destination, controller=controller)
    assert not destination.exists()


def test_exact_sdk_records_and_real_public_close_after_cut(tmp_path):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(tmp_path / "index"), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.create_collection(name="derivation_policy_v1", embedding_function=None)
        collection.add(ids=["U", "N-only"], documents=["N ordinary", "new item"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{"fact_policy": 1, "revision": 1}]*2)
        before = payload._sdk_records(collection)
        chosen = [dict(row, document="O ordinary") for row in before if row["id"] == "U"]
        cut = payload.IndexCut("index", "derivation_policy_v1", "unused", True, tuple(before), tuple(chosen))
        payload._edit_collection(client, (cut,))
        assert payload._sdk_records(collection) == chosen
    finally:
        client.close()


def _cut_history(latest, ordinary, row, *, row_hash=None, mutate=None):
    """Hash-qualified image codec fixture, never an admission capability."""
    descriptor = {"path": str(payload._root(ordinary) / "data/graph/events.db"), "table": "events_policy_v1"}
    target = hashlib.sha256(json.dumps(["events", 1,
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"))]).encode()).hexdigest()
    proof = {"schema_version": 1, "kind": "ordinary_payload_cut",
        "ordinary_binding_id": ordinary.source_binding_id, "ordinary_manifest_sha256": ordinary.manifest_sha256,
        "latest_binding_id": ordinary.source_binding_id, "latest_manifest_sha256": ordinary.manifest_sha256,
        "plan_sha256": hashlib.sha256(b"codec-only synthetic plan, not an admission proof").hexdigest(),
        "destination_root": latest.source_binding["state_root"],
        "destination_identity": latest.source_binding["state_identity"],
        "removed_rows": [{"database": PRIVACY, "table": "events_policy_v1", "primary_key": ["id"],
            "identity": [row["id"]], "row_sha256": row_hash or payload._digest(row), "family": "events"}],
        "removed_files": []}
    if mutate is not None:
        mutate(proof)
    archive = {"schema_version": 1, "source_binding": ordinary.source_binding_id,
        "source_binding_record": ordinary.source_binding, "snapshot_manifest_sha256": ordinary.manifest_sha256,
        "current_execution_grant": False,
        "targets": [{"id": target, "participant": "events", "version": 1, "descriptor": descriptor}],
        "payload_cut_proof": proof}
    raw = snapshots._json(archive) + b"\n"
    sha = hashlib.sha256(raw).hexdigest()
    reference = {"source_binding": ordinary.source_binding_id, "target_id": target, "archive_sha256": sha}
    ownership = {"schema_version": 1, "source_bindings": [ordinary.source_binding_id],
        "owned_targets": [], "delegated_targets": [], "historical_targets": [reference]}
    policy = snapshots._json(ownership) + b"\n"
    manifest = json.loads(json.dumps(latest.manifest))
    binding = dict(latest.source_binding, ownership_sha256=hashlib.sha256(policy).hexdigest())
    identity = payload._digest(binding)
    manifest.update(source_binding_id=identity, source_binding=binding)
    archive_key = "history/" + sha + ".json"
    for key, value, kind in ((archive_key, raw, "history_archive"),
            ("state/.kza-target-ownership.json", policy, "file")):
        path = latest.snapshot_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        manifest["files"][key] = {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value), "kind": kind}
    manifest["history_archives"] = {sha: {"path": archive_key, "sha256": sha, "size": len(raw),
        "source_binding": ordinary.source_binding_id, "references": [reference], "source_path": "/absent-original",
        "source_identity": None}}
    (latest.snapshot_root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    return snapshots.ClosedStateSnapshot(latest.snapshot_root, identity, binding, manifest)


@pytest.mark.parametrize("qualified", [True, False])
def test_forward_absence_requires_exact_row_from_binding_pinned_archive(tmp_path, qualified):
    source = memory_source(now=NOW-1)
    ordinary = _image(tmp_path, "N", statements=(("INSERT INTO events_policy_v1 VALUES(1,?,?)",
        (json.dumps([source.to_dict()]), source.expires_at)),))
    latest = _image(tmp_path, "R")
    row, = ordinary.manifest["databases"][PRIVACY]["tables"]["events_policy_v1"]["rows"]
    latest = _cut_history(latest, ordinary, row, row_hash=None if qualified else "0"*64)
    if not qualified:
        with pytest.raises(ValueError, match="payload_ordinary_absence_ambiguous"):
            _select(ordinary, latest)
    else:
        selected, _ = _select(ordinary, latest)
        restored, = selected["events_policy_v1"].values()
        assert restored.values == row and restored.source == "ordinary"
        assert json.loads(restored.values["sources_json"]) == [source.to_dict()]


@pytest.mark.parametrize("mutation,code", [
    ("missing_field", "schema_invalid"), ("bool_version", "schema_invalid"),
    ("bad_digest", "digest_invalid"), ("inode_bool", "destination_invalid"),
    ("identity_bool", "identity_invalid"), ("wrong_family", "row_invalid"),
    ("source_link", "source_mismatch"), ("ordinary_scope", "ordinary_source_missing"),
    ("wrong_pk", "identity_invalid"),
])
def test_qualified_archive_cannot_supply_incomplete_or_out_of_scope_cut(tmp_path, mutation, code):
    source = memory_source(now=NOW-1)
    ordinary = _image(tmp_path, "N", statements=(("INSERT INTO events_policy_v1 VALUES(1,?,?)",
        (json.dumps([source.to_dict()]), source.expires_at)),))
    latest = _image(tmp_path, "R")
    row, = ordinary.manifest["databases"][PRIVACY]["tables"]["events_policy_v1"]["rows"]
    def mutate(proof):
        if mutation == "missing_field":
            del proof["plan_sha256"]
        elif mutation == "bool_version":
            proof["schema_version"] = True
        elif mutation == "bad_digest":
            proof["plan_sha256"] = "A" * 64
        elif mutation == "inode_bool":
            proof["destination_identity"] = dict(proof["destination_identity"], inode=True)
        elif mutation == "identity_bool":
            proof["removed_rows"][0]["identity"] = [True]
        elif mutation == "wrong_family":
            proof["removed_rows"][0]["family"] = "voices"
        elif mutation == "source_link":
            proof["latest_binding_id"] = "0" * 64
        elif mutation == "ordinary_scope":
            proof["ordinary_binding_id"] = "0" * 64
        else:
            proof["removed_rows"][0]["primary_key"] = ["not_the_actual_pk"]
    latest = _cut_history(latest, ordinary, row, mutate=mutate)
    with pytest.raises(ValueError, match="payload_cut_history_" + code):
        payload._qualified_exclusions(ordinary, latest)


def test_forward_context_uses_pinned_N_bytes_when_R_absence_is_qualified(tmp_path, monkeypatch):
    from src.orchestrator.context_persister import ContextPersister
    source = memory_source(now=NOW-1)
    event = (("INSERT INTO events_policy_v1 VALUES(1,?,?)",
        (json.dumps([source.to_dict()]), source.expires_at)),)
    ordinary, latest = _image(tmp_path, "N", statements=event), _image(tmp_path, "R", statements=event)
    persister = ContextPersister(payload._root(ordinary) / "data/graph/contexts", clock=lambda: NOW)
    document = {"version": 3, "user_id": "N_only", "user_name": "Synthetic user", "last_seen": NOW-1,
        "session_count": 1, "compacted_summary": "N-only ordinary summary", "preserved_ids": [],
        "summary_sources": [source.to_dict()]}
    persister.save_payload(document)
    actual = persister.base_path / "N_only.json"
    captured = actual.read_bytes()
    relative = actual.relative_to(payload._root(ordinary)).as_posix()
    key = "state/" + relative
    copy = ordinary.snapshot_root / key
    copy.parent.mkdir(parents=True)
    copy.write_bytes(captured)
    manifest = json.loads(json.dumps(ordinary.manifest))
    manifest["files"][key] = {"sha256": hashlib.sha256(captured).hexdigest(), "size": len(captured),
        "kind": "file", "source_path": str(actual), "source_identity": snapshots._identity(actual)}
    (ordinary.snapshot_root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    ordinary = snapshots.ClosedStateSnapshot(ordinary.snapshot_root, ordinary.source_binding_id,
        ordinary.source_binding, manifest)
    layout = _layout(tmp_path / "F")
    assert key not in latest.manifest["files"]
    with pytest.raises(ValueError, match="payload_ordinary_absence_ambiguous"):
        payload.plan_payload_cut(ordinary, latest, now=NOW, layout=layout)
    row, = ordinary.manifest["databases"][PRIVACY]["tables"]["events_policy_v1"]["rows"]
    def file_proof(proof):
        proof["removed_rows"] = []
        proof["removed_files"] = [{"relative_path": relative, "sha256": hashlib.sha256(captured).hexdigest(),
            "family": "contexts"}]
    latest = _cut_history(latest, ordinary, row, mutate=file_proof)
    read_file = payload._read_file
    def checked_read(snapshot, relative_key):
        assert not (snapshot is latest and relative_key == key), "absent R context must never be read"
        return read_file(snapshot, relative_key)
    monkeypatch.setattr(payload, "_read_file", checked_read)
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=layout)
    selected, = (cut for cut in plan.files if cut.destination_relative == relative)
    assert selected.source == "ordinary" and selected.snapshot_relative == key
    assert read_file(ordinary, selected.snapshot_relative) == captured
    assert json.loads(captured) == document
    assert json.loads(captured)["summary_sources"] == [source.to_dict()]
    assert not layout.destination_root.exists()


def test_sdk_close_error_after_real_release_keeps_unconfirmed_receipt(tmp_path, monkeypatch):
    import chromadb
    factory = chromadb.PersistentClient
    clients = []
    def delegated(**kwargs):
        client = factory(**kwargs)
        client.create_collection(name="derivation_policy_v1", embedding_function=None)
        close = client.close
        def fail_after_return():
            close()
            raise OSError("synthetic metadata failure after real release")
        monkeypatch.setattr(client, "close", fail_after_return)
        clients.append(client)
        return client
    cut = payload.IndexCut("index", "derivation_policy_v1", payload.importlib.metadata.version("chromadb"),
        True, (), ())
    reservation = {"sdk": []}
    with pytest.raises(OSError, match="synthetic metadata failure"):
        payload._apply_indexes(SimpleNamespace(indexes=(cut,)), tmp_path, reservation, delegated)
    owner, = reservation["sdk"]
    assert owner["client"] is clients[0]
    assert owner["close_attempted"] and not owner["close_returned"]
    # The delegate really closed. No retry and no assertion that this error
    # means the SDK is still alive; only its public receipt is unconfirmed.
