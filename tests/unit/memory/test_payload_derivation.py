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
from src.memory.fact_ledger import FactDependency, FactLedger, MemoryAssigner, MemoryOwnerActor, MemoryReviewer
from tests.mocks.memory_sources import memory_source


NOW = 1000.0
PRIVACY = "data/graph/privacy.db"
DIAGNOSTICS = "data/graph/metrics.jsonl.sqlite3"
EXTRA_SCHEMA = """
CREATE TABLE utterances(id INTEGER PRIMARY KEY AUTOINCREMENT,room_id TEXT NOT NULL,
 t0 REAL NOT NULL,t1 REAL NOT NULL,text TEXT NOT NULL,speaker TEXT NOT NULL DEFAULT 'unknown',
 speaker_confidence REAL NOT NULL DEFAULT 0,azimuth REAL,azimuth_stability REAL NOT NULL DEFAULT 0,
 source TEXT NOT NULL DEFAULT 'unknown',confidence REAL,no_speech_prob REAL,vad_prob REAL,
 lang TEXT,lang_prob REAL,lang_ok INTEGER,during_tts INTEGER NOT NULL DEFAULT 0,
 distilled INTEGER NOT NULL DEFAULT 0,text_empty INTEGER NOT NULL DEFAULT 0,audio_path TEXT,
 capture_id TEXT,expires_at REAL,conversation_id TEXT,created_at REAL NOT NULL);
CREATE TABLE capture_conversations(id TEXT PRIMARY KEY,room_id TEXT NOT NULL,
 started_at REAL NOT NULL,last_seen_at REAL NOT NULL);
CREATE TABLE wake_clips_v1(sequence INTEGER PRIMARY KEY,id TEXT UNIQUE,target_id TEXT,
 bucket TEXT,room_id TEXT,score REAL,sources_json TEXT,captured_at REAL,expires_at REAL,published INTEGER);
CREATE TABLE stt_corrections_v1(id TEXT PRIMARY KEY,corrected_text TEXT,user_id TEXT,used INTEGER);
CREATE TABLE conversation_turns_v1(id TEXT PRIMARY KEY,quality TEXT);
CREATE TABLE conversation_export_parents_v1(export_id TEXT,parent_id TEXT,PRIMARY KEY(export_id,parent_id));
CREATE TABLE events_policy_v1(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp REAL NOT NULL,
 event_type TEXT NOT NULL,entity_id TEXT NOT NULL,action TEXT NOT NULL,user_id TEXT,user_name TEXT,
 trigger_phrase TEXT,context_json TEXT,hour INTEGER,minute INTEGER,weekday INTEGER,is_weekend INTEGER,
 time_of_day TEXT,sources_json TEXT NOT NULL,metrics_json TEXT NOT NULL,expires_at REAL NOT NULL,
 text_expires_at REAL);
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


def _image(tmp_path, name, *, statements=(), seed=None, diagnostic_seed=None, indexed=True):
    source = tmp_path / (name + "-source")
    (source / "data/graph").mkdir(parents=True)
    ledger = FactLedger(str(source / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        if seed is not None:
            seed(ledger)
        if diagnostic_seed is not None:
            diagnostic_seed(ledger, source / "data/graph/metrics.jsonl")
    finally:
        ledger.close()
    with closing(sqlite3.connect(source / PRIVACY)) as db, db:
        db.executescript(EXTRA_SCHEMA.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS "))
        for sql, values in statements:
            db.execute(sql, values)
    with closing(sqlite3.connect(source / DIAGNOSTICS)) as db, db:
        db.executescript(("CREATE TABLE metrics(id TEXT PRIMARY KEY,born REAL,expires REAL,payload TEXT,"
            "partition INTEGER,bytes INTEGER);CREATE TABLE lineage(record_id TEXT,conversation_id TEXT,"
            "PRIMARY KEY(record_id,conversation_id));CREATE TABLE state(key TEXT PRIMARY KEY,value INTEGER);")
            .replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS "))
    return _snapshot_codec(source, tmp_path / (name + "-image"), (PRIVACY, DIAGNOSTICS), indexed=indexed)


def _snapshot_codec(source, root, database_paths, *, indexed=True):
    """Pin real table bytes for codec tests, without an admission capability.

    indexed="live" exports one exact index record per live automatic/confirmed
    fact (the reconciled state), for plans that keep such facts.
    """
    databases, files, schemas, contracts = {}, {}, {}, {}
    for relative in database_paths:
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
            "relative_path": "data/graph/chroma", "collection": "unit_policy_v1", "present": indexed == "live",
            "metadata": None, "sdk_version": None, "fields": ["ids", "documents", "metadatas", "embeddings"],
            "records": _live_index_records(source) if indexed == "live" else []}] if indexed else []}
    (root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    return snapshots.ClosedStateSnapshot(root, identity, binding, manifest)


def _live_index_records(source):
    with closing(sqlite3.connect(source / PRIVACY)) as db:
        return [{"id": identity, "document": content, "metadata": {"fact_policy": 1, "revision": revision},
                 "embedding": [1.0, 0.0]} for identity, content, revision in db.execute(
                     "SELECT id,content,revision FROM facts WHERE state IN ('automatic','confirmed') ORDER BY id")]


def _layout(destination):
    roots = (("ambient_capture", "data/graph/audio"), ("collectors", "data/graph/clips"),
        ("contexts", "data/graph/contexts"), ("diagnostics", "data/graph/metrics.jsonl"))
    return payload.DerivationLayout(destination, PRIVACY,
        tuple(payload.DatabaseRelocation(path, path, path) for path in (PRIVACY, DIAGNOSTICS)),
        tuple(payload.FileRootRelocation(family, path, path, path) for family, path in roots),
        (payload.IndexRelocation("data/graph/chroma", "data/graph/chroma", "data/graph/chroma", "unit_policy_v1"),), {})


def _event_statement(encoded, expiry):
    return ("INSERT INTO events_policy_v1(id,timestamp,event_type,entity_id,action,sources_json,"
        "metrics_json,expires_at,text_expires_at) VALUES(1,999,'command','light.synthetic','turn_on',?,'{}',?,?)",
        (encoded, expiry, expiry))


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


def test_shortened_latest_deadlines_are_never_renewed(tmp_path):
    from src.analytics.event_storage import event_descriptor, sweep_events
    source = memory_source(now=NOW-1)
    statements = (_event_statement(json.dumps([source.to_dict()]), source.expires_at),
        ("UPDATE events_policy_v1 SET user_id='synthetic-account',user_name='Raw synthetic name',"
            "trigger_phrase='Raw synthetic phrase',context_json=?,metrics_json=? WHERE id=1",
            ('{"spoken":"raw synthetic context"}', '{"latency_ms":12}')))
    ordinary = _image(tmp_path, "O", statements=statements)
    row, = ordinary.manifest["databases"][PRIVACY]["tables"]["events_policy_v1"]["rows"]
    ledger = FactLedger(str(payload._root(ordinary) / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        for ordinal, deadline in enumerate((NOW-1, NOW, NOW+5)):
            latest = dict(row, expires_at=NOW+20, text_expires_at=deadline)
            oracle = tmp_path / ("oracle-" + str(ordinal) + ".db")
            snapshots._backup(payload._root(ordinary) / PRIVACY, oracle)
            with closing(sqlite3.connect(oracle)) as db, db:
                db.execute("UPDATE events_policy_v1 SET expires_at=?,text_expires_at=? WHERE id=1",
                    (latest["expires_at"], latest["text_expires_at"]))
            sweep = sweep_events(event_descriptor(oracle), 0, ledger, 90)
            with closing(sqlite3.connect(oracle)) as db:
                db.row_factory = sqlite3.Row
                expected = dict(db.execute("SELECT * FROM events_policy_v1 WHERE id=1").fetchone())
            cut = payload._blend(row, latest, NOW, table="events_policy_v1")
            assert cut == expected
            assert cut["expires_at"] == NOW+20 and cut["text_expires_at"] == deadline
            assert sweep.scrubbed == int(deadline <= NOW)
            if deadline <= NOW:
                assert cut["trigger_phrase"] is None and cut["user_name"] is None
                assert cut["context_json"] == cut["metrics_json"]
            else:
                assert cut["trigger_phrase"] == row["trigger_phrase"] and cut["context_json"] == row["context_json"]
            assert cut["timestamp"] == row["timestamp"] and cut["user_id"] == "synthetic-account"
            assert json.loads(cut["sources_json"]) == [source.to_dict()]
    finally:
        ledger.close()


def test_real_tombstone_blocks_raw_capture_and_retains_deleted_fact(tmp_path):
    source = memory_source(now=NOW-1)
    def seed(ledger):
        ledger.store("Synthetic ordinary fact", "fact", .8, (source,))
    def revoke(ledger):
        seed(ledger)
        ledger.delete_conversation(source.conversation_id, MemoryReviewer("synthetic-reviewer"))
    capture = (("INSERT INTO utterances(id,room_id,t0,t1,text,capture_id,expires_at,conversation_id,created_at)"
        " VALUES(1,'synthetic-room',?,?,'historical raw text',?,?,?,?)",
        (source.captured_at, NOW, source.capture_id, source.expires_at, source.conversation_id, source.captured_at)),
        ("INSERT INTO capture_conversations VALUES(?,'synthetic-room',?,?)",
            (source.conversation_id, source.captured_at, NOW)))
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
        return (_event_statement(encoded, NOW+50),)
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


def _cut_history(latest, ordinary, row, *, row_hash=None, mutate=None,
                 table="events_policy_v1", family="events", target_records=None):
    """Hash-qualified image codec fixture, never an admission capability."""
    descriptor = {"path": str(payload._root(ordinary) / "data/graph/events.db"), "table": "events_policy_v1"}
    target = hashlib.sha256(json.dumps(["events", 1,
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"))]).encode()).hexdigest()
    keys = ordinary.manifest["databases"][PRIVACY]["tables"][table]["primary_key"]
    targets = ([{"id": target, "participant": "events", "version": 1, "descriptor": descriptor}]
        if target_records is None else target_records)
    proof = {"schema_version": 1, "kind": "ordinary_payload_cut",
        "ordinary_binding_id": ordinary.source_binding_id, "ordinary_manifest_sha256": ordinary.manifest_sha256,
        "latest_binding_id": ordinary.source_binding_id, "latest_manifest_sha256": ordinary.manifest_sha256,
        "plan_sha256": hashlib.sha256(b"codec-only synthetic plan, not an admission proof").hexdigest(),
        "destination_root": latest.source_binding["state_root"],
        "destination_identity": latest.source_binding["state_identity"],
        "removed_rows": [{"database": PRIVACY, "table": table, "primary_key": list(keys),
            "identity": [row[key] for key in keys], "row_sha256": row_hash or payload._digest(row), "family": family}],
        "removed_files": []}
    if mutate is not None:
        mutate(proof)
    archive = {"schema_version": 1, "source_binding": ordinary.source_binding_id,
        "source_binding_record": ordinary.source_binding, "snapshot_manifest_sha256": ordinary.manifest_sha256,
        "current_execution_grant": False,
        "targets": targets,
        "payload_cut_proof": proof}
    raw = snapshots._json(archive) + b"\n"
    sha = hashlib.sha256(raw).hexdigest()
    references = [{"source_binding": ordinary.source_binding_id, "target_id": item["id"], "archive_sha256": sha}
        for item in targets]
    ownership = {"schema_version": 1, "source_bindings": [ordinary.source_binding_id],
        "owned_targets": [], "delegated_targets": [], "historical_targets": references}
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
        "source_binding": ordinary.source_binding_id, "references": references, "source_path": "/absent-original",
        "source_identity": None}}
    (latest.snapshot_root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    return snapshots.ClosedStateSnapshot(latest.snapshot_root, identity, binding, manifest)


@pytest.mark.parametrize("qualified", [True, False])
def test_forward_absence_requires_exact_row_from_binding_pinned_archive(tmp_path, qualified):
    source = memory_source(now=NOW-1)
    ordinary = _image(tmp_path, "N", statements=(
        _event_statement(json.dumps([source.to_dict()]), source.expires_at),))
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
    ordinary = _image(tmp_path, "N", statements=(
        _event_statement(json.dumps([source.to_dict()]), source.expires_at),))
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
    event = (_event_statement(json.dumps([source.to_dict()]), source.expires_at),)
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


@pytest.mark.parametrize("condition", ["retained", "revoked", "dependency_deleted", "metadata_expired"])
def test_real_numeric_diagnostics_keep_anchored_90d_and_activity_authority(tmp_path, condition):
    from src.obs.diagnostic_store import DiagnosticStore
    from src.obs.metrics_emitter import project_diagnostic
    original_source = memory_source(now=NOW-1)
    parent_source = memory_source(now=NOW-1)
    sources, parents, records = [], [], []
    def seed(ledger):
        parent = ledger.store("Synthetic diagnostic dependency", "fact", .9, (parent_source,))
        source = original_source
        if condition == "dependency_deleted":
            source = replace(source, fact_dependencies=(FactDependency(parent.id, parent.revision, parent.expires_at),))
        if sources:
            assert sources[0] == source and parents[0] == parent.id
        else:
            sources.append(source)
            parents.append(parent.id)
        ledger.store("Synthetic diagnostic source metadata", "fact", .9, (source,))
    def publish(ledger, path):
        store = DiagnosticStore(path, clock=lambda: NOW, retention_days=90, rotate_bytes=1024*1024,
            project=project_diagnostic, policy=ledger)
        try:
            if not records:
                admitted = store.record({"event_type": "request", "success": True, "total_ms": 12.,
                    "text": "raw synthetic transcript", "user_id": "synthetic-user", "secret": "synthetic-secret"},
                    (sources[0],))
                assert admitted is not None
                records.append(admitted)
            publication = store.append(records[0])
            assert publication is not None
            assert publication.expires_at == records[0].recorded_at + 90*86400
            assert publication.expires_at > sources[0].expires_at
            assert set(records[0].payload) == {"event_type", "diagnostic_schema", "success", "total_ms"}
            assert "raw synthetic" not in path.read_text() and "synthetic-secret" not in path.read_text()
        finally:
            store.close()
        assert store.status()["writer_lease"] is False
    ordinary = _image(tmp_path, "O", seed=seed, diagnostic_seed=publish)
    def latest_publish(ledger, path):
        publish(ledger, path)
        if condition == "revoked":
            ledger.delete_conversation(sources[0].conversation_id, MemoryReviewer("synthetic-reviewer"))
        elif condition == "dependency_deleted":
            ledger.delete_fact(parents[0], MemoryReviewer("synthetic-reviewer"))
    latest = _image(tmp_path, "N", seed=seed, diagnostic_seed=latest_publish)
    admitted = records[0]
    now = admitted.expires_at if condition == "metadata_expired" else sources[0].expires_at + 1
    policy = payload._Policy(ordinary, latest, ordinary.manifest["databases"][PRIVACY],
        latest.manifest["databases"][PRIVACY], now)
    policy.database = DIAGNOSTICS
    selected = payload._select_database(ordinary.manifest["databases"][DIAGNOSTICS],
        latest.manifest["databases"][DIAGNOSTICS], policy)
    authority = FactLedger(str(payload._root(latest) / PRIVACY), clock=lambda: now)
    authority.initialize()
    checker = DiagnosticStore(tmp_path / "never-opened.jsonl", clock=lambda: now, retention_days=90,
        rotate_bytes=1024*1024, project=project_diagnostic, policy=authority)
    try:
        assert policy.activity(tuple(sources)) == checker.available(tuple(sources))
        assert policy.live(tuple(sources)) is False
        assert bool(selected["metrics"]) is (condition == "retained")
        if condition == "retained":
            image, = selected["metrics"].values()
            assert image.values["born"] == admitted.recorded_at
            assert image.values["expires"] == admitted.expires_at == NOW + 90*86400
            assert json.loads(image.values["payload"]) == admitted.payload
            lineage, = selected["lineage"].values()
            assert lineage.values == {"record_id": admitted.identifier, "conversation_id": sources[0].conversation_id}
            assert sources[0].capture_id == original_source.capture_id and sources[0].expires_at == original_source.expires_at
    finally:
        checker.close()
        authority.close()


@pytest.mark.parametrize("failure_phase", ["qualification_receipt", "payload_beforeimage"])
def test_failed_preparation_keeps_real_sidecar_inadmissible(tmp_path, failure_phase):
    """Exercise preparation below admission, without manufacturing close ACKs."""
    from scripts.stage_runtime import validate_target_ownership
    from src.memory.erasure_journal import ErasureParticipant
    from src.memory.state_journal_derivation import JournalDerivationReceipt
    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    payload._clone_latest(plan, destination)
    ledger = FactLedger(str(destination / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        registered = ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,
            {"kind": "wake_clips", "root": str(destination / "data/graph/wake")})
    finally:
        ledger.close()
    # Real registered SQL and a physical blocked sidecar. This preparation DTO
    # grants no controller run, writer closure, SDK return or cleanup receipt.
    sidecar = destination / ".kza-target-ownership.json"
    blocked = snapshots._json({"schema_version": 1, "source_bindings": [], "owned_targets": [],
        "delegated_targets": [], "historical_targets": []}) + b"\n"
    sidecar.write_bytes(blocked)
    journal = JournalDerivationReceipt(latest.source_binding_id, destination, PRIVACY,
        "0"*64, tmp_path / "unissued-history.json", (), {}, (registered,), {})
    if failure_phase == "qualification_receipt":
        journal = replace(journal, registered_target_ids=("1"*64,))
        code = "payload_journal_receipt_mismatch"
    else:
        # An unaccounted payload write after qualification must fail the real
        # SQLite beforeimage check, while the registered inventory stays sealed.
        source = memory_source(now=NOW-1)
        sql, values = _event_statement(json.dumps([source.to_dict()]), source.expires_at)
        with closing(sqlite3.connect(destination / PRIVACY)) as db, db:
            db.execute(sql, values)
        code = "payload_destination_payload_changed"
    with pytest.raises(ValueError, match=code):
        payload._finish_prepared_payload(plan, destination, journal, {"sdk": []}, None)
    assert sidecar.read_bytes() == blocked
    binding = {"state_root": str(destination), "ownership_sha256": hashlib.sha256(blocked).hexdigest()}
    with pytest.raises(ValueError, match="runtime_admission_active_inventory_mismatch"):
        validate_target_ownership(tmp_path, binding)
    assert not (destination / ".kza-payload-preparation.json").exists()


@pytest.mark.asyncio
async def test_real_mixed_export_and_all_parents_stay_indivisible_in_R_and_F(tmp_path):
    """Real collector output; table replay below admission, not a fake Graph."""
    from pathlib import Path
    from tests.mocks.conversations import graph
    async with graph(tmp_path / "live") as g:
        async def image(name, database=None):
            root = tmp_path / name
            path = root / PRIVACY
            def copy():
                snapshots._backup(Path(database or g.privacy.ledger.path), path)
                return snapshots._database(path, root)
            data = await g.privacy.acall(copy)
            # A data-only codec view, without any closed-state capability.
            return SimpleNamespace(snapshot_root=root, manifest={"databases": {PRIVACY: data}})
        assert await g.add(user_input="O ordinary input", assistant_response="O ordinary output")
        ordinary = await image("O")
        cursor = max(row["sequence"] for row in ordinary.manifest["databases"][PRIVACY]["tables"]["wake_clips_v1"]["rows"])
        mixed_id = await g.add(user_input="N mixed unmarked input", assistant_response="N mixed unmarked output")
        old_manual = await g.add(user_input="N reviewed input", assistant_response="N reviewed output")
        views = await g.collector.get_turns()
        try:
            manual_view = next(view for view in views if view.id == old_manual)
            assert await g.collector.mark_response(manual_view, "corrected", "Latest manual correction",
                sources=(g.base.source(),))
        finally:
            for view in views:
                view.clear()
        views = await g.collector.get_turns()
        try:
            manual, = (view for view in views if view.quality.value == "corrected")
            manual_id = manual.id
            assert manual_id != old_manual and manual.correction == "Latest manual correction"
            assert next(view for view in views if view.id == mixed_id).quality.value == "unmarked"
        finally:
            for view in views:
                view.clear()
        exported = await g.collector.export_for_training(str(tmp_path / "live/exports"),
            only_marked=False, after_sequence=cursor)
        assert exported is not None
        try:
            assert exported.available
            export_id, export_path = exported.id, Path(exported.output_path)
            export_bytes = export_path.read_bytes()
            assert b"Latest manual correction" in export_bytes and b"N mixed unmarked output" in export_bytes
        finally:
            exported.clear()
        latest = await image("N")
        n_tables = latest.manifest["databases"][PRIVACY]["tables"]
        edges = [row for row in n_tables["conversation_export_parents_v1"]["rows"] if row["export_id"] == export_id]
        assert {row["parent_id"] for row in edges} == {manual_id, mixed_id}
        original_rows = {row["id"]: row for row in n_tables["wake_clips_v1"]["rows"]}
        async def replay(older, current, name):
            before = current.manifest["databases"][PRIVACY]
            policy = payload._Policy(older, current, older.manifest["databases"][PRIVACY], before, g.clock.now)
            policy.database = PRIVACY
            selected = payload._select_database(older.manifest["databases"][PRIVACY], before, policy)
            clips = {item.values["id"]: item.values for item in selected["wake_clips_v1"].values()}
            assert {export_id, manual_id, mixed_id} <= clips.keys()
            for identity in (export_id, manual_id, mixed_id):
                assert clips[identity] == original_rows[identity]
            selected_edges = [item.values for item in selected["conversation_export_parents_v1"].values()
                if item.values["export_id"] == export_id]
            assert sorted(selected_edges, key=payload._json) == sorted(edges, key=payload._json)
            output = await image(name, current.snapshot_root / PRIVACY)
            cuts = tuple(payload.TableCut(PRIVACY, table, tuple(before["tables"][table]["primary_key"]),
                tuple(before["tables"][table]["rows"]), tuple(values.values())) for table, values in selected.items())
            targets = {row["id"]: row["id"] for row in before["tables"]["erasure_targets"]["rows"]}
            codec_plan = SimpleNamespace(tables=cuts, ordinary=older, latest=current)
            journal_map = SimpleNamespace(target_id_map=targets, event_dependency_target_map={})
            await g.privacy.acall(payload._apply_tables, codec_plan, output.snapshot_root, journal_map)
            return await image(name + "-result", output.snapshot_root / PRIVACY)
        rollback = await replay(ordinary, latest, "R")
        forward = await replay(latest, rollback, "F")
        assert export_path.read_bytes() == export_bytes
        for point in (rollback, forward):
            table = point.manifest["databases"][PRIVACY]["tables"]
            actual = [row for row in table["conversation_export_parents_v1"]["rows"] if row["export_id"] == export_id]
            assert sorted(actual, key=payload._json) == sorted(edges, key=payload._json)


@pytest.mark.asyncio
@pytest.mark.parametrize("table,record,denied", [
    ("wake_clips_v1", "mixed_export", True),
    ("wake_clips_v1", "mixed_parent", True),
    ("conversation_turns_v1", "mixed_parent", True),
    ("conversation_export_parents_v1", "mixed_parent", True),
    ("conversation_export_parents_v1", "manual_parent", True),
    ("wake_clips_v1", "ordinary_export", False),
    ("conversation_export_parents_v1", "ordinary_parent", False),
])
async def test_qualified_history_cannot_exclude_real_manual_export_component(tmp_path, table, record, denied):
    """Exact qualified codec proof, without manufacturing a runtime close ACK."""
    from tests.mocks.conversations import graph
    source = tmp_path / "N-source"
    async with graph(source / "data/graph") as g:
        mixed_parent = await g.add(user_input="Mixed unmarked input", assistant_response="Mixed unmarked output")
        old_manual = await g.add(user_input="Manual input", assistant_response="Manual output")
        views = await g.collector.get_turns()
        try:
            selected = next(view for view in views if view.id == old_manual)
            assert await g.collector.mark_response(selected, "corrected", "Manual correction",
                sources=(g.base.source(),))
        finally:
            for view in views:
                view.clear()
        views = await g.collector.get_turns()
        try:
            selected, = (view for view in views if view.quality.value == "corrected")
            manual_parent = selected.id
            assert manual_parent != old_manual and selected.available
        finally:
            for view in views:
                view.clear()
        mixed = await g.collector.export_for_training(str(source / "data/graph/exports"), only_marked=False)
        assert mixed is not None
        try:
            assert mixed.available
            mixed_export = mixed.id
        finally:
            mixed.clear()
        cursor = max(row["sequence"] for row in await g.base.rows())
        ordinary_parent = await g.add(user_input="Independent ordinary input", assistant_response="Independent ordinary output")
        ordinary_export = await g.collector.export_for_training(str(source / "data/graph/exports"),
            only_marked=False, after_sequence=cursor)
        assert ordinary_export is not None
        try:
            assert ordinary_export.available
            ordinary_export_id = ordinary_export.id
        finally:
            ordinary_export.clear()
    assert not g.privacy.resource_status()["writer_lease"]
    # Snapshot only after actual fixture owners have closed. These image DTOs
    # are data codecs; they cannot be used as admitted controller witnesses.
    ordinary = _snapshot_codec(source, tmp_path / "N-image", (PRIVACY,), indexed=False)
    latest = _image(tmp_path, "R")
    tables = ordinary.manifest["databases"][PRIVACY]["tables"]
    edges = tables["conversation_export_parents_v1"]["rows"]
    assert {row["parent_id"] for row in edges if row["export_id"] == mixed_export} == {manual_parent, mixed_parent}
    assert {row["parent_id"] for row in edges if row["export_id"] == ordinary_export_id} == {ordinary_parent}
    identities = {"mixed_export": mixed_export, "mixed_parent": mixed_parent,
        "manual_parent": manual_parent, "ordinary_export": ordinary_export_id, "ordinary_parent": ordinary_parent}
    if table == "conversation_export_parents_v1":
        export_id = ordinary_export_id if record == "ordinary_parent" else mixed_export
        row, = (item for item in tables[table]["rows"]
            if item["export_id"] == export_id and item["parent_id"] == identities[record])
    else:
        row, = (item for item in tables[table]["rows"] if item["id"] == identities[record])
    targets = [dict(item, descriptor=json.loads(item["descriptor"])) for item in tables["erasure_targets"]["rows"]]
    latest = _cut_history(latest, ordinary, row, table=table, family="collectors", target_records=targets)
    if denied:
        with pytest.raises(ValueError, match="payload_cut_history_manual_row"):
            payload._qualified_exclusions(ordinary, latest)
    else:
        excluded = payload._qualified_exclusions(ordinary, latest)
        key = (PRIVACY, table, payload._digest(row))
        assert set(excluded) == {key}
        assert excluded[key]["identity"] == [row[name] for name in tables[table]["primary_key"]]
    assert snapshots.verify_snapshot(ordinary) == ordinary.manifest


def test_catalog_reservation_rejection_precedes_clone_and_ram_entry(tmp_path, monkeypatch):
    """Inject a rejected reservation; no capability or close receipt is issued."""
    from scripts.stage_runtime import AdmissionController
    from src.memory import copy_catalog, state_journal_derivation

    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    # Source validation has its own real-binding negative above. Isolate the
    # subsequent rejection boundary without manufacturing a successful binding.
    monkeypatch.setattr(state_journal_derivation, "_source", lambda *args, **kwargs: None)

    def reject(*args, **kwargs):
        assert not destination.exists() and str(destination) not in payload._INCOMPLETE
        raise OSError("catalog reservation unavailable")

    def forbidden_clone(*args):
        pytest.fail("clone dispatched after a rejected catalog reservation")

    monkeypatch.setattr(copy_catalog, "reserve_copy", reject)
    monkeypatch.setattr(payload, "_clone_latest", forbidden_clone)
    with AdmissionController(tmp_path / "stage") as controller:
        with pytest.raises(OSError, match="catalog reservation unavailable"):
            payload.materialize_derived_state(plan, destination, controller=controller)
    assert not destination.exists() and str(destination) not in payload._INCOMPLETE


@pytest.mark.parametrize("distinct_physical_root", [False, True])
def test_catalog_refs_deduplicate_only_the_complete_snapshot_descriptor(
        tmp_path, monkeypatch, distinct_physical_root):
    """Inspect a rejected reservation, never mint a closed-binding capability."""
    from shutil import copytree

    from scripts.stage_runtime import AdmissionController
    from src.memory import copy_catalog, state_journal_derivation

    ordinary = _image(tmp_path, "O")
    if distinct_physical_root:
        root = tmp_path / "O-physical-copy"
        copytree(ordinary.snapshot_root, root)
        latest = snapshots.ClosedStateSnapshot(root, ordinary.source_binding_id,
            ordinary.source_binding, ordinary.manifest)
    else:
        latest = ordinary
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    observed = []

    def reject(controller, kind, target, latest_binding_id, source_binding_ids, *, source_snapshot_refs):
        assert kind == "derived_state" and target == destination and not destination.exists()
        assert latest_binding_id == ordinary.source_binding_id
        assert source_binding_ids == (ordinary.source_binding_id,)
        observed.extend(source_snapshot_refs)
        assert len(observed) == (2 if distinct_physical_root else 1)
        assert all(reference["source_binding_id"] == ordinary.source_binding_id for reference in observed)
        assert all(reference["manifest_sha256"] == ordinary.manifest_sha256 for reference in observed)
        if distinct_physical_root:
            assert observed[0]["snapshot_root"] != observed[1]["snapshot_root"]
            assert observed[0]["root_identity"] != observed[1]["root_identity"]
        raise OSError("catalog reservation deliberately rejected")

    monkeypatch.setattr(state_journal_derivation, "_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(copy_catalog, "reserve_copy", reject)
    with AdmissionController(tmp_path / "stage") as controller:
        with pytest.raises(OSError, match="catalog reservation deliberately rejected"):
            payload.materialize_derived_state(plan, destination, controller=controller)
    assert observed and not destination.exists() and str(destination) not in payload._INCOMPLETE
    assert snapshots.verify_snapshot(ordinary) == ordinary.manifest
    assert snapshots.verify_snapshot(latest) == latest.manifest


@pytest.mark.parametrize("catalog_failure", [False, True])
def test_partial_clone_notifies_catalog_before_local_record(tmp_path, monkeypatch, catalog_failure):
    """Observe caller failure propagation only, without issuing admission/SDK ACKs."""
    from scripts.stage_runtime import AdmissionController
    from src.memory import copy_catalog, state_journal_derivation

    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    events = []
    failure_record = tmp_path / "observed-failure.json"

    class FailureObserver:
        # This fault observer deliberately has no SDK, prepared or cleanup API.
        def create_root(self):
            assert events == ["reserve"]
            destination.mkdir()
            events.append("root")

        def extent(self, relative, *, family, representation):
            assert (relative, family, representation) == (".", "managed_payload", "derived_state")
            events.append("extent")

        def failed(self, code):
            assert code == "OSError"
            assert not (destination / ".kza-payload-preparation.json").exists()
            assert any(path.is_file() for path in destination.rglob("*"))
            failure_record.write_text(json.dumps({"code": code, "cleanup": "pending", "serving": False}))
            events.append("failed")
            if catalog_failure:
                raise OSError("catalog failure marker unavailable")

    def observe_reservation(*args, **kwargs):
        assert not destination.exists()
        events.append("reserve")
        return FailureObserver()

    actual_write = payload._atomic_file
    writes = 0

    def fail_after_first_copy(path, raw):
        nonlocal writes
        assert events == ["reserve", "root", "extent"]
        writes += 1
        if writes == 2:
            raise OSError("second clone write failed")
        actual_write(path, raw)

    monkeypatch.setattr(state_journal_derivation, "_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(copy_catalog, "reserve_copy", observe_reservation)
    monkeypatch.setattr(payload, "_atomic_file", fail_after_first_copy)
    try:
        with AdmissionController(tmp_path / "stage") as controller:
            if catalog_failure:
                with pytest.raises(BaseExceptionGroup) as caught:
                    payload.materialize_derived_state(plan, destination, controller=controller)
                assert [str(error) for error in caught.value.exceptions] == [
                    "second clone write failed", "catalog failure marker unavailable"]
            else:
                with pytest.raises(OSError, match="second clone write failed"):
                    payload.materialize_derived_state(plan, destination, controller=controller)
        assert json.loads(failure_record.read_text()) == {"code": "OSError", "cleanup": "pending", "serving": False}
        assert events == ["reserve", "root", "extent", "failed"]
        assert payload._INCOMPLETE[str(destination)]["phase"] == "failed"
        assert writes == 2
    finally:
        # The observer never entered an SDK or owned native work. Its copied
        # bytes remain in place; this removes only this test's RAM observation.
        payload._INCOMPLETE.pop(str(destination), None)


def test_rejected_catalog_sdk_entry_never_dispatches_factory(tmp_path):
    calls = []

    class RejectEntry:
        def extent(self, relative, *, family, representation):
            assert (relative, family, representation) == ("index", "semantic_index", "sdk_work")
            return object()

        def sdk_enter(self, extent, *, sdk_version):
            raise OSError("SDK entry was not persisted")

    def forbidden_factory(**kwargs):
        calls.append(kwargs)
        pytest.fail("SDK constructor dispatched without durable entry")

    cut = payload.IndexCut("index", "derivation_policy_v1", payload.importlib.metadata.version("chromadb"),
        True, (), ())
    reservation = {"sdk": []}
    with pytest.raises(OSError, match="SDK entry was not persisted"):
        payload._apply_indexes(SimpleNamespace(indexes=(cut,)), tmp_path, reservation, forbidden_factory,
            operation=RejectEntry())
    owner, = reservation["sdk"]
    assert not calls and owner["client"] is None
    assert not owner["entered"] and not owner["close_attempted"] and not owner["close_returned"]


def test_ownership_publication_failure_after_prepared_is_recorded_without_finish(tmp_path, monkeypatch):
    """Fault-observe publication only; no runtime, controller-close or SDK ACK is issued."""
    from scripts.stage_runtime import AdmissionController, validate_target_ownership
    from src.memory import copy_catalog, state_journal_derivation
    from src.memory.erasure_journal import ErasureParticipant
    from src.memory.state_journal_derivation import JournalDerivationReceipt

    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    sidecar = destination / ".kza-target-ownership.json"
    blocked = snapshots._json({"schema_version": 1, "source_bindings": [], "owned_targets": [],
        "delegated_targets": [], "historical_targets": []}) + b"\n"
    failure_record = tmp_path / "publication-failure.json"
    events = []

    class PublicationObserver:
        def create_root(self):
            destination.mkdir()

        def extent(self, relative, *, family, representation):
            assert (relative, family, representation) == (".", "managed_payload", "derived_state")

        def prepared(self, record_path):
            assert sidecar.read_bytes() == blocked
            record = json.loads(record_path.read_bytes())
            assert record["phase"] == "materialized" and record["sdk_close_returns"] == 0
            assert record["serving"] is False and record["retained_copy_cleanup_ack"] is False
            events.append("prepared")

        def finish(self):
            events.append("finish")
            pytest.fail("publication failure must not finish preparation")

        def failed(self, code):
            assert events == ["prepared", "publication_failed"] and sidecar.read_bytes() == blocked
            assert code == "OSError"
            failure_record.write_text(json.dumps({"code": code, "cleanup": "pending", "serving": False}))
            events.append("failed")

    def qualify(*args, **kwargs):
        # Real target registration and a blocked policy exercise publication.
        # The receipt is local preparation data, with no closure capability.
        ledger = FactLedger(str(destination / PRIVACY), clock=lambda: NOW)
        ledger.initialize()
        try:
            registered = ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,
                {"kind": "wake_clips", "root": str(destination / "data/graph/wake")})
        finally:
            ledger.close()
        sidecar.write_bytes(blocked)
        return JournalDerivationReceipt(latest.source_binding_id, destination, PRIVACY,
            "0"*64, tmp_path / "unissued-history.json", (), {}, (registered,), {})

    actual_write = payload._atomic_file

    def fail_ownership(path, raw):
        if path == sidecar:
            assert json.loads(raw)["owned_targets"]
            assert events == ["prepared"]
            events.append("publication_failed")
            raise OSError("ownership publication unavailable")
        actual_write(path, raw)

    monkeypatch.setattr(state_journal_derivation, "_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(state_journal_derivation, "qualify_and_register_derived_journal", qualify)
    monkeypatch.setattr(copy_catalog, "reserve_copy", lambda *args, **kwargs: PublicationObserver())
    monkeypatch.setattr(payload, "_atomic_file", fail_ownership)
    try:
        with AdmissionController(tmp_path / "stage") as controller:
            with pytest.raises(OSError, match="ownership publication unavailable"):
                payload.materialize_derived_state(plan, destination, controller=controller)
        assert events == ["prepared", "publication_failed", "failed"]
        assert json.loads(failure_record.read_text()) == {"code": "OSError", "cleanup": "pending", "serving": False}
        assert sidecar.read_bytes() == blocked
        assert payload._INCOMPLETE[str(destination)]["phase"] == "failed"
        binding = {"state_root": str(destination), "ownership_sha256": hashlib.sha256(blocked).hexdigest()}
        with pytest.raises(ValueError, match="runtime_admission_active_inventory_mismatch"):
            validate_target_ownership(tmp_path, binding)
    finally:
        # No SDK handle or native task was entered by this fault observer.
        payload._INCOMPLETE.pop(str(destination), None)


def test_catalog_prepared_rejection_keeps_real_ownership_blocked(tmp_path):
    from scripts.stage_runtime import validate_target_ownership
    from src.memory.erasure_journal import ErasureParticipant
    from src.memory.state_journal_derivation import JournalDerivationReceipt

    ordinary, latest = _image(tmp_path, "O"), _image(tmp_path, "N")
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    payload._clone_latest(plan, destination)
    ledger = FactLedger(str(destination / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        registered = ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,
            {"kind": "wake_clips", "root": str(destination / "data/graph/wake")})
    finally:
        ledger.close()
    sidecar = destination / ".kza-target-ownership.json"
    blocked = snapshots._json({"schema_version": 1, "source_bindings": [], "owned_targets": [],
        "delegated_targets": [], "historical_targets": []}) + b"\n"
    sidecar.write_bytes(blocked)
    journal = JournalDerivationReceipt(latest.source_binding_id, destination, PRIVACY,
        "0"*64, tmp_path / "unissued-history.json", (), {}, (registered,), {})

    class RejectPrepared:
        def prepared(self, record_path):
            assert sidecar.read_bytes() == blocked
            record = json.loads(record_path.read_bytes())
            assert record["phase"] == "materialized" and not record["serving"]
            assert record["sdk_close_returns"] == 0 and not record["retained_copy_cleanup_ack"]
            raise OSError("catalog prepared record unavailable")

    with pytest.raises(OSError, match="catalog prepared record unavailable"):
        payload._finish_prepared_payload(plan, destination, journal, {"sdk": []}, None,
            operation=RejectPrepared())
    assert sidecar.read_bytes() == blocked
    binding = {"state_root": str(destination), "ownership_sha256": hashlib.sha256(blocked).hexdigest()}
    with pytest.raises(ValueError, match="runtime_admission_active_inventory_mismatch"):
        validate_target_ownership(tmp_path, binding)


# --- U3b-1a: owner tables are not payload; their facts are never cut (W1, W9) ---------

_DROP_OWNER_TABLES = (("DROP TABLE fact_owners", ()), ("DROP TABLE fact_owner_revisions", ()),
                      ("DROP TABLE fact_owner_schema", ()))
_NO_JOURNAL = SimpleNamespace(target_id_map={}, event_dependency_target_map={})


def _materialize_tables(tmp_path, ordinary, latest):
    """Clone latest and apply the planned table cut: the SQL half of materialization."""
    destination = tmp_path / "R"
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(destination))
    payload._clone_latest(plan, destination)
    payload._apply_tables(plan, destination, _NO_JOURNAL)
    connection = sqlite3.connect(destination / PRIVACY)
    connection.row_factory = sqlite3.Row
    return connection


def _two_proposals(ledger, sources):
    return [ledger.store(content, "fact", 1.0, (source,), proposed=True).id
            for content, source in zip(("Owned synthetic proposal", "Unowned synthetic proposal"), sources)]


def test_w1_fact_with_only_an_ownership_history_row_is_never_cut(tmp_path):
    """No 1a API produces this row (voice stage does); it is inserted by SQL.

    Reverting the W1 protection in payload_derivation._select_facts makes this fail.
    """
    sources = (memory_source(now=NOW - 1, ttl=10), memory_source(now=NOW - 1, ttl=10))
    ids = []
    expire = ("UPDATE facts SET expires_at=? WHERE state='proposed'", (NOW - .5,))
    history = ("INSERT INTO fact_owner_revisions SELECT id,1,'voice_proposed',0,? FROM facts "
               "WHERE content='Owned synthetic proposal'", (NOW,))
    ordinary = _image(tmp_path, "O", statements=(expire,), seed=lambda ledger: ids.extend(_two_proposals(ledger, sources)))
    latest = _image(tmp_path, "N", statements=(history,), seed=lambda ledger: _two_proposals(ledger, sources))
    owned, unowned = ids
    selected, _ = _select(ordinary, latest)
    assert selected["facts"][(owned,)].source == "latest"
    assert (unowned,) not in selected["facts"]  # The ordinary cut really removes an expired proposal.
    with closing(_materialize_tables(tmp_path, ordinary, latest)) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert [row["id"] for row in db.execute("SELECT id FROM facts WHERE state='proposed'")] == [owned]
        assert [tuple(row) for row in db.execute("SELECT fact_id,last_change FROM fact_owner_revisions")] == [
            (owned, "voice_proposed")]


_DROP_LOOKUP_INDEX = (("DROP INDEX idx_erasure_target_participant_id", ()),)


@pytest.mark.parametrize("missing", ["owner_schema", "lookup_index", "both"])
def test_mixed_cut_restores_owner_rows_only_from_the_latest_image(tmp_path, missing):
    """W9 + R1: the ordinary snapshot predates U3b, the 014b422 lookup index, or both.

    'both' is a b0866e8-era ordinary image (014b422 is not an ancestor of b0866e8);
    the former one-rule-at-a-time compatibility check refused it.
    """
    source, kept = memory_source(now=NOW - 1), memory_source(now=NOW - 1)
    operator = MemoryAssigner("synthetic-operator")
    allow = {"retired": lambda ids: False, "active": lambda person: True}

    def seed(ledger, *, assign):
        confirmed = ledger.confirm(ledger.store("Owned confirmed", "fact", 1.0, (source,), proposed=True).id,
                                   MemoryReviewer("synthetic-reviewer"))
        automatic = ledger.store("Automatic to assign", "fact", .9, (kept,))
        if assign:
            ledger.assign_owner(confirmed.id, "user_one", confirmed.revision, 0, operator, **allow)
            ledger.assign_owner(automatic.id, "user_two", automatic.revision, 0, operator, **allow)
            ledger.return_owned(automatic.id, "user_two", MemoryOwnerActor("synthetic-home", "user_two"))

    drops = {"owner_schema": _DROP_OWNER_TABLES, "lookup_index": _DROP_LOOKUP_INDEX,
             "both": _DROP_OWNER_TABLES + _DROP_LOOKUP_INDEX}[missing]
    ordinary = _image(tmp_path, "O", statements=drops, seed=lambda ledger: seed(ledger, assign=False),
                      indexed="live")
    latest = _image(tmp_path, "N", seed=lambda ledger: seed(ledger, assign=True), indexed="live")
    names = {row["name"] for row in ordinary.manifest["databases"][PRIVACY]["schema"]}
    assert ("fact_owners" in names) is (missing == "lookup_index")
    assert ("idx_erasure_target_participant_id" in names) is (missing == "owner_schema")
    with closing(sqlite3.connect(tmp_path / "N-source" / PRIVACY)) as db:
        expected = {table: sorted(db.execute("SELECT * FROM " + table))
                    for table in ("fact_owners", "fact_owner_revisions", "fact_owner_schema")}
    assert len(expected["fact_owners"]) == 1 and len(expected["fact_owner_revisions"]) == 2
    with closing(_materialize_tables(tmp_path, ordinary, latest)) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert {table: sorted(map(tuple, db.execute("SELECT * FROM " + table))) for table in expected} == expected


_DROP_VOICE_ATTRIBUTION_TABLES = tuple(("DROP TABLE " + table, ()) for table in (
    "voice_attribution_consents", "voice_verification_attempts", "voice_attribution_events",
    "voice_attribution_suspensions", "voice_attribution_control", "voice_attribution_schema"))


@pytest.mark.parametrize("missing", ["voice_attribution", "voice_attribution_and_owner_schema"])
def test_mixed_cut_restores_voice_attribution_rows_only_from_the_latest_image(tmp_path, missing):
    """U3b-4a-0: the ordinary snapshot predates the voice-attribution group (and possibly the owner group).

    The rows are protected authority, never payload: they are cloned from the
    latest image, aged rows included (nothing time-based prunes them in a cut).
    """
    from tests.unit.memory.test_voice_attribution_schema import seed_voice_attribution_rows

    drops = _DROP_VOICE_ATTRIBUTION_TABLES + (_DROP_OWNER_TABLES if missing != "voice_attribution" else ())
    ordinary = _image(tmp_path, "O", statements=drops)
    latest = _image(tmp_path, "N", seed=lambda ledger: seed_voice_attribution_rows(ledger._db, NOW))
    names = {row["name"] for row in ordinary.manifest["databases"][PRIVACY]["schema"]}
    assert "voice_attribution_consents" not in names
    assert ("fact_owners" in names) is (missing == "voice_attribution")
    tables = ("voice_attribution_consents", "voice_verification_attempts", "voice_attribution_events",
              "voice_attribution_suspensions", "voice_attribution_control", "voice_attribution_schema")
    with closing(sqlite3.connect(tmp_path / "N-source" / PRIVACY)) as db:
        expected = {table: sorted(map(tuple, db.execute("SELECT * FROM " + table))) for table in tables}
    assert all(expected[table] for table in tables)
    with closing(_materialize_tables(tmp_path, ordinary, latest)) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert {table: sorted(map(tuple, db.execute("SELECT * FROM " + table))) for table in expected} == expected


def test_pre_u3b_latest_image_has_no_owned_facts_and_plans_normally(tmp_path):
    ordinary = _image(tmp_path, "O", statements=_DROP_OWNER_TABLES)
    latest = _image(tmp_path, "N", statements=_DROP_OWNER_TABLES)
    assert payload._owned_fact_ids(latest.manifest["databases"][PRIVACY]["tables"]) == set()
    plan = payload.plan_payload_cut(ordinary, latest, now=NOW, layout=_layout(tmp_path / "R"))
    assert plan.coverage == tuple(sorted(payload.FAMILIES))


@pytest.mark.parametrize("dropped", ["fact_owners", "fact_owner_revisions"])
def test_latest_image_with_only_one_owner_table_is_invalid(tmp_path, dropped):
    ordinary = _image(tmp_path, "O")
    latest = _image(tmp_path, "N", statements=(("DROP TABLE " + dropped, ()),))
    with pytest.raises(payload.PayloadDerivationError, match="payload_table_schema_invalid"):
        _select(ordinary, latest)


def test_only_exact_additive_schema_groups_are_a_compatible_mixed_cut(tmp_path):
    post = _image(tmp_path, "N").manifest["databases"][PRIVACY]["schema"]
    index, owner, voice, ledger = payload._privacy_schema_additions()
    assert len(index) == 1 and len(owner) == 7 and len(voice) == 11 and len(ledger) == 3
    assert all(row in post for row in index + owner + voice + ledger)

    def without(*groups):
        return [row for row in post if not any(row in group for group in groups)]

    for older in (without(index), without(owner), without(voice), without(index, owner),
                  without(owner, voice), without(index, voice), without(index, owner, voice),
                  without(ledger), without(voice, ledger), without(index, owner, voice, ledger)):
        assert payload._privacy_schema_added(older, post)
        assert not payload._privacy_schema_added(post, older)  # Never towards an older latest image.
    assert not payload._privacy_schema_added(post, post)  # Equality is handled by the caller.
    # U3b-0d3: the ledger group is all-or-nothing too (no salt table alone, no autoindex alone).
    assert not payload._privacy_schema_added(without(ledger), without(ledger) + [ledger[0]])
    pre = without(index, owner, voice, ledger)
    assert not payload._privacy_schema_added(pre, [row for row in post if row["name"] != "fact_owners_person"])
    assert not payload._privacy_schema_added(pre, [row for row in post
                                                   if row["name"] != "voice_attribution_events_profile"])
    assert not payload._privacy_schema_added(without(owner), without(owner) + [owner[0]])  # Partial group.
    assert not payload._privacy_schema_added(without(voice), without(voice) + [voice[0]])  # Partial group.
    extra = {"type": "table", "name": "unrelated", "tbl_name": "unrelated", "sql": "CREATE TABLE unrelated (x)"}
    assert not payload._privacy_schema_added(pre, post + [extra])
    changed = [dict(row, sql=row["sql"].replace("NOT NULL PRIMARY KEY", "PRIMARY KEY"))
               if row["name"] == "fact_owners" else row for row in post]
    assert not payload._privacy_schema_added(pre, changed)
    retyped = [dict(row, sql=row["sql"] + " ") if row["name"] == "idx_erasure_target_participant_id" else row
               for row in post]
    assert not payload._privacy_schema_added(without(index), retyped)


def test_a_non_privacy_database_admits_only_the_deletion_ledger_group():
    """U3b-0d3: dashboard_auth.db gains the ledger at migrate_schema, so an older ordinary image lacks it."""
    base = [{"type": "table", "name": "auth_meta", "tbl_name": "auth_meta",
             "sql": "CREATE TABLE auth_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"}]
    (ledger,) = payload._authority_schema_additions()
    _index, owner, _voice, privacy_ledger = payload._privacy_schema_additions()
    assert ledger == privacy_ledger
    groups = payload._authority_schema_additions()
    assert payload._privacy_schema_added(base, base + list(ledger), groups)
    assert not payload._privacy_schema_added(base + list(ledger), base, groups)
    assert not payload._privacy_schema_added(base, base + list(ledger[:1]), groups)  # partial group
    assert not payload._privacy_schema_added(base, base + list(owner), groups)  # privacy-only groups
    extra = {"type": "table", "name": "unrelated", "tbl_name": "unrelated", "sql": "CREATE TABLE unrelated (x)"}
    assert not payload._privacy_schema_added(base, base + list(ledger) + [extra], groups)
