"""Offline image integrity and copied-store exports, without synthetic ACKs."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from scripts.stage_runtime import AdmissionController
from src.memory import state_snapshot as snapshots


def test_snapshot_requires_actual_closed_controller_before_creating_destination(tmp_path):
    destination = tmp_path / "image"
    with AdmissionController(tmp_path / "admission") as controller:
        with pytest.raises(ValueError, match="runtime_admission_not_closed"):
            snapshots.snapshot_closed_state(controller, "a" * 64, destination)
    assert not destination.exists()


def test_sqlite_backup_keeps_committed_wal_and_exact_blob_values(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "state.db"
    output = tmp_path / "image" / "state" / "state.db"
    image_root = tmp_path / "image"
    with sqlite3.connect(source) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA application_id=12345")
        writer.execute("PRAGMA user_version=7")
        writer.execute("CREATE TABLE payload (id TEXT PRIMARY KEY, value BLOB, deadline REAL)")
        writer.execute("INSERT INTO payload VALUES('original',?,172900)", (b"\x00\xffexact",))
        writer.commit()
        assert Path(str(source) + "-wal").exists()
        snapshots._backup(source, output)
        manifest = snapshots._database(output, image_root)
        assert not Path(str(output) + "-wal").exists()
        assert not Path(str(output) + "-shm").exists()
        with sqlite3.connect(output) as copied:
            assert copied.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert manifest["application_id"] == 12345 and manifest["user_version"] == 7
    table = manifest["tables"]["payload"]
    assert table["primary_key"] == ["id"]
    assert table["rows"][0]["id"] == "original"
    assert table["rows"][0]["deadline"] == 172900
    # This codec DTO is not an admitted closed-state witness.
    dto = snapshots.ClosedStateSnapshot(image_root, "", {}, {})
    blob = table["rows"][0]["value"]
    assert snapshots.read_blob(dto, blob) == b"\x00\xffexact"
    (image_root / blob["$blob"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="state_snapshot_blob_changed"):
        snapshots.read_blob(dto, blob)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "traversal"])
def test_image_paths_reject_redirected_or_shared_files(tmp_path, kind):
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("private")
    if kind == "symlink":
        (root / "redirected").symlink_to(outside)
        with pytest.raises(ValueError, match="state_snapshot_symlink"):
            snapshots._inventory(root)
    elif kind == "hardlink":
        (root / "shared").hardlink_to(outside)
        with pytest.raises(ValueError, match="state_snapshot_special_file"):
            snapshots._inventory(root)
    else:
        with pytest.raises(ValueError, match="state_snapshot_unsafe_path"):
            snapshots._relative("state/../outside")


def _integrity_image(tmp_path):
    root = tmp_path / "image"
    (root / "state").mkdir(parents=True)
    data = root / "state" / "ordinary.json"
    data.write_text('{"ordinary":"original"}')
    binding = {"state_root": str(tmp_path / "original")}
    identity = hashlib.sha256(snapshots._json(binding)).hexdigest()
    manifest = {"schema_version": 1, "source_binding_id": identity, "source_binding": binding,
                "files": {"state/ordinary.json": {"sha256": snapshots._sha(data),
                                                    "size": data.stat().st_size}}}
    (root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    return snapshots.ClosedStateSnapshot(root, identity, binding, manifest)


@pytest.mark.parametrize("mutation", ["payload", "manifest", "extra_file", "dto"])
def test_integrity_verifier_rejects_changed_image_or_dto(tmp_path, mutation):
    image = _integrity_image(tmp_path)
    assert snapshots.verify_snapshot(image) == image.manifest
    if mutation == "payload":
        (image.snapshot_root / "state/ordinary.json").write_text("different")
    elif mutation == "manifest":
        path = image.snapshot_root / "manifest.json"
        value = json.loads(path.read_text())
        value["source_binding_id"] = "b" * 64
        path.write_text(json.dumps(value))
    elif mutation == "extra_file":
        (image.snapshot_root / "untracked").write_text("unknown")
    else:
        image.manifest["untracked"] = "changed"
    with pytest.raises(ValueError, match="state_snapshot_(manifest|file|inventory)_changed"):
        snapshots.verify_snapshot(image)


def test_unlisted_sidecar_is_rejected_before_manifest_publication(tmp_path):
    image = _integrity_image(tmp_path)
    (image.snapshot_root / "manifest.json").unlink()
    (image.snapshot_root / "state/ordinary.json-shm").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="state_snapshot_inventory_changed"):
        snapshots._verify_files(image.snapshot_root, image.manifest["files"], manifest_present=False)
    assert not (image.snapshot_root / "manifest.json").exists()


def test_sdk_logical_export_uses_only_an_isolated_copy_and_public_close(tmp_path, monkeypatch):
    import chromadb
    from chromadb.config import Settings

    original = tmp_path / "original-sdk"
    client = chromadb.PersistentClient(path=str(original), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.create_collection(name="snapshot_policy_v1", embedding_function=None,
                                              metadata={"purpose": "snapshot"})
        collection.add(ids=["fact"], documents=["original document"],
                       metadatas=[{"fact_policy": True, "revision": 3}], embeddings=[[1.0, 0.0]])
    finally:
        client.close()
    before = snapshots._inventory(original)
    raw, work = tmp_path / "raw", tmp_path / "work"
    raw.mkdir()
    shutil.copytree(original, raw / "index")
    factory = chromadb.PersistentClient
    calls, closed = [], []

    def tracked(*args, **kwargs):
        path = Path(kwargs["path"])
        assert path.is_relative_to(work) and not path.is_relative_to(original)
        calls.append(path)
        actual = factory(*args, **kwargs)
        close = actual.close

        def record_close():
            close()
            closed.append(path)

        monkeypatch.setattr(actual, "close", record_close)
        return actual

    monkeypatch.setattr(chromadb, "PersistentClient", tracked)
    descriptor = {"source_path": str(original), "relative_path": "index",
                  "source_identity": snapshots._identity(original), "collection": "snapshot_policy_v1", "present": True}
    version = importlib.metadata.version("chromadb")
    result = snapshots._export_indexes(raw, [descriptor], {"sdk_versions": {"chromadb": version}}, work)
    assert len(calls) == 1 and closed == calls
    assert snapshots._inventory(original) == before
    assert result[0]["source_path"] == str(original)
    assert result[0]["sdk_version"] == version
    assert result[0]["metadata"] == {"purpose": "snapshot"}
    assert result[0]["records"] == [{"id": "fact", "document": "original document",
        "metadata": {"fact_policy": True, "revision": 3}, "embedding": [1.0, 0.0]}]


def test_wrong_sdk_version_rejects_before_client_construction(tmp_path, monkeypatch):
    import chromadb

    def forbidden(*args, **kwargs):
        pytest.fail("SDK constructor ran before version rejection")

    monkeypatch.setattr(chromadb, "PersistentClient", forbidden)
    with pytest.raises(ValueError, match="state_snapshot_sdk_version"):
        snapshots._export_indexes(tmp_path, [{"present": True}],
                                  {"sdk_versions": {"chromadb": "unsupported"}}, tmp_path / "work")


def _history_data():
    descriptor = {"path": "/qualified-source/data/events.db", "table": "events_policy_v1"}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    target_id = hashlib.sha256(json.dumps(["events", 1, encoded]).encode()).hexdigest()
    source_id = "a" * 64
    document = {"schema_version": 1, "source_binding": source_id,
        "targets": [{"id": target_id, "participant": "events", "version": 1, "descriptor": descriptor}],
        "payload_cut_proof": {"opaque_forward_proof": "retained-exactly"}}
    raw = snapshots._json(document) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    refs = [{"source_binding": source_id, "target_id": target_id, "archive_sha256": digest}]
    policy = {"schema_version": 1, "source_bindings": [source_id], "owned_targets": [],
              "delegated_targets": [], "historical_targets": refs}
    return document, raw, digest, refs, policy


def test_history_references_require_exact_ownership_hash():
    _, _, digest, refs, policy = _history_data()
    raw = snapshots._json(policy) + b"\n"
    binding = {"ownership_sha256": hashlib.sha256(raw).hexdigest()}
    assert snapshots._history_references(raw, binding) == {digest: refs}
    with pytest.raises(ValueError, match="state_snapshot_ownership_changed"):
        snapshots._history_references(raw + b" ", binding)


@pytest.mark.parametrize("mutation", ["bytes", "source_binding", "target_missing", "descriptor"])
def test_history_hash_and_source_target_qualification_are_exact(mutation):
    document, raw, digest, refs, _ = _history_data()
    if mutation == "bytes":
        raw += b" "
    elif mutation == "source_binding":
        document["source_binding"] = "b" * 64
        raw = snapshots._json(document) + b"\n"
        digest = hashlib.sha256(raw).hexdigest()
        refs[0]["archive_sha256"] = digest
    elif mutation == "target_missing":
        refs[0]["target_id"] = "c" * 64
    else:
        document["targets"][0]["descriptor"]["path"] = "/other-source/events.db"
        raw = snapshots._json(document) + b"\n"
        digest = hashlib.sha256(raw).hexdigest()
        refs[0]["archive_sha256"] = digest
    with pytest.raises(ValueError, match="state_snapshot_history_(hash|qualification|target)"):
        snapshots._qualified_history(raw, digest, refs)


def _history_image(tmp_path, *, referenced=True):
    document, payload, digest, refs, policy = _history_data()
    if not referenced:
        policy["historical_targets"] = []
    root = tmp_path / "history-image"
    (root / "state").mkdir(parents=True)
    (root / "history").mkdir()
    policy_raw = snapshots._json(policy) + b"\n"
    (root / "state/.kza-target-ownership.json").write_bytes(policy_raw)
    archive_path = "history/" + digest + ".json"
    (root / archive_path).write_bytes(payload)
    binding = {"state_root": str(tmp_path / "closed-source"),
               "ownership_sha256": hashlib.sha256(policy_raw).hexdigest()}
    binding_id = hashlib.sha256(snapshots._json(binding)).hexdigest()
    catalog = {"path": archive_path, "sha256": digest, "size": len(payload),
               "source_binding": document["source_binding"], "references": refs,
               "source_path": str(tmp_path / "absent-original.json"), "source_identity": None}
    manifest = {"schema_version": 1, "source_binding_id": binding_id, "source_binding": binding,
        "files": {archive_path: {"sha256": digest, "size": len(payload), "kind": "history_archive"},
            "state/.kza-target-ownership.json": {"sha256": hashlib.sha256(policy_raw).hexdigest(),
                                               "size": len(policy_raw), "kind": "file"}},
        "history_archives": {digest: catalog}}
    (root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    return snapshots.ClosedStateSnapshot(root, binding_id, binding, manifest), digest, document


def test_history_reader_uses_pinned_copy_without_original_or_sdk(tmp_path):
    # This data-only image makes no claim to a serving/close receipt.
    image, digest, document = _history_image(tmp_path)
    assert snapshots.read_history_archive(image, digest) == document
    assert snapshots.read_history_archive(image, digest)["payload_cut_proof"] == document["payload_cut_proof"]


def test_copied_but_unreferenced_archive_cannot_expand_history(tmp_path):
    image, digest, _ = _history_image(tmp_path, referenced=False)
    with pytest.raises(ValueError, match="state_snapshot_history_unreferenced"):
        snapshots.read_history_archive(image, digest)
