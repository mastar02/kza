"""Checkpoint cleanup IO/protocol tests, without manufactured admission ACKs.

Successful public authority is exercised by the compiled integration. Synthetic
wire snapshots below are inert parser inputs, never controller capabilities.
"""
import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle
import threading
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.core.owned_work import OwnedBlockingWork
from src.core.runtime_admission import _stage
from src.memory import checkpoint_cleanup as cleanup


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    if type(value) is dict:
        return {"k": "dict", "v": [[key, wire(item)] for key, item in value.items()]}
    if type(value) in (list, tuple):
        return {"k": "tuple" if type(value) is tuple else "list", "v": [wire(v) for v in value]}
    return value


def dto(name, fields):
    return {"k": "dto", "n": name, "v": wire(fields)}


def source(conversation):
    return {"capture_id": str(uuid4()), "conversation_id": conversation,
        "captured_at": 1_800_000_000., "expires_at": 1_800_172_800., "origin": "checkpoint-unit"}


def cohort(source_value):
    fields = {"source": None, "fact": "", "ambient_id": 0, "archive_path": "", "clip_ids": [],
        "event_ids": [], "suggestion_ids": [], "voice_ids": [], "user": "", "browser_client": ""}
    result = dto("Cohort", fields)
    result["v"]["v"][0][1] = dto("MemorySource", source_value)
    return result


def exchange(tmp_path):
    root = (tmp_path / "exchange").resolve()
    root.mkdir()
    blobs = root / "blobs"
    blobs.mkdir()
    sources = (source(str(uuid4())), source(str(uuid4())))
    raw = {"private": b"withdrawn private synthetic WAV", "shared": b"shared synthetic WAV",
        "foreign": b"foreign manual synthetic WAV"}
    refs = {}
    for key, value in raw.items():
        fingerprint = digest(value)
        (blobs / (fingerprint + ".bin")).write_bytes(value)
        refs[key] = {"k": "bytes", "sha256": fingerprint, "size": len(value)}
    specs, pins = {}, []
    for branch, mode in (("O", "seed-O"), ("N", "advance-N"), ("R", "rollback-R"), ("F", "forward-F")):
        rows, files = [], []
        for key, which in (("private", 0), ("shared", 0), ("foreign", 1)):
            identity = str(uuid4())
            rows.append({"id": identity, "target_id": "f" * 64, "sources_json": json.dumps([sources[which]])})
            files.append([identity, refs[key]])
        observed = wire({"collectors": {"rows": rows, "files": {}}})
        observed["v"][0][1]["v"][1][1]["v"] = files
        manual_id = str(uuid4())
        manual = wire({"id": manual_id, "current_file_locator": {"root": str(root / "collector"),
            "target_id": "e" * 64, "relative": "policy_v1/" + manual_id + ".wav"},
            "collectors": {"rows": [{"id": manual_id, "target_id": "e" * 64,
                "sources_json": json.dumps([sources[1]])}], "files": {}}})
        manual["v"][2][1]["v"][1][1]["v"] = [[manual_id, refs["shared"]]]
        payload = {"k": "dict", "v": [[branch + "_observed", observed],
            ["binary_observed", {"k": "dict", "v": [["U_shared_manual", manual]]}]]}
        if branch == "O":
            payload["v"].extend([["subject", cohort(sources[0])], ["foreign", cohort(sources[1])],
                ["ordinary_sources", {"k": "tuple", "v": []}]])
        payload["v"].extend([[prior + "_source_sha256", spec["expected_sha256"]] for prior, spec in specs.items()])
        metadata = {"unit": "T024-F1/material-code-data-reconciliation", "mode": mode,
            "source_commit": "1" * 40 if branch in {"O", "R"} else "2" * 40,
            "runtime_root": str(tmp_path.resolve() / branch / "runtime"),
            "state_root": str(tmp_path.resolve() / branch / "state/data/graph"),
            "release_sha256": "3" * 64, "helper_manifest_sha256": "4" * 64}
        document = dict(metadata, schema_version=1, payload=payload,
            blob_manifest={ref["sha256"]: ref["size"] for ref in refs.values()})
        path = root / (branch + "-state.json")
        path.write_bytes(json.dumps(document, sort_keys=True).encode() + b"\n")
        specs[branch] = {"path": path, "expected_sha256": digest(path.read_bytes()), "expected_pins": metadata}
        pins.append(cleanup.CheckpointPin(branch, path.name, specs[branch]["expected_sha256"], metadata))
    layout = cleanup.CheckpointBlobLayout("data/graph/privacy.db", "blobs", tuple(pins), sources,
        (sources[1]["conversation_id"],))
    inventory = cleanup.catalog._inventory(root)
    return {"root": root, "layout": layout, "inventory": inventory, "refs": refs, "sources": sources}


@pytest.fixture
def sample(tmp_path):
    return exchange(tmp_path)


def plan(sample):
    return cleanup._plan(sample["root"], sample["layout"], sample["inventory"], sample["sources"][0]["conversation_id"])


def test_original_typed_scope_and_global_union_select_only_private_bytes(sample):
    scope = cleanup._source_scope(sample["root"], sample["layout"])
    assert scope["source_anchors"] == sample["sources"]
    selected = plan(sample)
    assert set(selected["selected"]) == {sample["refs"]["private"]["sha256"]}
    assert selected["inventory"] == sample["inventory"]
    assert all(path.exists() for path in sample["root"].rglob("*.bin"))


@pytest.mark.parametrize("change", ["anchors", "preserved", "checkpoint-pin"])
def test_layout_sources_and_pins_cannot_relabel_original_scope(sample, change):
    layout = sample["layout"]
    if change == "anchors":
        layout = replace(layout, source_anchors=tuple(reversed(layout.source_anchors)))
    elif change == "preserved":
        layout = replace(layout, preserved_conversation_ids=(str(uuid4()),))
    else:
        layout = replace(layout, checkpoints=(replace(layout.checkpoints[0], sha256="a" * 64), *layout.checkpoints[1:]))
    with pytest.raises(ValueError):
        cleanup._source_scope(sample["root"], layout)
    assert cleanup.catalog._inventory(sample["root"]) == sample["inventory"]


def test_layout_defensively_detaches_nested_sources_and_checkpoint_pins(sample):
    original = sample["layout"]
    detached = cleanup._copy_layout(original)
    original.source_anchors[0]["origin"] = "caller mutation"
    original.checkpoints[0].expected_pins["runtime_root"] = "/caller/relabel"
    assert detached.source_anchors[0]["origin"] == "checkpoint-unit"
    assert detached.checkpoints[0].expected_pins["runtime_root"] != "/caller/relabel"
    assert cleanup._source_scope(sample["root"], detached)["source_anchors"] == detached.source_anchors


@pytest.mark.parametrize("extra", ["file", "empty-directory", "unknown-reference"])
def test_plan_rejects_unregistered_bytes_directories_and_unknown_references(sample, extra):
    if extra == "file":
        (sample["root"] / "unregistered.bin").write_bytes(b"unknown bytes")
    elif extra == "empty-directory":
        (sample["root"] / "extra").mkdir()
    else:
        pin = sample["layout"].checkpoints[-1]
        path = sample["root"] / pin.relative
        value = json.loads(path.read_text())
        value["payload"]["v"][0][0] = "arbitrary_namespace"
        path.write_bytes(json.dumps(value).encode() + b"\n")
        sample["layout"] = replace(sample["layout"], checkpoints=(*sample["layout"].checkpoints[:-1],
            replace(pin, sha256=digest(path.read_bytes()))))
        sample["inventory"] = cleanup.catalog._inventory(sample["root"])
    with pytest.raises((ValueError, OSError)):
        plan(sample)
    assert len(list((sample["root"] / "blobs").glob("*.bin"))) == 3


@dataclass
class FileOperation:
    """Only the private synchronous unlink seam, never an admission owner."""
    destination: Path
    _native_fds: set


def test_real_unlink_and_post_analysis_preserve_all_shared_and_foreign_bytes(sample):
    selected = plan(sample)
    fingerprint, pin = next(iter(selected["selected"].items()))
    operation = FileOperation(sample["root"], set())
    assert cleanup._unlink_blob(operation, pin["relative"], pin)
    assert not operation._native_fds
    remaining, unavailable = cleanup._inventory(sample["root"], selected["inventory"], selected["selected"], {fingerprint})
    assert remaining == {k: v for k, v in sample["inventory"].items() if k != pin["relative"]}
    assert unavailable == {fingerprint: {"size": pin["size"], "identity": pin["identity"]}}
    report = cleanup._analyze(sample["root"], sample["layout"], sample["sources"][0]["conversation_id"], unavailable)
    assert report["blobs"][fingerprint]["classification"] == "S_only"
    assert report["blobs"][sample["refs"]["shared"]["sha256"]]["classification"] == "shared"
    assert report["deletion_authorized"] is False and report["cleanup_ack"] is False


@pytest.mark.parametrize("change", ["sha256", "size", "inode", "symlink", "hardlink"])
def test_unlink_requires_the_original_unaliased_physical_file(sample, change):
    selected = plan(sample)
    _, original = next(iter(selected["selected"].items()))
    pin = deepcopy(original)
    path = sample["root"] / pin["relative"]
    if change == "sha256":
        pin["sha256"] = "a" * 64
    elif change == "size":
        pin["size"] += 1
    elif change == "inode":
        pin["identity"]["inode"] += 1
    else:
        borrowed = sample["root"].parent / "borrowed.bin"
        if change == "symlink":
            path.rename(borrowed)
            path.symlink_to(borrowed)
        else:
            borrowed.hardlink_to(path)
    operation = FileOperation(sample["root"], set())
    with pytest.raises((ValueError, OSError)):
        cleanup._unlink_blob(operation, pin["relative"], pin)
    assert path.exists() and not operation._native_fds


def test_missing_blob_requires_its_original_durable_intent_not_an_unavailable_DTO(sample):
    selected = plan(sample)
    fingerprint, pin = next(iter(selected["selected"].items()))
    (sample["root"] / pin["relative"]).unlink()
    with pytest.raises(ValueError, match="inventory_changed"):
        cleanup._inventory(sample["root"], selected["inventory"], selected["selected"], set())
    _, absent = cleanup._inventory(sample["root"], selected["inventory"], selected["selected"], {fingerprint})
    assert set(absent) == {fingerprint}
    with pytest.raises((ValueError, OSError)):
        plan(sample)


def protocol(sample):
    """A codec fixture, never an executable claim or a closed Graph receipt."""
    payload = plan(sample)
    return {"schema_version": 1, "scope": cleanup.SCOPE, "observation_id": uuid4().hex,
        "claim_id": uuid4().hex, "operation_id": str(uuid4()),
        "conversation_id": sample["sources"][0]["conversation_id"],
        "authority_head": {"schema_version": 1, "generation": 1, "binding_id": "a" * 64,
            "status": "closed", "run_id": None}, "observation_revision": 2,
        "observation_record_sha256": "b" * 64, "layout_sha256": cleanup._layout(sample["layout"]),
        "tombstone_sha256": "c" * 64, "plan": payload, "plan_sha256": cleanup._sha(payload),
        "phase": "claimed", "native_started": False, "native_returned": False,
        "removed": [], "inflight": None, "receipt": None, "error_code": None}


def test_journal_CAS_prevents_stale_overwrite_and_receipt_before_native_return(sample, tmp_path):
    directory = (tmp_path / "journal").resolve()
    directory.mkdir()
    first = protocol(sample)
    head = cleanup._append(directory, first, None)
    running = dict(first, phase="running", native_started=True)
    latest = cleanup._append(directory, running, head)
    with pytest.raises(ValueError, match="claim_cas"):
        cleanup._append(directory, running, head)
    with pytest.raises(ValueError, match="journal_receipt"):
        cleanup._append(directory, dict(running, phase="complete"), latest)
    saved, actual, orphans = cleanup._read_state(directory)
    assert saved == running and actual == latest and not orphans
    assert saved["receipt"] is None and saved["native_returned"] is False
    assert all(b"withdrawn private synthetic WAV" not in p.read_bytes() for p in directory.iterdir())


@pytest.mark.parametrize("field", ["version", "native-bool", "observation-revision", "plan-sha", "selection"])
def test_malformed_protocol_cannot_publish_an_event(sample, tmp_path, field):
    directory = (tmp_path / "journal").resolve()
    directory.mkdir()
    first = protocol(sample)
    if field == "version":
        first["schema_version"] = True
    elif field == "native-bool":
        first["native_started"] = 0
    elif field == "observation-revision":
        first["observation_revision"] = 0
    elif field == "plan-sha":
        first["plan_sha256"] = "d" * 64
    else:
        first["plan"]["selected"].clear()
        first["plan_sha256"] = cleanup._sha(first["plan"])
    with pytest.raises(ValueError):
        cleanup._append(directory, first, None)
    assert not list(directory.iterdir())


def test_crash_after_event_before_head_preserves_committed_prefix_without_repair(sample, tmp_path, monkeypatch):
    directory = (tmp_path / "journal").resolve()
    directory.mkdir()
    first = protocol(sample)
    head = cleanup._append(directory, first, None)
    original = (directory / "head.json").read_bytes()
    attempted = []

    def unavailable(path, value, **_kwargs):
        assert path == directory / "head.json" and value["revision"] == 2
        assert path.read_bytes() == original
        events = list(directory.glob("000000000002-*.json"))
        assert len(events) == 1
        event = json.loads(events[0].read_text())
        assert event["state"]["phase"] == "running" and event["previous"] == head["sha256"]
        assert events[0].name == "000000000002-" + digest(events[0].read_bytes()) + ".json"
        attempted.append(value)
        raise OSError("synthetic head publication fault")

    with monkeypatch.context() as patch:
        patch.setattr(_stage(), "_atomic_record", unavailable)
        with pytest.raises(OSError):
            cleanup._append(directory, dict(first, phase="running", native_started=True), head)
    assert len(attempted) == 1
    saved, actual, orphans = cleanup._read_state(directory)
    assert saved == first and actual == head and len(orphans) == 1
    assert (directory / "head.json").read_bytes() == original
    with pytest.raises(ValueError, match="claim_cas"):
        cleanup._append(directory, dict(first, phase="running", native_started=True), head)


def test_resume_preserves_original_selection_and_never_clears_an_inflight_intent(sample, tmp_path):
    directory = (tmp_path / "journal").resolve()
    directory.mkdir()
    first = protocol(sample)
    head = cleanup._append(directory, first, None)
    fingerprint = next(iter(first["plan"]["selected"]))
    running = dict(first, phase="running", native_started=True, inflight=fingerprint)
    current = cleanup._append(directory, running, head)
    resumed = dict(running, claim_id=uuid4().hex, phase="claimed", native_started=False)
    latest = cleanup._append(directory, resumed, current)
    assert cleanup._read_state(directory)[0]["inflight"] == fingerprint
    changed = deepcopy(resumed)
    changed.update(claim_id=uuid4().hex)
    changed["plan"]["selected"].clear()
    changed["plan_sha256"] = cleanup._sha(changed["plan"])
    with pytest.raises(ValueError):
        cleanup._append(directory, changed, latest)


@pytest.mark.parametrize("purported", [None, "historical-controller", {"status": "closed"}])
def test_historical_metadata_cannot_act_as_current_cleanup_authority(sample, purported):
    before = cleanup.catalog._inventory(sample["root"])
    with pytest.raises(ValueError, match="copy_catalog_controller"):
        cleanup.CheckpointBlobCleanup(purported, uuid4().hex, layout=sample["layout"])
    assert cleanup.catalog._inventory(sample["root"]) == before


def test_nominal_controller_without_actual_closed_HEAD_cannot_dispatch(sample, tmp_path):
    with stage.AdmissionController(tmp_path / "staging") as controller:
        with pytest.raises(ValueError, match="closed_head"):
            cleanup.CheckpointBlobCleanup(controller, uuid4().hex, layout=sample["layout"])
    assert cleanup.catalog._inventory(sample["root"]) == sample["inventory"]


def test_claim_identity_cannot_be_transferred_by_serialization():
    value = cleanup.CheckpointCleanupClaim(uuid4().hex, uuid4().hex, str(uuid4()), str(uuid4()),
        2, "a" * 64, 1, "b" * 64, {}, "c" * 64, "d" * 64)
    with pytest.raises(TypeError, match="not_serializable"):
        pickle.dumps(value)


async def test_cancelled_unlink_observer_does_not_replace_actual_native_return(sample):
    payload = plan(sample)
    _, pin = next(iter(payload["selected"].items()))
    operation = FileOperation(sample["root"], set())
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    work = OwnedBlockingWork(capacity=1, close_timeout_s=.03, isolate_context=True)

    def native():
        try:
            entered.set()
            assert release.wait(3)
            return cleanup._unlink_blob(operation, pin["relative"], pin)
        finally:
            returned.set()

    observer = asyncio.create_task(work.run(native))
    closing = None
    try:
        for _ in range(300):
            if entered.is_set():
                break
            await asyncio.sleep(.001)
        assert entered.is_set()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        closing = work.begin_close()
        await asyncio.sleep(.005)
        assert not returned.is_set() and not closing.done()
        assert (sample["root"] / pin["relative"]).exists()
    finally:
        release.set()
        await asyncio.gather(observer, return_exceptions=True)
        if closing is None:
            closing = work.begin_close()
        await closing
    assert returned.is_set() and not operation._native_fds
    assert not (sample["root"] / pin["relative"]).exists()
