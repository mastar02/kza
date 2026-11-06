"""Current-authority collector cleanup of two real, non-serving O/O copies."""
from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

TABLES = ("wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
          "conversation_export_parents_v1")
KINDS = {"wake_clips": ".wav", "stt_corrections": ".wav",
         "conversations": ".json", "conversation_exports": ".jsonl"}
PRIVACY = "data/graph/privacy.db"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(value):
    return {"blob_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)} if type(value) is bytes else value


def _tables(database):
    assert database.is_file() and database.resolve(strict=True) == database
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {name: [{key: _value(row[key]) for key in row.keys()}
                       for row in db.execute('SELECT * FROM "' + name.replace('"', '""') + '"')]
                for name in names}


def _collector_targets(tables, destination):
    targets = {}
    for row in tables["erasure_targets"]:
        if row["participant"] != "collectors" or row["version"] != 1:
            continue
        descriptor = json.loads(row["descriptor"])
        if set(descriptor) != {"kind", "root"} or descriptor["kind"] not in KINDS:
            continue
        root = Path(descriptor["root"])
        assert root.is_relative_to(destination) and root.resolve() == root
        targets[row["id"]] = {"kind": descriptor["kind"], "root": root}
    assert {value["kind"] for value in targets.values()} == set(KINDS)
    return targets


def _observe(destination):
    tables = _tables(destination / PRIVACY)
    targets = _collector_targets(tables, destination)
    files = {}
    for row in tables["wake_clips_v1"]:
        owner = targets[row["target_id"]]
        file = owner["root"] / "policy_v1" / (row["id"] + KINDS[owner["kind"]])
        if file.is_file():
            files[row["id"]] = {"relative": file.relative_to(destination).as_posix(),
                                "sha256": _sha(file), "size": file.stat().st_size}
    ordinary_files = {}
    selected_database_files = {destination / PRIVACY, destination / (PRIVACY + "-wal"),
        destination / (PRIVACY + "-shm")}
    for file in destination.rglob("*"):
        # This store's SQL is compared in full below; its WAL/SHM are physical
        # representations changed by the actual selective transaction/checkpoint.
        if not file.is_file() or file.is_symlink() or file in selected_database_files:
            continue
        if any(file.is_relative_to(value["root"]) for value in targets.values()):
            continue
        ordinary_files[file.relative_to(destination).as_posix()] = _sha(file)
    return {"tables": {name: tables[name] for name in TABLES}, "files": files,
        "other_tables_sha256": _digest({name: sorted(rows, key=_json) for name, rows in tables.items()
                                       if name not in TABLES}),
        "other_files_sha256": _digest(ordinary_files)}


@dataclass(frozen=True)
class Probe:
    slot: str
    destination: Path
    copy_id: str
    before: dict
    seed: dict


def prepare_cleanup_probes(controller, prepared, image_O, seed):
    """Install empty placeholders, then materialize O/O under the current lease."""
    from scripts.stage_runtime import prepare_slot, read_admission_head
    from src.memory.copy_catalog import recover_copies
    from src.memory.payload_derivation import materialize_derived_state, plan_payload_cut
    from tests.mocks.slot_reconciliation import clear_fresh_placeholder, fixture_layout

    assert "copy_cleanup" in seed
    assert "src/memory/copy_cleanup.pyc" in prepared["new_manifest"]["files"]
    source = prepared["full_runtimes"]["F"].parent / "state"
    assert all(path.is_dir() or path == source / "config/settings.yaml" for path in source.rglob("*"))
    before_head = read_admission_head(controller.root)
    probes = []
    for slot in ("cleanup-normal", "cleanup-cancelled"):
        prepare_slot(prepared["archive"], controller.root, slot, source)
        destination = controller.root / "slots" / slot / "state"
        clear_fresh_placeholder(destination, (source / "config/settings.yaml").read_bytes())
        layout = fixture_layout(image_O, image_O, destination)
        plan = plan_payload_cut(image_O, image_O, now=seed["clock_now"], layout=layout)
        derived = materialize_derived_state(plan, destination, controller=controller)
        assert derived.serving is False and read_admission_head(controller.root) == before_head
        record, = (item for item in recover_copies(controller) if item["destination"] == str(destination))
        assert record["preparation"] == "prepared" and record["finished"] is True
        probes.append(Probe(slot, destination, record["copy_id"], _observe(destination), seed["copy_cleanup"]))
    return tuple(probes)


def _layout(probe):
    from src.memory.copy_cleanup import CollectorCopyLayout, CollectorRoot

    tables = _tables(probe.destination / PRIVACY)
    targets = _collector_targets(tables, probe.destination)
    return CollectorCopyLayout(PRIVACY, tuple(CollectorRoot(identity, value["kind"],
        value["root"].relative_to(probe.destination).as_posix()) for identity, value in sorted(targets.items())))


def _assert_selective(probe):
    after = _observe(probe.destination)
    subject = probe.seed["S_source"]["conversation_id"]
    before = probe.before
    removed = {row["id"] for row in before["tables"]["wake_clips_v1"]
        if any(source["conversation_id"] == subject for source in json.loads(row["sources_json"]))}
    assert {probe.seed["s_turn_id"], probe.seed["mixed_export_id"]} <= removed
    kept = {row["id"] for row in before["tables"]["wake_clips_v1"]} - removed
    assert {probe.seed["u_turn_id"], probe.seed["u_correction_id"], probe.seed["u_only_export_id"]} <= kept
    for name, rows in before["tables"].items():
        expected = ([row for row in rows if row["export_id"] not in removed]
                    if name == "conversation_export_parents_v1" else [row for row in rows if row["id"] not in removed])
        assert sorted(after["tables"][name], key=_json) == sorted(expected, key=_json), name
    assert after["files"] == {identity: record for identity, record in before["files"].items() if identity in kept}
    for identity in removed:
        if identity in before["files"]:
            assert not (probe.destination / before["files"][identity]["relative"]).exists()
    assert after["other_tables_sha256"] == before["other_tables_sha256"]
    assert after["other_files_sha256"] == before["other_files_sha256"]
    wal = probe.destination / (PRIVACY + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    assert probe.destination.is_dir()
    return {"removed_ids": sorted(removed), "preserved_ids": sorted(kept),
            "preserved_rows_sha256": _digest(after["tables"]), "preserved_files_sha256": _digest(after["files"])}


def _assert_fenced(controller, probe):
    """The real serving preflight must reject before any target store opens."""
    try:
        controller.prepare(probe.slot)
    except ValueError as error:
        assert str(error) == "runtime_admission_copy_invalidated"
    else:
        raise AssertionError("cleaned copy obtained a new serving binding")


def _current_state(controller):
    from scripts.stage_runtime import read_admission_head, verify_closed_binding
    from src.memory.state_snapshot import _inventory

    head = read_admission_head(controller.root)
    assert head["status"] == "closed"
    binding = verify_closed_binding(controller.root, head["binding_id"])
    return {"head": head, "binding": binding, "files": _inventory(Path(binding["state_root"]))}


def _assert_receipt(receipt, probe, operation):
    value = receipt.to_dict()
    assert set(value) == {"scope", "copy_id", "claim_id", "operation_id", "conversation_id",
        "representation_complete", "aggregate_copy_complete", "other_representations",
        "cleanup_revision", "record_sha256", "fence_sha256", "removed_records",
        "removed_exports", "removed_files", "preserved_rows_sha256", "preserved_files_sha256",
        "native_returned"}
    assert value["copy_id"] == probe.copy_id and value["operation_id"] == operation["id"]
    assert value["conversation_id"] == operation["conversation_id"] == probe.seed["S_source"]["conversation_id"]
    assert value["scope"] == "collector_rows_files_v1"
    assert value["representation_complete"] is True and value["native_returned"] is True
    assert value["aggregate_copy_complete"] is False and value["other_representations"] == "blocked"
    assert type(value["cleanup_revision"]) is int and value["cleanup_revision"] > 0
    for field in ("removed_records", "removed_exports", "removed_files"):
        assert type(value[field]) is int and value[field] > 0
    for field in ("record_sha256", "fence_sha256", "preserved_rows_sha256", "preserved_files_sha256"):
        assert type(value[field]) is str and len(value[field]) == 64
        assert all(character in "0123456789abcdef" for character in value[field])
    return value


async def _claim(controller, probe, operation):
    from scripts.stage_runtime import read_admission_head
    from src.memory.copy_catalog import recover_copies
    from src.memory.copy_cleanup import CollectorCopyCleanup

    record, = (item for item in recover_copies(controller) if item["copy_id"] == probe.copy_id)
    assert record["preparation"] == "prepared" and record["finished"] is True
    owner = CollectorCopyCleanup(controller, probe.copy_id, layout=_layout(probe))
    try:
        claim = owner.claim(operation["id"], expected_catalog_revision=record["revision"],
            expected_record_sha256=record["record_sha256"], expected_cleanup_revision=0)
    except BaseException:
        await owner.aclose()
        raise
    value = claim.to_dict()
    assert value["copy_id"] == probe.copy_id and value["operation_id"] == operation["id"]
    assert value["conversation_id"] == operation["conversation_id"]
    assert value["authority_head"] == read_admission_head(controller.root)
    assert value["catalog_revision"] == record["revision"]
    assert value["catalog_record_sha256"] == record["record_sha256"]
    _assert_fenced(controller, probe)
    return owner, claim


async def _clean_normal(controller, probe, operation):
    from src.memory.copy_cleanup import read_copy_cleanup

    owner, claim = await _claim(controller, probe, operation)
    try:
        receipt = _assert_receipt(await owner.clean(claim), probe, operation)
    finally:
        await owner.aclose()
    status = read_copy_cleanup(controller, probe.copy_id)
    _assert_cleanup_status(status, probe)
    assert status["native_returned"] is True and status["receipt"] == receipt
    _assert_fenced(controller, probe)
    return {"receipt": receipt, "final_status": status, "selective": _assert_selective(probe)}


def _assert_cleanup_status(status, probe):
    assert set(status) == {"schema_version", "scope", "copy_id", "phase", "cleanup_revision",
        "record_sha256", "claim_id", "operation_id", "conversation_id", "native_started",
        "native_returned", "receipt", "error_code", "capability", "aggregate_copy_complete",
        "other_representations"}
    assert type(status["schema_version"]) is int and status["schema_version"] == 1
    assert status["copy_id"] == probe.copy_id and status["capability"] is False
    assert status["scope"] == "collector_rows_files_v1"
    assert status["aggregate_copy_complete"] is False and status["other_representations"] == "blocked"


async def _clean_cancelled(controller, probe, operation, current):
    import pytest
    from src.memory import copy_cleanup

    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    actual_unlink = copy_cleanup._unlink_file
    owner, claim = await _claim(controller, probe, operation)
    observer = joined = None

    def hold_after_native(root, identifier, suffix):
        result = actual_unlink(root, identifier, suffix)
        if (result is True and suffix == ".json" and Path(root).is_relative_to(probe.destination)
                and identifier == probe.seed["s_turn_id"]):
            entered.set()
            if not release.wait(15):
                raise TimeoutError("copy_cleanup_test_native_gate_timeout")
            returned.set()
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(copy_cleanup, "_unlink_file", hold_after_native)
        try:
            observer = asyncio.create_task(owner.clean(claim))
            assert await asyncio.to_thread(entered.wait, 10), "real collector unlink did not reach gate"
            observer.cancel()
            outcome, = await asyncio.gather(observer, return_exceptions=True)
            assert isinstance(outcome, asyncio.CancelledError)
            assert not returned.is_set()
            held_status = copy_cleanup.read_copy_cleanup(controller, probe.copy_id)
            _assert_cleanup_status(held_status, probe)
            assert held_status["native_started"] is True and held_status["native_returned"] is False
            assert held_status["receipt"] is None and held_status["operation_id"] == operation["id"]
            _assert_fenced(controller, probe)
            assert _current_state(controller) == current
            joined = asyncio.create_task(owner.clean(claim))
            await asyncio.sleep(.02)
            assert not joined.done() and not returned.is_set()
            release.set()
            receipt = _assert_receipt(await joined, probe, operation)
            assert returned.is_set()
        finally:
            release.set()
            if observer is not None:
                await asyncio.gather(observer, return_exceptions=True)
            if joined is not None:
                await asyncio.gather(joined, return_exceptions=True)
            await owner.aclose()
    status = copy_cleanup.read_copy_cleanup(controller, probe.copy_id)
    _assert_cleanup_status(status, probe)
    assert status["native_returned"] is True and status["receipt"] == receipt
    _assert_fenced(controller, probe)
    return {"receipt": receipt, "cancelled_observer": True, "native_returned": returned.is_set(),
        "held_status": held_status, "final_status": status, "selective": _assert_selective(probe)}


def exercise_collector_copy_cleanup(task, evidence, repo, *, new_revision):
    """Use current F authority to retire S from two retained O/O collector copies."""
    from scripts.stage_runtime import AdmissionController
    from src.memory.copy_catalog import recover_copies
    from tests.mocks.compiled_restore import write_json
    from tests.mocks.copy_cleanup_reader import verify_compiled_copy_cleanup
    from tests.mocks.slot_reconciliation import exercise_material_reconciliation

    captured = {}

    def prepare_probes(controller, prepared, image_O, seed):
        captured.update(staging=controller.root, seed=seed,
            probes=prepare_cleanup_probes(controller, prepared, image_O, seed))

    evidence = Path(evidence)
    report = {"schema_version": 1, "unit": "T024-F1/collector-copy-cleanup",
        "complete": False, "F1_delivered": False, "goal_complete": False,
        "scope": "collector representation only; other copy obligations remain blocked",
        "cleanup_invocation": "public production owner in parent; authority from actual compiled F closure"}
    try:
        material = exercise_material_reconciliation(task, evidence, repo,
            new_revision=new_revision, before_advance_n=prepare_probes, copy_cleanup_seed=True)
        assert material["complete"] is True and len(material["phases"]) == 4
        report["material"] = material
        with AdmissionController(captured["staging"]) as controller:
            records = recover_copies(controller)
            assert len(records) == 8
            assert all(record["preparation"] == "prepared" and record["finished"] is True for record in records)
            current = _current_state(controller)
            assert current["binding"]["slot"] == "forward"
            subject = captured["seed"]["copy_cleanup"]["S_source"]["conversation_id"]
            tables = _tables(Path(current["binding"]["state_root"]) / PRIVACY)
            operation, = (row for row in tables["conversation_erasures"] if row["conversation_id"] == subject)
            assert any(row["conversation_id"] == subject for row in tables["conversation_tombstones"])
            assert any(row["operation_id"] == operation["id"] and row["state"] == "pending"
                for row in tables["erasure_steps"])

            async def clean_both():
                normal = await _clean_normal(controller, captured["probes"][0], operation)
                assert _current_state(controller) == current
                cancelled = await _clean_cancelled(controller, captured["probes"][1], operation, current)
                assert _current_state(controller) == current
                return {"normal": normal, "cancelled": cancelled}

            report["cleanup"] = asyncio.run(clean_both())
            report["current_F_binding_id"] = current["head"]["binding_id"]
            report["current_F_preserved"] = True
            report["catalog_copy_count"] = len(recover_copies(controller))
            assert report["catalog_copy_count"] == 8
            report["aggregate_copy_complete"] = False
        probes = [{"copy_id": probe.copy_id, "slot": probe.slot, "destination": str(probe.destination)}
            for probe in captured["probes"]]
        expected = [report["cleanup"][mode]["receipt"] for mode in ("normal", "cancelled")]
        reader = verify_compiled_copy_cleanup(Path(task), evidence, probes, expected)
        assert reader["complete"] is True and reader["actual_wait_returned"] is True
        assert type(reader["exit_code"]) is int and reader["exit_code"] == 0
        assert all(reader[field] is False for field in ("containment_only", "aggregate_copy_complete",
            "cleanup_ack", "graph_closure_ack"))
        assert len(reader["copies"]) == len(reader["admission_denials"]) == 2
        report["fresh_reader"] = reader
        report["complete"] = True
        return report
    finally:
        write_json(evidence / "copy-cleanup-parent-proof.json", report)
