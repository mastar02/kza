"""Actual S-only retirement with unchanged historical checkpoint pins."""
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from uuid import uuid4

import pytest

from scripts.stage_runtime import AdmissionController
from src.memory import checkpoint_cleanup as cleanup
from src.memory.checkpoint_lineage import PIN_FIELDS, analyze_checkpoints, checkpoint_source_scope, normalize_wire
from src.memory.copy_catalog import _inventory
from src.memory.copy_observation import ObservedExtent, observe_existing_copy, recover_observations
from tests.integration.test_checkpoint_binary_lineage import _current
from tests.integration.test_compiled_material_reconciliation import _run_catalog_child
from tests.mocks.compiled_restore import write_json
from tests.mocks.slot_reconciliation import exercise_material_reconciliation


_POST_RETIREMENT_READER = """
import hashlib, importlib.machinery, importlib.util, json, os, stat, sys
from pathlib import Path
runtime, admission, control_path = map(Path, sys.argv[1:4])
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def regular(path):
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    assert path.resolve(strict=True) == path
    return path.read_bytes()
raw = regular(control_path)
assert sha(raw) == sys.argv[4]
control = json.loads(raw)
release_raw = regular(runtime / 'release.json')
assert sha(release_raw) == control['release_sha256']
release = json.loads(release_raw)
relative = 'src/memory/checkpoint_lineage.pyc'
parser_path = runtime / relative
parser_sha = sha(regular(parser_path))
assert parser_sha == release['files'][relative] == control['parser_sha256']
module_spec = importlib.util.spec_from_file_location('inert_retired_checkpoint_checker', parser_path)
assert isinstance(module_spec.loader, importlib.machinery.SourcelessFileLoader)
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
journal = Path(control['journal_directory'])
raw = regular(journal / 'head.json')
assert sha(raw) == control['journal_head_sha256']
head = json.loads(raw)
assert type(head['schema_version']) is int and head['schema_version'] == 1
assert head['revision'] == control['cleanup']['cleanup_revision']
assert head['sha256'] == control['cleanup']['record_sha256']
previous, committed, record = None, {'head.json'}, None
for revision in range(1, head['revision'] + 1):
    matches = list(journal.glob(f'{revision:012d}-*.json'))
    assert len(matches) == 1
    path = matches[0]
    raw = regular(path)
    digest = sha(raw)
    assert path.name == f'{revision:012d}-{digest}.json'
    event = json.loads(raw)
    assert set(event) == {'schema_version','revision','previous','state'}
    assert type(event['schema_version']) is int and event['schema_version'] == 1
    assert type(event['revision']) is int and event['revision'] == revision
    assert event['previous'] == previous
    record, previous = event['state'], digest
    committed.add(path.name)
assert previous == head['sha256'] and {p.name for p in journal.iterdir()} == committed
assert record['phase'] == 'complete' and record['native_returned'] is True
assert record['scope'] == 'checkpoint_blobs_s_only_v1'
assert record['receipt'] == control['cleanup']['receipt']
assert record['observation_id'] == control['cleanup']['observation_id']
assert record['operation_id'] == control['operation_id']
assert record['conversation_id'] == control['subject']
assert sha(canonical(record['plan'])) == record['plan_sha256'] == record['receipt']['plan_sha256']
assert set(record['removed']) == set(record['plan']['selected']) == {control['private_S_digest']}
assert record['inflight'] is None
unavailable = {}
for digest, pin in record['plan']['selected'].items():
    assert pin['sha256'] == digest
    assert pin['relative'] == record['plan']['blob_relative'] + '/' + digest + '.bin'
    assert record['plan']['inventory'][pin['relative']] == {
        key:pin[key] for key in ('sha256','size','identity')}
    assert not (Path(control['blob_root']) / (digest + '.bin')).exists()
    unavailable[digest] = {key:pin[key] for key in ('size','identity')}
assert unavailable == control['original_retired_pins']
checkpoints = {branch:dict(value,path=Path(value['path']))
    for branch,value in control['checkpoints'].items()}
arguments = dict(source_anchors=tuple(control['source_anchors']),
    withdrawn_conversation_id=control['subject'],
    preserved_conversation_ids=tuple(control['preserved']))
try:
    module.analyze_checkpoints(checkpoints, Path(control['blob_root']), **arguments)
except FileNotFoundError:
    strict_missing_rejected = True
else:
    raise AssertionError('strict analysis accepted an absent referenced blob')
report = module.analyze_checkpoints(checkpoints, Path(control['blob_root']),
    unavailable_blobs=unavailable, **arguments)
assert sha(canonical(report)) == record['receipt']['analysis_sha256'] == control['analysis_sha256']
assert report['blobs'][control['private_S_digest']]['classification'] == 'S_only'
assert report['blobs'][control['shared_S_U_digest']]['classification'] == 'shared'
assert report['deletion_authorized'] is report['cleanup_ack'] is report['capability'] is False
retired = [ref['physical'] for ref in report['references'] if ref['sha256'] in unavailable]
assert retired and all(item['available'] is False and item['supplied_retirement_pins'] is True
    and item['identity'] == unavailable[control['private_S_digest']]['identity'] for item in retired)
for branch, checkpoint in checkpoints.items():
    assert sha(regular(checkpoint['path'])) == control['checkpoints'][branch]['expected_sha256']
assert not any(name == 'src' or name.startswith('src.') or name == 'chromadb'
    or name.startswith('chromadb.') or name == 'tests.mocks.compiled_restore_child'
    for name in sys.modules)
print(json.dumps({'pid':os.getpid(),'scope':'checkpoint_blobs_s_only_v1',
    'parser_relative':relative,'parser_sha256':parser_sha,'parser_loader':type(module_spec.loader).__name__,
    'cleanup_record_sha256':head['sha256'],'analysis_sha256':control['analysis_sha256'],
    'retired_blobs':len(unavailable),'strict_missing_rejected':strict_missing_rejected,
    'original_checkpoint_pins_preserved':True,'retirement_pins_from_owner_journal':True,
    'project_or_SDK_imported':False,'application_DTO_instantiated':False,
    'capability':False,'cleanup_ack':False,'aggregate_copy_complete':False,'graph_closure_ack':False}))
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _checkpoint_specs(task, evidence, material):
    specs, decoded = {}, {}
    modes = {"O": "seed-O", "N": "advance-N", "R": "rollback-R", "F": "forward-F"}
    for branch, mode in modes.items():
        path = task / "exchange" / (branch + "-state.json")
        control = json.loads((evidence / (branch + "-control.json")).read_text())
        assert _sha(path) == material["phases"][branch]["checkpoint_sha256"]
        pins = {key: control[key] for key in PIN_FIELDS - {"unit", "mode", "helper_manifest_sha256"}}
        pins.update(unit="T024-F1/material-code-data-reconciliation", mode=mode,
            helper_manifest_sha256=hashlib.sha256(json.dumps(control["island_files"], sort_keys=True).encode()).hexdigest())
        specs[branch] = {"path": path, "expected_sha256": _sha(path), "expected_pins": pins}
        decoded[branch], _, _ = normalize_wire(json.loads(path.read_text())["payload"])
    return specs, decoded


async def _retire(controller, observation, layout, operation_id, monkeypatch, original_inventory, private):
    owner = cleanup.CheckpointBlobCleanup(controller, observation.observation_id, layout=layout)
    entered, release = threading.Event(), threading.Event()
    calls = []
    native = cleanup._unlink_blob

    def held_native(actual_owner, relative, pin):
        returned = native(actual_owner, relative, pin)
        calls.append({"relative": relative, "pin": dict(pin), "returned": returned})
        entered.set()
        if not release.wait(10):
            raise TimeoutError("checkpoint integration gate was not released")
        return returned

    try:
        arguments = {"expected_observation_revision": observation.revision,
            "expected_record_sha256": observation.record_sha256}
        with pytest.raises(cleanup.CheckpointCleanupError, match="checkpoint_cleanup_current_tombstone"):
            owner.claim(str(uuid4()), **arguments)
        assert cleanup.read_checkpoint_cleanup(controller, observation.observation_id)["phase"] == "unclaimed"
        assert _inventory(owner.destination) == original_inventory
        claim = owner.claim(operation_id, **arguments)
        for metadata in (claim.to_dict(), cleanup.CheckpointCleanupClaim(**claim.to_dict()),
                         {"operation_id": operation_id, "deleted": True}):
            with pytest.raises(cleanup.CheckpointCleanupError, match="checkpoint_cleanup_local_claim_required"):
                await owner.clean(metadata)
        closing = None
        with monkeypatch.context() as patch:
            patch.setattr(cleanup, "_unlink_blob", held_native)
            observer = asyncio.create_task(owner.clean(claim))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                pending = cleanup.read_checkpoint_cleanup(controller, observation.observation_id)
                assert pending["phase"] == "pending" and pending["native_started"] is True
                assert pending["native_returned"] is False and pending["receipt"] is None
                assert pending["capability"] is False and pending["aggregate_copy_complete"] is False
                assert not (owner.destination / layout.blob_relative / (private + ".bin")).exists()
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                closing = asyncio.create_task(owner.aclose())
                await asyncio.sleep(0)
                assert not closing.done()
                assert cleanup.read_checkpoint_cleanup(controller, observation.observation_id) == pending
            finally:
                release.set()
            receipt = await owner.clean(claim)
            assert await owner.clean(claim) is receipt
            await closing
        view = cleanup.read_checkpoint_cleanup(controller, observation.observation_id)
        assert view["phase"] == "complete" and view["native_returned"] is True
        assert view["receipt"] == receipt.to_dict()
        assert len(calls) == 1 and calls[0]["returned"] is True
        assert calls[0]["pin"]["sha256"] == private
        assert receipt.removed_files == receipt.newly_removed_files == 1
        assert receipt.scope == "checkpoint_blobs_s_only_v1" and receipt.representation_complete is True
        assert receipt.aggregate_copy_complete is False and receipt.other_representations == "blocked"
        assert receipt.native_returned is True
        assert await owner.clean(claim) is receipt
        assert cleanup.read_checkpoint_cleanup(controller, observation.observation_id) == view
        return receipt.to_dict(), view, calls, pending
    finally:
        release.set()
        await owner.aclose()


def test_compiled_checkpoint_cleanup_unlinks_only_withdrawn_blob_and_preserves_historical_pins(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    task = tmp_path.resolve()
    selected = os.environ.get("KZA_CHECKPOINT_CLEANUP_EVIDENCE_ROOT")
    evidence = task / "evidence" if selected is None else Path(selected).resolve() / ("attempt-" + uuid4().hex)
    evidence.mkdir(mode=0o700, parents=True)
    proof = {"schema_version": 1, "unit": "T024-F1/checkpoint-blob-cleanup", "complete": False,
        "cleanup_ack": False, "aggregate_copy_complete": False, "F1_delivered": False, "goal_complete": False}
    try:
        material = exercise_material_reconciliation(task, evidence, repo,
            new_revision=revision, binary_lineage_seed=True)
        assert material["complete"] is True
        proof["material"] = material
        before = _current(task)
        specs, decoded = _checkpoint_specs(task, evidence, material)
        original_json = {branch: spec["path"].read_bytes() for branch, spec in specs.items()}
        scope = checkpoint_source_scope(specs["O"]["path"], expected_sha256=specs["O"]["expected_sha256"])
        subject = scope["withdrawn_conversation_id"]
        source_arguments = {"source_anchors": scope["source_anchors"], "withdrawn_conversation_id": subject,
            "preserved_conversation_ids": scope["preserved_conversation_ids"]}
        blob_root = task / "exchange/reconciliation-blobs"
        report = analyze_checkpoints(specs, blob_root, **source_arguments)
        roles = decoded["O"]["binary_lineage"]["records"]
        private = roles["S_private"]["file_locator"]["sha256"]
        shared = roles["S_shared"]["file_locator"]["sha256"]
        assert {digest for digest, value in report["blobs"].items() if value["classification"] == "S_only"} == {private}
        assert shared == roles["U_shared_manual"]["file_locator"]["sha256"]
        assert report["blobs"][shared]["classification"] == "shared"
        assert any("manual" in value["categories"] for value in report["blobs"].values())
        assert all(value["classification"] != "blocked" for value in report["blobs"].values())
        operation = decoded["N"]["operation"]
        assert operation["conversation_id"] == subject and operation["revoked"] is True
        original_inventory = _inventory(task / "exchange")
        retired_relative = "reconciliation-blobs/" + private + ".bin"
        original_retired_pins = {private: {key: original_inventory[retired_relative][key] for key in ("size", "identity")}}
        layout = cleanup.CheckpointBlobLayout("data/graph/privacy.db", "reconciliation-blobs",
            tuple(cleanup.CheckpointPin(branch, spec["path"].name, spec["expected_sha256"], spec["expected_pins"])
                for branch, spec in specs.items()), scope["source_anchors"], scope["preserved_conversation_ids"])
        with AdmissionController(task / "staging") as controller:
            observation = observe_existing_copy(controller, kind="phase_checkpoint",
                existing_root=task / "exchange", latest_binding_id=before["head"]["binding_id"],
                extents=(ObservedExtent(".", "history", "raw"),),
                source_binding_ids=tuple(material["phases"][branch]["closed_binding_id"] for branch in specs))
            observed, = recover_observations(controller)
            receipt, view, calls, pending = asyncio.run(_retire(controller, observation, layout,
                operation["id"], monkeypatch, original_inventory, private))
            assert receipt["retained_blobs"] == len(report["blobs"]) - 1
            assert recover_observations(controller) == (observed,)
            expected = {key: value for key, value in original_inventory.items() if key != retired_relative}
            assert _inventory(task / "exchange") == expected
            assert receipt["preserved_inventory_sha256"] == hashlib.sha256(_canonical(expected)).hexdigest()

            async def reject_recovered_grants():
                owner = cleanup.CheckpointBlobCleanup(controller, observation.observation_id, layout=layout)
                try:
                    for metadata in (observed, view, receipt):
                        with pytest.raises(cleanup.CheckpointCleanupError, match="checkpoint_cleanup_local_claim_required"):
                            await owner.clean(metadata)
                    with pytest.raises(cleanup.CheckpointCleanupError, match="checkpoint_cleanup_resume_blocked"):
                        owner.claim(operation["id"], expected_observation_revision=observation.revision,
                            expected_record_sha256=observation.record_sha256,
                            expected_cleanup_revision=view["cleanup_revision"])
                finally:
                    await owner.aclose()
            asyncio.run(reject_recovered_grants())
            assert cleanup.read_checkpoint_cleanup(controller, observation.observation_id) == view
            assert _inventory(task / "exchange") == expected
        assert _current(task) == before
        assert all(specs[branch]["path"].read_bytes() == raw for branch, raw in original_json.items())
        after = analyze_checkpoints(specs, blob_root, unavailable_blobs=original_retired_pins, **source_arguments)
        analysis_sha = hashlib.sha256(_canonical(after)).hexdigest()
        assert analysis_sha == receipt["analysis_sha256"]
        runtime = task / "staging/slots/forward/runtime"
        journal = task / "staging/admission/checkpoint-cleanup/v1/claims" / observation.observation_id
        control = {"checkpoints": {key: dict(value, path=str(value["path"])) for key, value in specs.items()},
            "blob_root": str(blob_root), "source_anchors": scope["source_anchors"], "subject": subject,
            "preserved": scope["preserved_conversation_ids"], "operation_id": operation["id"],
            "private_S_digest": private, "shared_S_U_digest": shared,
            "original_retired_pins": original_retired_pins, "cleanup": view,
            "journal_directory": str(journal), "journal_head_sha256": _sha(journal / "head.json"),
            "analysis_sha256": analysis_sha, "release_sha256": _sha(runtime / "release.json"),
            "parser_sha256": _sha(runtime / "src/memory/checkpoint_lineage.pyc")}
        control_path = evidence / "checkpoint-cleanup-reader-control.json"
        write_json(control_path, control)
        reader_path = _run_catalog_child(_POST_RETIREMENT_READER, runtime, task / "staging", evidence,
            "checkpoint-cleanup-inert-reader", control_path, _sha(control_path))
        reader = json.loads(reader_path.read_text())
        assert reader["exit_code"] == 0 and reader["actual_wait_returned"] is True
        assert reader["strict_missing_rejected"] is True and reader["retired_blobs"] == 1
        assert reader["project_or_SDK_imported"] is reader["application_DTO_instantiated"] is False
        assert reader["cleanup_ack"] is reader["graph_closure_ack"] is reader["capability"] is False
        assert _current(task) == before
        assert _inventory(task / "exchange") == expected
        proof.update(observation=asdict(observation), cleanup=receipt, cleanup_view=view, pending_view=pending,
            native_unlinks=calls, reader=reader, current_F_unchanged=True,
            original_checkpoint_pins_preserved=True, retained_binary_inventory_unchanged=True,
            private_S_digest=private, shared_S_U_digest=shared, complete=True)
    finally:
        write_json(evidence / "checkpoint-cleanup-parent-proof.json", proof)
