"""Observe existing synthetic copies after one real compiled O/N/R/F run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from tests.integration.test_compiled_material_reconciliation import (
    _CATALOG_CHILD_BOOTSTRAP,
    _probe_compiled_sdk_constructor_failure,
    _probe_compiled_sdk_metadata_faults,
    _recover_compiled_catalog,
    _run_catalog_child,
)


_COMMON = _CATALOG_CHILD_BOOTSTRAP + """
from dataclasses import asdict
from src.memory.copy_observation import (
    ObservationReceipt, ObservedExtent, observe_existing_copy, recover_observations,
)
control_path = Path(sys.argv[3])
assert hashlib.sha256(control_path.read_bytes()).hexdigest() == sys.argv[4]
control = json.loads(control_path.read_text())
assert control['schema_version'] == 1 and control['admission_root'] == str(admission)
assert control['runtime_root'] == str(runtime)

def no_sdk_modules():
    assert not any(name == 'chromadb' or name.startswith('chromadb.') for name in sys.modules)

def assert_observed(record, spec, receipt):
    assert record['observation_id'] == receipt['observation_id']
    assert record['phase'] == 'observed' and record['revision'] == receipt['revision'] == 2
    assert record['record_sha256'] == receipt['record_sha256']
    assert record['observed_root'] == receipt['observed_root'] == spec['path']
    assert record['inventory'] == spec['inventory']
    assert record['inventory_sha256'] == receipt['inventory_sha256']
    assert receipt['inventory_sha256'] == hashlib.sha256(copy_catalog._json(spec['inventory'])).hexdigest()
    assert record['extents'] == spec['extents'] and record['kind'] == spec['kind']
    assert receipt['retrospective'] is True and receipt['before_first_byte'] is False
    assert receipt['non_serving'] is True and receipt['capability'] is False
    assert record['root_identity'] == spec['identity'] and record['authority'] == control['sentinel']['head']
    assert record['retrospective'] is True and record['before_first_byte'] is False
    assert record['non_serving'] is True and record['capability'] is False
    assert record['lineage_complete'] is False and record['SDK_retirement_confirmed'] is False
    assert record['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
    assert record['existing_copy'] == spec['existing_copy']
    assert record['source_binding_ids'] == spec['source_binding_ids']

def assert_unchanged(controller):
    assert state_sentinel(controller) == control['sentinel']
    assert copy_catalog._inventory(admission / 'admission/copies') == control['catalog_inventory']
    for spec in control['roots']:
        assert copy_catalog._identity(Path(spec['path'])) == spec['identity']
        assert copy_catalog._inventory(Path(spec['path'])) == spec['inventory']

def deny_adoption(controller, specs):
    # This deliberately non-live validation input uses a real current lease
    # and historical run identity. It can never grant admission: HEAD is closed.
    # The observation guard must reject before marker/claim/store checks.
    ticket = dict(control['denial_ticket'], lease_fd=controller._descriptor,
        lease_identity=copy_catalog._identity(admission / '.admission.lock'),
        controller_pid=os.getpid())
    denied = []
    for spec in specs:
        root = Path(spec['path'])
        for mode in ('reserve_copy', 'ticket_check_state_false'):
            try:
                if mode == 'reserve_copy':
                    copy_catalog.reserve_copy(controller, 'snapshot', root,
                        control['sentinel']['binding_id'], (control['sentinel']['binding_id'],))
                else:
                    stage.validate_admission_ticket(root, ticket, check_state=False)
            except ValueError as error:
                assert str(error) == 'runtime_admission_copy_observed', str(error)
                denied.append({'role': spec['role'], 'mode': mode, 'error': str(error)})
            else:
                raise AssertionError('Observation metadata must never grant store admission')
    return denied
no_sdk_modules()
"""


_OBSERVE_COMPILED = _COMMON + """
with stage.AdmissionController(admission) as controller:
    assert_unchanged(controller)
    assert recover_observations(controller) == ()
    before = copy_catalog._inventory(admission / 'admission')
    namespace = admission / 'admission/copy-observations'
    assert not namespace.exists()
    spec = control['roots'][0]
    kwargs = dict(kind=spec['kind'], existing_root=Path(spec['path']),
        extents=tuple(ObservedExtent(**row) for row in spec['extents']))
    negatives = []
    try:
        observe_existing_copy(controller, latest_binding_id=control['old_binding_id'], **kwargs)
    except ValueError as error:
        assert str(error) == 'copy_catalog_authority_changed', str(error)
        negatives.append({'case': 'stale_closed_HEAD', 'error': str(error)})
    else:
        raise AssertionError('Historical binding must not authorize observation')
    actual_fd = controller._descriptor
    duplicate = os.dup(actual_fd)
    try:
        controller._descriptor = duplicate
        try:
            observe_existing_copy(controller, latest_binding_id=control['sentinel']['binding_id'], **kwargs)
        except ValueError as error:
            assert str(error) == 'copy_catalog_controller_fd', str(error)
            negatives.append({'case': 'substituted_FD', 'error': str(error)})
        else:
            raise AssertionError('A duplicate descriptor is not the controller capability')
    finally:
        controller._descriptor = actual_fd
        os.close(duplicate)
    # Exercise the actual 16MiB producer ceiling with valid canonical
    # descriptors, before any namespace/payload mutation or receipt exists.
    oversized = tuple(ObservedExtent('oversized-%06d-' % index + 'x' * 150,
        'history', 'raw') for index in range(75000))
    try:
        observe_existing_copy(controller, kind=spec['kind'], existing_root=Path(spec['path']),
            latest_binding_id=control['sentinel']['binding_id'], extents=oversized)
    except ValueError as error:
        assert str(error) == 'copy_catalog_observation_record_size', str(error)
    else:
        raise AssertionError('An oversized intent cannot publish a usable observation')
    assert not namespace.exists() and copy_catalog._inventory(admission / 'admission') == before
    assert_unchanged(controller)
    receipts = []
    for spec in control['roots']:
        receipt = observe_existing_copy(controller, kind=spec['kind'], existing_root=Path(spec['path']),
            latest_binding_id=control['sentinel']['binding_id'],
            extents=tuple(ObservedExtent(**row) for row in spec['extents']),
            source_binding_ids=tuple(spec['source_binding_ids']), existing_copy_id=spec['existing_copy_id'])
        assert type(receipt) is ObservationReceipt
        receipts.append(asdict(receipt))
    recovered = list(recover_observations(controller))
    assert len(recovered) == len(receipts) == 3
    for spec, receipt in zip(control['roots'], receipts, strict=True):
        record, = (row for row in recovered if row['observation_id'] == receipt['observation_id'])
        assert_observed(record, spec, receipt)
    denials = deny_adoption(controller, control['roots'])
    assert_unchanged(controller)
    no_sdk_modules()
    result = {'pid': os.getpid(), 'complete': True, 'scope': 'retrospective metadata only',
        'receipts': receipts, 'recovered': recovered, 'negative_authority': negatives,
        'oversized_intent_rejected_before_namespace': True,
        'admission_denials': denials, 'state_sentinel': control['sentinel'],
        'catalog_inventory_unchanged': True, 'module_origins': audit_origins(),
        'SDK_opened': False, 'cleanup_ack': False, 'graph_closure_ack': False,
        'aggregate_copy_complete': False, 'F1_delivered': False, 'goal_complete': False}
print(json.dumps(result, sort_keys=True))
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sentinel(controller):
    from scripts.stage_runtime import read_admission_head, verify_closed_binding
    from src.memory import copy_catalog

    head = read_admission_head(controller.root)
    binding = verify_closed_binding(controller.root, head["binding_id"])
    return {"head": head, "binding_id": head["binding_id"], "state_root": binding["state_root"],
        "state_identity": binding["state_identity"],
        "authority_sha256": hashlib.sha256(copy_catalog._json(binding["authority"])).hexdigest(),
        "inventory": copy_catalog._inventory(Path(binding["state_root"]))}


def _spec(root, role, kind, source_ids, *, existing=None):
    from src.memory import copy_catalog

    root = root.resolve(strict=True)
    inventory = copy_catalog._inventory(root)
    assert inventory
    extents = []
    for relative, pin in sorted(inventory.items()):
        if role == "failed_SDK_work" and relative.startswith("index/"):
            family, representation = "semantic_index", "sdk_work"
        else:
            family = "managed_payload"
            representation = "blob" if relative.endswith(".bin") else "raw"
        extents.append({"relative": relative, "family": family, "representation": representation,
            "source_sha256": pin["sha256"]})
    return {"role": role, "kind": kind, "path": str(root), "identity": copy_catalog._identity(root),
        "inventory": inventory, "extents": extents, "source_binding_ids": list(source_ids),
        "existing_copy_id": None if existing is None else existing["copy_id"],
        "existing_copy": None if existing is None else {key: existing[key]
            for key in ("copy_id", "revision", "record_sha256")}}


def _parent_denials(controller, control):
    from scripts import stage_runtime as stage
    from src.memory import copy_catalog

    before = copy_catalog._inventory(controller.root / "admission")
    ticket = dict(control["denial_ticket"], lease_fd=controller._descriptor,
        lease_identity=copy_catalog._identity(controller.root / ".admission.lock"),
        controller_pid=os.getpid())
    denials = []
    for spec in control["roots"]:
        for mode in ("reserve_copy", "ticket_check_state_false"):
            try:
                if mode == "reserve_copy":
                    copy_catalog.reserve_copy(controller, "snapshot", Path(spec["path"]),
                        control["sentinel"]["binding_id"], (control["sentinel"]["binding_id"],))
                else:
                    stage.validate_admission_ticket(Path(spec["path"]), ticket, check_state=False)
            except ValueError as error:
                assert str(error) == "runtime_admission_copy_observed", str(error)
                denials.append({"role": spec["role"], "mode": mode, "error": str(error)})
            else:
                raise AssertionError("Observed copies must remain non-serving")
    assert copy_catalog._inventory(controller.root / "admission") == before
    return denials


def _prepare_control(task, evidence, audit, material):
    from scripts import stage_runtime as stage
    from src.memory import copy_catalog

    constructor = json.loads((audit / "catalog-sdk-constructor-failure.json").read_text())
    with stage.AdmissionController(task / "staging") as controller:
        sentinel = _sentinel(controller)
        assert sentinel == constructor["state_sentinel"]
        records = copy_catalog.recover_copies(controller)
        failed, = (row for row in records if row["copy_id"] == constructor["copy_id"])
        assert failed["record_sha256"] == constructor["copy_record_sha256"]
        assert failed["preparation"] == "unknown" and failed["finished"] is False
        assert failed["cleanup"] == {"phase": "blocked", "owner": None, "receipt": None}
        source_ids = tuple(material["phases"][branch]["closed_binding_id"] for branch in ("O", "N", "R", "F"))
        assert source_ids[-1] == sentinel["binding_id"] and source_ids[0] != source_ids[-1]
        roots = [_spec(task / "exchange", "checkpoints_and_blobs", "phase_checkpoint", source_ids),
            _spec(audit, "closed_audit_export", "forensic_audit", (sentinel["binding_id"],)),
            _spec(Path(failed["destination"]), "failed_SDK_work", "failed_work",
                (sentinel["binding_id"],), existing=failed)]
        # Check original checkpoint provenance and every shared BLOB without
        # instantiating DataCodec/DTOs or changing any source/capture deadline.
        for branch in ("O", "N", "R", "F"):
            document = json.loads((task / "exchange" / (branch + "-state.json")).read_text())
            assert document["source_commit"] == material["code_pair"][branch]
            for digest, size in document["blob_manifest"].items():
                blob = task / "exchange/reconciliation-blobs" / (digest + ".bin")
                assert _sha(blob) == digest and blob.stat().st_size == size
        closure = json.loads((controller.root / "admission/closures" /
            (sentinel["binding_id"] + ".json")).read_text())["running_head"]
        return {"schema_version": 1, "admission_root": str(controller.root),
            "runtime_root": str(task / "staging/slots/forward/runtime"), "sentinel": sentinel,
            "catalog_inventory": copy_catalog._inventory(controller.root / "admission/copies"),
            "roots": roots, "old_binding_id": source_ids[0],
            "denial_ticket": {"schema_version": 1, "admission_root": str(controller.root),
                "binding_id": closure["binding_id"], "generation": closure["generation"],
                "run_id": closure["run_id"]}}


def exercise_copy_observation(task, evidence, repo, *, new_revision):
    """Keep original material oracles and observe three already-created roots."""
    from scripts.stage_runtime import AdmissionController
    from src.memory import copy_catalog
    from tests.mocks.compiled_restore import write_json
    from tests.mocks.slot_reconciliation import exercise_material_reconciliation

    task, evidence = Path(task).resolve(), Path(evidence).resolve()
    report = {"schema_version": 1, "unit": "T024-F1/existing-copy-observation", "complete": False,
        "F1_delivered": False, "goal_complete": False, "aggregate_copy_complete": False,
        "scope": "retrospective metadata for existing checkpoint/audit/failed SDK roots only"}
    try:
        material = exercise_material_reconciliation(task, evidence, repo, new_revision=new_revision)
        assert material["complete"] is True
        report["material"] = material
        _recover_compiled_catalog(task, evidence)
        audit = evidence / "existing-audit"
        audit.mkdir(mode=0o700)
        _probe_compiled_sdk_metadata_faults(task, audit)
        _probe_compiled_sdk_constructor_failure(task, audit)
        control = _prepare_control(task, evidence, audit, material)
        control_path = evidence / "copy-observation-control.json"
        write_json(control_path, control)
        runtime = Path(control["runtime_root"])
        assert (runtime / "src/memory/copy_observation.pyc").is_file()
        admission = task / "staging"
        observed_path = _run_catalog_child(_OBSERVE_COMPILED, runtime, admission, evidence,
            "copy-observation", control_path, _sha(control_path))
        observed = json.loads(observed_path.read_text())
        assert observed["complete"] is True and len(observed["receipts"]) == 3
        assert observed["SDK_opened"] is False and len(observed["negative_authority"]) == 2
        assert observed["oversized_intent_rejected_before_namespace"] is True
        report["observer"] = observed
        with AdmissionController(admission) as controller:
            assert _sentinel(controller) == control["sentinel"]
            report["parent_admission_denials"] = _parent_denials(controller, control)
            assert copy_catalog._inventory(admission / "admission/copies") == control["catalog_inventory"]
        # A new process reads the same persisted IDs; it never issues an SDK or
        # Graph close receipt and does not refresh/promote the observation.
        reader_path = _run_catalog_child(_READ_COMPILED, runtime, admission, evidence,
            "copy-observation-reader", control_path, _sha(control_path), observed_path)
        reader = json.loads(reader_path.read_text())
        assert reader["complete"] is True and reader["SDK_opened"] is False
        assert reader["record_sha256s"] == sorted(row["record_sha256"] for row in observed["receipts"])
        assert reader["admission_denials"] == observed["admission_denials"] == report["parent_admission_denials"]
        assert all(reader[key] is False for key in ("cleanup_ack", "graph_closure_ack", "capability",
            "before_first_byte", "aggregate_copy_complete", "F1_delivered", "goal_complete"))
        report["reader"] = reader
        report["current_F_unchanged"] = True
        report["original_catalog_unchanged"] = True
        report["observed_root_count"] = 3
        report["complete"] = True
        return report
    finally:
        write_json(evidence / "copy-observation-parent-proof.json", report)


_READ_COMPILED = _COMMON + """
probe = json.loads(Path(sys.argv[5]).read_text())
assert probe['actual_wait_returned'] is True and probe['exit_code'] == 0
assert probe['containment_only'] is False and probe['pid'] != os.getpid()
assert probe['complete'] is True and probe['SDK_opened'] is False
with stage.AdmissionController(admission) as controller:
    assert_unchanged(controller)
    recovered = list(recover_observations(controller))
    assert recovered == probe['recovered']
    for spec, receipt in zip(control['roots'], probe['receipts'], strict=True):
        record, = (row for row in recovered if row['observation_id'] == receipt['observation_id'])
        assert_observed(record, spec, receipt)
    before = copy_catalog._inventory(admission / 'admission')
    denials = deny_adoption(controller, control['roots'])
    assert copy_catalog._inventory(admission / 'admission') == before
    assert_unchanged(controller)
    no_sdk_modules()
    result = {'pid': os.getpid(), 'complete': True, 'observation_ids': sorted(row['observation_id'] for row in recovered),
        'record_sha256s': sorted(row['record_sha256'] for row in recovered),
        'admission_denials': denials, 'state_sentinel': control['sentinel'],
        'catalog_inventory_unchanged': True, 'module_origins': audit_origins(),
        'retrospective': True, 'before_first_byte': False, 'non_serving': True,
        'capability': False, 'SDK_opened': False, 'cleanup_ack': False,
        'graph_closure_ack': False, 'aggregate_copy_complete': False,
        'F1_delivered': False, 'goal_complete': False}
print(json.dumps(result, sort_keys=True))
"""
