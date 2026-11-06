"""Actual opt-in binary acquisition and inert historical BLOB lineage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from tests.integration.test_compiled_material_reconciliation import _run_catalog_child
from src.memory.checkpoint_lineage import PIN_FIELDS, analyze_checkpoints, normalize_wire
from tests.mocks.compiled_restore import write_json
from tests.mocks.slot_reconciliation import exercise_material_reconciliation


_INERT_READER = """
import hashlib, importlib.util, json, os, sys
from pathlib import Path
control_path, helper_path = Path(sys.argv[3]), Path(sys.argv[5])
assert hashlib.sha256(control_path.read_bytes()).hexdigest() == sys.argv[4]
assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == sys.argv[6]
spec = importlib.util.spec_from_file_location('inert_binary_checker', helper_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
control = json.loads(control_path.read_text())
checkpoints = {branch:dict(value, path=Path(value['path']))
    for branch,value in control['checkpoints'].items()}
report = module.analyze_checkpoints(checkpoints, Path(control['blob_root']),
    source_anchors=tuple(control['source_anchors']),
    withdrawn_conversation_id=control['subject'],
    preserved_conversation_ids=tuple(control['preserved']))
assert not any(name == 'src' or name.startswith('src.') or name == 'chromadb'
    or name.startswith('chromadb.') or name == 'tests.mocks.compiled_restore_child'
    for name in sys.modules)
canonical = json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
assert hashlib.sha256(canonical).hexdigest() == control['report_sha256']
print(json.dumps({'pid':os.getpid(),'report_sha256':control['report_sha256'],
    'project_or_SDK_imported':False,'application_DTO_instantiated':False,
    'cleanup_ack':False,'graph_closure_ack':False,'capability':False}))
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _current(task):
    from scripts.stage_runtime import AdmissionController, read_admission_head, verify_closed_binding
    from src.memory.copy_catalog import _inventory

    with AdmissionController(task / "staging") as controller:
        head = read_admission_head(controller.root)
        binding = verify_closed_binding(controller.root, head["binding_id"])
        return {"head": head, "state": binding["state_root"],
            "inventory": _inventory(Path(binding["state_root"]))}


def test_compiled_binary_source_lineage_distinguishes_withdrawn_shared_and_manual(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    task = tmp_path.resolve()
    selected = os.environ.get("KZA_BINARY_LINEAGE_EVIDENCE_ROOT")
    evidence = task / "evidence" if selected is None else Path(selected).resolve() / ("attempt-" + uuid4().hex)
    evidence.mkdir(mode=0o700, parents=True)
    proof = {"schema_version": 1, "unit": "T024-F1/typed-checkpoint-binary-lineage", "complete": False,
        "cleanup_ack": False, "aggregate_copy_complete": False, "F1_delivered": False, "goal_complete": False}
    try:
        material = exercise_material_reconciliation(task, evidence, repo,
            new_revision=revision, binary_lineage_seed=True)
        assert material["complete"] is True
        proof["material"] = material
        before = _current(task)
        checkpoint_specs, decoded = {}, {}
        for branch in ("O", "N", "R", "F"):
            path = task / "exchange" / (branch + "-state.json")
            document = json.loads(path.read_text())
            control = json.loads((evidence / (branch + "-control.json")).read_text())
            assert _sha(path) == material["phases"][branch]["checkpoint_sha256"]
            pins = {key: control[key] for key in PIN_FIELDS - {"unit", "mode", "helper_manifest_sha256"}}
            pins.update(unit="T024-F1/material-code-data-reconciliation", mode={
                "O": "seed-O", "N": "advance-N", "R": "rollback-R", "F": "forward-F"}[branch],
                helper_manifest_sha256=hashlib.sha256(json.dumps(control["island_files"], sort_keys=True).encode()).hexdigest())
            checkpoint_specs[branch] = {"path": path, "expected_sha256": _sha(path), "expected_pins": pins}
            decoded[branch], _, _ = normalize_wire(document["payload"])
        seed = decoded["O"]
        anchors = (seed["subject"]["source"], seed["foreign"]["source"], *seed["ordinary_sources"])
        subject = seed["subject"]["source"]["conversation_id"]
        preserved = tuple(sorted({source["conversation_id"] for source in anchors} - {subject}))
        original = seed["binary_lineage"]
        roles = original["records"]
        assert set(roles) == {"S_private", "S_shared", "U_shared_manual"}
        assert roles["S_private"]["sources"][0] == roles["S_shared"]["sources"][0] == anchors[0]
        assert roles["U_shared_manual"]["sources"][0] == anchors[1]
        for branch in ("N", "R", "F"):
            assert decoded[branch]["binary_lineage"] == original
            assert not decoded[branch]["binary_observed"]["S_private"]["available"]
            assert not decoded[branch]["binary_observed"]["S_shared"]["available"]
            assert decoded[branch]["binary_observed"]["U_shared_manual"]["available"]
        report = analyze_checkpoints(checkpoint_specs, task / "exchange/reconciliation-blobs",
            source_anchors=anchors, withdrawn_conversation_id=subject, preserved_conversation_ids=preserved)
        private = roles["S_private"]["file_locator"]["sha256"]
        shared = roles["S_shared"]["file_locator"]["sha256"]
        assert private != shared == roles["U_shared_manual"]["file_locator"]["sha256"]
        assert report["blobs"][private]["classification"] == "S_only"
        assert report["blobs"][shared]["classification"] == "shared"
        assert {"S", "U"} <= set(report["blobs"][shared]["categories"])
        assert any("manual" in value["categories"] for value in report["blobs"].values())
        assert all(value["classification"] != "blocked" for value in report["blobs"].values())
        assert _sha(task / "exchange/reconciliation-blobs" / (private + ".bin")) == private
        assert report["complete_reference_scope"] is True and report["deletion_authorized"] is False
        assert report["cleanup_ack"] is False and report["capability"] is False
        proof.update(lineage=report, private_S_digest=private, shared_S_U_digest=shared)
        from dataclasses import asdict
        from scripts.stage_runtime import AdmissionController
        from src.memory.copy_observation import ObservedExtent, observe_existing_copy, recover_observations

        with AdmissionController(task / "staging") as controller:
            observation = observe_existing_copy(controller, kind="phase_checkpoint",
                existing_root=task / "exchange", latest_binding_id=before["head"]["binding_id"],
                extents=(ObservedExtent(".", "history", "raw"),),
                source_binding_ids=tuple(material["phases"][branch]["closed_binding_id"] for branch in ("O", "N", "R", "F")))
            recovered, = recover_observations(controller)
            assert recovered["phase"] == "observed" and recovered["capability"] is False
            assert recovered["before_first_byte"] is False and recovered["cleanup"]["receipt"] is None
            proof.update(observation=asdict(observation), observation_record_sha256=recovered["record_sha256"])
        control = {"checkpoints": {k: dict(v, path=str(v["path"])) for k, v in checkpoint_specs.items()},
            "blob_root": str(task / "exchange/reconciliation-blobs"), "source_anchors": anchors,
            "subject": subject, "preserved": preserved, "report_sha256": hashlib.sha256(_canonical(report)).hexdigest()}
        path = evidence / "binary-lineage-reader-control.json"
        write_json(path, control)
        helper = repo / "src/memory/checkpoint_lineage.py"
        result = _run_catalog_child(_INERT_READER, task / "staging/slots/forward/runtime", task / "staging",
            evidence, "binary-lineage-inert-reader", path, _sha(path), helper, _sha(helper))
        reader = json.loads(result.read_text())
        assert reader["report_sha256"] == control["report_sha256"] and reader["exit_code"] == 0
        assert reader["project_or_SDK_imported"] is False and reader["application_DTO_instantiated"] is False
        assert _current(task) == before
        proof.update(reader=reader, current_F_unchanged=True, complete=True)
    finally:
        write_json(evidence / "binary-lineage-parent-proof.json", proof)
