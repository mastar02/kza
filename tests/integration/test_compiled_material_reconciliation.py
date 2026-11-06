"""Real old/new bytecode, material ordinary rollback, and protected forward recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from tests.mocks.slot_reconciliation import exercise_material_reconciliation


_RECOVER_COMPILED = """
import json, os, sys
from pathlib import Path
runtime, admission = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime))
from scripts.stage_runtime import AdmissionController
from scripts.stage_runtime import read_admission_head
from src.memory import copy_catalog
assert Path(copy_catalog.__file__).suffix == '.pyc'
assert Path(copy_catalog.__file__).resolve().is_relative_to(runtime)
with AdmissionController(admission) as controller:
    records = copy_catalog.recover_copies(controller)
assert len(records) == 6
assert sum(r['kind'] == 'snapshot' for r in records) == 3
assert sum(r['kind'] == 'derived_state' for r in records) == 3
assert all(r['preparation'] == 'prepared' and r['capability'] is False for r in records)
assert all(r['finished'] is True for r in records)
assert all(r['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None} for r in records)
assert all(r['preparation_owner']['process_id'] != os.getpid() for r in records)
sdk = [entry for r in records for entry in r['sdk'].values()]
assert len(sdk) >= 6 and all(entry['phase'] == 'close_returned' for entry in sdk)
import importlib.metadata
import chromadb
from chromadb.config import Settings
with AdmissionController(admission) as controller:
    binding = read_admission_head(admission)['binding_id']
    probe = copy_catalog.reserve_copy(controller, 'snapshot',
        admission.parent / 'catalog-borrowed-probe', binding, (binding,))
    probe.create_root()
    extent = probe.extent('index', family='semantic_index', representation='sdk_work')
    entry = probe.sdk_enter(extent, sdk_version=importlib.metadata.version('chromadb'))
    borrowed = chromadb.PersistentClient(path=str(extent.path),
        settings=Settings(anonymized_telemetry=False))
    dispatched = []
    def borrowed_factory(**kwargs):
        dispatched.append(True)
        return borrowed
    try:
        try:
            entry.construct(borrowed_factory, path=str(extent.path),
                settings=Settings(anonymized_telemetry=False))
        except copy_catalog.CopyCatalogError:
            pass
        else:
            raise AssertionError('Borrowed factory must not become an owned constructor')
        assert not dispatched and borrowed.heartbeat() > 0
    finally:
        borrowed.close()
    probe.failed('BorrowedFactoryRejected')
    pending = next(r for r in copy_catalog.recover_copies(controller)
        if r['copy_id'] == probe.copy_id)
    assert pending['capability'] is False and pending['finished'] is False
    assert pending['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
    assert all(r['phase'] != 'close_returned' for r in pending['sdk'].values())
print(json.dumps({'pid': os.getpid(), 'module': copy_catalog.__file__,
    'copy_ids': sorted(r['copy_id'] for r in records),
    'record_hashes': sorted(r['record_sha256'] for r in records),
    'prepared_copies': len(records), 'sdk_close_returns': len(sdk),
    'borrowed_factory_rejected_before_dispatch': True,
    'retained_probe_copy_id': pending['copy_id'],
    'recovered_capability': False, 'cleanup_ack': False, 'graph_closure_ack': False}))
"""


def _recover_compiled_catalog(task, evidence):
    runtime = (task / "staging/slots/forward/runtime").resolve(strict=True)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _RECOVER_COMPILED,
         str(runtime), str(task / "staging")],
        cwd=runtime, text=True, capture_output=True, timeout=20, check=False,
    )
    (evidence / "catalog-recovery.stdout").write_text(completed.stdout)
    (evidence / "catalog-recovery.stderr").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    recovered = json.loads(completed.stdout)
    recovered.update(actual_wait_returned=True, exit_code=completed.returncode)
    (evidence / "catalog-recovery.json").write_text(json.dumps(recovered, indent=2) + "\n")


def test_compiled_material_rollback_and_forward_preserve_latest_authority(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    selected = os.environ.get("KZA_RECONCILIATION_EVIDENCE_ROOT")
    evidence = (tmp_path / "evidence" if selected is None else
        Path(selected).absolute() / ("attempt-" + uuid4().hex))
    evidence.mkdir(mode=0o700, parents=True)
    proof = exercise_material_reconciliation(tmp_path, evidence.resolve(), repo, new_revision=revision)
    assert proof["complete"] is True
    _recover_compiled_catalog(tmp_path.resolve(), evidence.resolve())
