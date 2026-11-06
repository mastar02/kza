"""Real old/new bytecode, material ordinary rollback, and protected forward recovery."""
import hashlib
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


_CATALOG_CHILD_BOOTSTRAP = """
import hashlib, json, os, sys
from pathlib import Path
runtime, admission = map(Path, sys.argv[1:3])
sys.path.insert(0, str(runtime))
from scripts import stage_runtime as stage
from src.memory import copy_catalog
assert Path(copy_catalog.__file__).suffix == '.pyc'
assert Path(copy_catalog.__file__).resolve().is_relative_to(runtime)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()

def state_sentinel(controller):
    head = stage.read_admission_head(admission)
    binding = stage.verify_closed_binding(admission, head['binding_id'])
    state = Path(binding['state_root'])
    return {'head': head, 'binding_id': head['binding_id'], 'state_root': str(state),
        'state_identity': binding['state_identity'],
        'authority_sha256': hashlib.sha256(canonical(binding['authority'])).hexdigest(),
        'inventory': copy_catalog._inventory(state)}

def audit_origins():
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != 'src' and not name.startswith('src.'):
            continue
        path = Path(module.__file__).resolve()
        assert path.suffix == '.pyc' and path.is_relative_to(runtime), (name, str(path))
        origins[name] = str(path)
    return origins
"""


_SDK_METADATA_FAULT_COMPILED = _CATALOG_CHILD_BOOTSTRAP + """
import importlib.metadata
import chromadb
from chromadb.api.client import Client
from chromadb.config import Settings

fault_phase = sys.argv[3]
assert fault_phase in {'handle_returned', 'close_entered'}
constructor, public_close = chromadb.PersistentClient, Client.close
constructor_code, close_code = constructor.__code__, public_close.__code__
fault_error = OSError('injected_catalog_' + fault_phase)
faults, events, observed_handle = [], [], []

def trace(frame, event, value):
    if frame.f_code is constructor_code:
        label = 'PersistentClient'
    elif (frame.f_code is close_code and observed_handle
            and frame.f_locals.get('self') is observed_handle[0]):
        label = 'Client.close'
    else:
        return None
    if event == 'call':
        events.append({'function': label, 'event': 'call'})
    elif event == 'exception':
        events.append({'function': label, 'event': 'exception', 'kind': value[0].__name__})
    elif event == 'return':
        events.append({'function': label, 'event': 'return',
            'none': value is None, 'type': type(value).__name__})
        if label == 'PersistentClient' and type(value) is Client:
            observed_handle.append(value)
    return trace

with stage.AdmissionController(admission) as controller:
    sentinel = state_sentinel(controller)
    operation = copy_catalog.reserve_copy(controller, 'snapshot',
        admission.parent / ('catalog-sdk-fault-' + fault_phase),
        sentinel['binding_id'], (sentinel['binding_id'],))
    operation.create_root()
    extent = operation.extent('index', family='semantic_index', representation='sdk_work')
    operation.extent('probe.json', family='managed_payload', representation='preparation')
    record_path = operation.destination / 'probe.json'
    record_path.write_text(json.dumps({'kind': 'metadata_fault_probe', 'phase': fault_phase}))
    entry = operation.sdk_enter(extent, sdk_version=importlib.metadata.version('chromadb'))
    actual_append = copy_catalog._append

    def append(directory, record, expected):
        selected = record['sdk'].get(entry.identity)
        if (not faults and record['copy_id'] == operation.copy_id
                and selected is not None and selected['phase'] == fault_phase):
            faults.append({'copy_id': operation.copy_id, 'phase': fault_phase,
                'directory': str(directory), 'error_kind': type(fault_error).__name__})
            raise fault_error
        return actual_append(directory, record, expected)

    construction_error = close_error = None
    previous_trace = sys.gettrace()
    copy_catalog._append = append
    sys.settrace(trace)
    try:
        try:
            entry.construct(constructor, path=str(extent.path),
                settings=Settings(anonymized_telemetry=False))
        except BaseException as error:
            construction_error = error
        finally:
            # The entry retains the actual returned owner even when writing
            # handle_returned failed. Never close an unrelated borrowed handle.
            try:
                entry.close()
            except BaseException as error:
                close_error = error
    finally:
        sys.settrace(previous_trace)
        copy_catalog._append = actual_append
    assert len(faults) == 1
    if fault_phase == 'handle_returned':
        assert construction_error is fault_error, repr(construction_error)
        assert close_error is None, repr(close_error)
    else:
        assert construction_error is None, repr(construction_error)
        assert close_error is fault_error, repr(close_error)
    assert chromadb.PersistentClient is constructor and constructor.__code__ is constructor_code
    assert Client.close is public_close and public_close.__code__ is close_code
    assert len(observed_handle) == 1 and type(observed_handle[0]) is Client
    assert events == [
        {'function': 'PersistentClient', 'event': 'call'},
        {'function': 'PersistentClient', 'event': 'return', 'none': False, 'type': 'Client'},
        {'function': 'Client.close', 'event': 'call'},
        {'function': 'Client.close', 'event': 'return', 'none': True, 'type': 'NoneType'},
    ], events
    # A genuine public close return cannot erase metadata uncertainty.
    for action in (lambda: operation.prepared(record_path), operation.finish):
        try:
            action()
        except copy_catalog.CopyCatalogError:
            pass
        else:
            raise AssertionError('Metadata uncertainty must prevent preparation/finish')
    operation.failed('SDKMetadataFault')
    pending = next(record for record in copy_catalog.recover_copies(controller)
        if record['copy_id'] == operation.copy_id)
    assert pending['preparation'] == 'unknown' and pending['finished'] is False
    assert pending['capability'] is False
    assert pending['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
    assert state_sentinel(controller) == sentinel
    origins = audit_origins()
    result = {'pid': os.getpid(), 'copy_id': operation.copy_id,
        'fault': faults[0], 'events': events, 'fault_kind': 'catalog_metadata_persistence_only',
        'native_close_failure_claimed': False, 'state_sentinel': sentinel,
        'sdk_phase': pending['sdk'][entry.identity]['phase'], 'module_origins': origins,
        'trusted_close': {'module': public_close.__module__, 'qualname': public_close.__qualname__,
            'path': str(Path(close_code.co_filename).resolve()),
            'source_sha256': hashlib.sha256(Path(close_code.co_filename).read_bytes()).hexdigest()},
        'cleanup_ack': False, 'graph_closure_ack': False}
print(json.dumps(result, sort_keys=True))
"""


_VERIFY_SDK_METADATA_FAULTS_COMPILED = _CATALOG_CHILD_BOOTSTRAP + """
probes = [json.loads(Path(path).read_text()) for path in sys.argv[3:]]
assert len(probes) == 2 and {probe['fault']['phase'] for probe in probes} == {
    'handle_returned', 'close_entered'}
with stage.AdmissionController(admission) as controller:
    sentinel = state_sentinel(controller)
    recovered = {record['copy_id']: record for record in copy_catalog.recover_copies(controller)}
    selected = []
    for probe in probes:
        assert probe['pid'] != os.getpid() and probe['actual_wait_returned'] is True
        assert probe['exit_code'] == 0 and probe['containment_only'] is False
        assert probe['state_sentinel'] == sentinel
        record = recovered[probe['copy_id']]
        assert record['preparation'] == 'unknown' and record['finished'] is False
        assert record['capability'] is False
        assert record['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
        assert record['authority'] == sentinel['head']
        destination = Path(record['destination'])
        assert copy_catalog._identity(destination) == record['root_identity']
        assert any(path.is_file() for path in (destination / 'index').rglob('*'))
        sdk, = record['sdk'].values()
        assert sdk['phase'] == probe['sdk_phase']
        expected = 'close_returned' if probe['fault']['phase'] == 'handle_returned' else 'unknown'
        assert sdk['phase'] == expected
        selected.append({'copy_id': record['copy_id'], 'record_sha256': record['record_sha256'],
            'preparation': record['preparation'], 'finished': record['finished'],
            'sdk_phase': sdk['phase'], 'cleanup': record['cleanup'], 'capability': False})
    assert state_sentinel(controller) == sentinel
    origins = audit_origins()
print(json.dumps({'pid': os.getpid(), 'copies': selected, 'module_origins': origins,
    'state_inventory_sha256': hashlib.sha256(canonical(sentinel)).hexdigest(),
    'cleanup_ack': False, 'graph_closure_ack': False}, sort_keys=True))
"""


def _run_catalog_child(script, runtime, admission, evidence, name, *arguments):
    """Keep actual wait evidence separate from SDK/native cleanup claims."""
    command = [sys.executable, "-I", "-B", "-c", script, str(runtime), str(admission), *map(str, arguments)]
    process = subprocess.Popen(command, cwd=runtime, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    containment = False
    primary = None
    try:
        stdout, stderr = process.communicate(timeout=25)
    except BaseException as error:
        primary = error
        containment = True
        try:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        finally:
            stdout, stderr = process.communicate()
    finally:
        exit_code = process.wait()
    (evidence / (name + ".stdout")).write_text(stdout)
    (evidence / (name + ".stderr")).write_text(stderr)
    wait = {"pid": process.pid, "exit_code": exit_code, "actual_wait_returned": True,
        "containment_only": containment, "cleanup_ack": False,
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "runtime_root": str(runtime), "admission_root": str(admission)}
    (evidence / (name + "-wait.json")).write_text(json.dumps(wait, indent=2) + "\n")
    if primary is not None:
        raise primary
    assert not containment, "compiled catalog child exceeded its 25 second cap"
    assert exit_code == 0, stderr
    result = json.loads(stdout)
    assert result["pid"] == process.pid
    result.update(wait)
    path = evidence / (name + ".json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def _probe_compiled_sdk_metadata_faults(task, evidence):
    runtime = (task / "staging/slots/forward/runtime").resolve(strict=True)
    admission = (task / "staging").resolve(strict=True)
    probes = [_run_catalog_child(_SDK_METADATA_FAULT_COMPILED, runtime, admission, evidence,
        "catalog-sdk-fault-" + phase, phase) for phase in ("handle_returned", "close_entered")]
    _run_catalog_child(_VERIFY_SDK_METADATA_FAULTS_COMPILED, runtime, admission, evidence,
        "catalog-sdk-fault-recovery", *probes)


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
    _probe_compiled_sdk_metadata_faults(tmp_path.resolve(), evidence.resolve())
