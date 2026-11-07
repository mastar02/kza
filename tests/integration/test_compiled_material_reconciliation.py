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


_SDK_CONSTRUCTOR_FAILURE_COMPILED = _CATALOG_CHILD_BOOTSTRAP + """
import importlib.metadata
import chromadb
from chromadb.api.client import AdminClient, Client
from chromadb.config import Settings
from chromadb.errors import NotFoundError
constructor, public_close = chromadb.PersistentClient, Client.close
constructor_code, close_code = constructor.__code__, public_close.__code__
events, handles, errors = [], [], []
lookup_code = AdminClient.get_tenant.__code__
lookup_events, lookup_errors = [], []

def trace(frame, event, value):
    if frame.f_code is lookup_code:
        if event == 'call':
            lookup_events.append(['call', frame.f_locals['name']])
        elif event == 'exception':
            lookup_events.append(['exception', value[0].__name__])
            lookup_errors.append(value[1])
        elif event == 'return':
            lookup_events.append(['return', type(value).__name__])
        return trace
    if frame.f_code not in (constructor_code, close_code):
        return None
    name = 'constructor' if frame.f_code is constructor_code else 'public_close'
    if event == 'exception':
        events.append([name, event, value[0].__name__])
        errors.append(value[1])
    elif event == 'return':
        events.append([name, event, type(value).__name__])
        if type(value) is Client:
            handles.append(value)
    elif event == 'call':
        events.append([name, event])
    return trace

with stage.AdmissionController(admission) as controller:
    sentinel = state_sentinel(controller)
    operation = copy_catalog.reserve_copy(controller, 'snapshot',
        admission.parent / 'catalog-sdk-constructor-failure',
        sentinel['binding_id'], (sentinel['binding_id'],))
    operation.create_root()
    extent = operation.extent('index', family='semantic_index', representation='sdk_work')
    operation.extent('probe.json', family='managed_payload', representation='preparation')
    record_path = operation.destination / 'probe.json'
    record_path.write_text(json.dumps({'kind': 'genuine_sdk_constructor_failure'}))
    tenant = 'kza_absent_tenant_' + operation.copy_id
    entry = operation.sdk_enter(extent, sdk_version=importlib.metadata.version('chromadb'))
    failure, previous_trace = None, sys.gettrace()
    sys.settrace(trace)
    try:
        try:
            entry.construct(constructor, path=str(extent.path), tenant=tenant,
                settings=Settings(anonymized_telemetry=False))
        except BaseException as error:
            failure = error
        finally:
            # Only an actual returned owner can receive public retirement.
            if handles:
                entry.close()
    finally:
        sys.settrace(previous_trace)
    assert type(failure) is NotFoundError and errors == [failure], repr(failure)
    assert lookup_errors == [failure] and lookup_events == [
        ['call', tenant], ['exception', 'NotFoundError'], ['return', 'NoneType']], lookup_events
    assert not handles and events == [['constructor', 'call'],
        ['constructor', 'exception', 'NotFoundError'], ['constructor', 'return', 'NoneType']], events
    assert chromadb.PersistentClient is constructor and constructor.__code__ is constructor_code
    assert Client.close is public_close and public_close.__code__ is close_code
    assert (extent.path / 'chroma.sqlite3').is_file()
    for action in (lambda: operation.prepared(record_path), operation.finish):
        try:
            action()
        except copy_catalog.CopyCatalogError:
            pass
        else:
            raise AssertionError('An SDK exception cannot confirm preparation/retirement')
    operation.failed('SDKConstructorError')
    pending = next(record for record in copy_catalog.recover_copies(controller)
        if record['copy_id'] == operation.copy_id)
    assert pending['preparation'] == 'unknown' and pending['finished'] is False
    assert pending['capability'] is False
    assert pending['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
    sdk, = pending['sdk'].values()
    assert sdk['phase'] == 'unknown' and state_sentinel(controller) == sentinel
    result = {'pid': os.getpid(), 'copy_id': operation.copy_id,
        'fault_kind': 'actual_nominal_constructor_exception_before_handle_return',
        'exception_kind': type(failure).__name__, 'tenant': tenant, 'events': events,
        'tenant_lookup_events': lookup_events,
        'returned_handles': 0, 'public_close_calls': 0, 'sdk_internal_retirement_confirmed': False,
        'state_sentinel': sentinel, 'copy_record_sha256': pending['record_sha256'],
        'copy_inventory': copy_catalog._inventory(operation.destination),
        'module_origins': audit_origins(), 'cleanup_ack': False, 'graph_closure_ack': False}
print(json.dumps(result, sort_keys=True))
"""


_VERIFY_SDK_CONSTRUCTOR_FAILURE_COMPILED = _CATALOG_CHILD_BOOTSTRAP + """
probe = json.loads(Path(sys.argv[3]).read_text())
assert probe['pid'] != os.getpid() and probe['actual_wait_returned'] is True
assert probe['exit_code'] == 0 and probe['containment_only'] is False
assert probe['fault_kind'] == 'actual_nominal_constructor_exception_before_handle_return'
assert probe['exception_kind'] == 'NotFoundError' and probe['returned_handles'] == 0
assert probe['tenant_lookup_events'] == [['call', probe['tenant']],
    ['exception', 'NotFoundError'], ['return', 'NoneType']]
assert probe['public_close_calls'] == 0 and probe['sdk_internal_retirement_confirmed'] is False
with stage.AdmissionController(admission) as controller:
    sentinel = state_sentinel(controller)
    assert probe['state_sentinel'] == sentinel
    record, = (record for record in copy_catalog.recover_copies(controller)
        if record['copy_id'] == probe['copy_id'])
    assert record['record_sha256'] == probe['copy_record_sha256']
    assert record['preparation'] == 'unknown' and record['finished'] is False
    assert record['capability'] is False
    assert record['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
    assert record['authority'] == sentinel['head']
    destination = Path(record['destination'])
    assert copy_catalog._identity(destination) == record['root_identity']
    assert copy_catalog._inventory(destination) == probe['copy_inventory']
    assert (destination / 'index/chroma.sqlite3').is_file()
    sdk, = record['sdk'].values()
    assert sdk['phase'] == 'unknown' and state_sentinel(controller) == sentinel
    result = {'pid': os.getpid(), 'copy_id': record['copy_id'],
        'record_sha256': record['record_sha256'], 'preparation': record['preparation'],
        'finished': record['finished'], 'sdk_phase': sdk['phase'],
        'cleanup': record['cleanup'], 'capability': False,
        'state_inventory_sha256': hashlib.sha256(canonical(sentinel)).hexdigest(),
        'module_origins': audit_origins(), 'sdk_internal_retirement_confirmed': False,
        'cleanup_ack': False, 'graph_closure_ack': False}
print(json.dumps(result, sort_keys=True))
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


def _probe_compiled_sdk_constructor_failure(task, evidence):
    runtime = (task / "staging/slots/forward/runtime").resolve(strict=True)
    admission = (task / "staging").resolve(strict=True)
    probe = _run_catalog_child(_SDK_CONSTRUCTOR_FAILURE_COMPILED, runtime, admission, evidence,
        "catalog-sdk-constructor-failure")
    _run_catalog_child(_VERIFY_SDK_CONSTRUCTOR_FAILURE_COMPILED, runtime, admission, evidence,
        "catalog-sdk-constructor-failure-recovery", probe)


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


_RECOVER_OUTPUT_REGISTRATION = """
import asyncio, hashlib, json, os, sys
from pathlib import Path
runtime, admission, evidence = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime))
from src.core.runtime_admission import _stage
from src.memory import output_registration, copy_catalog
from src.lists.list_store import ListStore
from src.reminders.reminder_store import ReminderStore
import aiosqlite
stage = _stage()
assert Path(output_registration.__file__).suffix == '.pyc'
assert Path(output_registration.__file__).resolve().is_relative_to(runtime)
assert Path(stage.__file__).suffix == '.pyc'
with stage.AdmissionController(admission) as controller:
    before_head = stage.read_admission_head(admission)
    current = stage.read_binding(admission, before_head['binding_id'])
    before_state = copy_catalog._inventory(Path(current['state_root']))
    manifest = json.loads((runtime / 'release.json').read_text())
    assert hashlib.sha256((runtime / 'release.json').read_bytes()).hexdigest() == current['release_sha256']
    expected = {}
    for mode in ('advance-N', 'forward-F'):
        parent = json.loads((evidence / (mode + '-parent-receipt.json')).read_text())
        closure_path = admission / 'admission/closures' / (parent['closed_binding_id'] + '.json')
        assert hashlib.sha256(closure_path.read_bytes()).hexdigest() == parent['closure_sha256']
        running = json.loads(closure_path.read_text())['running_head']
        expected[running['run_id']] = (running['binding_id'], running['generation'])
    records = output_registration.read_output_inventory(controller)
    assert {r['run_id']: (r['header']['binding_id'], r['header']['generation']) for r in records} == expected
    assert len(records) == 2, records
    assert all(r['phase'] == 'observed' for r in records)
    assert all(r['version'] == 'v2' and r['header']['schema_version'] == 2 for r in records)
    assert all(r['journal']['revision'] > 10 for r in records)
    expected_owners = {'lists', 'reminders', 'privacy_authority', 'authority_sessions', 'routines',
        'events', 'events.eraser', 'events.retention', 'suggestions', 'suggestions.eraser', 'suggestions.retention'}
    assert all(set(r['stores']) == expected_owners for r in records)
    assert all(set(r['instrumented_producers']) == {'lists', 'reminders', 'privacy_authority',
        'authority_sessions', 'routines', 'events', 'suggestions'} for r in records)
    assert all(set(r['partial_producers']) == {'privacy_authority', 'authority_sessions', 'routines',
        'events', 'suggestions'} and r['material_observations'] for r in records)
    assert all(s['phase'] == 'closed' for r in records for s in r['stores'].values())
    assert all(len(r['required_producers']) == 32 and len(r['uncovered_producers']) == 30 for r in records)
    assert all(r['capability'] is False and r['global_inventory_complete'] is False
        and r['aggregate_copy_complete'] is False for r in records)
    assert all(r['configuration']['effective_digest'] == 'observed_only' for r in records)
    receipts = [stage._read_json(admission / 'admission/receipts' / (r['run_id'] + '.json')) for r in records]
    assert all(r['child_pid'] != os.getpid() for r in receipts)
    # A completed real run and serialized registration metadata grant no new
    # open. This fresh process has a closed-controller lease, not a replayed
    # direct-child runtime ticket.
    connects = []
    original_connect = aiosqlite.connect
    aiosqlite.connect = lambda *a, **k: connects.append(True)
    async def rejected_replay():
        for record in records:
            for name in ('lists', 'reminders'):
                state = record['stores'][name]
                store_type = ListStore if name == 'lists' else ReminderStore
                store = store_type(state['plan']['path'], registration=state)
                try:
                    await store.initialize()
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError('Completed run metadata admitted a native connection')
                assert store._db is None
        assert connects == []
    try:
        asyncio.run(rejected_replay())
    finally:
        aiosqlite.connect = original_connect
    assert output_registration.read_output_inventory(controller) == records
    assert stage.read_admission_head(admission) == before_head
    assert copy_catalog._inventory(Path(current['state_root'])) == before_state
    module_hashes = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'src' or name.startswith('src.'):
            origin = Path(module.__file__).resolve()
            assert origin.suffix == '.pyc' and origin.is_relative_to(runtime)
            relative = str(origin.relative_to(runtime))
            digest = hashlib.sha256(origin.read_bytes()).hexdigest()
            assert manifest['files'][relative] == digest
            module_hashes[name] = digest
print(json.dumps({'pid': os.getpid(), 'module': output_registration.__file__,
    'stage_module': stage.__file__, 'module_hashes': module_hashes, 'unchanged_head_state': True, 'run_ids': sorted(r['run_id'] for r in records),
    'stores_closed': 22, 'instrumented_per_run': 7, 'required_producers': 32, 'uncovered_per_run': 30,
    'completed_run_metadata_rejected_before_connect': True,
    'capability': False, 'global_inventory_complete': False, 'aggregate_copy_complete': False}))
"""


def _recover_compiled_output_registration(task, evidence):
    runtime = (task / "staging/slots/forward/runtime").resolve(strict=True)
    return _run_catalog_child(_RECOVER_OUTPUT_REGISTRATION, runtime, task / "staging",
        evidence, "output-registration-recovery", evidence)


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
    privacy = "data/graph/privacy.db"
    index = {"type": "index", "name": "idx_erasure_target_participant_id",
        "tbl_name": "erasure_targets",
        "sql": "CREATE INDEX idx_erasure_target_participant_id ON erasure_targets(participant, id)"}
    images = {branch: json.loads((tmp_path / "closed-snapshots" / branch / "manifest.json").read_text())
        for branch in ("O", "N", "R")}
    assert not any(row["name"] == index["name"] for row in images["O"]["databases"][privacy]["schema"])
    for branch in ("N", "R"):
        assert [row for row in images[branch]["databases"][privacy]["schema"]
            if row["name"] == index["name"]] == [index]
    # F has a genuine closed binding rather than another snapshot directory.
    # Its exact schema digest must retain the same latest index after forward
    # materialization and the original compiled application's owned close.
    final_binding = json.loads((tmp_path / "staging/admission/bindings" /
        (proof["phases"]["F"]["closed_binding_id"] + ".json")).read_text())
    assert final_binding["schemas"][privacy] == images["N"]["source_binding"]["schemas"][privacy]
    _recover_compiled_output_registration(tmp_path.resolve(), evidence.resolve())
    _recover_compiled_catalog(tmp_path.resolve(), evidence.resolve())
    _probe_compiled_sdk_metadata_faults(tmp_path.resolve(), evidence.resolve())
    _probe_compiled_sdk_constructor_failure(tmp_path.resolve(), evidence.resolve())
