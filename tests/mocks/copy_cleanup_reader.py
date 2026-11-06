"""Fresh compiled metadata observation; receipt history never grants admission."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


_READ_COMPILED = """
import hashlib, json, os, sys
from pathlib import Path

runtime, admission, control_path = map(Path, sys.argv[1:4])
raw = control_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == sys.argv[4]
control = json.loads(raw)
assert control['schema_version'] == 1 and control['parent_pid'] == os.getppid()
assert control['runtime_root'] == str(runtime) and control['admission_root'] == str(admission)
sys.path.insert(0, str(runtime))
from scripts import stage_runtime as stage
from src.memory import copy_catalog, copy_cleanup

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def origin(module):
    path = Path(module.__file__).resolve(strict=True)
    assert path.suffix == '.pyc' and path.is_relative_to(runtime), str(path)
    return str(path)

required_origins = {'copy_cleanup': origin(copy_cleanup),
    'copy_catalog': origin(copy_catalog), 'stage_runtime': origin(stage)}
scope = 'collector_rows_files_v1'
receipt_keys = {'scope', 'copy_id', 'claim_id', 'operation_id', 'conversation_id',
    'representation_complete', 'aggregate_copy_complete', 'other_representations',
    'cleanup_revision', 'record_sha256', 'fence_sha256', 'removed_records',
    'removed_exports', 'removed_files', 'preserved_rows_sha256',
    'preserved_files_sha256', 'native_returned'}
copies, denials = [], []
with stage.AdmissionController(admission) as controller:
    head_before = stage.read_admission_head(admission)
    historical = {record['copy_id']: record for record in copy_catalog.recover_copies(controller)}
    assert len(control['probes']) == 2
    for probe in control['probes']:
        identity, slot = probe['copy_id'], probe['slot']
        expected = probe['expected_receipt']
        assert Path(probe['destination']) == admission / 'slots' / slot / 'state'
        assert set(expected) == receipt_keys and expected['copy_id'] == identity
        assert expected['scope'] == scope and expected['representation_complete'] is True
        assert expected['native_returned'] is True and expected['aggregate_copy_complete'] is False
        assert expected['other_representations'] == 'blocked'
        assert digest(expected) == probe['expected_receipt_sha256']
        for key in ('removed_records', 'removed_exports', 'removed_files', 'cleanup_revision'):
            assert type(expected[key]) is int and expected[key] > 0
        for key in ('record_sha256', 'fence_sha256', 'preserved_rows_sha256', 'preserved_files_sha256'):
            assert copy_catalog._hash(expected[key])

        observed = copy_cleanup.read_copy_cleanup(controller, identity)
        assert observed['scope'] == scope and observed['phase'] == 'complete'
        assert observed['native_started'] is True and observed['native_returned'] is True
        assert observed['capability'] is False and observed['error_code'] is None
        assert observed['aggregate_copy_complete'] is False
        assert observed['other_representations'] == 'blocked'
        assert observed['receipt'] == expected
        assert digest(observed['receipt']) == probe['expected_receipt_sha256']
        assert observed['cleanup_revision'] == expected['cleanup_revision']

        # The receipt names the preceding running event, not its own hash.
        directory = admission / 'admission/copy-cleanup/v1/claims' / identity
        event_path = directory / (f"{observed['cleanup_revision']:012d}-{observed['record_sha256']}.json")
        event_raw = event_path.read_bytes()
        assert hashlib.sha256(event_raw).hexdigest() == observed['record_sha256']
        event = json.loads(event_raw)
        assert event['previous'] == expected['record_sha256']
        assert event['state']['receipt'] == expected
        fence_path = admission / 'admission/copy-cleanup/v1/fences' / (identity + '.json')
        assert hashlib.sha256(fence_path.read_bytes()).hexdigest() == expected['fence_sha256']

        history = historical[identity]
        assert history['kind'] == 'derived_state' and history['preparation'] == 'prepared'
        assert history['finished'] is True and history['capability'] is False
        assert history['destination'] == probe['destination']
        assert history['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
        assert history['prepared_record'] is not None
        try:
            controller.prepare(slot)
        except ValueError as error:
            assert str(error) == 'runtime_admission_copy_invalidated', str(error)
            denials.append({'copy_id': identity, 'slot': slot, 'error': str(error)})
        else:
            raise AssertionError('Historical prepared copy must remain invalidated')
        copies.append({'copy_id': identity, 'slot': slot, 'scope': scope,
            'phase': observed['phase'], 'native_returned': observed['native_returned'],
            'receipt': observed['receipt'], 'receipt_sha256': digest(observed['receipt']),
            'expected_receipt_sha256': probe['expected_receipt_sha256'],
            'cleanup_record_sha256': observed['record_sha256'],
            'catalog_record_sha256': history['record_sha256'],
            'catalog_prepared_history_only': True, 'capability': False,
            'aggregate_copy_complete': False, 'other_representations': 'blocked'})
    assert stage.read_admission_head(admission) == head_before
    assert all(record['cleanup'] == {'phase': 'blocked', 'owner': None, 'receipt': None}
        and record['capability'] is False for record in historical.values())
    source_origins = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'src' or name.startswith('src.'):
            source_origins[name] = origin(module)
print(json.dumps({'schema_version': 1, 'pid': os.getpid(), 'complete': True,
    'complete_scope': 'fresh compiled collector cleanup metadata and admission denial only',
    'copies': copies, 'admission_denials': denials, 'catalog_copy_count': len(historical),
    'required_origins': required_origins, 'module_origins': source_origins,
    'authority_head_unchanged': head_before, 'parent_pid': control['parent_pid'],
    'aggregate_copy_complete': False, 'cleanup_ack': False, 'graph_closure_ack': False,
    'goal_complete': False, 'F1_delivered': False}, sort_keys=True))
"""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _write_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_compiled_copy_cleanup(task, evidence, probes, expected_receipts) -> dict:
    """Observe two receipts with a new process after the parent lease is closed.

    Expected receipt hashes select metadata to compare. They grant no cleanup,
    admission, Graph closure or authority. A timeout only contains/reaps the
    reader and cannot produce an accepted proof.
    """
    task, evidence = Path(task).resolve(strict=True), Path(evidence).resolve(strict=True)
    runtime = (task / "staging/slots/forward/runtime").resolve(strict=True)
    admission = (task / "staging").resolve(strict=True)
    if (type(probes) is not list or type(expected_receipts) is not list
            or len(probes) != 2 or len(expected_receipts) != 2):
        raise ValueError("compiled_copy_cleanup_reader_requires_two_receipts")
    selected = []
    for probe, receipt in zip(probes, expected_receipts, strict=True):
        if (type(probe) is not dict or set(probe) != {"copy_id", "slot", "destination"}
                or any(type(probe[key]) is not str for key in probe)
                or len(probe["copy_id"]) != 32
                or any(character not in "0123456789abcdef" for character in probe["copy_id"])
                or type(receipt) is not dict or receipt.get("copy_id") != probe["copy_id"]
                or Path(probe["destination"]) != admission / "slots" / probe["slot"] / "state"):
            raise ValueError("compiled_copy_cleanup_reader_probe_mismatch")
        selected.append(dict(probe, expected_receipt=receipt,
            expected_receipt_sha256=hashlib.sha256(_json(receipt)).hexdigest()))
    if (len({probe["copy_id"] for probe in selected}) != 2
            or len({probe["slot"] for probe in selected}) != 2):
        raise ValueError("compiled_copy_cleanup_reader_duplicate_probe")
    control = {"schema_version": 1, "runtime_root": str(runtime), "admission_root": str(admission),
        "parent_pid": os.getpid(), "probes": selected}
    prefix = "copy-cleanup-compiled-reader"
    control_path = evidence / (prefix + "-control.json")
    _write_json(control_path, control)
    control_sha = hashlib.sha256(control_path.read_bytes()).hexdigest()
    command = [sys.executable, "-I", "-B", "-c", _READ_COMPILED,
        str(runtime), str(admission), str(control_path), control_sha]
    process = subprocess.Popen(command, cwd=runtime, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    primary = None
    containment = False
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
    (evidence / (prefix + ".stdout")).write_text(stdout)
    (evidence / (prefix + ".stderr")).write_text(stderr)
    wait = {"pid": process.pid, "actual_wait_returned": True, "exit_code": exit_code,
        "containment_only": containment, "cleanup_ack": False, "graph_closure_ack": False,
        "script_sha256": hashlib.sha256(_READ_COMPILED.encode()).hexdigest(),
        "control_sha256": control_sha, "runtime_root": str(runtime), "admission_root": str(admission)}
    _write_json(evidence / (prefix + "-wait.json"), wait)
    if primary is not None:
        raise primary
    assert not containment and exit_code == 0, stderr
    result = json.loads(stdout)
    assert result["pid"] == process.pid and result["complete"] is True
    assert len(result["copies"]) == len(result["admission_denials"]) == 2
    assert result["aggregate_copy_complete"] is False
    assert result["cleanup_ack"] is False and result["graph_closure_ack"] is False
    result.update(wait)
    _write_json(evidence / (prefix + ".json"), result)
    return result
