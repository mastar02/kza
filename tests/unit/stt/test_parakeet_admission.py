"""Operation metadata faults through the real preloaded Parakeet wrapper."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from src.stt.parakeet_stt import ParakeetSTT


class OperationControl(BaseException):
    pass


class LocalLeaf:
    def __init__(self, control, *, get_fault=0, set_fault=0):
        self.local = threading.local()
        self.control = control
        self.get_fault = get_fault
        self.set_fault = set_fault
        self.gets = 0
        self.sets = 0

    @property
    def lease(self):
        self.gets += 1
        if self.gets == self.get_fault:
            raise self.control
        return getattr(self.local, 'lease', None)

    @lease.setter
    def lease(self, value):
        self.sets += 1
        self.local.lease = value
        if self.sets == self.set_fault:
            raise self.control


class ConditionLeaf:
    def __init__(self, condition, control, phase):
        self.condition = condition
        self.control = control
        self.phase = phase
        self.enters = 0
        self.exits = 0
        self.notifies = 0

    def __enter__(self):
        self.enters += 1
        if self.phase == 'cleanup_enter' and self.enters == 2:
            raise self.control
        return self.condition.__enter__()

    def __exit__(self, *args):
        result = self.condition.__exit__(*args)
        self.exits += 1
        expected = {'admission_exit': 1, 'cleanup_exit': 2}.get(self.phase)
        if self.exits == expected:
            raise self.control
        return result

    def notify_all(self):
        self.notifies += 1
        if self.phase == 'cleanup_notify' and self.notifies == 1:
            raise self.control
        return self.condition.notify_all()

    def __getattr__(self, name):
        return getattr(self.condition, name)


@pytest.mark.parametrize('phase', [
    'admission_exit', 'tls_previous_read', 'tls_install', 'tls_restore',
    'cleanup_enter', 'cleanup_exit', 'cleanup_notify',
])
def test_public_load_metadata_fault_releases_only_its_admission(phase):
    model = ParakeetSTT()
    sentinel = object()
    model._model = sentinel
    control = OperationControl()
    model._parakeet_condition = ConditionLeaf(
        model._parakeet_condition, control, phase,
    )
    model._parakeet_local = LocalLeaf(
        control, get_fault=2 if phase == 'tls_previous_read' else 0,
        set_fault={'tls_install': 1, 'tls_restore': 2}.get(phase, 0),
    )

    with pytest.raises(OperationControl) as caught:
        model.load()
    assert caught.value is control
    assert model._parakeet_active == 0
    assert model._model is sentinel
    # The one-shot metadata fault does not seal or break an ordinary operation.
    model.load()
    assert model._parakeet_active == 0
    model.close()
    assert model._model is None


class ModelLockLeaf:
    def __init__(self, lock, hook):
        self.lock = lock
        self.hook = hook
        self.armed = True

    def __enter__(self):
        result = self.lock.__enter__()
        if self.armed:
            self.armed = False
            try:
                self.hook()
            except BaseException:
                self.lock.__exit__(*sys.exc_info())
                raise
        return result

    def __exit__(self, *args):
        return self.lock.__exit__(*args)


def test_close_guard_uses_the_actual_worker_when_tls_was_lost():
    model = ParakeetSTT()
    model._model = object()

    def from_body():
        del model._parakeet_local.lease
        with pytest.raises(RuntimeError, match='^parakeet_close_from_worker$'):
            model.close()
        assert not model._parakeet_sealed
        assert model._parakeet_active == 1

    model._model_lock = ModelLockLeaf(model._model_lock, from_body)
    model.load()
    assert model._parakeet_active == 0
    model.close()
    assert model._model is None


def test_nested_restore_fault_never_retires_the_parent_operation():
    model = ParakeetSTT()
    model._model = object()
    control = OperationControl()
    model._parakeet_local = LocalLeaf(control, set_fault=3)

    def nested_body():
        model.request_close()
        with pytest.raises(OperationControl) as caught:
            model.load()
        assert caught.value is control
        assert model._parakeet_active == 1
        with pytest.raises(RuntimeError, match='^parakeet_close_from_worker$'):
            model.begin_close()

    model._model_lock = ModelLockLeaf(model._model_lock, nested_body)
    model.load()
    assert model._parakeet_active == 0
    model.close()
    assert model._model is None


def test_foreign_thread_tls_cannot_borrow_an_admitted_operation():
    model = ParakeetSTT()
    model._model = object()
    entered = threading.Event()
    release = threading.Event()
    worker_errors = []
    captured = []

    def body():
        captured.append(model._parakeet_local.lease)
        entered.set()
        assert release.wait(3)

    def load():
        try:
            model.load()
        except BaseException as error:
            worker_errors.append(type(error).__name__)

    model._model_lock = ModelLockLeaf(model._model_lock, body)
    worker = threading.Thread(target=load, daemon=True)
    worker.start()
    try:
        assert entered.wait(3)
        model.request_close()
        model._parakeet_local.lease = captured[0]
        with pytest.raises(RuntimeError, match='^parakeet_closed$'):
            model.load()
        assert model._parakeet_active == 1
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not worker_errors
    assert model._parakeet_active == 0
    model.close()


_LOST_NOTIFICATION = r'''
import json
import threading
from src.stt.parakeet_stt import ParakeetSTT

model = ParakeetSTT()
model._model = object()
entered = threading.Event()
release = threading.Event()
waited = threading.Event()
operation_finished = threading.Event()
closed = threading.Event()
errors = []
original_lock = model._model_lock
original_condition = model._parakeet_condition

class LockLeaf:
    armed = True
    def __enter__(self):
        result = original_lock.__enter__()
        if self.armed:
            self.armed = False
            entered.set()
            if not release.wait(3):
                original_lock.__exit__(None, None, None)
                raise AssertionError('operation_gate_timeout')
        return result
    def __exit__(self, *args):
        return original_lock.__exit__(*args)

class ConditionLeaf:
    suppressed = 0
    def __enter__(self):
        return original_condition.__enter__()
    def __exit__(self, *args):
        return original_condition.__exit__(*args)
    def notify_all(self):
        self.suppressed += 1
    def wait(self, *args, **kwargs):
        waited.set()
        return original_condition.wait(*args, **kwargs)

def operation():
    try:
        model.load()
    except BaseException as error:
        errors.append(type(error).__name__)
    finally:
        operation_finished.set()

def close():
    try:
        model.close()
    except BaseException as error:
        errors.append(type(error).__name__)
    else:
        closed.set()

model._model_lock = LockLeaf()
condition = ConditionLeaf()
model._parakeet_condition = condition
worker = threading.Thread(target=operation, daemon=True)
closer = threading.Thread(target=close, daemon=True)
worker.start()
try:
    assert entered.wait(3)
    closer.start()
    assert waited.wait(3)
finally:
    release.set()
    worker.join(3)
    if closer.ident is not None:
        closer.join(3)
assert not worker.is_alive() and not closer.is_alive()
assert operation_finished.is_set() and closed.is_set() and not errors
assert model._parakeet_active == 0 and model._model is None
assert condition.suppressed >= 2
print(json.dumps({'actual_wait_observed': waited.is_set(), 'actual_threads_joined': True,
                  'closed': True, 'suppressed': condition.suppressed}))
'''


def test_actual_close_progresses_when_admission_notifications_are_lost():
    result = subprocess.run(
        [sys.executable, '-c', _LOST_NOTIFICATION],
        cwd=Path(__file__).resolve().parents[3],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['actual_wait_observed'] and receipt['actual_threads_joined']
    assert receipt['closed'] and receipt['suppressed'] >= 2
