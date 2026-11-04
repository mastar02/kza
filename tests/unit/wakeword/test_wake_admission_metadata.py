"""Public metadata operations retain real Wake admission identities."""
import threading

import pytest

from src.wakeword.detector import WakeWordDetector
from tests.unit.stt.test_parakeet_admission import ConditionLeaf, LocalLeaf, ModelLockLeaf
from tests.unit.stt.test_parakeet_close_protocol import CloseControl


def registered():
    detector = WakeWordDetector()
    calls = []

    def callback(*args):
        calls.append(1)

    detector.on_detection(callback)
    return detector, callback, calls


@pytest.mark.parametrize('phase', [
    'admission_exit', 'tls_previous_read', 'tls_install', 'tls_restore',
    'cleanup_enter', 'cleanup_exit', 'cleanup_notify',
])
def test_public_threshold_fault_does_not_leave_an_active_admission(phase):
    detector, callback, calls = registered()
    control = CloseControl()
    detector._wake_condition = ConditionLeaf(detector._wake_condition, control, phase)
    detector._wake_local = LocalLeaf(
        control, get_fault=2 if phase == 'tls_previous_read' else 0,
        set_fault={'tls_install': 1, 'tls_restore': 2}.get(phase, 0),
    )
    with pytest.raises(CloseControl) as caught:
        detector.set_threshold(0.61)
    assert caught.value is control
    assert detector._wake_active == 0 and detector._on_detection is callback
    assert not calls and not detector._loaded and detector._oww_model is None
    detector.set_threshold(0.72)
    assert detector.threshold == 0.72 and detector._wake_active == 0
    detector.close()
    assert detector._on_detection is None


def test_actual_worker_guard_survives_missing_tls_before_context_dispatch():
    detector, _, calls = registered()

    def body():
        del detector._wake_local.lease
        for close in (detector.close, detector.begin_close):
            with pytest.raises(RuntimeError, match='^wake_close_from_worker$'):
                close()
        assert not detector._wake_sealed and detector._wake_active == 1

    detector._model_lock = ModelLockLeaf(detector._model_lock, body)
    detector.set_threshold(0.61)
    assert detector._wake_active == 0 and not calls
    detector.close()


def test_nested_restore_fault_keeps_the_parent_live_after_seal():
    detector, _, calls = registered()
    control = CloseControl()
    detector._wake_local = LocalLeaf(control, set_fault=3)

    def nested():
        detector.request_close()
        with pytest.raises(CloseControl) as caught:
            detector.set_threshold(0.73)
        assert caught.value is control and detector._wake_active == 1
        with pytest.raises(RuntimeError, match='^wake_close_from_worker$'):
            detector.begin_close()

    detector._model_lock = ModelLockLeaf(detector._model_lock, nested)
    detector.set_threshold(0.61)
    assert detector._wake_active == 0 and not calls
    detector.close()


def test_foreign_thread_cannot_borrow_the_live_wake_lease_from_tls():
    detector, _, calls = registered()
    entered, release = threading.Event(), threading.Event()
    errors, captured = [], []

    def body():
        captured.append(detector._wake_local.lease)
        entered.set()
        assert release.wait(3)

    def operation():
        try:
            detector.set_threshold(0.61)
        except BaseException as error:
            errors.append(type(error).__name__)

    detector._model_lock = ModelLockLeaf(detector._model_lock, body)
    worker = threading.Thread(target=operation, daemon=True)
    worker.start()
    try:
        assert entered.wait(3)
        detector.request_close()
        detector._wake_local.lease = captured[0]
        with pytest.raises(RuntimeError, match='^wake_closed$'):
            detector.set_threshold(0.73)
        assert detector._wake_active == 1
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors and not calls
    assert detector._wake_active == 0
    detector.close()


def test_reset_remains_reusable_and_does_not_release_the_callback():
    detector, callback, calls = registered()
    for _ in range(2):
        detector.reset()
        detector.reset_refractory()
        detector.set_threshold(0.61)
        assert detector._on_detection is callback and not detector._wake_sealed
    assert not calls
    detector.close()
    assert detector._on_detection is None
    with pytest.raises(RuntimeError, match='^wake_closed$'):
        detector.reset()
