"""Wake close faults distinguish disposer entry, return and publication."""
import asyncio
import threading

import pytest

from src.application import WakeModelOwner
from tests.unit.stt.test_parakeet_close_protocol import CloseCall, CloseControl
from tests.unit.wakeword.test_wake_admission_metadata import registered


def tracked():
    detector, callback, calls = registered()
    entries, returned = [], []
    original = detector._wake_release_resources

    def dispose():
        entries.append(threading.current_thread())
        result = original()
        returned.append(threading.current_thread())
        return result

    detector._wake_release_resources = dispose
    return detector, callback, calls, entries, returned


def assert_failed(call):
    assert not call.returned
    assert type(call.error) is RuntimeError
    assert call.error.args == ('wake_close_failed',)


class ConditionFault:
    def __init__(self, detector, entries, returned, phase, control):
        self.detector = detector
        self.entries = entries
        self.returned = returned
        self.phase = phase
        self.control = control
        self.condition = detector._wake_condition
        self.armed = True
        self.faults = 0

    def fail(self):
        self.armed = False
        self.faults += 1
        raise self.control

    def owned_pending(self):
        selected = self.detector._wake_close
        return (selected is not None and selected.result is None
                and selected.owner is threading.current_thread())

    def __enter__(self):
        if self.armed:
            if (self.phase == 'before_election' and self.detector._wake_sealed
                    and self.detector._wake_close is None):
                self.fail()
            if self.owned_pending():
                if (self.phase == 'disposer_enter' and self.entries
                        and not self.returned):
                    self.fail()
                if self.phase == 'finalization_enter' and self.returned:
                    self.fail()
        return self.condition.__enter__()

    def __exit__(self, *args):
        result = self.condition.__exit__(*args)
        if self.armed and self.owned_pending():
            if self.phase == 'election_exit' and not self.entries:
                self.fail()
            if self.phase == 'disposer_exit' and self.entries and not self.returned:
                self.fail()
            if self.phase == 'finalization_exit' and self.returned:
                self.fail()
        return result

    def notify_all(self):
        result = self.condition.notify_all()
        if (self.armed and self.owned_pending() and self.returned
                and self.phase == 'finalization_notify'):
            self.fail()
        return result

    def __getattr__(self, name):
        return getattr(self.condition, name)


def test_unpublished_close_control_keeps_callback_and_allows_later_close():
    detector, callback, calls, entries, returned = tracked()
    control = CloseControl()
    fault = ConditionFault(detector, entries, returned, 'before_election', control)
    detector._wake_condition = fault
    with pytest.raises(CloseControl) as caught:
        detector.close()
    assert caught.value is control and fault.faults == 1
    assert detector._wake_close is None and detector._on_detection is callback
    assert not entries and not returned and not calls
    detector.close()
    assert len(entries) == len(returned) == 1 and detector._on_detection is None


@pytest.mark.parametrize('phase', [
    'election_exit', 'disposer_enter', 'disposer_exit',
    'finalization_enter', 'finalization_notify', 'finalization_exit',
])
def test_close_failure_has_phase_accurate_ownership_and_never_retries(phase):
    detector, callback, calls, entries, returned = tracked()
    fault = ConditionFault(detector, entries, returned, phase, CloseControl())
    detector._wake_condition = fault
    assert_failed(CloseCall(detector).start().join())
    receipt = detector._wake_close
    assert receipt.result == ('wake_close_failed',) and fault.faults == 1
    if phase == 'election_exit':
        assert not entries and not returned and detector._on_detection is callback
    elif phase.startswith('disposer_'):
        assert len(entries) == 1 and not returned
        assert detector._on_detection is None
    else:
        assert len(entries) == len(returned) == 1
        assert detector._on_detection is None
    before = (len(entries), len(returned))
    assert_failed(CloseCall(detector).start().join())
    assert detector._wake_close is receipt
    assert (len(entries), len(returned)) == before and not calls


def test_actual_close_progresses_when_operation_and_final_notifications_are_lost():
    detector, _, calls, entries, returned = tracked()
    entered, release, waited = threading.Event(), threading.Event(), threading.Event()
    original_lock = detector._model_lock
    original_condition = detector._wake_condition
    errors = []

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

        def wait(self, *args, **kwargs):
            waited.set()
            return original_condition.wait(*args, **kwargs)

        def notify_all(self):
            self.suppressed += 1

    def operation():
        try:
            detector.set_threshold(0.61)
        except BaseException as error:
            errors.append(type(error).__name__)

    detector._model_lock = LockLeaf()
    condition = ConditionLeaf()
    detector._wake_condition = condition
    worker = threading.Thread(target=operation, daemon=True)
    closer = CloseCall(detector)
    worker.start()
    try:
        assert entered.wait(3)
        closer.start()
        assert waited.wait(3)
    finally:
        release.set()
        worker.join(3)
        if closer.thread.ident is not None:
            closer.join()
    assert not worker.is_alive() and not errors
    assert closer.returned and closer.error is None
    assert len(entries) == len(returned) == 1 and not calls
    assert detector._on_detection is None and condition.suppressed >= 3


@pytest.mark.asyncio
async def test_actual_wake_owner_task_reports_failure_and_retains_callback():
    detector, callback, calls, entries, returned = tracked()
    detector._wake_condition = ConditionFault(
        detector, entries, returned, 'election_exit', CloseControl(),
    )
    owner = WakeModelOwner()
    owner.add_model(detector)
    closing = owner.begin_close()
    with pytest.raises(RuntimeError, match='^wake_close_failed$'):
        await asyncio.wait_for(asyncio.shield(closing), 3)
    assert closing.done() and not closing.cancelled() and owner.begin_close() is closing
    assert owner._models[id(detector)] is detector
    assert detector._on_detection is callback and not entries and not returned
    with pytest.raises(RuntimeError, match='^wake_close_failed$'):
        await detector.begin_close()
    assert detector._on_detection is callback and not entries and not calls


def test_success_is_not_visible_before_final_notification_exit():
    detector, _, calls, entries, returned = tracked()
    dispose_entered, dispose_release = threading.Event(), threading.Event()
    wait_entered, metadata_entered = threading.Event(), threading.Event()
    metadata_release, pending_observed = threading.Event(), threading.Event()
    original_dispose = detector._wake_release_resources
    original_condition = detector._wake_condition
    owner, follower = CloseCall(detector), CloseCall(detector)

    def dispose():
        dispose_entered.set()
        assert dispose_release.wait(3)
        return original_dispose()

    class ConditionLeaf:
        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, *args):
            result = original_condition.__exit__(*args)
            # A returned receipt excludes the Condition INSIDE Wake disposal.
            if (threading.current_thread() is owner.thread and returned
                    and detector._wake_close.result is None):
                metadata_entered.set()
                assert metadata_release.wait(3)
                raise CloseControl()
            return result

        def wait(self, *args, **kwargs):
            if threading.current_thread() is follower.thread:
                wait_entered.set()
            result = original_condition.wait(*args, **kwargs)
            if threading.current_thread() is follower.thread and metadata_entered.is_set():
                pending_observed.set()
            return result

        def notify_all(self):
            return original_condition.notify_all()

    detector._wake_release_resources = dispose
    detector._wake_condition = ConditionLeaf()
    owner.start()
    try:
        assert dispose_entered.wait(3)
        follower.start()
        assert wait_entered.wait(3)
        dispose_release.set()
        assert metadata_entered.wait(3) and pending_observed.wait(3)
        assert len(entries) == len(returned) == 1
        assert detector._on_detection is None and not calls
        assert not follower.finished.wait(0.05)
        assert detector._wake_close.result is None
    finally:
        dispose_release.set()
        metadata_release.set()
        owner.join()
        if follower.thread.ident is not None:
            follower.join()
    assert_failed(owner)
    assert_failed(follower)
    assert len(entries) == len(returned) == 1


def test_follower_control_does_not_change_the_elected_wake_result():
    detector, _, calls, entries, returned = tracked()
    entered, release = threading.Event(), threading.Event()
    original_dispose = detector._wake_release_resources
    original_condition = detector._wake_condition
    owner, follower = CloseCall(detector), CloseCall(detector)
    control = CloseControl()

    def dispose():
        entered.set()
        assert release.wait(3)
        return original_dispose()

    class ConditionLeaf:
        follower_exits = 0

        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, *args):
            result = original_condition.__exit__(*args)
            if threading.current_thread() is follower.thread:
                self.follower_exits += 1
                if self.follower_exits == 3:
                    assert detector._wake_close.owner is owner.thread
                    assert detector._wake_close.result is None
                    raise control
            return result

        def __getattr__(self, name):
            return getattr(original_condition, name)

    detector._wake_release_resources = dispose
    condition = ConditionLeaf()
    detector._wake_condition = condition
    owner.start()
    try:
        assert entered.wait(3)
        follower.start().join()
        assert follower.error is control and not follower.returned
        assert condition.follower_exits == 3 and detector._wake_close.result is None
    finally:
        release.set()
        owner.join()
    assert owner.returned and owner.error is None
    assert len(entries) == len(returned) == 1 and not calls
    detector.close()
    assert len(entries) == 1 and detector._wake_close.result == (None,)
