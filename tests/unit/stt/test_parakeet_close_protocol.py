"""Close election and publication through real preloaded Parakeet adapters."""
from __future__ import annotations

import asyncio
import threading

import pytest

from src.application import STTExecutorOwner
from src.stt.parakeet_stt import ParakeetSTT


class CloseControl(BaseException):
    pass


class CloseCall:
    def __init__(self, model):
        self.error = None
        self.returned = False
        self.finished = threading.Event()

        def invoke():
            try:
                model.close()
            except BaseException as error:
                self.error = error
            else:
                self.returned = True
            finally:
                self.finished.set()

        self.thread = threading.Thread(target=invoke, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def join(self):
        self.thread.join(3)
        assert not self.thread.is_alive() and self.finished.is_set()
        return self


def assert_failed(call):
    assert not call.returned
    assert type(call.error) is RuntimeError
    assert call.error.args == ('parakeet_close_failed',)


def preloaded():
    model = ParakeetSTT()
    sentinel = object()
    model._model = sentinel
    calls = []
    original = model._parakeet_release_resources

    def dispose():
        calls.append(threading.current_thread())
        return original()

    model._parakeet_release_resources = dispose
    return model, sentinel, calls


class ConditionFault:
    def __init__(self, model, calls, phase, control):
        self.model = model
        self.calls = calls
        self.phase = phase
        self.control = control
        self.condition = model._parakeet_condition
        self.armed = True
        self.faults = 0

    def fail(self):
        self.armed = False
        self.faults += 1
        raise self.control

    def owned_pending(self):
        closing = self.model._parakeet_close
        return (closing is not None and closing.result is None
                and closing.owner is threading.current_thread())

    def __enter__(self):
        if self.armed:
            if (self.phase == 'before_election' and self.model._parakeet_sealed
                    and self.model._parakeet_close is None):
                self.fail()
            if (self.phase == 'finalization_enter' and self.calls
                    and self.owned_pending()):
                self.fail()
        return self.condition.__enter__()

    def __exit__(self, *args):
        result = self.condition.__exit__(*args)
        if self.armed and self.owned_pending():
            if self.phase == 'election_exit' and not self.calls:
                self.fail()
            if self.phase == 'finalization_exit' and self.calls:
                self.fail()
        return result

    def notify_all(self):
        result = self.condition.notify_all()
        if (self.armed and self.phase == 'finalization_notify' and self.calls
                and self.owned_pending()):
            self.fail()
        return result

    def __getattr__(self, name):
        return getattr(self.condition, name)


def test_unpublished_election_control_preserves_caller_and_can_close_later():
    model, sentinel, calls = preloaded()
    control = CloseControl()
    fault = ConditionFault(model, calls, 'before_election', control)
    model._parakeet_condition = fault
    with pytest.raises(CloseControl) as caught:
        model.close()
    assert caught.value is control and fault.faults == 1
    assert model._parakeet_close is None and model._model is sentinel
    assert not calls
    model.close()
    assert model._model is None and len(calls) == 1
    model.close()
    assert len(calls) == 1


@pytest.mark.parametrize('phase', [
    'election_exit', 'finalization_enter', 'finalization_notify',
    'finalization_exit',
])
def test_elected_metadata_failure_is_one_terminal_outcome_without_retry(phase):
    model, sentinel, calls = preloaded()
    fault = ConditionFault(model, calls, phase, CloseControl())
    model._parakeet_condition = fault
    assert_failed(CloseCall(model).start().join())
    receipt = model._parakeet_close
    assert receipt.result == ('parakeet_close_failed',) and fault.faults == 1
    expected = 0 if phase == 'election_exit' else 1
    assert len(calls) == expected
    assert model._model is (sentinel if expected == 0 else None)
    assert_failed(CloseCall(model).start().join())
    assert model._parakeet_close is receipt and len(calls) == expected


@pytest.mark.parametrize('after_release', [False, True])
def test_disposal_control_never_retries_or_infers_success_from_model_none(after_release):
    model, sentinel, calls = preloaded()
    original = model._parakeet_release_resources
    attempts = []

    def dispose():
        attempts.append(threading.current_thread())
        if after_release:
            original()
        raise CloseControl()

    model._parakeet_release_resources = dispose
    assert_failed(CloseCall(model).start().join())
    receipt = model._parakeet_close
    assert receipt.result == ('parakeet_close_failed',)
    assert len(attempts) == 1 and len(calls) == int(after_release)
    assert model._model is (None if after_release else sentinel)
    assert_failed(CloseCall(model).start().join())
    assert model._parakeet_close is receipt and len(attempts) == 1


def test_failed_drain_retains_model_until_actual_operation_exits_without_retry():
    model, sentinel, calls = preloaded()
    entered = threading.Event()
    release = threading.Event()
    worker_errors = []
    original_lock = model._model_lock

    class LockLeaf:
        def __enter__(self):
            result = original_lock.__enter__()
            entered.set()
            if not release.wait(3):
                original_lock.__exit__(None, None, None)
                raise AssertionError('operation_gate_timeout')
            return result

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    class WaitFault(ConditionFault):
        def wait(self, *args, **kwargs):
            if self.armed and self.owned_pending() and model._parakeet_active:
                self.fail()
            return self.condition.wait(*args, **kwargs)

    def operation():
        try:
            model.load()
        except BaseException as error:
            worker_errors.append(type(error).__name__)

    model._model_lock = LockLeaf()
    fault = WaitFault(model, calls, 'drain_wait', CloseControl())
    model._parakeet_condition = fault
    worker = threading.Thread(target=operation, daemon=True)
    worker.start()
    try:
        assert entered.wait(3) and model._parakeet_active == 1
        assert_failed(CloseCall(model).start().join())
        assert fault.faults == 1 and not calls
        assert model._model is sentinel and model._parakeet_active == 1
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not worker_errors
    assert model._parakeet_active == 0
    assert_failed(CloseCall(model).start().join())
    assert not calls and model._model is sentinel


def test_success_is_not_visible_during_fallible_notification_exit():
    model, _, calls = preloaded()
    dispose_entered = threading.Event()
    dispose_release = threading.Event()
    wait_entered = threading.Event()
    metadata_exit_entered = threading.Event()
    metadata_exit_release = threading.Event()
    pending_observed = threading.Event()
    original_dispose = model._parakeet_release_resources
    original_condition = model._parakeet_condition
    owner = CloseCall(model)
    follower = CloseCall(model)
    control = CloseControl()

    def dispose():
        dispose_entered.set()
        assert dispose_release.wait(3)
        original_dispose()

    class ConditionLeaf:
        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, *args):
            result = original_condition.__exit__(*args)
            if (threading.current_thread() is owner.thread and calls
                    and model._parakeet_close.result is None):
                metadata_exit_entered.set()
                assert metadata_exit_release.wait(3)
                raise control
            return result

        def wait(self, *args, **kwargs):
            if threading.current_thread() is follower.thread:
                wait_entered.set()
            result = original_condition.wait(*args, **kwargs)
            if (threading.current_thread() is follower.thread
                    and metadata_exit_entered.is_set()):
                pending_observed.set()
            return result

        def notify_all(self):
            return original_condition.notify_all()

    model._parakeet_release_resources = dispose
    model._parakeet_condition = ConditionLeaf()
    owner.start()
    try:
        assert dispose_entered.wait(3)
        follower.start()
        assert wait_entered.wait(3)
        dispose_release.set()
        assert metadata_exit_entered.wait(3)
        assert pending_observed.wait(3)
        assert model._model is None and len(calls) == 1
        assert not follower.finished.wait(0.05)
        assert model._parakeet_close.result is None
    finally:
        dispose_release.set()
        metadata_exit_release.set()
        owner.join()
        if follower.thread.ident is not None:
            follower.join()
    assert_failed(owner)
    assert_failed(follower)
    assert len(calls) == 1
    assert model._parakeet_close.result == ('parakeet_close_failed',)


def test_follower_control_does_not_poison_the_elected_owner():
    model, _, calls = preloaded()
    dispose_entered = threading.Event()
    dispose_release = threading.Event()
    original_dispose = model._parakeet_release_resources
    original_condition = model._parakeet_condition
    owner = CloseCall(model)
    follower = CloseCall(model)
    control = CloseControl()

    def dispose():
        dispose_entered.set()
        assert dispose_release.wait(3)
        original_dispose()

    class ConditionLeaf:
        follower_exits = 0

        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, *args):
            result = original_condition.__exit__(*args)
            if threading.current_thread() is follower.thread:
                self.follower_exits += 1
                if self.follower_exits == 3:
                    assert model._parakeet_close.owner is owner.thread
                    assert model._parakeet_close.result is None
                    raise control
            return result

        def __getattr__(self, name):
            return getattr(original_condition, name)

    model._parakeet_release_resources = dispose
    leaf = ConditionLeaf()
    model._parakeet_condition = leaf
    owner.start()
    try:
        assert dispose_entered.wait(3)
        follower.start().join()
        assert follower.error is control and not follower.returned
        assert leaf.follower_exits == 3
        assert model._parakeet_close.result is None
    finally:
        dispose_release.set()
        owner.join()
    assert owner.returned and owner.error is None
    assert model._model is None and len(calls) == 1
    model.close()
    assert len(calls) == 1 and model._parakeet_close.result == (None,)


@pytest.mark.asyncio
async def test_actual_model_owner_reports_terminal_failure_and_keeps_model():
    model, sentinel, calls = preloaded()
    model._parakeet_condition = ConditionFault(
        model, calls, 'election_exit', CloseControl(),
    )
    owner = STTExecutorOwner()
    owner.add_model(model)
    closing = owner.begin_close()
    with pytest.raises(RuntimeError, match='^parakeet_close_failed$'):
        await asyncio.wait_for(asyncio.shield(closing), 3)
    assert closing.done() and not closing.cancelled()
    assert owner.begin_close() is closing
    assert owner._models[id(model)] is model
    assert model._model is sentinel and not calls
    with pytest.raises(RuntimeError, match='^parakeet_close_failed$'):
        await model.begin_close()
    assert model._model is sentinel and not calls
