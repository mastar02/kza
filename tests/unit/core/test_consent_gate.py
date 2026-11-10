"""CloudConsentGate: a one-way, process-local runtime latch (OQ-8)."""
import asyncio
import copy
import pickle
import threading

import pytest

from src.core.egress import CloudConsentGate, ConsentWithdrawalSource, EgressDenied


def test_withdraw_is_one_way_and_idempotent():
    gate = CloudConsentGate()
    assert gate.withdrawn is False and gate.source is None
    gate.require_open()
    assert gate.withdraw(ConsentWithdrawalSource.ACTOR) is True
    assert gate.withdrawn is True and gate.source is ConsentWithdrawalSource.ACTOR
    assert gate.withdraw(ConsentWithdrawalSource.DURABLE_MARKER) is False
    assert gate.source is ConsentWithdrawalSource.ACTOR  # The first source is kept.
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        gate.require_open()
    assert not any(name in dir(gate) for name in ("reopen", "reset", "grant", "clear"))


def test_withdraw_rejects_unknown_source_without_closing():
    gate = CloudConsentGate()
    with pytest.raises(ValueError):
        gate.withdraw("someone")
    assert gate.withdrawn is False


def test_subscribers_before_and_after_withdrawal_are_called_exactly_once():
    gate = CloudConsentGate()
    calls = []
    gate.subscribe(lambda: calls.append("early"))
    assert calls == []
    gate.withdraw(ConsentWithdrawalSource.ACTOR)
    assert calls == ["early"]
    gate.subscribe(lambda: calls.append("late"))
    assert calls == ["early", "late"]
    gate.withdraw(ConsentWithdrawalSource.ACTOR)
    assert calls == ["early", "late"]


def test_subscriber_exception_neither_propagates_nor_reopens(caplog):
    gate = CloudConsentGate()
    calls = []

    def broken():
        raise RuntimeError("synthetic subscriber failure")
    gate.subscribe(broken)
    gate.subscribe(lambda: calls.append("next"))
    assert gate.withdraw(ConsentWithdrawalSource.ACTOR) is True
    assert gate.withdrawn is True and calls == ["next"]
    gate.subscribe(broken)  # Immediate call after withdrawal is guarded too.
    assert gate.withdrawn is True
    assert "synthetic subscriber failure" in caplog.text


def test_bound_method_subscribers_are_held_weakly():
    gate = CloudConsentGate()
    calls = []

    class Owner:
        def on_withdrawn(self):
            calls.append(self)
    owner = Owner()
    gate.subscribe(owner.on_withdrawn)
    del owner  # Refcounting alone releases it: the gate holds no strong reference.
    gate.withdraw(ConsentWithdrawalSource.ACTOR)
    assert calls == []


def test_copy_and_deepcopy_preserve_identity_and_pickle_is_refused():
    gate = CloudConsentGate()
    assert copy.copy(gate) is gate and copy.deepcopy(gate) is gate
    assert copy.deepcopy({"gate": gate})["gate"] is gate
    with pytest.raises(TypeError):
        pickle.dumps(gate)


async def test_withdrawal_in_a_worker_thread_is_visible_on_the_loop():
    gate = CloudConsentGate()
    assert await asyncio.to_thread(gate.withdraw, ConsentWithdrawalSource.ACTOR) is True
    assert gate.withdrawn is True


def test_concurrent_subscribe_and_withdraw_never_loses_a_subscriber():
    for _ in range(200):
        gate = CloudConsentGate()
        calls = []
        barrier = threading.Barrier(9)

        def subscribe(index):
            barrier.wait()
            gate.subscribe(lambda: calls.append(index))

        def withdraw():
            barrier.wait()
            gate.withdraw(ConsentWithdrawalSource.ACTOR)
        threads = [threading.Thread(target=subscribe, args=(index,)) for index in range(8)]
        threads.append(threading.Thread(target=withdraw))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(calls) == list(range(8))
