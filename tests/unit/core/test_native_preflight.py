"""Actual worker-entry ordering; runtime authority payloads are synthetic here."""
from __future__ import annotations

import asyncio
from contextlib import closing
from contextvars import copy_context
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from src.core import owned_work, runtime_admission
from src.core.owned_work import OwnedBlockingWork, current_native_execution, validate_native_execution


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [False, True], ids=["run", "native_step_after_seal"])
async def test_preflight_enters_before_original_callback_and_outside_native_lock(tmp_path, sequence):
    work = OwnedBlockingWork()
    order, executions = [], []
    material_lock = threading.RLock()
    def original(value, *, marker):
        assert order == ["preflight"]
        with material_lock, closing(sqlite3.connect(tmp_path / "original.db")) as database, database:
            database.execute("CREATE TABLE value (n INTEGER)")
            database.execute("INSERT INTO value VALUES (?)", (value,))
        order.append("callback")
        return marker
    def preflight(execution):
        assert current_native_execution() is execution
        assert validate_native_execution(execution, work=work, function=original)
        assert not owned_work._NATIVE_LOCK._is_owned()
        assert not material_lock._is_owned()
        assert execution.retained_task in work._tasks
        assert not (tmp_path / "original.db").exists()
        executions.append(execution)
        order.append("preflight")
    owner = object()
    work._install_native_preflight(owner, preflight)
    work._install_native_preflight(owner, lambda execution: pytest.fail("observer replaced"))
    with pytest.raises(RuntimeError, match="native_preflight_owner_changed"):
        work._install_native_preflight(object(), preflight)
    marker = object()
    async def admitted_sequence():
        work.begin_close()
        return await work.native_step(original, 7, marker=marker)
    try:
        result = (await work.run_async(admitted_sequence) if sequence
            else await work.run(original, 7, marker=marker))
        assert result is marker and order == ["preflight", "callback"]
        assert len(executions) == 1 and not validate_native_execution(executions[0])
        with closing(sqlite3.connect(tmp_path / "original.db")) as database:
            assert database.execute("SELECT n FROM value").fetchall() == [(7,)]
    finally:
        await work.aclose()
    assert work.status()["closed"] and work.status()["operations"] == 0
    assert work._native_executions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [False, True], ids=["run", "native_step"])
async def test_failed_preflight_never_calls_original_and_releases_actual_provenance(sequence):
    work, called, saved = OwnedBlockingWork(), [], []
    failure = RuntimeError("full validation failed")
    def original():
        called.append(True)
    def preflight(execution):
        saved.append(execution)
        raise failure
    work._install_native_preflight(object(), preflight)
    async def admitted_sequence():
        return await work.native_step(original)
    try:
        with pytest.raises(RuntimeError) as error:
            if sequence:
                await work.run_async(admitted_sequence)
            else:
                await work.run(original)
        assert error.value is failure and called == []
        assert len(saved) == 1 and not validate_native_execution(saved[0])
        assert current_native_execution() is None and work._native_executions == {}
    finally:
        await work.aclose()
    assert work.status()["operations"] == 0


@pytest.mark.asyncio
async def test_runtime_witness_is_exact_execution_and_keeps_its_full_observation(tmp_path, monkeypatch):
    binding = {"state_root": str(tmp_path)}
    admission = runtime_admission.RuntimeAdmission({}, binding, str(tmp_path / "privacy.db"), "unit")
    snapshots, live = [], []
    def snapshot(state, ticket):
        value = object()
        snapshots.append(value)
        return value
    def validate_live(state, ticket, value):
        live.append(value)
        return binding
    monkeypatch.setattr(runtime_admission, "_stage", lambda: SimpleNamespace(
        admission_validation_snapshot=snapshot, validate_admission_live=validate_live))
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: True})
    work, saved = OwnedBlockingWork(), []
    def original():
        execution = current_native_execution()
        saved.append(execution)
        admission._revalidate_native(execution)
        original_snapshot = snapshots[-1]
        # A later full validation updates the shared view. The running
        # invocation's live check must still use its own earlier observation.
        admission.revalidate()
        assert snapshots[-1] is not original_snapshot
        admission._revalidate_live()
        assert live[-1] is original_snapshot
        cloned = owned_work.NativeExecution(work, execution.function, execution.task, execution.parent_sequence)
        with pytest.raises(RuntimeError, match="runtime_admission_native_owner"):
            admission._revalidate_native(cloned)
        copied = copy_context()
        async def foreign():
            with pytest.raises(RuntimeError, match="runtime_admission_native_owner"):
                admission._revalidate_native(execution)
        copied.run(asyncio.run, foreign())
    def later_without_preflight():
        with pytest.raises(RuntimeError, match="runtime_admission_native_owner"):
            admission._revalidate_native(saved[0])
        with pytest.raises(RuntimeError, match="runtime_admission_native_owner"):
            admission._revalidate_native(current_native_execution())
    def preflight(execution):
        if execution.function is original:
            admission._preflight_native(execution)
    work._install_native_preflight(admission, preflight)
    try:
        await work.run(original)
        await work.run(later_without_preflight)
    finally:
        await work.aclose()
    assert len(snapshots) == 2


@pytest.mark.asyncio
async def test_retained_completed_witnesses_are_pruned_but_live_native_owners_are_not(tmp_path, monkeypatch):
    binding = {"state_root": str(tmp_path)}
    admission = runtime_admission.RuntimeAdmission({}, binding, str(tmp_path / "privacy.db"), "unit")
    monkeypatch.setattr(runtime_admission, "_stage", lambda: SimpleNamespace(
        admission_validation_snapshot=lambda state, ticket: object(),
        validate_admission_live=lambda state, ticket, snapshot: binding))
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: True})
    monkeypatch.setattr(runtime_admission, "_MAX_NATIVE_WITNESSES", 2)
    work, saved, entered = OwnedBlockingWork(), [], [threading.Event(), threading.Event()]
    release = threading.Event()
    def original(index=None):
        execution = current_native_execution()
        saved.append(execution)
        admission._revalidate_native(execution)
        if index is not None:
            entered[index].set()
            if not release.wait(5):
                raise TimeoutError("test did not release original native work")
    work._install_native_preflight(admission, admission._preflight_native)
    pending = []
    try:
        for _ in range(5):
            await work.run(original)
        assert len(saved) == 5 and len(admission._native_validations) <= 2
        pending = [asyncio.create_task(work.run(original, index)) for index in range(2)]
        assert await asyncio.to_thread(entered[0].wait, 3)
        assert await asyncio.to_thread(entered[1].wait, 3)
        live = tuple(admission._native_validations)
        assert len(live) == 2 and all(not execution.retained_task.done() for execution in live)
        with pytest.raises(RuntimeError, match="^runtime_admission_native_capacity$"):
            await work.run(original)
        assert set(admission._native_validations) == set(live)
        assert len(saved) == 7 and all(not task.done() for task in pending)
        release.set()
        await asyncio.gather(*pending)
        await work.run(original)
        assert len(saved) == 8 and len(admission._native_validations) == 1
    finally:
        release.set()
        await asyncio.gather(*pending, return_exceptions=True)
        await work.aclose()
    assert work.status()["operations"] == 0
