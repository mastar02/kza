"""Real SQLite/owner ordering under the documented synthetic admission fixture."""
from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from src.analytics.event_logger import EventLogger
from src.analytics.event_storage import event_descriptor
from src.core import runtime_admission
from src.core.owned_work import current_native_execution
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory.privacy_services import build_privacy_services
from tests.unit.memory.test_output_registration_families import (
    assert_failed_registry_cleanup, managed, observations,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["primary_nested", "original_retention"])
async def test_full_inventory_precedes_real_sqlite_transactions_and_locked_observations(tmp_path, monkeypatch, route):
    from src.analytics.erasure import build_event_erasure
    from src.analytics.retention import EventRetention
    from src.memory.erasure_coordinator import ErasureCoordinator
    monkeypatch.setattr(EventRetention, "start", lambda self: None)
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        scanned, connections, supported = [], [], []
        real_inventory = runtime_admission._stage()._target_inventory
        real_connect = sqlite3.connect
        def connect(*args, **kwargs):
            database = real_connect(*args, **kwargs)
            connections.append((threading.get_ident(), database))
            return database
        def full_validation():
            assert not owner._lock._is_owned()
            privacy_binding = owner._roles.get("privacy_authority")
            if privacy_binding is not None:
                assert not privacy_binding.instance._lock._is_owned()
                assert not privacy_binding.instance.ledger._lock._is_owned()
            for thread, database in connections:
                if thread != threading.get_ident():
                    continue
                try:
                    assert not database.in_transaction, "full inventory entered its own target transaction"
                except sqlite3.ProgrammingError:
                    pass  # An original transient close already returned.
            # Run the real SQLite inventory, not a replacement no-op scan.
            real_inventory(owner.admission.state_root)
            scanned.append(current_native_execution())
        monkeypatch.setattr(sqlite3, "connect", connect)
        monkeypatch.setattr(owner.admission, "revalidate", full_validation)
        privacy = await build_privacy_services(config, shutdown=stack)
        events = EventLogger(config["analytics"]["events_db"], source_policy=privacy.ledger)
        stack.push_async_callback(events.aclose)
        privacy.add_borrower(events.begin_close)
        await registration.attach_output(events, "events")
        await events.acall(events.initialize)
        assert events._initialized and privacy._ready
        if route == "primary_nested":
            def original():
                with registration.native_connection(events, events.db_path, sqlite3.connect, events.db_path) as first:
                    first.execute("BEGIN IMMEDIATE")
                    # This second primary open occurs inside the same exact
                    # original invocation while the first SQLite writer is live.
                    with registration.native_connection(events, events.db_path, sqlite3.connect, events.db_path) as second:
                        assert second.execute("SELECT 1").fetchone() == (1,)
                        assert first.in_transaction
                    first.rollback()
            before = len(scanned)
            await events.acall(original)
            assert len(scanned) == before + 1
        else:
            erasures = ErasureCoordinator(privacy)
            stack.push_async_callback(erasures.aclose)
            driver = await build_event_erasure(config, privacy=privacy, owner=erasures)
            real_observe = registration.observe_history
            def observe(path, kind, state):
                if state == "supported":
                    live = []
                    for thread, database in connections:
                        if thread != threading.get_ident():
                            continue
                        try:
                            live.append(database.in_transaction)
                        except sqlite3.ProgrammingError:
                            pass
                    assert any(live), "original representation observation left its actual transaction"
                    supported.append(True)
                return real_observe(path, kind, state)
            monkeypatch.setattr(registration, "observe_history", observe)
            descriptor = event_descriptor(events.db_path)
            before = len(scanned)
            outcome = await driver.retention._run({"descriptor": descriptor}, 0)
            assert outcome.error is None and supported
            assert len(scanned) == before + 1
            records = observations(owner)
            assert any(r.get("kind") == "physical" and r.get("state") == "supported" for r in records)
        assert not owner._failed
        assert len(owner._open_extents) == 1
    assert owner._open_extents == {} and owner._opening_extents == 0
    assert privacy.ledger._db is None and privacy._lease is None


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
async def test_inherited_v2_async_store_keeps_full_gate_before_native_connect(tmp_path, monkeypatch, producer):
    import aiosqlite
    from src.lists.list_store import ListStore
    from src.reminders.reminder_store import ReminderStore
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = (ListStore if producer == "lists" else ReminderStore)(config[producer]["db_path"])
        stack.push_async_callback(store.close)
        await owner.register_store(store, producer)
        full, connected = [], []
        def changed_authority():
            assert not owner._lock._is_owned()
            full.append(True)
            raise RuntimeError("changed_full_authority")
        def forbidden(*args, **kwargs):
            connected.append(True)
            raise AssertionError("connector reached after full admission failed")
        monkeypatch.setattr(owner.admission, "revalidate", changed_authority)
        monkeypatch.setattr(aiosqlite, "connect", forbidden)
        with pytest.raises(RuntimeError, match="^changed_full_authority$"):
            await store.initialize()
        assert full == [True] and connected == [] and store._db is None


@pytest.mark.asyncio
async def test_completed_dispatches_do_not_accumulate_when_callers_keep_execution_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(registration, "_MAX_NATIVE_WITNESSES", 2)
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = await build_privacy_services(config, shutdown=stack)
        events = EventLogger(config["analytics"]["events_db"], source_policy=privacy.ledger)
        stack.push_async_callback(events.aclose)
        privacy.add_borrower(events.begin_close)
        await registration.attach_output(events, "events")
        await events.acall(events.initialize)
        saved = []
        def original():
            saved.append(current_native_execution())
            with registration.native_connection(events, events.db_path, sqlite3.connect, events.db_path) as database:
                return database.execute("SELECT 1").fetchone()
        for _ in range(6):
            assert await events.acall(original) == (1,)
            assert len(owner._dispatches) <= 2
        assert len(saved) == 6 and all(execution.retained_task.done() for execution in saved)
        assert not owner._failed and events._work.status()["operations"] == 0
    assert owner._open_extents == {} and privacy.ledger._db is None


@pytest.mark.asyncio
async def test_coordinator_capacity_preserves_the_original_live_dispatch_and_connection(tmp_path, monkeypatch):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            events = EventLogger(config["analytics"]["events_db"], source_policy=privacy.ledger)
            stack.push_async_callback(events.aclose)
            privacy.add_borrower(events.begin_close)
            await registration.attach_output(events, "events")
            await events.acall(events.initialize)
            # EventLogger owns two actual executor threads: one retained owner
            # can stay entered while the second thread exercises the cap.
            monkeypatch.setattr(registration, "_MAX_NATIVE_WITNESSES", 1)
            entered, release = threading.Event(), threading.Event()
            retained, forbidden = [], []
            def original():
                execution = current_native_execution()
                with registration.native_connection(events, events.db_path, sqlite3.connect, events.db_path) as database:
                    retained.append((execution, database))
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release actual event callback")
                    return database.execute("SELECT 1").fetchone()
            def must_not_enter():
                forbidden.append(True)
                raise AssertionError("capacity refusal called original body")
            operation = asyncio.create_task(events.acall(original))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                execution, database = retained[0]
                with pytest.raises(registration.OutputRegistrationError,
                        match="^output_registration_native_capacity$"):
                    await events.acall(must_not_enter)
                assert forbidden == [] and tuple(owner._dispatches) == (execution,)
                assert not operation.done() and not execution.retained_task.done()
                assert any(extent[2] is database for extent in owner._open_extents.values())
                assert privacy._lease is not None and privacy.ledger._db is not None
                assert owner._failed
                release.set()
                assert await operation == (1,)
            finally:
                release.set()
                await asyncio.gather(operation, return_exceptions=True)
            with pytest.raises(sqlite3.ProgrammingError):
                database.execute("SELECT 1")
            assert len(owner._open_extents) == 1 and owner._failed
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, privacy, cleanup.value)


@pytest.mark.asyncio
async def test_finished_native_steps_are_pruned_inside_same_parent_after_sealing(tmp_path, monkeypatch):
    monkeypatch.setattr(registration, "_MAX_NATIVE_WITNESSES", 2)
    monkeypatch.setattr(runtime_admission, "_MAX_NATIVE_WITNESSES", 2)
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = await build_privacy_services(config, shutdown=stack)
        events = EventLogger(config["analytics"]["events_db"], source_policy=privacy.ledger)
        stack.push_async_callback(events.aclose)
        privacy.add_borrower(events.begin_close)
        await registration.attach_output(events, "events")
        await events.acall(events.initialize)
        saved, parents = [], []
        def original():
            execution = current_native_execution()
            assert execution.parent_sequence is parents[0]
            assert not execution.retained_task.done()
            if saved:
                previous = saved[-1]
                assert previous.task.done() and not previous.retained_task.done()
                with pytest.raises(RuntimeError, match="^runtime_admission_native_owner$"):
                    owner.admission._revalidate_native(previous)
            saved.append(execution)
            with registration.native_connection(events, events.db_path, sqlite3.connect, events.db_path) as database:
                return database.execute("SELECT 1").fetchone()
        async def sequence():
            parent = asyncio.current_task()
            parents.append(parent)
            with registration.event_call(events, original):
                for index in range(6):
                    if index == 1:
                        events._work.begin_close()
                    assert await events._work.native_step(original) == (1,)
                    assert len(owner._dispatches) <= 2
                    assert len(owner.admission._native_validations) <= 2
                    assert parent in events._work._tasks and not parent.done()
                    assert not owner._failed
            return "original_sequence_returned"
        assert await events._work.run_async(sequence) == "original_sequence_returned"
        assert len(saved) == 6 and len(parents) == 1 and parents[0].done()
        assert all(execution.retained_task is parents[0] and execution.task.done() for execution in saved)
        assert events._work.status()["closed"] and events._work.status()["operations"] == 0
        assert len(owner._open_extents) == 1 and privacy._lease is not None
    assert owner._open_extents == {} and privacy.ledger._db is None and privacy._lease is None
    assert events._closing.done() and not events._closing.cancelled()
    assert events._closing.exception() is None
