"""Real historical callbacks and owners under explicit synthetic admission seams.

The shared managed fixture stubs full/live admission validation. This module
supplies immutable ownership observations as metadata, never fabricated binding
or controller-receipt files. Scope projection, native execution identity, target
registry joins, SQLite work and retained shutdown are production implementations.
The compiled compatibility scenario remains the real authority-ancestry oracle.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
from copy import deepcopy
from contextvars import copy_context
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from src.analytics.erasure import build_event_erasure
from src.analytics.event_storage import event_descriptor, existing_event_database, sweep_events
from src.analytics.retention import EventRetention
from src.analytics.services import build_event_logger
from src.core import owned_work, runtime_admission
from src.core.owned_work import OwnedBlockingWork, current_native_execution
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory.privacy_services import build_privacy_services
from tests.unit.memory.test_output_registration_families import (
    assert_failed_registry_cleanup,
    managed,
    observations,
)


def _seed_events(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("""CREATE TABLE events_policy_v1(
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
            event_type TEXT NOT NULL, entity_id TEXT NOT NULL, action TEXT NOT NULL,
            user_id TEXT, user_name TEXT, trigger_phrase TEXT, context_json TEXT,
            hour INTEGER, minute INTEGER, weekday INTEGER, is_weekend INTEGER,
            time_of_day TEXT, sources_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
            expires_at REAL NOT NULL, text_expires_at REAL)""")
        db.execute("""INSERT INTO events_policy_v1
            (timestamp,event_type,entity_id,action,context_json,sources_json,
             metrics_json,expires_at) VALUES (1,'synthetic','unit','test','{}','[]','{}',2)""")


def _observation(admission, source_root: Path, targets: list[dict], *, proof_label="original"):
    """An explicit metadata seam, with production canonical encodings and IDs."""
    stage = runtime_admission._stage()
    source = {
        "state_root": str(source_root),
        "state_identity": base._identity(source_root),
        "unit_metadata_only": True,
    }
    source_binding = stage._digest(source)
    proof = {"unit_metadata_observation": proof_label, "not_a_controller_receipt": True}
    target_map = {target["id"]: deepcopy(target) for target in targets}
    state_identity = base._identity(admission.state_root)
    record = {
        "schema_version": 1,
        "destination_state": str(admission.state_root),
        "destination_identity": state_identity,
        "source_binding": source_binding,
        "source_close_sha256": stage._digest(proof),
        "targets": target_map,
    }
    policy = {
        "schema_version": 1,
        "source_bindings": [source_binding],
        "owned_targets": [],
        "delegated_targets": sorted(target_map),
        "historical_targets": [],
    }
    ownership = {
        "targets": target_map,
        "policy": policy,
        "delegations": {identity: {
            "record": record, "source": source,
            "marker": {"binding_id": source_binding}, "proof": proof,
        } for identity in target_map},
    }
    policy_json = stage._canonical_json(policy) + b"\n"
    admission._binding["state_identity"] = state_identity
    admission._binding["ownership_sha256"] = hashlib.sha256(policy_json).hexdigest()
    snapshot = SimpleNamespace(ownership_json=stage._canonical_json(ownership))
    admission._validation_snapshot = snapshot
    scope = {
        "scope_schema_version": 1,
        "source_binding": source_binding,
        "delegation_sha256": stage._digest(record),
        "source_close_sha256": record["source_close_sha256"],
        "source_state_root": str(source_root),
        "source_state_identity": source["state_identity"],
    }
    return SimpleNamespace(snapshot=snapshot, ownership=ownership,
        scope=scope, scope_json=stage._canonical_json(scope), policy_json=policy_json)


async def _graph(owner, stack, config, tmp_path, monkeypatch):
    monkeypatch.setattr(EventRetention, "start", lambda self: None)
    privacy = await build_privacy_services(config, shutdown=stack)
    erasures = ErasureCoordinator(privacy)
    stack.push_async_callback(erasures.aclose)
    source = (tmp_path / "historical-source" / "state").resolve()
    path = source / "data" / "events.db"
    _seed_events(path)
    descriptor = event_descriptor(path)
    identity = await privacy.acall(privacy.ledger.erasures.register_target,
        ErasureParticipant.EVENTS, 1, descriptor)
    target = {"id": identity, "participant": "events", "version": 1, "descriptor": descriptor}
    observed = _observation(owner.admission, source, [target])
    driver = await build_event_erasure(config, privacy=privacy, owner=erasures)
    return SimpleNamespace(privacy=privacy, driver=driver, target=target,
        source=source, path=path, observed=observed)


def _privacy_is_only_open_extent(owner, privacy) -> None:
    assert len(owner._open_extents) == 1
    binding, _execution, connection = next(iter(owner._open_extents.values()))
    assert binding.instance is privacy and binding.producer == "privacy_authority"
    assert connection is privacy.ledger._db
    assert owner._opening_extents == 0


def _driver_is_closed(graph) -> None:
    for resource in (graph.driver, graph.driver.retention):
        assert resource._closing is not None and resource._closing.done()
        assert not resource._closing.cancelled() and resource._closing.exception() is None
        assert resource._work.status()["closed"] and resource._work.status()["operations"] == 0
        assert not any(thread.is_alive() for thread in resource._executor._threads)


@pytest.mark.asyncio
async def test_original_retention_uses_exact_delegation_without_journal_churn(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
        actual_connect = sqlite3.connect
        executions, closed = [], []
        class ObservedConnection(sqlite3.Connection):
            def close(self):
                super().close()
                closed.append(self)
        def observed_connect(*args, **kwargs):
            assert args[0] == graph.path.as_uri() + "?mode=rw"
            execution = current_native_execution()
            assert execution.function is sweep_events
            assert execution.work is graph.driver.retention._work
            executions.append(execution)
            kwargs["factory"] = ObservedConnection
            return actual_connect(*args, **kwargs)
        monkeypatch.setattr(sqlite3, "connect", observed_connect)
        outcome = await graph.driver.retention._run(graph.target, 0)
        assert outcome.error is None and outcome.receipt.deleted == 1
        first_count = len(observations(owner))
        for _ in range(3):
            repeated = await graph.driver.retention._run(graph.target, 0)
            assert repeated.error is None and repeated.receipt.deleted == 0
        assert len(observations(owner)) == first_count
        assert len(executions) == len(closed) == 4
        assert len({id(value) for value in executions}) == 4
        records = [event for event in observations(owner)
            if event.get("kind") == "delegated_target" and event.get("id") == graph.target["id"]]
        assert records and all(event["scope"] == graph.observed.scope for event in records)
        assert all(event["plan"]["path"] == str(graph.path) for event in records)
        policies = [event for event in observations(owner) if event.get("kind") == "delegation_policy"]
        assert len(policies) == 1
        assert base._encoded(policies[0]["policy"]) == graph.observed.policy_json
        assert policies[0]["ownership_sha256"] == owner.admission._binding["ownership_sha256"]
        assert json.loads(graph.observed.policy_json) == policies[0]["policy"]
        with closing(actual_connect(graph.path)) as db:
            assert db.execute("SELECT count(*) FROM events_policy_v1").fetchone() == (0,)
        _privacy_is_only_open_extent(owner, graph.privacy)
    _driver_is_closed(graph)
    assert not owner._open_extents and not owner._opening_extents


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source_close", "unselected", "destination"])
async def test_changed_delegation_is_rejected_before_original_retention_connect(tmp_path, monkeypatch, change):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
            before = graph.path.read_bytes()
            if change == "source_close":
                _observation(owner.admission, graph.source, [graph.target], proof_label="changed")
            else:
                changed = deepcopy(graph.observed.ownership)
                if change == "unselected":
                    changed["policy"]["delegated_targets"] = []
                else:
                    changed["delegations"][graph.target["id"]]["record"]["destination_state"] = str(tmp_path)
                owner.admission._validation_snapshot = SimpleNamespace(
                    ownership_json=runtime_admission._stage()._canonical_json(changed))
            connects = []
            actual_connect = sqlite3.connect
            def counted(*args, **kwargs):
                connects.append(args[0])
                return actual_connect(*args, **kwargs)
            monkeypatch.setattr(sqlite3, "connect", counted)
            outcome = await graph.driver.retention._run(graph.target, 0)
            assert outcome.error is not None and outcome.receipt is None
            assert connects == [] and graph.path.read_bytes() == before
            assert owner._failed
            _privacy_is_only_open_extent(owner, graph.privacy)
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, graph.privacy, cleanup.value)
    _driver_is_closed(graph)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["leaf_symlink", "parent_replaced"])
async def test_delegated_physical_mutation_is_rejected_before_connect(tmp_path, monkeypatch, change):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
            if change == "leaf_symlink":
                retained = graph.path.with_name("retained.db")
                graph.path.rename(retained)
                graph.path.symlink_to(retained)
            else:
                retained_parent = graph.path.parent.with_name("retained-data")
                graph.path.parent.rename(retained_parent)
                graph.path.parent.mkdir()
                graph.path.write_bytes((retained_parent / graph.path.name).read_bytes())
                retained = retained_parent / graph.path.name
            before = retained.read_bytes()
            connects = []
            actual_connect = sqlite3.connect
            def counted(*args, **kwargs):
                connects.append(args[0])
                return actual_connect(*args, **kwargs)
            monkeypatch.setattr(sqlite3, "connect", counted)
            outcome = await graph.driver.retention._run(graph.target, 0)
            assert outcome.error is not None and connects == []
            assert retained.read_bytes() == before and owner._failed
            _privacy_is_only_open_extent(owner, graph.privacy)
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, graph.privacy, cleanup.value)
    _driver_is_closed(graph)


@pytest.mark.asyncio
async def test_delegation_cannot_redirect_the_primary_event_builder(tmp_path, monkeypatch):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
            config["analytics"]["events_db"] = str(graph.path)
            connects = []
            actual_connect = sqlite3.connect
            def counted(*args, **kwargs):
                connects.append(args[0])
                return actual_connect(*args, **kwargs)
            monkeypatch.setattr(sqlite3, "connect", counted)
            with pytest.raises(registration.OutputRegistrationError, match="configured_path"):
                await build_event_logger(config, shutdown=stack, privacy=graph.privacy)
            assert connects == [] and owner._failed
            _privacy_is_only_open_extent(owner, graph.privacy)
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, graph.privacy, cleanup.value)
    _driver_is_closed(graph)


@pytest.mark.asyncio
async def test_selected_scope_does_not_grant_an_unadmitted_same_worker_callback(tmp_path, monkeypatch):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
            planned = owner.admission._plan_historical_target(graph.target)
            assert planned == (graph.observed.scope_json, graph.observed.policy_json)
            connects = []
            actual_connect = sqlite3.connect
            def counted(*args, **kwargs):
                connects.append(args[0])
                return actual_connect(*args, **kwargs)
            monkeypatch.setattr(sqlite3, "connect", counted)
            def foreign():
                with existing_event_database(graph.target["descriptor"]):
                    raise AssertionError("planning metadata admitted a foreign callable")
            with pytest.raises(registration.OutputRegistrationError, match="native_owner"):
                await graph.driver.retention._work.run(foreign)
            assert connects == [] and owner._failed
            _privacy_is_only_open_extent(owner, graph.privacy)
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, graph.privacy, cleanup.value)
    _driver_is_closed(graph)


@pytest.mark.asyncio
async def test_late_delegated_registry_root_cannot_enter_original_sweep_operation(tmp_path, monkeypatch):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
            late = graph.path.with_name("late.db")
            _seed_events(late)
            late_before = late.read_bytes()
            original_authority = graph.privacy.ledger.sources_committed_unavailable
            seen = []
            def after_original_authority(sources):
                result = original_authority(sources)
                execution = current_native_execution()
                assert execution.function is sweep_events
                descriptor = event_descriptor(late)
                identity = graph.privacy.ledger.erasures.register_target(ErasureParticipant.EVENTS, 1, descriptor)
                target = {"id": identity, "participant": "events", "version": 1, "descriptor": descriptor}
                _observation(owner.admission, graph.source, [graph.target, target])
                seen.append(execution)
                with existing_event_database(descriptor):
                    raise AssertionError("a late registry target entered the original sweep")
                return result
            monkeypatch.setattr(graph.privacy.ledger, "sources_committed_unavailable", after_original_authority)
            opened = []
            actual_connect = sqlite3.connect
            def counted(*args, **kwargs):
                opened.append(args[0])
                return actual_connect(*args, **kwargs)
            monkeypatch.setattr(sqlite3, "connect", counted)
            outcome = await graph.driver.retention._run(graph.target, 0)
            assert outcome.error is not None and len(seen) == 1
            assert opened == [graph.path.as_uri() + "?mode=rw"]
            assert late.read_bytes() == late_before and owner._failed
            _privacy_is_only_open_extent(owner, graph.privacy)
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, graph.privacy, cleanup.value)
    _driver_is_closed(graph)


@pytest.mark.asyncio
async def test_cancelled_observer_keeps_delegated_connection_until_actual_close(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
        entered, release = threading.Event(), threading.Event()
        closed, opened = [], []
        actual_connect = sqlite3.connect
        class HeldConnection(sqlite3.Connection):
            def close(self):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release original native close")
                super().close()
                closed.append(self)
        def held_connect(*args, **kwargs):
            assert args[0] == graph.path.as_uri() + "?mode=rw"
            assert current_native_execution().function is sweep_events
            kwargs["factory"] = HeldConnection
            connection = actual_connect(*args, **kwargs)
            opened.append(connection)
            return connection
        monkeypatch.setattr(sqlite3, "connect", held_connect)
        operation = asyncio.create_task(graph.driver.retention._run(graph.target, 0))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
            closing_task = graph.driver.begin_close()
            await asyncio.sleep(0)
            assert not closing_task.done() and closed == []
            assert graph.driver.retention._work.status()["operations"] == 1
            assert owner._open_extents[id(opened[0])][2] is opened[0]
            assert graph.privacy.ledger.authority_available and graph.privacy._lease is not None
        finally:
            release.set()
            await asyncio.gather(operation, return_exceptions=True)
            await graph.driver.begin_close()
        assert closed == opened and len(closed) == 1
        _privacy_is_only_open_extent(owner, graph.privacy)
    _driver_is_closed(graph)
    assert not owner._open_extents and not owner._opening_extents


@pytest.mark.asyncio
async def test_native_scope_uses_exact_issued_snapshot_and_rejects_replay(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
        admission = owner.admission
        work, saved = OwnedBlockingWork(), []
        expected = (graph.observed.scope_json, graph.observed.policy_json)
        def original():
            execution = current_native_execution()
            saved.append(execution)
            assert admission._native_historical_target(execution, graph.target) == expected
            _observation(admission, graph.source, [graph.target], proof_label="later-global-observation")
            assert admission._plan_historical_target(graph.target) != expected
            assert admission._native_historical_target(execution, graph.target) == expected
            clone = owned_work.NativeExecution(work, execution.function, execution.task, execution.parent_sequence)
            with pytest.raises(RuntimeError, match="runtime_admission_native_owner"):
                admission._native_historical_target(clone, graph.target)
            async def foreign_task():
                with pytest.raises(RuntimeError, match="runtime_admission_native_owner"):
                    admission._native_historical_target(execution, graph.target)
            copy_context().run(asyncio.run, foreign_task())
        def without_preflight():
            for execution in (saved[0], current_native_execution()):
                with pytest.raises(RuntimeError, match="runtime_admission_native_owner"):
                    admission._native_historical_target(execution, graph.target)
        def preflight(execution):
            if execution.function is original:
                admission._preflight_native(execution)
        work._install_native_preflight(admission, preflight)
        try:
            await work.run(original)
            await work.run(without_preflight)
        finally:
            await work.aclose()
        assert work.status()["closed"] and work.status()["operations"] == 0
    _driver_is_closed(graph)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["id", "version", "path", "marker", "proof", "policy_hash"])
async def test_scope_projection_refuses_inconsistent_selected_observation(tmp_path, monkeypatch, change):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        graph = await _graph(owner, stack, config, tmp_path, monkeypatch)
        admission = owner.admission
        target, ownership = deepcopy(graph.target), deepcopy(graph.observed.ownership)
        if change == "id":
            target["id"] = "f" * 64
        elif change == "version":
            target["version"] = 2
        elif change == "path":
            target["descriptor"]["path"] = str(graph.path.with_name("other.db"))
        elif change == "marker":
            ownership["delegations"][target["id"]]["marker"]["binding_id"] = "f" * 64
        elif change == "proof":
            ownership["delegations"][target["id"]]["proof"]["not_a_controller_receipt"] = False
        else:
            admission._binding["ownership_sha256"] = "f" * 64
        admission._validation_snapshot = SimpleNamespace(
            ownership_json=runtime_admission._stage()._canonical_json(ownership))
        before = dict(admission._native_validations)
        try:
            with pytest.raises(RuntimeError, match="runtime_admission_target_scope"):
                admission._plan_historical_target(target)
            assert dict(admission._native_validations) == before
        finally:
            admission._validation_snapshot = graph.observed.snapshot
            admission._binding["ownership_sha256"] = hashlib.sha256(graph.observed.policy_json).hexdigest()
        assert not owner._failed
        _privacy_is_only_open_extent(owner, graph.privacy)
    _driver_is_closed(graph)
