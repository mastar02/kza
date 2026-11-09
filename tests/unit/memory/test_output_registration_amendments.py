"""Real service regressions using the shared synthetic managed-admission fixture.

Decoder tests additionally stub receipt and immutable binding validation.
These unit cases do not establish compiled admission ancestry.
"""
from __future__ import annotations

import asyncio
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import aiosqlite
import pytest

from src.analytics.event_logger import EventLogger
from src.dashboard.authority import DashboardAuthority
from src.lists.list_store import ListStore
from src.memory import copy_catalog
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory import output_registration_v3 as registration_v3
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from src.routines.routine_store import RoutineStore
from tests.unit.memory.test_output_registration_families import (
    assert_failed_registry_cleanup, managed, observations, legacy_epoch,
)


def rewrite_events(directory: Path, events: list[dict]) -> None:
    """Recompute valid hashes so semantic oracles do not merely reject corruption."""
    header, raw = base._read(directory / "epoch.json")
    for path in directory.glob("[0-9]*.json"):
        path.unlink()
    previous = base._sha(raw)
    for revision, event in enumerate(events, 1):
        event = dict(event, revision=revision, previous_sha256=previous)
        raw = base._encoded(event)
        previous = base._sha(raw)
        base._write_new(directory / f"{revision:012d}-{previous}.json", raw)
    (directory / "HEAD.json").write_bytes(base._encoded({"revision": len(events), "sha256": previous}))


def decoder_authority_seam(owner, monkeypatch):
    """Only decoder classification is exercised; compiled tests prove real leases."""
    header = owner._epoch
    binding = {"state_root": header["state_root"], "state_identity": header["state_identity"],
        "authority": {"projection": {"configuration": {"config/settings.yaml": "b" * 64}}}}
    receipt = {"run_id": header["run_id"], "binding_id": header["binding_id"],
        "generation": header["generation"], "status": "closed", "resources": {"unit": True}}
    stage = SimpleNamespace(read_binding=lambda *_: binding, _read_json=lambda *_: receipt,
        _CLOSED_RESOURCES={"unit"})
    controller = object()
    root = Path(owner.admission._ticket["admission_root"])
    monkeypatch.setattr(copy_catalog, "_controller_lease", lambda actual: (stage, root, {"unit": "stable"}, None, None))
    return controller


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["material_read", "final_chain"])
async def test_late_recovery_failure_never_leaves_positive_observed_fields(tmp_path, monkeypatch, failure):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = ListStore(config["lists"]["db_path"])
        stack.push_async_callback(store.close)
        await owner.register_store(store, "lists")
        await store.initialize()
        await store.close()
        controller = decoder_authority_seam(owner, monkeypatch)
        before = {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        chain, read = registration_v3.chain, base._read
        parsed = []
        def late_chain(directory):
            if parsed and failure == "final_chain":
                raise RuntimeError("late consistency check failed")
            result = chain(directory)
            parsed.append(True)
            return result
        def late_read(path):
            if parsed and failure == "material_read" and path.name[:1].isdigit():
                raise OSError("late record read failed")
            return read(path)
        with monkeypatch.context() as patch:
            patch.setattr(registration_v3, "chain", late_chain)
            patch.setattr(base, "_read", late_read)
            record, = registration.read_inventory(controller)
        assert record["phase"] == "unknown" and record["reason"] == "invalid_or_unfinished_history"
        assert not {"header", "journal", "stores", "configuration", "instrumented_producers", "material_observations"} & record.keys()
        assert record["uncovered_producers"] == list(base.PRODUCERS)
        assert {p.name: p.read_bytes() for p in owner._directory.iterdir()} == before


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["active_none", "closed_changed", "active_existing_changed", "worker_lists"])
async def test_valid_hash_chain_rejects_false_sqlite_identity_and_worker_claims(tmp_path, monkeypatch, mutation):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        Path(config["lists"]["db_path"]).touch()
        store = ListStore(config["lists"]["db_path"])
        stack.push_async_callback(store.close)
        await owner.register_store(store, "lists")
        await store.initialize()
        await store.close()
        events = observations(owner)
        for event in events:
            if mutation == "worker_lists":
                event["plan"] = {"kind": "worker", "producer": "lists", "role": "primary"}
                event["native_identity"] = None
            elif event["phase"] == "active" and mutation == "active_none":
                event["native_identity"] = None
            elif (event["phase"] == "closed" and mutation == "closed_changed") or (event["phase"] == "active" and mutation == "active_existing_changed"):
                event["native_identity"]["inode"] += 1
        rewrite_events(owner._directory, events)
        owner._failed = True  # Preserve malformed evidence while original callbacks drain.
        with pytest.raises(RuntimeError, match="native_identity|plan"):
            registration_v3.chain(owner._directory)
        controller = decoder_authority_seam(owner, monkeypatch)
        record, = registration.read_inventory(controller)
        assert record["phase"] == "unknown" and "stores" not in record
        assert len(record["uncovered_producers"]) == 32


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("configured_path", "relative.db"), ("configured_path", "/state/../foreign.db"),
    ("missing_parents", ["invented"]), ("leases", []), ("classification", "mixed_source_policy"),
])
async def test_valid_hash_chain_rejects_false_declared_plan_semantics(tmp_path, monkeypatch, field, value):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, "authority_sessions")
        await store.initialize()
        await store.aclose()
        events = observations(owner)
        for event in events:
            event["plan"][field] = copy.deepcopy(value)
        rewrite_events(owner._directory, events)
        owner._failed = True
        with pytest.raises(RuntimeError, match="plan"):
            registration_v3.chain(owner._directory)
        record, = registration.read_inventory(decoder_authority_seam(owner, monkeypatch))
        assert record["phase"] == "unknown" and "stores" not in record


@pytest.mark.asyncio
async def test_selected_disabled_worker_is_explicit_and_cannot_turn_into_sqlite_closed(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch, analytics_enabled=False, dashboard_enabled=False) as (owner, stack, config):
        for producer in ("authority_sessions", "events", "suggestions"):
            plan = {"kind": "worker", "producer": producer, "role": "primary"}
            await owner._run(owner._append, producer, "disabled_never_opened", plan)
        states = registration_v3.chain(owner._directory)[2]
        assert all(s["phase"] == "disabled_never_opened" and s["native_identity"] is None for s in states.values())
        assert not any(Path(config["analytics"]["events_db"]).parent.glob("*.db"))
        events = observations(owner)
        events[0]["phase"] = "intent"
        rewrite_events(owner._directory, events)
        owner._failed = True
        with pytest.raises(RuntimeError, match="owner_phase"):
            registration_v3.chain(owner._directory)


def mutate_extent(path: Path, mutation: str) -> None:
    if mutation == "directory":
        path.mkdir()
    elif mutation == "fifo":
        os.mkfifo(path)
    elif mutation == "hardlink":
        other = path.with_name(path.name + ".other")
        other.write_bytes(b"synthetic existing extent")
        os.link(other, path)
    elif mutation == "existing_extra_link":
        os.link(path, path.with_name(path.name + ".other"))
    elif mutation == "sidecar_symlink":
        Path(str(path) + "-wal").symlink_to(path.with_name("foreign-sidecar"))
    else:
        raise AssertionError(mutation)


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["privacy_authority", "authority_sessions", "routines", "events"])
@pytest.mark.parametrize("mutation", ["directory", "fifo", "hardlink", "existing_extra_link", "sidecar_symlink"])
async def test_changed_extent_is_rejected_before_any_store_connector(tmp_path, monkeypatch, family, mutation):
    body_completed = False
    expected_cleanup = family == "privacy_authority"
    cleanup_context = (pytest.raises(registration.OutputRegistrationError,
        match="^output_registration_registry_unavailable$") if expected_cleanup else nullcontext())
    with cleanup_context as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            if family == "privacy_authority":
                store = PrivacyServices(config["memory"]["privacy_db_path"])
                path, close = Path(store.ledger.path), store.aclose
                initialize = lambda: store.acall(store.initialize)
            elif family == "authority_sessions":
                store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
                path, close, initialize = store.path, store.aclose, store.initialize
            elif family == "routines":
                store = RoutineStore(config["routines"]["db_path"])
                path, close, initialize = Path(store._db_path), store.close, store.initialize
            else:
                store = EventLogger(config["analytics"]["events_db"])
                path, close = store.db_path, store.aclose
                initialize = lambda: store.acall(store.initialize)
            stack.push_async_callback(close)
            if mutation == "existing_extra_link":
                path.touch()
            await registration.attach_output(store, family)
            mutate_extent(path, mutation)
            connects = []
            def forbidden(*args, **kwargs):
                connects.append(True)
                raise AssertionError("unsafe extent reached connector")
            monkeypatch.setattr(sqlite3, "connect", forbidden)
            monkeypatch.setattr(aiosqlite, "connect", forbidden)
            with pytest.raises(RuntimeError, match="destination_changed"):
                await initialize()
            assert connects == []
            assert owner._failed
            if expected_cleanup:
                assert owner._failed
            body_completed = True
    assert body_completed
    if expected_cleanup:
        assert_failed_registry_cleanup(owner, store, cleanup.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["directory", "fifo", "hardlink", "existing_extra_link", "sidecar_symlink"])
async def test_historical_mutation_after_snapshot_is_rejected_before_connect(tmp_path, monkeypatch, mutation):
    from src.analytics.erasure import build_event_erasure
    from src.analytics.event_storage import event_descriptor
    from src.analytics.retention import EventRetention
    from src.memory.erasure_coordinator import ErasureCoordinator
    from src.memory.erasure_journal import ErasureParticipant
    monkeypatch.setattr(EventRetention, "start", lambda self: None)
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            erasures = ErasureCoordinator(privacy)
            stack.push_async_callback(erasures.aclose)
            path = Path(config["analytics"]["events_db"])
            if mutation == "existing_extra_link":
                path.touch()
            driver = await build_event_erasure(config, privacy=privacy, owner=erasures)
            descriptor = event_descriptor(path)
            snapshot = await privacy.acall(privacy.ledger.erasures.target_snapshot)
            identity = next(e.id for e in snapshot.entries if e.participant is ErasureParticipant.EVENTS and e.descriptor == descriptor)
            actual_worker = driver.retention._work._worker
            async def changed_after_refresh(function, *args, **kwargs):
                mutate_extent(path, mutation)
                return await actual_worker(function, *args, **kwargs)
            monkeypatch.setattr(driver.retention._work, "_worker", changed_after_refresh)
            connects = []
            def forbidden(*args, **kwargs):
                connects.append(True)
                raise AssertionError("historical mutation reached connect")
            monkeypatch.setattr(sqlite3, "connect", forbidden)
            outcome = await driver.retention._run({"id": identity, "version": 1, "descriptor": descriptor}, 0)
            assert outcome.error is not None and owner._failed
            assert connects == []
            assert owner._failed
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, privacy, cleanup.value)


@pytest.mark.asyncio
async def test_post_admission_new_root_cannot_join_the_existing_native_operation(tmp_path, monkeypatch):
    from src.analytics import retention
    from src.analytics.erasure import build_event_erasure
    from src.analytics.event_storage import event_descriptor, existing_event_database
    from src.memory.erasure_coordinator import ErasureCoordinator
    from src.memory.erasure_journal import ErasureParticipant
    monkeypatch.setattr(retention.EventRetention, "start", lambda self: None)
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            erasures = ErasureCoordinator(privacy)
            stack.push_async_callback(erasures.aclose)
            first = Path(config["analytics"]["events_db"])
            late = first.with_name("late-existing.db")
            for path in (first, late):
                with sqlite3.connect(path) as database:
                    database.execute("CREATE TABLE legacy(value TEXT)")
            late_before = late.read_bytes()
            driver = await build_event_erasure(config, privacy=privacy, owner=erasures)
            first_descriptor, late_descriptor = event_descriptor(first), event_descriptor(late)
            entered, release = threading.Event(), threading.Event()
            opened = []
            actual_connect = sqlite3.connect
            def counted(*args, **kwargs):
                opened.append(args[0])
                return actual_connect(*args, **kwargs)
            monkeypatch.setattr(sqlite3, "connect", counted)
            def two_roots(*args):
                with existing_event_database(first_descriptor):
                    pass
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not register late target")
                with existing_event_database(late_descriptor):
                    raise AssertionError("post-admission root entered an old operation")
            monkeypatch.setattr(retention, "sweep_events", two_roots)
            operation = asyncio.create_task(registration.run_registered(driver.retention,
                driver.retention._work, retention.sweep_events))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                await privacy.acall(privacy.ledger.erasures.register_target,
                    ErasureParticipant.EVENTS, 1, late_descriptor)
                release.set()
                with pytest.raises(RuntimeError, match="target_unregistered"):
                    await operation
            finally:
                release.set()
                await asyncio.gather(operation, return_exceptions=True)
            assert len(opened) == 1 and str(first) in opened[0]
            assert late.read_bytes() == late_before and owner._failed
            assert len(owner._open_extents) == 1
            assert owner._failed
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, privacy, cleanup.value)  # Only the original live privacy ledger remains.


@pytest.mark.asyncio
async def test_metadata_failure_does_not_skip_original_dashboard_rotation_fences(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        token = {"value": "synthetic-token-before"}
        store = DashboardAuthority(config["dashboard"]["auth"]["db_path"], expected_token=lambda: token["value"])
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, "authority_sessions")
        await store.initialize()
        previous = store._persisted_epoch
        token["value"] = "synthetic-token-after"
        owner._failed = True
        connections = []
        actual_connect = sqlite3.connect
        def counted(*args, **kwargs):
            connections.append(args[0])
            return actual_connect(*args, **kwargs)
        monkeypatch.setattr(sqlite3, "connect", counted)
        await store.aclose()
        assert len(connections) == 1 and store._lease_fd is None
        assert store._closing.done() and not owner._open_extents
        with actual_connect(store.path) as database:
            epoch = int(database.execute("SELECT value FROM auth_meta WHERE key='legacy_epoch'").fetchone()[0])
        assert epoch > previous
        assert registration_v3.chain(owner._directory)[2]["authority_sessions"]["phase"] == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["initialize", "arguments", "guard"])
async def test_original_final_worker_rejects_every_operation_except_its_exact_rotation(tmp_path, monkeypatch, mutation):
    from src.core.owned_work import OwnedBlockingWork
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, "authority_sessions")
        await store.initialize()
        actual_step = OwnedBlockingWork.native_step
        async def altered(work, function, *args, **kwargs):
            if work is owner._roles["authority_sessions"].final_work:
                action, values, guard = args
                if mutation == "initialize":
                    action = "initialize"
                elif mutation == "arguments":
                    values = (values[0], values[1] + 1)
                else:
                    guard = lambda: True
                args = (action, values, guard)
            return await actual_step(work, function, *args, **kwargs)
        monkeypatch.setattr(OwnedBlockingWork, "native_step", altered)
        connections = []
        def forbidden(*args, **kwargs):
            connections.append(True)
            raise AssertionError("unregistered final operation reached connect")
        monkeypatch.setattr(sqlite3, "connect", forbidden)
        with pytest.raises(RuntimeError, match="close_operation"):
            await store.aclose()
        assert connections == [] and store._lease_fd is None and owner._failed
        with pytest.raises(RuntimeError, match="close_operation"):
            await stack.__aexit__(RuntimeError, RuntimeError("invalid final operation"), None)


@pytest.mark.asyncio
async def test_duplicate_run_versions_and_unknown_namespace_never_keep_positive_fields(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        legacy = legacy_epoch(owner)
        duplicate = legacy.with_name(owner._directory.name)
        legacy.rename(duplicate)
        header, _ = base._read(duplicate / "epoch.json")
        header["run_id"] = duplicate.name
        raw = base._encoded(header)
        (duplicate / "epoch.json").write_bytes(raw)
        (duplicate / "HEAD.json").write_bytes(base._encoded({"revision": 0, "sha256": base._sha(raw)}))
        unknown = owner._namespace / "v99"
        unknown.mkdir()
        (unknown / "opaque").write_bytes(b"unknown-version-history")
        before = {str(p.relative_to(owner._namespace)): p.read_bytes()
            for p in owner._namespace.rglob("*") if p.is_file()}
        records = registration.read_inventory(decoder_authority_seam(owner, monkeypatch))
        conflicts = [r for r in records if r.get("run_id") == owner._directory.name]
        assert len(conflicts) == 2
        assert all(r["phase"] == "unknown" and r["reason"] == "conflicting_run_versions" for r in conflicts)
        assert all(not {"header", "journal", "stores", "configuration"} & r.keys() for r in conflicts)
        assert any(r["version"] == "v99" and r["reason"] == "unsupported_version" for r in records)
        assert {str(p.relative_to(owner._namespace)): p.read_bytes()
            for p in owner._namespace.rglob("*") if p.is_file()} == before
        owner._failed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("family,suffix", [("privacy_authority", ".lock"), ("authority_sessions", ".lease")])
async def test_changed_lease_path_is_rejected_before_lease_or_database_open(tmp_path, monkeypatch, family, suffix):
    body_completed = False
    expected_cleanup = family == "privacy_authority"
    cleanup_context = (pytest.raises(registration.OutputRegistrationError,
        match="^output_registration_registry_unavailable$") if expected_cleanup else nullcontext())
    with cleanup_context as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            if family == "privacy_authority":
                store = PrivacyServices(config["memory"]["privacy_db_path"])
                path = Path(store.ledger.path)
                initialize = lambda: store.acall(store.initialize)
            else:
                store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
                path, initialize = store.path, store.initialize
            stack.push_async_callback(store.aclose)
            await registration.attach_output(store, family)
            Path(str(path) + suffix).symlink_to(path.with_name("foreign-lease"))
            connects = []
            monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: connects.append(True))
            with pytest.raises(RuntimeError, match="destination_changed"):
                await initialize()
            assert connects == [] and not path.exists()
            assert getattr(store, "_lease", None) is None and getattr(store, "_lease_fd", None) is None
            if expected_cleanup:
                assert owner._failed
            body_completed = True
    assert body_completed
    if expected_cleanup:
        assert_failed_registry_cleanup(owner, store, cleanup.value)
