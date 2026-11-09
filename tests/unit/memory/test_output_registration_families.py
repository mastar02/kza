"""Real SQLite owners with synthetic admission validation and verified-instance seams.

The managed fixture stubs revalidation and the verified admission registry;
compiled scenarios separately establish real admission ancestry.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4

import pytest

from src.core import runtime_admission
from src.core.owned_work import WorkCloseTimeout
from src.dashboard.authority import DashboardAuthority
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from src.analytics.event_logger import EventLogger
from src.analytics.services import build_event_logger
from src.analytics.suggestion_engine import SuggestionEngine
from src.analytics.suggestion_store import SuggestionStore
from src.routines.routine_store import RoutineStore


@asynccontextmanager
async def managed(tmp_path, monkeypatch, *, analytics_enabled=True, dashboard_enabled=True):
    root, state = tmp_path / "admission-root", tmp_path / "state"
    (root / "admission").mkdir(parents=True)
    data = state / "data"
    data.mkdir(parents=True)
    config = {
        "memory": {"privacy_db_path": str(data / "privacy.db")},
        "dashboard": {"auth": {"db_path": str(data / "authority.db")}},
        "routines": {"db_path": str(data / "routines.db")},
        "analytics": {"events_db": str(data / "events.db"), "suggestions_path": str(data / "suggestions.json")},
        **{name: {"db_path": str(data / (name + ".db"))} for name in ("lists", "reminders")},
    }
    config["analytics"]["enabled"] = analytics_enabled
    config["dashboard"]["enabled"] = dashboard_enabled
    ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": 1}
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(state)},
        config["memory"]["privacy_db_path"], runtime_admission._config_digest(config))
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    # This fixture supplies synthetic authority only; actual native execution
    # issuance, owner identity, opening and retained-close guards remain real.
    admission._validation_snapshot = object()
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    monkeypatch.setattr(runtime_admission, "_verified_runs", {(ticket["run_id"], ticket["binding_id"])})
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(base, "_ACTIVE", set())
    stack = AsyncExitStack()
    admission.install_on(stack)
    owner = await admission.begin_destination_registration(config)
    try:
        yield owner, stack, config
    finally:
        try:
            await stack.__aexit__(RuntimeError, RuntimeError("unit has no compiled graph receipt"), None)
        except RuntimeError as error:
            if str(error) != "output_registration_epoch_incomplete":
                raise


def assert_failed_registry_cleanup(
    owner: registration.OutputRegistrationV2,
    privacy: PrivacyServices,
    error: BaseException,
) -> None:
    """Verify the exact poisoned-epoch error after the original owners drained."""
    assert type(error) is registration.OutputRegistrationError
    assert str(error) == "output_registration_registry_unavailable"
    assert owner._failed
    for closing in (privacy._closing, owner._closing):
        assert closing is not None and closing.done() and not closing.cancelled()
        assert closing.exception() is error
    assert privacy._closed and privacy.ledger._db is None
    assert not privacy.ledger.authority_available and privacy._lease is None
    assert not any(thread.is_alive() for thread in privacy._executor._threads)
    for work in (privacy._work, privacy._close_work, owner._disk):
        status = work.status()
        assert status["closed"] and status["operations"] == 0
    assert not owner._open_extents and not owner._opening_extents

def observations(owner):
    return [base._read(p)[0] for p in sorted(owner._directory.glob("[0-9]*.json"))]


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["privacy_authority", "authority_sessions", "routines", "events"])
async def test_attachment_fault_prevents_real_owner_file_and_lease_creation(tmp_path, monkeypatch, family):
    body_completed = False
    expected_cleanup = family == "privacy_authority"
    cleanup_context = (pytest.raises(registration.OutputRegistrationError,
        match="^output_registration_registry_unavailable$") if expected_cleanup else nullcontext())
    with cleanup_context as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            if family == "privacy_authority":
                store = PrivacyServices(config["memory"]["privacy_db_path"])
                initialize = lambda: store.acall(store.initialize)
                path, close = Path(store.ledger.path), store.aclose
            elif family == "authority_sessions":
                store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
                initialize, path, close = store.initialize, store.path, store.aclose
            elif family == "routines":
                store = RoutineStore(config["routines"]["db_path"])
                initialize, path, close = store.initialize, Path(store._db_path), store.close
            else:
                store = EventLogger(config["analytics"]["events_db"])
                initialize = lambda: store.acall(store.initialize)
                path, close = store.db_path, store.aclose
            stack.push_async_callback(close)
            original = base._head
            def fail_attachment(directory, previous, value):
                if value["revision"] == 2:
                    raise OSError("synthetic attachment fault")
                return original(directory, previous, value)
            monkeypatch.setattr(base, "_head", fail_attachment)
            with pytest.raises(OSError):
                await registration.attach_output(store, family)
            before = {p.name: p.read_bytes() for p in owner._directory.iterdir()}
            with pytest.raises(RuntimeError):
                await initialize()
            assert not path.exists()
            assert not tuple(path.parent.glob(path.name + "*"))
            assert {p.name: p.read_bytes() for p in owner._directory.iterdir()} == before
            if expected_cleanup:
                assert owner._failed
            body_completed = True
    assert body_completed
    if expected_cleanup:
        assert_failed_registry_cleanup(owner, store, cleanup.value)


@pytest.mark.asyncio
async def test_dashboard_active_publication_fault_keeps_ready_false_and_original_lease_owner(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, "authority_sessions")
        original = owner._append
        def publish(producer, phase, plan):
            if phase == "active":
                raise OSError("synthetic active failure")
            return original(producer, phase, plan)
        monkeypatch.setattr(owner, "_append", publish)
        with pytest.raises(OSError):
            await store.initialize()
        assert not store._ready and store._lease_fd is not None
        assert not owner._open_extents  # Original transient connection already closed.
        with pytest.raises(RuntimeError, match="epoch_incomplete"):
            await stack.__aexit__(OSError, OSError("publication failed"), None)
        assert store._lease_fd is None and store._closing.done()
        assert base._chain(owner._directory)[2]["authority_sessions"]["phase"] == "opening"


@pytest.mark.asyncio
async def test_dashboard_final_rotation_uses_original_distinct_worker_and_records_only_real_close(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, "authority_sessions")
        await store.initialize()
        original_work = store._work
        await store.aclose()
        binding = owner._roles["authority_sessions"]
        assert binding.final_work is not None and binding.final_work is not original_work
        assert binding.final_work.status()["closed"] and not owner._open_extents
        assert store._lease_fd is None
        assert base._chain(owner._directory)[2]["authority_sessions"]["phase"] == "closed"


@pytest.mark.asyncio
async def test_live_ledger_can_be_borrowed_without_reopening_but_closed_ledger_cannot(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = await build_privacy_services(config, shutdown=stack)
        database = privacy.ledger._db
        events = await build_event_logger(config, shutdown=stack, privacy=privacy)
        assert privacy.ledger._db is database and database is not None
        assert base._chain(owner._directory)[2]["events"]["phase"] == "active"
        await privacy.aclose()
        with pytest.raises(RuntimeError):
            await asyncio.to_thread(privacy.ledger.initialize)
        assert privacy.ledger._db is None
        assert events._closing.done()


@pytest.mark.asyncio
async def test_suggestion_constructor_requires_real_reserved_dispatch_and_borrows_event_close(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = await build_privacy_services(config, shutdown=stack)
        events = await build_event_logger(config, shutdown=stack, privacy=privacy)
        path = config["analytics"]["suggestions_path"]
        engine = await registration.construct_suggestions(events, SuggestionEngine,
            event_logger=events, suggestions_path=path)
        state = base._chain(owner._directory)[2]["suggestions"]
        assert state["phase"] == "active" and Path(state["plan"]["path"]).exists()
        assert owner._roles["suggestions"].work is events._work
        assert engine.store is owner._roles["suggestions"].instance
        assert not hasattr(engine.store, "begin_close")
        await events.aclose()
        assert base._chain(owner._directory)[2]["suggestions"]["phase"] == "closed"
        database = privacy.ledger._db
        assert database is not None and privacy._lease is not None
        assert set(owner._open_extents) == {id(database)}
        extent = owner._open_extents[id(database)]
        assert extent[0] is owner._roles["privacy_authority"]
        assert extent[0].instance is privacy and extent[2] is database
        assert database.execute("SELECT 1").fetchone()[0] == 1
        assert events._closing.done() and events._closing.exception() is None
        assert not owner._opening_extents
        await privacy.aclose()
        assert privacy._closing.done() and privacy._closing.exception() is None
        assert privacy.ledger._db is None and privacy._lease is None
        assert not owner._open_extents and not owner._opening_extents


@pytest.mark.asyncio
async def test_unreserved_suggestion_constructor_rejected_before_register_or_file(tmp_path, monkeypatch):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            events = await build_event_logger(config, shutdown=stack, privacy=privacy)
            before = await privacy.acall(privacy.ledger.erasures.target_snapshot)
            path = config["analytics"]["suggestions_path"]
            with pytest.raises(RuntimeError, match="constructor_reservation"):
                await events.acall(SuggestionStore, events, path)
            assert not tuple(Path(path).parent.glob(Path(path).name + "*"))
            # The failed epoch is deliberately sealed, so inspect the original live
            # SQLite connection without asking it to reopen through managed gates.
            rows = privacy.ledger._db.execute("SELECT count(*) FROM erasure_targets").fetchone()[0]
            assert rows == len(before.entries)
            assert owner._failed
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, privacy, cleanup.value)


@pytest.mark.asyncio
async def test_same_worker_unadmitted_callable_cannot_open_events(tmp_path, monkeypatch):
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            events = await build_event_logger(config, shutdown=stack, privacy=privacy)
            # Keep this global connector spy specific to the foreign dispatch.
            # The real maintenance loop may already have admitted native work.
            maintenance, work = events._maintenance, events._work
            database, lease = privacy.ledger._db, privacy._lease
            assert maintenance is not None and database is not None and lease is not None
            maintenance.cancel()
            async with asyncio.timeout(1):
                await asyncio.gather(maintenance, return_exceptions=True)
                await work.drain_admitted()
            assert events._maintenance is maintenance and maintenance.cancelled()
            assert events._work is work and events._initialized and events._closing is None
            assert not work.status()["closed"] and work.status()["operations"] == 0
            assert owner._roles["events"].instance is events and owner._roles["events"].work is work
            assert privacy.ledger._db is database and privacy._lease == lease
            assert privacy._ready and privacy._closing is None and events.source_policy is privacy.ledger
            assert not owner._failed
            connections = []
            connect = sqlite3.connect
            def observed(*args, **kwargs):
                connections.append(args[0])
                return connect(*args, **kwargs)
            monkeypatch.setattr(sqlite3, "connect", observed)
            def foreign():
                with events._connect():
                    raise AssertionError("foreign callback reached native database")
            with pytest.raises(RuntimeError, match="native_owner"):
                await events._work.run(foreign)
            assert connections == []
            assert owner._failed
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, privacy, cleanup.value)


@pytest.mark.asyncio
async def test_transient_actual_close_timeout_keeps_owner_lease_and_unclosed_inventory(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, "authority_sessions")
        entered, release = threading.Event(), threading.Event()
        close_count = []
        connect = sqlite3.connect
        class HeldClose:
            def __init__(self, database):
                object.__setattr__(self, "database", database)
            def __getattr__(self, name):
                return getattr(self.database, name)
            def __setattr__(self, name, value):
                setattr(self.database, name, value)
            def __enter__(self):
                self.database.__enter__()
                return self
            def __exit__(self, *args):
                return self.database.__exit__(*args)
            def close(self):
                close_count.append(True)
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release native close")
                self.database.close()
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: HeldClose(connect(*a, **k)))
        initialize = asyncio.create_task(store.initialize())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            initialize.cancel()
            with pytest.raises(asyncio.CancelledError):
                await initialize
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await store.aclose()
            assert store._lease_fd is not None and not store._closing.done()
            assert owner._open_extents
            assert base._chain(owner._directory)[2]["authority_sessions"]["phase"] == "opening"
        finally:
            release.set()
            await store.aclose()
        assert store._lease_fd is None and not owner._open_extents
        assert len(close_count) == 2  # Initial connection and original final rotation.


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_historical_missing_file_and_existing_missing_table_are_distinct(tmp_path, monkeypatch, legacy):
    from src.analytics.erasure import build_event_erasure
    from src.analytics.event_storage import event_descriptor
    from src.analytics.retention import EventRetention
    from src.memory.erasure_coordinator import ErasureCoordinator
    from src.memory.erasure_journal import ErasureParticipant
    monkeypatch.setattr(EventRetention, "start", lambda self: None)
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = await build_privacy_services(config, shutdown=stack)
        erasures = ErasureCoordinator(privacy)
        stack.push_async_callback(erasures.aclose)
        driver = await build_event_erasure(config, privacy=privacy, owner=erasures)
        target = Path(config["analytics"]["events_db"]).with_name("historical.db")
        if legacy:
            with sqlite3.connect(target) as database:
                database.execute("CREATE TABLE legacy(value TEXT)")
                database.execute("INSERT INTO legacy VALUES ('synthetic-private-sentinel')")
            before = target.read_bytes()
        descriptor = event_descriptor(target)
        identity = await privacy.acall(privacy.ledger.erasures.register_target, ErasureParticipant.EVENTS, 1, descriptor)
        outcome = await driver.retention._run({"id": identity, "version": 1, "descriptor": descriptor}, 0)
        assert outcome.error is None
        physical = [e for e in observations(owner) if e.get("kind") == "physical" and e.get("path") == str(target)]
        assert physical[-1]["state"] == ("representation_absent" if legacy else "file_absent")
        assert not owner._open_extents or all(e[0].producer == "privacy_authority" for e in owner._open_extents.values())
        if legacy:
            assert target.read_bytes() == before
        else:
            assert not target.exists()
        assert all(b"synthetic-private-sentinel" not in p.read_bytes() for p in owner._directory.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [1, 2])
async def test_registry_change_between_publication_and_dispatch_retries_once_then_stays_pending(tmp_path, monkeypatch, changes):
    from src.analytics.erasure import build_event_erasure
    from src.analytics.event_storage import event_descriptor
    from src.analytics.retention import EventRetention
    from src.memory.erasure_coordinator import ErasureCoordinator
    from src.memory.erasure_journal import ErasureParticipant
    monkeypatch.setattr(EventRetention, "start", lambda self: None)
    body_completed = False
    expected_cleanup = changes == 2
    cleanup_context = (pytest.raises(registration.OutputRegistrationError,
        match="^output_registration_registry_unavailable$") if expected_cleanup else nullcontext())
    with cleanup_context as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            erasures = ErasureCoordinator(privacy)
            stack.push_async_callback(erasures.aclose)
            driver = await build_event_erasure(config, privacy=privacy, owner=erasures)
            descriptor = event_descriptor(config["analytics"]["events_db"])
            rows = await privacy.acall(privacy.ledger.erasures.target_snapshot)
            identity = next(e.id for e in rows.entries if e.descriptor == descriptor)
            actual_worker = driver.retention._work._worker
            dispatches = []
            async def racing_worker(function, *args, **kwargs):
                dispatches.append(function)
                if len(dispatches) <= changes:
                    path = Path(descriptor["path"]).with_name(f"new-root-{len(dispatches)}.db")
                    await privacy.acall(privacy.ledger.erasures.register_target, ErasureParticipant.EVENTS, 1, event_descriptor(path))
                return await actual_worker(function, *args, **kwargs)
            monkeypatch.setattr(driver.retention._work, "_worker", racing_worker)
            outcome = await driver.retention._run({"id": identity, "version": 1, "descriptor": descriptor}, 0)
            assert len(dispatches) == 2
            assert (outcome.error is None) is (changes == 1)
            assert owner._failed is (changes == 2)
            assert not Path(descriptor["path"]).exists()
            assert not tuple(Path(descriptor["path"]).parent.glob("new-root-*"))
            if expected_cleanup:
                assert owner._failed
            body_completed = True
    assert body_completed
    if expected_cleanup:
        assert_failed_registry_cleanup(owner, privacy, cleanup.value)


@pytest.mark.asyncio
async def test_borrowed_event_reader_reuses_one_native_snapshot_join_and_original_exitstack(tmp_path, monkeypatch):
    from src.analytics import suggestion_retention
    from src.analytics.event_evidence_reader import EventEvidenceReader
    from src.analytics.event_storage import event_descriptor
    from src.analytics.suggestion_erasure import build_suggestion_erasure
    from src.analytics.suggestion_retention import SuggestionRetention
    from src.analytics.suggestion_storage import existing_suggestion_database
    from src.memory.erasure_coordinator import ErasureCoordinator
    monkeypatch.setattr(SuggestionRetention, "start", lambda self: None)
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = await build_privacy_services(config, shutdown=stack)
        events = await build_event_logger(config, shutdown=stack, privacy=privacy)
        engine = await registration.construct_suggestions(events, SuggestionEngine,
            event_logger=events, suggestions_path=config["analytics"]["suggestions_path"])
        erasures = ErasureCoordinator(privacy)
        stack.push_async_callback(erasures.aclose)
        driver = await build_suggestion_erasure(config, privacy=privacy, owner=erasures)
        event = event_descriptor(config["analytics"]["events_db"])
        joins, held = [], []
        original_join = owner._join
        def joined(*args, **kwargs):
            joins.append(True)
            return original_join(*args, **kwargs)
        monkeypatch.setattr(owner, "_join", joined)
        # This connector-focused adapter uses the real retention worker and real
        # original context managers. Existing sweep tests cover row semantics.
        def inspect_connectors(ledger, target, cursor, event_targets, days):
            with existing_suggestion_database(target) as suggestions:
                with EventEvidenceReader(ledger, {"event": event}, days) as reader:
                    database = reader._database("event")
                    assert database is not None and suggestions is not None
                    held.append(database)
                    assert reader._database("event") is database
                    assert len(owner._open_extents) == 3  # Live ledger plus two transient databases.
                with pytest.raises(sqlite3.ProgrammingError):
                    database.execute("SELECT 1")
            return "connectors_closed"
        monkeypatch.setattr(suggestion_retention, "sweep_suggestions", inspect_connectors)
        result = await registration.run_registered(driver.retention, driver.retention._work,
            suggestion_retention.sweep_suggestions, privacy.ledger, engine.store.descriptor, None, {}, 90)
        assert result == "connectors_closed" and held and joins == [True]
        assert len(owner._open_extents) == 1
        revision = base._chain(owner._directory)[1]["revision"]
        for _ in range(8):
            assert await registration.run_registered(driver.retention, driver.retention._work,
                suggestion_retention.sweep_suggestions, privacy.ledger, engine.store.descriptor, None, {}, 90) == "connectors_closed"
        assert base._chain(owner._directory)[1]["revision"] == revision
        assert len(joins) == 9


@pytest.mark.asyncio
async def test_original_eraser_close_is_registered_before_attachment_can_fail(tmp_path, monkeypatch):
    from src.analytics.erasure import build_event_erasure
    from src.memory.erasure_coordinator import ErasureCoordinator
    from src.memory.erasure_journal import ErasureParticipant
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            erasures = ErasureCoordinator(privacy)
            stack.push_async_callback(erasures.aclose)
            original = registration.attach_output
            observed = []
            async def fail(owner_to_attach, producer, *, role="primary"):
                if role == "eraser":
                    assert erasures._drivers[ErasureParticipant.EVENTS].callback.__self__ is owner_to_attach
                    assert any(close.__self__ is owner_to_attach for close in erasures._resources)
                    observed.append(owner_to_attach)
                    raise OSError("attachment unavailable")
                return await original(owner_to_attach, producer, role=role)
            monkeypatch.setattr(registration, "attach_output", fail)
            with pytest.raises(OSError):
                await build_event_erasure(config, privacy=privacy, owner=erasures)
            await erasures.aclose()
            assert observed[0]._closing.done() and observed[0].retention._closing.done()
            assert owner._failed
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, privacy, cleanup.value)


def legacy_epoch(owner):
    directory = owner._namespace / "v1/runs" / uuid4().hex
    directory.mkdir(parents=True)
    header = {key: value for key, value in owner._epoch.items() if key != "selected"}
    header.update(schema_version=1, run_id=directory.name, namespace_identity=base._identity(directory))
    raw = base._encoded(header)
    base._write_new(directory / "epoch.json", raw)
    base._head(directory, None, {"revision": 0, "sha256": base._sha(raw)})
    return directory


@pytest.mark.asyncio
async def test_legacy_epoch_is_read_without_mutation_or_v2_promotion(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        legacy = legacy_epoch(owner)
        before = {p.name: p.read_bytes() for p in legacy.iterdir()}
        original = base._chain_v1(legacy)
        assert base._chain(legacy) == original
        assert original[0]["schema_version"] == 1 and original[2] == {}
        with pytest.raises(RuntimeError, match="header"):
            registration.chain(legacy)
        assert {p.name: p.read_bytes() for p in legacy.iterdir()} == before
        assert base._chain(owner._directory)[0]["schema_version"] == 3


@pytest.mark.asyncio
async def test_total_capacity_counts_legacy_and_unknown_bytes_before_new_open(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = await build_privacy_services(config, shutdown=stack)
        database = privacy.ledger._db
        legacy_epoch(owner)
        unknown = owner._namespace / "v99"
        unknown.mkdir()
        (unknown / "opaque").write_bytes(b"bounded-unknown-history")
        size_with_unknown = registration._namespace(owner._namespace)
        assert size_with_unknown >= len(b"bounded-unknown-history")
        (unknown / "opaque").unlink()
        unknown.rmdir()
        size = registration._namespace(owner._namespace)
        assert size_with_unknown - size == len(b"bounded-unknown-history")
        monkeypatch.setattr(base, "MAX_TOTAL_BYTES", size)
        store = RoutineStore(config["routines"]["db_path"])
        stack.push_async_callback(store.close)
        before = {str(p.relative_to(owner._namespace)): p.read_bytes()
                  for p in owner._namespace.rglob("*") if p.is_file()}
        with pytest.raises(RuntimeError, match="capacity"):
            await registration.attach_output(store, "routines")
        assert privacy.ledger._db is database and privacy._lease is not None
        assert database.execute("SELECT 1").fetchone()[0] == 1
        assert not Path(store._db_path).exists()
        assert {str(p.relative_to(owner._namespace)): p.read_bytes()
                for p in owner._namespace.rglob("*") if p.is_file()} == before
        assert owner._failed
        with pytest.raises(registration.OutputRegistrationError,
                match="^output_registration_registry_unavailable$") as cleanup:
            await stack.__aexit__(RuntimeError, RuntimeError("capacity exhausted"), None)
        assert_failed_registry_cleanup(owner, privacy, cleanup.value)
        with pytest.raises(sqlite3.ProgrammingError):
            database.execute("SELECT 1")


@pytest.mark.asyncio
async def test_unknown_epoch_schema_is_rejected_without_repair(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        legacy = legacy_epoch(owner)
        header, _ = base._read(legacy / "epoch.json")
        header["schema_version"] = 77
        raw = base._encoded(header)
        (legacy / "epoch.json").write_bytes(raw)
        (legacy / "HEAD.json").write_bytes(base._encoded({"revision": 0, "sha256": base._sha(raw)}))
        before = {p.name: p.read_bytes() for p in legacy.iterdir()}
        with pytest.raises(RuntimeError, match="schema"):
            base._chain(legacy)
        assert {p.name: p.read_bytes() for p in legacy.iterdir()} == before


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["privacy_authority", "events"])
async def test_native_initializer_returns_before_active_and_failed_publication_preserves_close_owner(tmp_path, monkeypatch, family):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        privacy = None
        if family == "events":
            privacy = await build_privacy_services(config, shutdown=stack)
            store = EventLogger(config["analytics"]["events_db"], source_policy=privacy.ledger)
            privacy.add_borrower(store.begin_close)
        else:
            store = PrivacyServices(config["memory"]["privacy_db_path"])
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, family)
        original = owner._append
        returned = []
        def publication(producer, phase, plan):
            if producer == family and phase == "active":
                execution = owner._roles[family].initialized_execution
                assert execution.task.done() and not execution.task.cancelled()
                assert execution.task.exception() is None
                assert not execution.retained_task.done()
                returned.append(True)
                raise OSError("active observation cannot publish")
            return original(producer, phase, plan)
        monkeypatch.setattr(owner, "_append", publication)
        with pytest.raises(OSError):
            await store.acall(store.initialize)
        assert returned == [True]
        assert not (store._ready if family == "privacy_authority" else store._initialized)
        if family == "privacy_authority":
            assert store.ledger._db is not None and store._lease is not None
        assert owner._failed
        with pytest.raises(registration.OutputRegistrationError,
                match="^output_registration_registry_unavailable$") as cleanup:
            await stack.__aexit__(OSError, OSError("publication failed"), None)
        assert_failed_registry_cleanup(owner, (store if family == "privacy_authority" else privacy), cleanup.value)
        assert store._closing.done() and not owner._open_extents
        assert not any(thread.is_alive() for thread in store._executor._threads)
