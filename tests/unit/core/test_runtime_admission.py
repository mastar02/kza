"""Managed construction must stop before a private owner opens stores."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
import threading
import time
from uuid import uuid4

import pytest

from src.core import runtime_admission
from src.memory.privacy_services import PrivacyServices


@asynccontextmanager
async def _construction_graph(tmp_path, monkeypatch):
    """Real graph/owners; protocol metadata here never issues a close receipt."""
    from src.memory.erasure_journal import ErasureParticipant
    from tests.mocks.memory_graph import Clock
    from tests.mocks.restore_application_graph import application_graph

    async with application_graph(tmp_path / "graph", monkeypatch, Clock(time.time())) as bundle:
        admission = runtime_admission.RuntimeAdmission({"run_id": uuid4().hex}, {},
            bundle.graph.privacy.ledger.path, runtime_admission._config_digest(bundle.graph.config))
        stack = AsyncExitStack()
        admission.install_on(stack)
        admission.attach_graph(bundle.graph)
        services = bundle.graph.erasure_services._drivers[ErasureParticipant.DIAGNOSTICS].callback.__self__
        try:
            yield bundle, admission, services
        finally:
            # The actual fixture closes every graph resource. This separate
            # observer stack has no admitted controller run and cannot emit a
            # native/SDK receipt, including on these positive seam assertions.
            await stack.__aexit__(RuntimeError, RuntimeError("unit construction seam only"), None)


@pytest.mark.asyncio
async def test_constructed_revalidation_waits_for_real_historical_catalog_close(tmp_path, monkeypatch):
    from scripts.stage_runtime import _assert_writers_closed
    from src.obs.diagnostic_store import DiagnosticStore
    from tests.mocks.restore_application_graph import until

    entered, release = threading.Event(), threading.Event()
    history = tmp_path / "historical" / "metrics.jsonl"
    actual_close = DiagnosticStore.close

    def held_close(store):
        try:
            if store.path == history:
                entered.set()
                assert release.wait(5)
        finally:
            actual_close(store)

    monkeypatch.setattr(DiagnosticStore, "close", held_close)
    async with _construction_graph(tmp_path, monkeypatch) as (bundle, admission, services):
        calls = []

        def validate():
            assert services._lock.locked()
            _assert_writers_closed(history.parent)
            calls.append("actual writer check returned")

        monkeypatch.setattr(admission, "revalidate", validate)
        history_observer = asyncio.create_task(services._run_history(str(history)))
        validation = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            with pytest.raises(ValueError, match="^runtime_admission_writer_active$"):
                _assert_writers_closed(history.parent)
            validation = asyncio.create_task(admission.revalidate_constructed())
            await until(lambda: admission._construction_revalidation is not None)
            assert not validation.done() and services._lock.locked() and calls == []
            release.set()
            await history_observer
            await validation
            assert calls == ["actual writer check returned"]
            assert not services._lock.locked()
            assert not services._closed and not services._disk.status()["closed"]
            # Construction serialization does not permanently stop maintenance
            # or future real historical work.
            await services._run_history(str(history))
            assert bundle.privacy.ledger.authority_available
        finally:
            release.set()
            await asyncio.gather(history_observer, return_exceptions=True)
            if validation is not None:
                await asyncio.gather(validation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error"])
async def test_cancelled_construction_observer_retains_lock_until_native_return(tmp_path, monkeypatch, outcome):
    entered, release = threading.Event(), threading.Event()
    async with _construction_graph(tmp_path, monkeypatch) as (_, admission, services):
        returned = threading.Event()

        def validate():
            entered.set()
            assert release.wait(5)
            returned.set()
            if outcome == "error":
                raise ValueError("synthetic native validation rejected")

        monkeypatch.setattr(admission, "revalidate", validate)
        observer = asyncio.create_task(admission.revalidate_constructed())
        contender_entered = asyncio.Event()

        async def contender():
            async with services._lock:
                contender_entered.set()

        waiting = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            native_owner = admission._construction_revalidation
            assert native_owner is not None and not native_owner.done()
            observer.cancel()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(.03):
                    await asyncio.shield(observer)
            assert not returned.is_set() and not observer.done() and services._lock.locked()
            waiting = asyncio.create_task(contender())
            assert not contender_entered.is_set()
            release.set()
            if outcome == "success":
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert native_owner.done() and native_owner.exception() is None
            else:
                with pytest.raises(BaseExceptionGroup) as caught:
                    await observer
                assert any(type(error) is ValueError for error in caught.value.exceptions)
                assert any(type(error) is asyncio.CancelledError for error in caught.value.exceptions)
                assert type(native_owner.exception()) is ValueError
            assert returned.is_set()
            await waiting
            assert contender_entered.is_set() and not services._lock.locked()
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)
            if waiting is not None:
                await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_construction_serialization_does_not_ignore_an_unrelated_real_writer(tmp_path, monkeypatch):
    from scripts.stage_runtime import _assert_writers_closed
    from src.obs.diagnostic_store import DiagnosticStore
    from src.obs.metrics_emitter import project_diagnostic

    async with _construction_graph(tmp_path, monkeypatch) as (bundle, admission, services):
        foreign_root = tmp_path / "unrelated"
        writer = DiagnosticStore(foreign_root / "metrics.jsonl", clock=bundle.privacy.ledger.clock,
            retention_days=90, rotate_bytes=1024*1024, project=project_diagnostic,
            policy=bundle.privacy.ledger)
        try:
            await asyncio.to_thread(writer.prune)
            assert writer.status()["writer_lease"]
            monkeypatch.setattr(admission, "revalidate", lambda: _assert_writers_closed(foreign_root))
            with pytest.raises(ValueError, match="^runtime_admission_writer_active$"):
                await admission.revalidate_constructed()
            assert not services._lock.locked() and writer.status()["writer_lease"]
            assert type(admission._construction_revalidation.exception()) is ValueError
        finally:
            await asyncio.to_thread(writer.close)


@pytest.mark.asyncio
async def test_concurrent_construction_observers_share_one_actual_pending_validation(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    async with _construction_graph(tmp_path, monkeypatch) as (_, admission, services):
        calls = []

        def validate():
            calls.append("entered")
            entered.set()
            assert release.wait(5)

        monkeypatch.setattr(admission, "revalidate", validate)
        first = asyncio.create_task(admission.revalidate_constructed())
        second = asyncio.create_task(admission.revalidate_constructed())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            owner = admission._construction_revalidation
            assert services._lock.locked() and not owner.done() and calls == ["entered"]
            release.set()
            await asyncio.gather(first, second)
            assert owner.done() and calls == ["entered"]
            # A later explicit validation is fresh; a completed successful
            # observation cannot substitute for the next requested check.
            await admission.revalidate_constructed()
            assert admission._construction_revalidation is not owner
            assert calls == ["entered", "entered"]
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_historical_writer_cannot_reenter_its_own_construction_lock(tmp_path, monkeypatch):
    async with _construction_graph(tmp_path, monkeypatch) as (_, admission, services):
        actual_run = services._disk.run
        rejected = []

        async def forwarding(function, *args, **kwargs):
            assert services._lock.locked()
            with pytest.raises(RuntimeError, match="^runtime_admission_construction_reentrant$"):
                await admission.revalidate_constructed()
            rejected.append(True)
            return await actual_run(function, *args, **kwargs)

        monkeypatch.setattr(services._disk, "run", forwarding)
        await services._run_history(str(tmp_path / "historical" / "metrics.jsonl"))
        assert rejected == [True] and not services._lock.locked()
        assert admission._construction_revalidation is None


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["callback", "privacy", "lock", "disk"])
async def test_constructed_diagnostic_owner_identity_cannot_be_substituted(tmp_path, monkeypatch, edge):
    from src.core.owned_work import OwnedBlockingWork
    from src.memory.erasure_coordinator import CleanupDriver
    from src.memory.erasure_journal import ErasureParticipant
    from src.obs.diagnostic_erasure import DiagnosticServices

    async with _construction_graph(tmp_path, monkeypatch) as (bundle, admission, services):
        called = []
        monkeypatch.setattr(admission, "revalidate", lambda: called.append(True))
        coordinator = bundle.graph.erasure_services
        original = (coordinator._drivers[ErasureParticipant.DIAGNOSTICS], services.privacy,
                    services._lock, services._disk)
        other = None
        try:
            if edge == "callback":
                other = DiagnosticServices(bundle.privacy)
                coordinator._drivers[ErasureParticipant.DIAGNOSTICS] = CleanupDriver(other.clean, frozenset({2}))
            elif edge == "privacy":
                other = PrivacyServices(str(tmp_path / "unrelated" / "privacy.db"))
                await other.acall(other.initialize)
                services.privacy = other
            elif edge == "lock":
                services._lock = asyncio.Lock()
            else:
                other = OwnedBlockingWork(capacity=1)
                services._disk = other
            with pytest.raises(RuntimeError, match="^runtime_admission_diagnostic_owner_changed$"):
                await admission.revalidate_constructed()
            assert called == [] and admission._construction_revalidation is None
        finally:
            coordinator._drivers[ErasureParticipant.DIAGNOSTICS], services.privacy, services._lock, services._disk = original
            if edge == "callback":
                await other.begin_close()
            elif edge in {"privacy", "disk"}:
                await other.aclose()


def _managed_path(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    (state / runtime_admission.MARKER).write_text('{}\n')
    return state / "data" / "graph" / "privacy.db"


def test_unmanaged_development_does_not_load_controller_or_create_data(tmp_path, monkeypatch):
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    def unexpected():
        raise AssertionError("unmanaged_configuration_loaded_controller")
    monkeypatch.setattr(runtime_admission, "_stage", unexpected)
    database = tmp_path / "unmanaged" / "privacy.db"
    assert runtime_admission.verify_runtime_admission({"memory": {"privacy_db_path": str(database)}}) is None
    assert not database.parent.exists()


@pytest.mark.asyncio
async def test_missing_ticket_prevents_sqlite_and_writer_lock_creation(tmp_path, monkeypatch):
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    database = _managed_path(tmp_path)
    services = PrivacyServices(str(database))
    try:
        with pytest.raises(RuntimeError, match="^runtime_admission_ticket_missing$"):
            await services.acall(services.initialize)
        assert not database.parent.exists()
        assert not Path(str(database) + ".lock").exists()
    finally:
        await services.aclose()
    assert services.resource_status()["close_phase"] == "complete"
    assert not services.resource_status()["writer_lease"]


@pytest.mark.asyncio
async def test_missing_ticket_stops_graph_before_leaf_constructor(tmp_path, monkeypatch):
    from src.main import ApplicationFactories, build_application

    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    database = _managed_path(tmp_path)
    calls = []
    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("unadmitted_graph_reached_native_constructor")
    config = {"memory": {"privacy_db_path": str(database)},
        "home_assistant": {"url": "http://127.0.0.1:1", "token": "synthetic-unused"}}
    async with AsyncExitStack() as shutdown:
        with pytest.raises(RuntimeError, match="^runtime_admission_ticket_missing$"):
            await build_application(config, shutdown=shutdown,
                factories=ApplicationFactories(ha_client=forbidden))
    assert calls == [] and not database.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["observation", "stop", "both", "cancellation"])
async def test_stop_observation_failure_still_attempts_cleanup_and_stays_unconfirmed(failure):
    """Exercise the stop seam only; no graph, native ACK or receipt is forged."""
    admission = runtime_admission.RuntimeAdmission({}, {}, "unused", "unused")
    calls = []

    class Observation:
        async def pre_close(self):
            calls.append("observe")
            if failure == "cancellation":
                raise asyncio.CancelledError
            if failure in {"observation", "both"}:
                raise RuntimeError("observation_failed")

    async def stop():
        calls.append("cleanup")
        if failure in {"stop", "both"}:
            raise RuntimeError("cleanup_failed")

    admission._closure_witness = Observation()
    admission._stop_original = stop
    expected = (BaseExceptionGroup if failure == "both" else
        asyncio.CancelledError if failure == "cancellation" else RuntimeError)
    with pytest.raises(expected):
        await admission._observe_service_stop()
    assert calls == ["observe", "cleanup"]
    assert admission._closure_failed is True
    with pytest.raises(RuntimeError, match="^runtime_admission_closure_unconfirmed$"):
        await admission._prepare_closure()
    assert calls == ["observe", "cleanup"]


@pytest.mark.parametrize("domain", ["memory", "stt", "new_store"])
def test_explicit_serving_database_cannot_escape_state_in_another_domain(tmp_path, domain):
    state = tmp_path / "state"
    state.mkdir()
    with pytest.raises(RuntimeError, match="^runtime_admission_foreign_serving_store:"):
        runtime_admission._check_serving_paths(
            {domain: {"db_path": str(tmp_path / "foreign.db")}}, state)
    assert not (tmp_path / "foreign.db").exists()


@pytest.mark.asyncio
async def test_authority_projection_accepts_schemas_created_by_the_actual_graph(tmp_path, monkeypatch):
    import time

    from scripts.stage_runtime import capture_state_authority
    from tests.mocks.memory_graph import Clock
    from tests.mocks.restore_application_graph import application_graph

    state = tmp_path / "state"
    (state / "config").mkdir(parents=True)
    (state / "config/settings.yaml").write_text("reasoner:\n  cloud:\n    consent: false\n")
    async with application_graph(state / "data/graph", monkeypatch, Clock(time.time())) as bundle:
        assert bundle.graph.privacy.ledger.authority_available
    assert not bundle.graph.privacy.ledger.authority_available
    projection = capture_state_authority(state)
    assert projection["schema_version"] == 1
    assert len(projection["digest"]) == 64
    assert projection == capture_state_authority(state)
