"""Managed construction must stop before a private owner opens stores."""
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from src.core import runtime_admission
from src.memory.privacy_services import PrivacyServices


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
