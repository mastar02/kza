"""T033-U1 through the real application factory (T010 graph).

The RequestRouter and the orchestrator's dispatcher share one read-only
VoiceRoutineService whose scheduler main binds late (it is built after both);
spoken routine questions read routines.db, match the web catalog and attach a
TaskOutcome, in both router modes, without blocking the event loop.
"""

from contextlib import AsyncExitStack

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from src.application import ApplicationFactories
from src.main import build_application
from src.monitoring.loop_lag import LoopLagConfig, LoopLagProbe
from src.pipeline.command_event import CommandEvent
from src.routines.models import RunStatus, parse_routine_spec
from src.routines.voice_routines import VoiceRoutineService
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config, offline_home
from tests.mocks.audio_output import AudioBackend

STORE_READS = ("list_all", "get", "get_run", "get_latest_run", "get_run_by_occurrence")


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    import src.tts.piper_tts as audio
    import src.vectordb.chroma_sync as index

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index, "SentenceTransformer", lambda *a, **kw: OfflineEncoder())
    monkeypatch.setattr(audio, "sd", AudioBackend())
    return ApplicationFactories(stt=lambda _: OfflineSTT(), tts=lambda _: OfflineTTS(), preflight=lambda: None)


def event():
    return CommandEvent(np.full(16000, .1, dtype=np.float32), room_id="")


def spec(name, *, enabled=True, schedule=None):
    return parse_routine_spec({"name": name, "description": "", "enabled": enabled,
                               "timezone": "America/Montevideo", "schedule": schedule, "conditions": [],
                               "actions": [{"entity_id": "light.living", "operation": "turn_on"}]})


def count_reads(monkeypatch, store) -> list[str]:
    calls: list[str] = []
    for name in STORE_READS:
        original = getattr(store, name)

        async def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return await _original(*args, **kwargs)
        monkeypatch.setattr(store, name, counted)
    return calls


async def seed(graph):
    """Three saved routines (one disabled) and one failed receipt, written to routines.db."""
    store = graph.routine_scheduler.store
    await store.save(spec("Buenos días", schedule={"at": "07:00", "weekdays": [0, 1, 2, 3, 4]}), actor="admin")
    await store.save(spec("Buenas noches", enabled=False), actor="admin")
    watering = await store.save(spec("Riego"), actor="admin")
    claim = await store.claim_run(watering.routine_id, "manual-1", actor="household", source="manual")
    await store.finish_run(claim.run.run_id, RunStatus.FAILED)


async def test_router_and_orchestrator_share_the_late_bound_scheduler(leaves, tmp_path):
    async with offline_home() as (url, _):
        async with AsyncExitStack() as stack:
            graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
            router = graph.pipeline.request_router
            service = router.voice_routines
            assert isinstance(service, VoiceRoutineService)
            assert router._orchestrator.dispatcher.voice_routines is service
            assert graph.routine_scheduler is not None
            assert service.scheduler is graph.routine_scheduler
            assert router._orchestrator.dispatcher.voice_routines.scheduler is graph.routine_scheduler


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_spoken_listing_matches_the_web_catalog(leaves, tmp_path, dashboard_auth, orchestrated):
    async with offline_home() as (url, observed):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await seed(graph)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url="http://test",
                                   headers=dashboard_auth) as client:
                response = await client.get("/api/routines")
            assert response.status_code == 200
            web = [(row["routine_id"], row["name"], row["enabled"], row["revision"],
                    (row["latest_run"] or {}).get("status")) for row in response.json()]
            entries = await graph.pipeline.request_router.voice_routines.catalog()
            assert [(e.routine_id, e.name, e.enabled, e.revision, e.latest_run_status) for e in entries] == web
            assert [row[1] for row in web] == ["Buenos días", "Buenas noches", "Riego"]

            graph.models.stt.text = "qué rutinas hay"
            result = await graph.pipeline.process_command(event())
            assert result["intent"] == "routine" and result["handled"]
            assert result["response"] == "Hay 3 rutinas: Buenos días, Buenas noches (desactivada) y Riego."
            assert result["task_completed"] and result["outcome"]["status"] == "completed"
            assert result["outcome"]["persistence"] == "not_required"
            if orchestrated:
                assert result["path"] == "fast_routine"

            graph.models.stt.text = "¿cómo salió la rutina riego?"
            consult = await graph.pipeline.process_command(event())
            assert consult["intent"] == "routine" and consult["task_completed"]
            assert consult["response"] == ("La rutina Riego está activada y no tiene horario: se ejecuta solo "
                                           "cuando se pide. La última ejecución falló.")
            assert observed["service_calls"] == 0


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_non_routine_turns_never_read_routines_db(leaves, tmp_path, monkeypatch, orchestrated):
    async with offline_home() as (url, _):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await seed(graph)
            reads = count_reads(monkeypatch, graph.routine_scheduler.store)
            texts = ["agrega leche a la lista de compras", "qué recordatorios tengo", "vacía la lista de inexistente",
                     "cancela el recordatorio de agua"]
            if not orchestrated:
                # The graph's orchestrator worker is not started, so an orchestrated slow-path turn would only
                # wait for its response timeout; these two are covered on the dispatcher with a stubbed cloud.
                texts += ["cuando juega Peñarol", "algo rutinario"]
            for text in texts:
                graph.models.stt.text = text
                result = await graph.pipeline.process_command(event())
                assert result.get("intent") != "routine", text
            assert reads == []
            graph.models.stt.text = "qué rutinas hay"
            await graph.pipeline.process_command(event())
            assert reads == ["list_all"] + ["get_latest_run"] * 3


async def test_routine_reads_keep_the_loop_responsive_and_survive_a_rebuild(leaves, tmp_path):
    async with offline_home() as (url, _):
        config = offline_config(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await seed(graph)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            probe = LoopLagProbe(LoopLagConfig(interval_s=0.02, report_interval_s=3600))
            probe.start()
            try:
                for text in ("qué rutinas hay", "¿cómo salió la rutina riego?", "¿se ejecutó la rutina buenos días?"):
                    graph.models.stt.text = text
                    result = await graph.pipeline.process_command(event())
                    assert result["intent"] == "routine" and result["task_completed"], text
            finally:
                await probe.stop()
            assert probe.snapshot()["max_ms"] < 200, probe.snapshot()
            assert result["response"] == ("La rutina Buenos días está activada y corre a las 7:00 de lunes a "
                                          "viernes. Todavía no se ejecutó.")
