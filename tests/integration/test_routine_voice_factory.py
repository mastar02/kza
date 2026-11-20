"""T033-U1 through the real application factory (T010 graph).

The RequestRouter and the orchestrator's dispatcher share one read-only
VoiceRoutineService whose scheduler main binds late (it is built after both);
spoken routine questions read routines.db, match the web catalog and attach a
TaskOutcome, in both router modes, without blocking the event loop.
"""

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from src.application import ApplicationFactories
from src.main import build_application
from src.monitoring.loop_lag import LoopLagConfig, LoopLagProbe
from src.nlu.llm_router import CommandClassification
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


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_routine_state_questions_reach_the_read_in_the_real_graph(leaves, tmp_path, orchestrated):
    """T033-U1-RV1: the command gate's non-immediate rule and the dispatcher's domotics keywords
    ('activa') no longer drop or divert a routine state question, in either word order."""
    async with offline_home() as (url, observed):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await seed(graph)
            watering = ("La rutina Riego está activada y no tiene horario: se ejecuta solo cuando se pide. "
                        "La última ejecución falló.")
            for text, said in (("¿está activa la rutina riego?", watering),
                               ("¿la rutina riego está activa?", watering),
                               ("nexa, ¿está activa la rutina riego?", watering),
                               ("¿está desactivada la rutina buenas noches?",
                                "La rutina Buenas noches está desactivada. Todavía no se ejecutó.")):
                graph.models.stt.text = text
                result = await graph.pipeline.process_command(event())
                assert (result["intent"], result["response"]) == ("routine", said), text
                assert result["task_completed"] and result["outcome"]["status"] == "completed", text
                if orchestrated:
                    assert result["path"] == "fast_routine", text
            assert observed["service_calls"] == 0


async def test_general_questions_about_routines_are_not_answered_as_unknown_routines(leaves, tmp_path):
    """T033-U1-RV2 in the legacy graph (an orchestrated slow-path turn would only wait for the
    unstarted worker's timeout; the dispatcher test covers it with a stubbed cloud)."""
    async with offline_home() as (url, observed):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = False
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await seed(graph)
            for text in ("¿cuál es una buena rutina de ejercicios?", "¿qué rutina de skincare me recomendás?",
                         "¿qué hago en mi rutina de ejercicios?"):
                graph.models.stt.text = text
                result = await graph.pipeline.process_command(event())
                assert result["intent"] != "routine", text
                assert "No encontré una rutina" not in (result["response"] or ""), text
            assert observed["service_calls"] == 0


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_list_and_consult_phrases_reach_the_read_in_the_real_graph(leaves, tmp_path, orchestrated):
    """T033-U1-RV6: the phrases the unit tests list as reads reach the read in the graph, not
    only detect(). Orchestrated: 'lista de rutinas' is no longer taken by the shared lists
    (FAST_LIST). Legacy: the read runs before stage 2, so a consult after an answer is no
    longer taken as feedback on that answer."""
    async with offline_home() as (url, observed):
        config = offline_config(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await seed(graph)
            listing = "Hay 3 rutinas: Buenos días, Buenas noches (desactivada) y Riego."
            night = "La rutina Buenas noches está desactivada. Todavía no se ejecutó."
            list_phrases = (
                "qué rutinas tengo", "lista de rutinas", "muestra las rutinas",
                "cuáles son mis rutinas", "mis automatizaciones",
            )
            turns = [(text, listing) for text in list_phrases] + [
                ("¿Se ejecutó la rutina Buenas Noches?", night),
                ("¿qué hace la rutina buenas noches?", night),
                ("¿la rutina buenas noches anduvo bien hoy?", night),
                ("¿cómo salió la rutina que se llama riego?",
                 "La rutina Riego está activada y no tiene horario: se ejecuta solo cuando se pide. "
                 "La última ejecución falló."),
                ("¿qué rutina apaga las luces?", listing),
            ]
            for text, said in turns:
                graph.models.stt.text = text
                result = await graph.pipeline.process_command(event())
                assert (result["intent"], result["response"]) == ("routine", said), text
                assert result["outcome"]["status"] == "completed", text
                if orchestrated:
                    assert result["path"] == "fast_routine", text
            assert observed["service_calls"] == 0


# T033-U1b (user answer 2026-11-19, "T033-U1 routines"): production runs the 7B LLMCommandRouter
# (nlu.llm_router.enabled); the offline graph has none, so these tests attach one that rejects every turn.
REJECTING_7B = CommandClassification(is_command=False, confidence=0.1, intent="noise", rejection_reason="noise")
LISTING = "Hay 3 rutinas: Buenos días, Buenas noches (desactivada) y Riego."
WATERING = ("La rutina Riego está activada y no tiene horario: se ejecuta solo cuando se pide. "
            "La última ejecución falló.")
NIGHT = "La rutina Buenas noches está desactivada. Todavía no se ejecutó."
# Routine questions the read always answers (VoiceRoutineService.claims), with what they say.
ROUTINE_QUESTIONS = [
    ("qué rutinas hay", LISTING),
    ("¿qué rutina apaga las luces?", LISTING),
    # T033-U1b-RV1: list questions the household layer claims (R1 routine query) with more words than
    # household_intents.unvetoed_max_words (6) skip its 7B veto too; the clock bypass has no length limit either.
    ("¿cuáles son las rutinas que tengo guardadas ahora?", LISTING),
    ("mostrame las rutinas que hay en la casa por favor", LISTING),
    ("¿está activa la rutina riego?", WATERING),
    ("¿la rutina riego está activa?", WATERING),
    ("¿cómo salió la rutina riego?", WATERING),
    ("¿qué hace la rutina buenas noches?", NIGHT),
    ("¿está desactivada la rutina buenas noches?", NIGHT),
]
# How each capture reaches the router (MultiRoomAudioLoop tags it; see test_request_router_clock).
OWN_WAKE = {"capture_trigger": "wake", "wake_score": 0.9}
WITHOUT_A_CLEAR_OWN_WAKE = {
    "follow_up": {"capture_trigger": "follow_up", "wake_score": 0.9},
    "barge_in": {"capture_trigger": "barge_in", "wake_score": 0.9},
    "untagged": {"wake_score": 0.9},
    "textual": {"capture_trigger": "wake", "wake_score": 0.9, "textual_wake": True},
    "ambient_strict": {"capture_trigger": "wake", "wake_score": 0.9, "ambient_strict": True},
    "low_score": {"capture_trigger": "wake", "wake_score": 0.42},
}


def captured(**fields) -> CommandEvent:
    return CommandEvent(np.full(16000, .1, dtype=np.float32), room_id="", **fields)


async def graph_with_rejecting_7b(stack, tmp_path, url, leaves):
    graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
    await seed(graph)
    router = graph.pipeline.request_router
    assert router.orchestrator_enabled and router.wake_acoustically_confirmed
    router.llm_command_router = MagicMock()
    router.llm_command_router.classify = AsyncMock(return_value=REJECTING_7B)
    return graph, router.llm_command_router


async def test_routine_questions_on_a_clear_own_wake_skip_the_7b_veto(leaves, tmp_path):
    """T033-U1-FV2: a 7B that rejects the turn no longer silences a routine question woken by its own wake.
    The voice list keeps disabled routines marked '(desactivada)', like the web catalog."""
    async with offline_home() as (url, observed):
        async with AsyncExitStack() as stack:
            graph, veto = await graph_with_rejecting_7b(stack, tmp_path, url, leaves)
            for text, said in ROUTINE_QUESTIONS:
                graph.models.stt.text = text
                result = await graph.pipeline.process_command(captured(**OWN_WAKE))
                assert (result["intent"], result["response"], result["path"]) == ("routine", said, "fast_routine"), text
                assert result["outcome"]["status"] == "completed", text
                assert "llm_router" not in result["timings"], text
            veto.classify.assert_not_awaited()
            assert observed["service_calls"] == 0


@pytest.mark.parametrize("capture", sorted(WITHOUT_A_CLEAR_OWN_WAKE))
async def test_routine_questions_without_a_clear_own_wake_keep_the_7b_veto(leaves, tmp_path, monkeypatch, capture):
    """T037-U5: a follow-up or barge-in capture borrows the last wake's score, so a TV saying a routine
    question in that window is vetoed; so is every weak wake the clock bypass excludes (T037-U4)."""
    async with offline_home() as (url, observed):
        async with AsyncExitStack() as stack:
            graph, veto = await graph_with_rejecting_7b(stack, tmp_path, url, leaves)
            reads = count_reads(monkeypatch, graph.routine_scheduler.store)
            household = graph.pipeline.request_router.household_intents
            texts = [text for text, _ in ROUTINE_QUESTIONS
                     # An untagged capture is not a TV signal for the household layer (P3-F2 S6), which answers
                     # a short claimed turn ('qué rutinas hay') with no veto; unclaimed and longer claimed ones
                     # reach the 7B.
                     if capture != "untagged" or household.classify(text) is None
                     or len(text.split()) > household.settings.unvetoed_max_words]
            assert len(texts) >= len(ROUTINE_QUESTIONS) - 1
            for text in texts:
                graph.models.stt.text = text
                veto.classify.reset_mock()
                result = await graph.pipeline.process_command(captured(**WITHOUT_A_CLEAR_OWN_WAKE[capture]))
                assert veto.classify.await_count == 1, text
                assert result["intent"].startswith("llm_rejected:"), (text, result["intent"])
                assert result["success"] is False and result["response"] == "", text
            assert reads == []
            assert observed["service_calls"] == 0


@pytest.mark.parametrize("text", [
    "contame un chiste",
    # T033-U1-RV2: general questions that only mention "rutina" are not routine reads.
    "¿cuál es una buena rutina de ejercicios?",
    # An uncued consult that names no saved routine: the read would decline it, so it keeps the gate.
    "¿qué hago en mi rutina de ejercicios?",
])
async def test_other_turns_on_a_clear_own_wake_still_meet_the_7b(leaves, tmp_path, monkeypatch, text):
    async with offline_home() as (url, observed):
        async with AsyncExitStack() as stack:
            graph, veto = await graph_with_rejecting_7b(stack, tmp_path, url, leaves)
            reads = count_reads(monkeypatch, graph.routine_scheduler.store)
            graph.models.stt.text = text
            result = await graph.pipeline.process_command(captured(**OWN_WAKE))
            assert veto.classify.await_count == 1
            assert result["intent"] == "llm_rejected:noise"
            assert result["success"] is False and result["response"] == ""
            assert reads == []
            assert observed["service_calls"] == 0


@pytest.mark.parametrize("text", [
    "decime las rutinas y prendé la luz",
    "decime las rutinas y abrí la puerta",
])
async def test_compound_turns_naming_the_routines_keep_the_household_veto(leaves, tmp_path, monkeypatch, text):
    """T033-U1b-RV2: a compound the household layer claims (R5) is not a routine query, so even on a clear own
    wake a turn over unvetoed_max_words meets its 7B veto; a rejecting 7B ends it silently with no read or action."""
    async with offline_home() as (url, observed):
        async with AsyncExitStack() as stack:
            graph, veto = await graph_with_rejecting_7b(stack, tmp_path, url, leaves)
            router = graph.pipeline.request_router
            assert router.household_intents.classify(text).family.value == "compound"
            assert router.voice_routines.claims(text)
            reads = count_reads(monkeypatch, graph.routine_scheduler.store)
            graph.models.stt.text = text
            result = await graph.pipeline.process_command(captured(**OWN_WAKE))
            assert veto.classify.await_count == 1
            assert result["intent"] == "llm_rejected:household"
            assert result["success"] is False and result["response"] == ""
            assert reads == []
            assert observed["service_calls"] == 0


@pytest.mark.parametrize("text", [
    "¿qué hay en la lista de rutinas?",
    "mostrame la lista de rutinas que hay en la casa",
])
async def test_other_household_queries_naming_the_routines_keep_the_household_veto(leaves, tmp_path, monkeypatch,
                                                                                    text):
    """T033-U1b-FV1: only a claim of the routine family skips the household veto. A shared-list query (R1, family
    list) that names the routines is also accepted by VoiceRoutineService.claims, yet over unvetoed_max_words it
    meets the 7B on a clear own wake; a rejecting 7B ends it silently with no read or action."""
    async with offline_home() as (url, observed):
        async with AsyncExitStack() as stack:
            graph, veto = await graph_with_rejecting_7b(stack, tmp_path, url, leaves)
            router = graph.pipeline.request_router
            household = router.household_intents.classify(text)
            assert (household.family.value, household.operation.value) == ("list", "query")
            assert router.voice_routines.claims(text)
            assert len(text.split()) > router.household_intents.settings.unvetoed_max_words
            reads = count_reads(monkeypatch, graph.routine_scheduler.store)
            graph.models.stt.text = text
            result = await graph.pipeline.process_command(captured(**OWN_WAKE))
            assert veto.classify.await_count == 1
            assert result["intent"] == "llm_rejected:household"
            assert result["success"] is False and result["response"] == ""
            assert reads == []
            assert observed["service_calls"] == 0
