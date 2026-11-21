"""P6 offline acceptance through the real application factory and HTTP routes."""

import asyncio
from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager, closing
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import yaml

from src.application import ApplicationFactories, ApplicationGraph
from src.dashboard.authority import DashboardRole
from src.lists.list_store import OwnerType
from src.llm import reasoner
from src.main import build_application
from src.pipeline.command_event import CapturedTranscript, CommandEvent, TranscriptSource
from src.tts import piper_tts
from src.users.user_manager import PermissionLevel
from src.vectordb import chroma_sync
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, OfflineWake, offline_config
from tests.mocks.audio_output import AudioBackend
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio

ROOT = Path(__file__).resolve().parents[2]
PCM = np.full(16000, 0.1, dtype=np.float32)
HA_TOKEN = "synthetic-p6-ha-bearer"
ROUTER_TOKEN = "synthetic-p6-model-bearer"
PRIVATE_ITEM = "synthetic-p6-private-list-item"
PRIVATE_REMINDER = "synthetic-p6-private-reminder"
RESIDENT_NAME = "P6 synthetic resident"
PRIVATE_LISTS = ("pendientes", "viajes", "compras", "agenda", "tareas")
ROOMS = ("living", "cocina", "hall", "escritorio", "dormitorio")


@dataclass(frozen=True)
class NegativeTurn:
    family: str
    text: str
    room: str = "living"


def negative_turns() -> tuple[NegativeTurn, ...]:
    """Build the fixed C02 matrix from utterances and independent capture conditions."""
    imperatives = (
        "prendé la luz", "apagá la luz", "poné un timer de cinco minutos",
        "cerrá la puerta", "desactivá la alarma", "pausá la tele",
    )
    outside_domain = (
        "a qué hora es el partido de", "quién ganó el campeonato en",
        "cuándo abre el museo de", "quién escribió la historia de",
        "dónde queda la estación de", "cuánto sale el ómnibus a",
    )
    ambiguous = (
        "prendé la luz y apagá la luz y prendé la luz y apagá la luz",
        "apagá la luz y prendé la luz y apagá la luz y prendé la luz",
        "poné un timer de cinco minutos y apagá la luz y prendé la luz y apagá la luz",
        "apagá la luz y poné un timer de cinco minutos y apagá la luz y prendé la luz",
        "prendé la luz y poné un timer de cinco minutos y prendé la luz y apagá la luz",
        "apagá la luz y prendé la luz y poné un timer de cinco minutos y apagá la luz",
    )
    visitor = (
        "leé la lista de", "mostrame la lista de", "qué hay en la lista de",
        "borrá la lista de", "vaciá la lista de", "agregá pan a la lista de",
    )
    cases = [
        NegativeTurn(family, phrase, room)
        for family, phrases in (
            ("incomplete_capture", imperatives),
            ("stale_transcript", outside_domain),
            ("ambiguous_compound", ambiguous),
            ("tv", imperatives),
            ("barge_in", imperatives),
        )
        for phrase in phrases
        for room in ROOMS
    ]
    cases.extend(NegativeTurn("outside_domain", f"{query} {place}")
                 for query in outside_domain for place in ("Maldonado", "Madrid", "Lima", "Salto", "Roma"))
    cases.extend(NegativeTurn("visitor", f"{query} {name}") for query in visitor for name in PRIVATE_LISTS)
    return tuple(cases)


@pytest.fixture
def settings() -> dict:
    with (ROOT / "config/settings.yaml").open(encoding="utf-8") as source:
        return yaml.safe_load(source)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict):
    """Keep real policy, stores, queue, HTTP and PCM delivery around offline leaves."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(chroma_sync, "SentenceTransformer", lambda *_args, **_kwargs: OfflineEncoder())
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, "sd", backend)
    monkeypatch.setenv("LLAMA_API_KEY", ROUTER_TOKEN)
    model = Path(settings["router"]["model"]).name
    host = settings["router"]["base_url"]
    provider_calls: list[tuple[str, str]] = []
    client = httpx.Client
    classified_as_requests = {repr("qué tiempo hace")}
    classified_as_requests.update(repr(case.text) for case in negative_turns() if case.family == "ambiguous_compound")

    def model_server(request: httpx.Request) -> httpx.Response:
        provider_calls.append((request.url.host, request.url.path))
        assert request.headers["Authorization"] == f"Bearer {ROUTER_TOKEN}"
        if request.url.host == "reasoner.p6.invalid":
            assert request.url.path == "/v1/models"
            return httpx.Response(503)
        assert request.url.port == 8101
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": model, "object": "model"}]})
        assert (request.method, request.url.path) == ("POST", "/v1/completions")
        prompt = json.loads(request.content)["prompt"]
        assert all(secret not in prompt for secret in (PRIVATE_ITEM, PRIVATE_REMINDER, RESIDENT_NAME))
        is_request = prompt.rpartition("\nTEXTO: ")[2] in classified_as_requests
        verdict = {
            "is_command": is_request, "confidence": 0.99,
            "rejection_reason": None if is_request else "incomplete",
        }
        return httpx.Response(200, json={
            "id": "p6-test", "object": "text_completion", "created": 0, "model": model,
            "choices": [{"text": json.dumps(verdict), "index": 0, "finish_reason": "stop"}],
        })

    def open_model_client(**kwargs: object) -> httpx.Client:
        return client(transport=httpx.MockTransport(model_server), **kwargs)

    monkeypatch.setattr(reasoner, "httpx", SimpleNamespace(Client=open_model_client))

    @asynccontextmanager
    async def open_graph(ha: TaskTransport, *, remote: bool = False):
        async with ha.serve() as url, AsyncExitStack() as stack:
            config = offline_config(tmp_path, url, rooms=True)
            for zone, room in enumerate(ROOMS, 1):
                config["rooms"][room] = {"name": room, "mic_device_index": zone, "ma1260_zone": zone}
            config["rooms"]["living"]["default_light"] = "light.living"
            config["presence"] = {"enabled": False}
            config["tts"]["streaming"]["default_output_zone"] = "zone_living"
            config.update({key: deepcopy(settings[key]) for key in ("command_gate", "household_intents", "nlu")})
            config["router"] = {"enabled": True, "destination": "local", "base_url": host, "model": model}
            config["home_assistant"].update(token=HA_TOKEN, weather_entity="weather.synthetic")
            selection = tmp_path / "synthetic-private-selection.json"
            selection.write_text(json.dumps({
                "selection_approved": True, "live_api_verified": True, "allowed_entities": [{
                    "entity_id": "light.living", "operations": ["turn_on", "turn_off"],
                    "live_verified": True, "indirect_effects_verified": True,
                }],
            }), encoding="utf-8")
            config["home_assistant"]["control_selection_path"] = str(selection)
            if remote:
                config["reasoner"].update(
                    http_base_url="https://reasoner.p6.invalid/v1", destination="cloud",
                    api_key_env="LLAMA_API_KEY", cloud={"consent": True},
                )
            stt, tts = OfflineSTT(), OfflineTTS()
            amp = MagicMock(audio_output_device=7)
            factories = ApplicationFactories(
                stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None,
                wake_detector=OfflineWake, amplifier=lambda **_: amp,
            )
            graph = await build_application(config, shutdown=stack, factories=factories)
            assert graph.models.stt is stt and graph.models.tts is tts
            yield graph

    return open_graph, backend, provider_calls


async def say(graph: ApplicationGraph, text: str, room: str = "living", **capture) -> dict:
    """Drive the production command entry at the PCM-to-STT boundary."""
    graph.models.stt.text = text
    event = CommandEvent(PCM.copy(), room_id=room, **{"capture_complete": True, "capture_trigger": "wake", **capture})
    return await graph.pipeline.process_command(event)


async def start_commands(graph: ApplicationGraph) -> None:
    """Run the command queue and timer manager with their normal factory cleanup."""
    await graph.pipeline.request_router._orchestrator.start()
    await graph.voice_services.features.timer_manager.start()


def durable_effects(root: Path) -> dict[str, list[tuple]]:
    """Read all effect and receipt rows, including hidden or cancelled records."""
    tables = {
        "lists": ("lists", "list_items"),
        "reminders": ("reminders", "reminder_home_actions", "timers"),
        "routines": ("routines", "routine_runs", "routine_run_actions"),
    }
    result = {}
    for owner, names in tables.items():
        with closing(sqlite3.connect(f"{(root / (owner + '.db')).as_uri()}?mode=ro", uri=True)) as db:
            for name in names:
                result[name] = db.execute(f"SELECT * FROM {name} ORDER BY rowid").fetchall()
    return result


async def test_p6_real_factory_rebuilds_shared_owners_with_committed_placement(rig, settings: dict):
    assert settings["stt"]["device"] == settings["tts"]["kokoro"]["device"] == "cuda:1"
    assert settings["speaker_id"]["device"] == "cuda:1"
    assert settings["ambient"]["stt"]["device"] == "cuda:0"
    assert (settings["embeddings"]["model"], settings["embeddings"]["device"]) == ("BAAI/bge-m3", "cpu")
    # The 7B device belongs to the external service. KZA binds its declared local endpoint.
    assert settings["router"]["destination"] == "local"
    assert settings["router"]["base_url"] == "http://127.0.0.1:8101/v1"
    assert "7B" in settings["router"]["model"]
    conventions = (ROOT / "docs/SERVER_CONVENTIONS.md").read_text(encoding="utf-8")
    router_declaration, = (line for line in conventions.splitlines() if line.startswith("| :8101 |"))
    assert "7B" in router_declaration and "cuda:1" in router_declaration
    open_graph, _, _ = rig
    ha = TaskTransport()
    previous = None
    for _ in range(2):
        async with open_graph(ha) as graph:
            pipe = graph.pipeline
            voice = pipe.request_router
            orchestrator = voice._orchestrator
            assert graph.home_actions is voice.home_actions is orchestrator.dispatcher.home_actions
            assert graph.home_actions is graph.dashboard.home_actions
            assert graph.ha_client is graph.dashboard.ha is graph.home_actions.ha
            assert graph.home_capabilities is voice.home_voice_resolver.capabilities
            assert pipe.response_handler is voice.response_handler is graph.voice_services.responses
            assert graph.dashboard.user_manager is pipe.command_processor.user_manager
            assert graph.voice_registry is graph.dashboard.user_manager.voice_registry
            assert graph.voice_registry.privacy is graph.privacy is pipe.command_processor._privacy
            assert graph.dashboard.authority is graph.dashboard_authority
            assert voice.llm_command_router.router.policy.consent_gate is graph.consent_gate
            assert graph.models.speaker_identifier is None and voice.llm is None
            assert graph.models.stt.loads == graph.models.tts.loads == 0
            assert not pipe._running and not graph.routine_scheduler._running
            if previous is not None:
                assert previous.home_actions is not graph.home_actions
                assert previous.privacy is not graph.privacy
                assert previous.dashboard_authority is not graph.dashboard_authority
            previous = graph
        assert graph.voice_services.responses.get_delivery_status()["closed"]
    assert ha.sends == []


async def test_p6_negative_captures_have_zero_home_effects_and_private_disclosures(
    rig, tmp_path: Path, dashboard_auth: dict[str, str],
):
    open_graph, _, provider_calls = rig
    ha = TaskTransport()
    cases = negative_turns()
    assert len(set(cases)) == len(cases) == 210
    assert Counter(case.family for case in cases) == {
        "incomplete_capture": 30, "stale_transcript": 30, "ambiguous_compound": 30,
        "tv": 30, "barge_in": 30, "outside_domain": 30, "visitor": 30,
    }

    async with open_graph(ha) as graph:
        await start_commands(graph)
        users = graph.dashboard.user_manager
        person, _ = await asyncio.to_thread(users.add_user, RESIDENT_NAME, PermissionLevel.ADULT)
        store = graph.list_manager._store
        private_lists = []
        for name in PRIVATE_LISTS:
            private = await store.create_list(name, OwnerType.USER, person.user_id)
            await store.add_item(private.id, PRIVATE_ITEM)
            private_lists.append(private)
        await graph.reminder_manager.create(person.user_id, PRIVATE_REMINDER, time.time() + 3600)
        authority = graph.dashboard_authority
        owner = await authority.issue_credential(
            authority.resolve(dashboard_auth["Authorization"]), subject="p6-resident-session",
            role=DashboardRole.HOME, user_id=person.user_id, expires_at=time.time() + 120,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=graph.dashboard.app),
                                     base_url="http://testserver") as client:
            for private in private_lists:
                endpoint = f"/api/lists/{private.id}/items"
                reply = await client.get(endpoint, headers={"Authorization": f"Bearer {owner.token}"})
                assert reply.status_code == 200 and PRIVATE_ITEM in reply.text
        timers = graph.voice_services.features.timer_manager
        seeded_timer = await say(graph, "poné un timer de 60 minutos")
        assert seeded_timer["task_completed"], seeded_timer
        existing, = timers.get_all_timers()
        before = await asyncio.to_thread(durable_effects, tmp_path)
        stale = CapturedTranscript.from_audio("prendé la luz", TranscriptSource.WAKE, PCM * 2)
        seen_veto: set[str] = set()
        visitor_refusals = 0
        for case in cases:
            capture = {
                "incomplete_capture": {"capture_complete": False, "early_dispatch": True},
                "stale_transcript": {"transcript": stale},
                "tv": {"textual_wake": True, "ambient_strict": True, "wake_score": 0.4},
                "barge_in": {"capture_trigger": "barge_in"},
            }.get(case.family, {})
            spoken_before = len(graph.models.tts.texts)
            stt_before = graph.models.stt.calls
            calls_before = len(provider_calls)
            result = await say(graph, case.text, case.room, **capture)
            assert result["user"] is None, case
            assert result["outcome"]["status"] in {"failed", "unsupported", "needs_input"}, (case, result)
            assert not result["task_completed"] and not result["success"], (case, result)
            assert result["outcome"]["persistence"] != "committed", case
            assert result["outcome"]["dispatch"] != "accepted" and result["outcome"]["effect"] != "observed", case
            reply = result["response"]
            assert graph.models.tts.texts[spoken_before:] == ([reply] if reply else []), case
            for secret in (PRIVATE_ITEM, PRIVATE_REMINDER, person.user_id, RESIDENT_NAME, owner.token):
                assert secret not in json.dumps(result), case
            if case.family == "incomplete_capture":
                assert result["intent"] == "gate_rejected:incomplete_capture" and not reply, case
                assert graph.models.stt.calls == stt_before, case
            if case.family == "stale_transcript":
                assert result["text"] == case.text and graph.models.stt.calls == stt_before + 1, case
            if case.family == "ambiguous_compound":
                assert result["intent"] == "compound" and reply == "Decímelo de a una cosa.", case
            if len(provider_calls) > calls_before and result["intent"] == "llm_rejected:incomplete":
                seen_veto.add(case.family)
            if case.family == "visitor" and result["intent"] == "list" and reply:
                visitor_refusals += 1
            assert ha.sends == [] and ha.forecast_reads == [], case
            assert ha.states["light.living"]["state"] == "off", case
            assert timers.get_all_timers() == [existing] and existing.state == "running", case

        assert {"tv", "barge_in", "outside_domain", "stale_transcript"} <= seen_veto
        assert visitor_refusals > 0
        assert await asyncio.to_thread(durable_effects, tmp_path) == before

        # A real eligible turn proves the action policy/transport were armed for the matrix.
        control = await say(graph, "prendé la luz del living")
        assert control["task_completed"] and control["outcome"]["effect"] == "observed", control
        assert len(ha.sends) == 1 and ha.states["light.living"]["state"] == "on"


async def test_q22_http_projects_real_ha_observation_and_protects_diagnostics(
    rig, tmp_path: Path, dashboard_auth: dict[str, str],
):
    open_graph, _, _ = rig
    ha = TaskTransport()
    entity = "/api/entities/light.living"
    diagnostics = ("/api/health", "/api/subsystems", "/api/metrics")
    secrets = (str(tmp_path), HA_TOKEN, ROUTER_TOKEN, dashboard_auth["Authorization"].split()[-1])
    async with open_graph(ha) as graph:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=graph.dashboard.app),
                                     base_url="http://testserver") as client:
            for route in (*diagnostics, entity):
                assert (await client.get(route)).status_code == 401
            authority = graph.dashboard_authority
            visitor = await authority.issue_credential(
                authority.resolve(dashboard_auth["Authorization"]), subject="p6-home-without-diagnostics",
                role=DashboardRole.HOME, expires_at=time.time() + 120,
            )
            for route in (*diagnostics, entity):
                assert (await client.get(route, headers={"Authorization": f"Bearer {visitor.token}"})).status_code == 403
            assert ha.entity_reads == []

            observed = await client.get(entity, headers=dashboard_auth)
            assert observed.status_code == 200 and observed.headers["cache-control"] == "no-store"
            assert observed.headers["x-kza-source-id"] == "ha-rest-entity"
            assert observed.headers["x-kza-observation-state"] == "fresh"
            assert observed.json() == graph.ha_client.get_entity_observation("light.living").state
            assert observed.json()["state"] == "off" and ha.entity_reads == ["light.living"]

            for status, expected_state, phase in (
                (200, "connected", "ready"), (503, "disconnected", "failed"), (200, "connected", "ready"),
            ):
                ha.states_status = status
                snapshot = await graph.ha_client.get_state_snapshot()
                assert snapshot.available is (status == 200)
                native = graph.ha_client.get_health_status()
                assert str(native.rest.state) == expected_state and native.rest.phase == phase

                responses = [await client.get(route, headers=dashboard_auth) for route in diagnostics]
                assert all(response.status_code == 200 for response in responses)
                health, subsystems, metrics = (response.json() for response in responses)
                if status == 503:
                    assert health["status"] == "unhealthy"
                for payload in (health, subsystems):
                    projected = next(row for row in payload["subsystems"] if row["name"] == "home_assistant")
                    assert projected["status"] == ("healthy" if status == 200 else "unhealthy")
                    rest = projected["extra"]["channels"]["rest"]
                    assert (rest["state"], rest["phase"]) == (expected_state, phase)
                    assert rest["observed_at"] == native.rest.observed_at
                    assert rest["expires_at"] == native.rest.expires_at
                    assert projected["extra"]["error_count"] == native.error_count
                queue = graph.pipeline.request_router._orchestrator.queue
                assert metrics["queue_depth"] == queue.get_stats()["queue_size"]
                for response in (observed, *responses):
                    assert all(secret not in response.text for secret in (*secrets, visitor.token))
    assert ha.sends == [] and ha.forecast_reads == []


async def test_h2_local_voice_stays_available_during_ha_and_remote_reasoner_failure(rig, record_property):
    open_graph, backend, provider_calls = rig
    ha = TaskTransport(states_status=503)
    async with open_graph(ha, remote=True) as graph:
        assert ("reasoner.p6.invalid", "/v1/models") in provider_calls
        assert graph.pipeline.request_router.llm is None
        await start_commands(graph)
        loop = asyncio.get_running_loop()
        lags = []
        async with asyncio.timeout(10):
            for _ in range(10):
                ha.states_entered = asyncio.Event()
                ha.states_release = asyncio.Event()
                ha.states_hold = True
                # Keep the actual HA REST read in flight at the transport boundary.
                async with asyncio.TaskGroup() as tasks:
                    weather = tasks.create_task(say(graph, "qué tiempo hace"))
                    await ha.states_entered.wait()
                    tasks.create_task(release_ha_after_stall(ha))
                    deadline = loop.time() + 0.005
                    beat_count = 0
                    while not weather.done():
                        await asyncio.sleep(max(0, deadline - loop.time()))
                        now = loop.time()
                        lags.append(max(0, now - deadline))
                        deadline = now + 0.005
                        beat_count += 1
                    assert beat_count >= 20
                result = weather.result()
                assert result["outcome"]["status"] == "failed" and not result["task_completed"]
                assert result["source_observation"]["reason"] == "source_unavailable"
                assert result["source_observation"]["observed_at"] is None
                assert result["response"] == "No tengo el dato del clima ahora mismo."
                assert result["response_completed"]
        ordered = sorted(lags)
        p95_ms = ordered[int(0.95 * (len(ordered) - 1))] * 1000
        max_ms = ordered[-1] * 1000
        record_property("p6_h2_ha_heartbeat", json.dumps({
            "repetitions": 10, "stall_ms": 200, "interval_ms": 5,
            "samples": len(ordered), "p95_ms": p95_ms, "max_ms": max_ms,
        }))
        assert p95_ms <= 25 and max_ms <= 75

        before = len(backend.heard())
        timer = await say(graph, "poné un timer de 60 minutos")
        assert timer["task_completed"] and timer["response_completed"]
        np.testing.assert_array_equal(backend.heard()[before:, 0], np.full(240, 0.125, dtype=np.float32))
        ha.states_status = 200
        ha.states_hold = False
        ha.states["weather.synthetic"] = {
            "entity_id": "weather.synthetic", "state": "sunny", "attributes": {"temperature": 22},
        }
        weather = await say(graph, "qué tiempo hace")
        assert weather["task_completed"] and weather["source_observation"]["observed_at"] is not None
        assert weather["response"] == "Hay 22 grados, soleado."
        assert ha.sends == [] and ha.forecast_reads == []
    assert graph.voice_services.responses.get_delivery_status()["closed"]


async def release_ha_after_stall(ha: TaskTransport) -> None:
    """Release one simulated 200 ms dependency stall."""
    await asyncio.sleep(0.2)
    ha.states_release.set()
