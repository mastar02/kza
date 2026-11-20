"""T037 acceptance cases through the application factory and its real entry points."""

from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import time

import httpx
import numpy as np
import pytest

from src.application import ApplicationFactories, ApplicationGraph
from src.dashboard.authority import DashboardRole
from src.home_assistant.control_policy import HomeActor
from src.llm import reasoner
from src.main import build_application
from src.pipeline.command_event import CommandEvent
from tests.integration.test_capability_application import leaves as capability_leaves
from tests.integration.test_task_outcomes import selection
from tests.mocks.application_graph import OfflineWake, offline_config
from tests.mocks.task_transport import TaskTransport

WEATHER = "weather.synthetic"
# Tuesday 4 August, 23:30 in the household. UTC is already Wednesday.
BEFORE_MIDNIGHT = datetime(2026, 8, 5, 2, 30, tzinfo=UTC)
AFTER_MIDNIGHT = datetime(2026, 8, 5, 4, 7, tzinfo=UTC)
NO_CURRENT = "No tengo el dato del clima ahora mismo."
NO_FORECAST = "No tengo el pronóstico ahora mismo."

CURRENT_WEATHER = (
    pytest.param("qué tiempo hace", "sunny", {"temperature": 22}, 200,
                 "Hay 22 grados, soleado.", id="conditions"),
    pytest.param("cómo está el clima", "cloudy", {"temperature": 19.6, "humidity": 58.1}, 200,
                 "Hay 20 grados, nublado, 58 por ciento de humedad.", id="temperature-and-humidity"),
    pytest.param("qué temperatura hace afuera", "unknown_condition", {"temperature": "17.2"}, 200,
                 "Hay 17 grados.", id="temperature-without-known-condition"),
    pytest.param("llueve ahora", "rainy", {}, 200, "Hay lluvioso.", id="rain-condition-only"),
    pytest.param("qué tiempo hace hoy", "clear-night", {"humidity": 0}, 200,
                 "Hay despejado, 0 por ciento de humedad.", id="present-today"),
    pytest.param("qué tiempo hace esta mañana", "fog", {"temperature": 9}, 200,
                 "Hay 9 grados, con niebla.", id="present-this-morning"),
    pytest.param("cuántos grados hay afuera", "cloudy", {"temperature": True, "humidity": "unknown"}, 200,
                 "Hay nublado.", id="discard-invalid-metrics"),
    pytest.param("qué tiempo hace", None, {}, 200, NO_CURRENT, id="missing-exact-source"),
    pytest.param("qué tiempo hace", "unavailable", {"temperature": 22}, 200, NO_CURRENT, id="unavailable-state"),
    pytest.param("qué tiempo hace", "sunny", {"temperature": 22}, 503, NO_CURRENT, id="state-backend-down"),
    pytest.param("qué tiempo hace", "sunny", {"temperature": 22}, 403, NO_CURRENT, id="C02-weather-source-denied"),
)

TODAY = {"datetime": "2026-08-04T15:00:00+00:00", "condition": "sunny", "temperature": 22}
TOMORROW = {"datetime": "2026-08-05T15:00:00+00:00", "condition": "rainy", "temperature": 25}
DAY_AFTER = {"datetime": "2026-08-06T15:00:00+00:00", "condition": "cloudy", "temperature": 19}
SATURDAY = {"datetime": "2026-08-08T15:00:00+00:00", "condition": "pouring", "temperature": 12}
UNORDERED = [SATURDAY, DAY_AFTER, TODAY, TOMORROW]

# A None response reason for an unsuccessful read means that HA returned a
# valid, timestamped collection but the requested day could not be selected.
FORECAST = (
    pytest.param("va a llover hoy", UNORDERED, 200, "Hoy: soleado, 22 grados.", "completed", None,
                 id="future-today-first-record-is-saturday"),
    pytest.param("va a llover mañana", UNORDERED, 200, "Mañana: lluvioso, 25 grados.", "completed", None,
                 id="reordered-tomorrow-is-last"),
    pytest.param("qué tiempo hace pasado mañana", UNORDERED, 200, "Pasado mañana: nublado, 19 grados.",
                 "completed", None, id="day-after-tomorrow"),
    pytest.param("va a llover el sábado", UNORDERED, 200, "El sábado: lloviendo fuerte, 12 grados.",
                 "completed", None, id="named-weekday"),
    pytest.param("va a llover mañana", [TODAY, DAY_AFTER], 200, "No tengo el pronóstico para mañana.",
                 "failed", None, id="missing-requested-day"),
    pytest.param("va a llover mañana", [{**TOMORROW, "datetime": "2026-08-05T00:00:00+00:00"}], 200,
                 "No tengo el pronóstico para mañana.", "failed", None, id="utc-midnight-is-still-today"),
    pytest.param("va a llover mañana", [TOMORROW, {**TOMORROW, "temperature": 28}], 200,
                 "No tengo el pronóstico para mañana.", "failed", None, id="conflicting-same-day-entries"),
    pytest.param("va a llover mañana", [{**TOMORROW, "datetime": "2026-08-05T15:00:00"}], 200,
                 NO_FORECAST, "failed", "source_invalid", id="naive-provider-date"),
    pytest.param("va a llover mañana", [{**TOMORROW, "datetime": "mañana"}], 200,
                 NO_FORECAST, "failed", "source_invalid", id="unparseable-provider-date"),
    pytest.param("va a llover mañana", UNORDERED, 503, NO_FORECAST, "failed", "source_unavailable",
                 id="forecast-provider-down"),
    pytest.param("va a llover hoy y mañana", None, 200, "Preguntame por un día a la vez.", "needs_input", None,
                 id="C02-ambiguous-forecast"),
    pytest.param("va a llover la semana que viene", None, 200,
                 "No tengo ese pronóstico. Puedo decirte el de hoy, mañana, pasado mañana o un día de la semana.",
                 "unsupported", None, id="C02-unsupported-horizon"),
)

TIME_DATE = (
    pytest.param("qué hora es", BEFORE_MIDNIGHT, "Son las 23:30.", id="time-is-local"),
    pytest.param("decime la hora", AFTER_MIDNIGHT, "Es la 1:07.", id="singular-and-zero-padded-minute"),
    pytest.param("qué hora es ahora mismo", datetime(2026, 8, 5, 3, tzinfo=UTC), "Son las 0:00.",
                 id="local-midnight"),
    pytest.param("Nexa, ¿qué día es hoy?", BEFORE_MIDNIGHT, "Hoy es martes 4 de agosto.", id="wake-and-local-date"),
    pytest.param("¿A qué fecha estamos?", BEFORE_MIDNIGHT, "Hoy es martes 4 de agosto de 2026.", id="date-with-year"),
    pytest.param("che, ¿me podés decir qué día de la semana es mañana, por favor?", BEFORE_MIDNIGHT,
                 "Mañana es miércoles 5 de agosto.", id="long-own-wake-question"),
    pytest.param("qué día es pasado mañana", BEFORE_MIDNIGHT, "Pasado mañana es jueves 6 de agosto.",
                 id="day-after-tomorrow"),
    pytest.param("qué día fue ayer", BEFORE_MIDNIGHT, "Ayer fue lunes 3 de agosto.", id="yesterday"),
    pytest.param("qué fecha es mañana", datetime(2026, 12, 31, 15, tzinfo=UTC),
                 "Mañana es viernes 1 de enero de 2027.", id="year-rollover"),
    pytest.param("qué día es pasado mañana", datetime(2028, 2, 27, 15, tzinfo=UTC),
                 "Pasado mañana es martes 29 de febrero.", id="leap-day"),
    pytest.param("qué hora es", None, "No pude leer el reloj de la casa.", id="invalid-clock-source"),
)

DEVICE_STATE = (
    pytest.param("light.synthetic", "on", "diagnostics", 200, id="light-on"),
    pytest.param("light.synthetic", "off", "diagnostics", 200, id="light-off"),
    pytest.param("fan.synthetic", "on", "diagnostics", 200, id="fan-state"),
    pytest.param("cover.synthetic", "closed", "diagnostics", 200, id="cover-state"),
    pytest.param("climate.synthetic", "heat", "diagnostics", 200, id="climate-mode"),
    pytest.param("switch.synthetic", "off", "diagnostics", 200, id="switch-state"),
    pytest.param("light.synthetic", "unavailable", "diagnostics", 200, id="provider-unavailable-state"),
    pytest.param("light.synthetic", 42, "diagnostics", 503, id="invalid-provider-state"),
    pytest.param("light.synthetic", "on", "anonymous", 401, id="C02-device-anonymous"),
    pytest.param("light.synthetic", "on", "home", 403, id="C02-device-home-role"),
)


@dataclass(frozen=True)
class Part:
    """Expected reply, verdict and HA receipt for one spoken part."""

    segment: str
    entity: str | None = None
    service: str | None = None
    receipt: str | None = None
    verdict: str = "done"
    status: str = "completed"
    dispatch: str = "accepted"
    effect: str = "observed"
    speech: str = "hecho."


LIVING_OFF = Part("Apagá la luz del living", "light.living", "turn_off", "confirmed")
LIVING_ON = Part("Prendé la luz del living", "light.living", "turn_on", "confirmed")
KITCHEN_OFF = Part("Apagá la luz de la cocina", "light.second", "turn_off", "confirmed")
KITCHEN_ON = Part("Prendé la luz de la cocina", "light.second", "turn_on", "confirmed")
DESK_OFF = Part("Apagá la luz del escritorio", "light.third", "turn_off", "confirmed")
LOCAL_OFF = replace(LIVING_OFF, segment="Apagá la luz")
SECURITY_REFUSAL = Part("Abrí la cerradura", verdict="not_done", status="unsupported",
                        dispatch="not_sent", effect="not_observed",
                        speech="no lo hice. Las cerraduras, las puertas y la alarma no las manejo por voz.")
UNKNOWN_ROOM = Part("Apagá la luz del quincho", verdict="not_done", status="needs_input",
                    dispatch="not_sent", effect="not_observed",
                    speech="no lo hice. ¿En qué ambiente querés controlar la luz?")
DUPLICATE = Part("Apagá la luz del living", verdict="duplicate", status="cancelled",
                 dispatch="not_sent", effect="not_observed",
                 speech="no lo repetí porque era lo mismo que lo anterior.")
HA_403 = replace(LIVING_OFF, receipt="failed", verdict="not_done", status="failed",
                 dispatch="rejected", effect="not_observed", speech="no lo hice. No pude hacerlo.")
HA_503 = replace(LIVING_OFF, receipt="uncertain", verdict="uncertain", status="uncertain",
                 dispatch="uncertain", effect="uncertain",
                 speech="no sé si se hizo; fijate antes de repetirlo.")

COMPOUND_ELLIPSIS = (
    pytest.param("apagá la luz del living y prendé la luz de la cocina", (LIVING_OFF, KITCHEN_ON),
                 "completed", "confirmed", False, id="explicit-two-actions"),
    pytest.param("prendé la luz de la cocina y apagá la luz del living", (KITCHEN_ON, LIVING_OFF),
                 "completed", "confirmed", False, id="reverse-spoken-order"),
    pytest.param("apagá la luz del living y de la cocina", (LIVING_OFF, KITCHEN_OFF),
                 "completed", "confirmed", False, id="two-room-ellipsis"),
    pytest.param("apagá las luces del living y de la cocina y del escritorio", (
        replace(LIVING_OFF, segment="Apagá las luces del living"),
        replace(KITCHEN_OFF, segment="Apagá las luces de la cocina"),
        replace(DESK_OFF, segment="Apagá las luces del escritorio"),
    ), "completed", "confirmed", False, id="three-room-ellipsis"),
    pytest.param("prendé la luz del living y apagá la luz del escritorio", (LIVING_ON, DESK_OFF),
                 "completed", "confirmed", True, id="same-turn-receipt-replay"),
    pytest.param("apagá la luz del living y abrí la cerradura", (LIVING_OFF, SECURITY_REFUSAL),
                 "failed", "confirmed", False, id="C02-security-part-refused"),
    pytest.param("apagá la luz del quincho y prendé la luz de la cocina", (UNKNOWN_ROOM, KITCHEN_ON),
                 "failed", "confirmed", False, id="C02-unresolved-room-does-not-stop-other-part"),
    pytest.param("apagá la luz y apagá la luz del living", (LOCAL_OFF, DUPLICATE),
                 "failed", "confirmed", False, id="C02-resolved-duplicate"),
    pytest.param("apagá la luz del living y prendé la luz de la cocina", (HA_403, KITCHEN_ON),
                 "failed", "rejected", False, id="C02-ha-denies-one-part"),
    pytest.param("apagá la luz del living y prendé la luz de la cocina", (HA_503, KITCHEN_ON),
                 "uncertain", "uncertain", True, id="C02-uncertain-part-not-resent"),
)

FAMILIES = {
    "current_weather": CURRENT_WEATHER,
    "forecast": FORECAST,
    "time_date": TIME_DATE,
    "device_state": DEVICE_STATE,
    "compound_ellipsis": COMPOUND_ELLIPSIS,
}


@pytest.fixture
def leaves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ApplicationFactories:
    return replace(capability_leaves.__wrapped__(monkeypatch, tmp_path), wake_detector=OfflineWake)


@pytest.mark.parametrize("family", tuple(FAMILIES))
def test_acceptance_minimum_by_family(family: str) -> None:
    cases = FAMILIES[family]
    assert len(cases) >= 10, family
    assert len({case.id for case in cases}) == len(cases)


async def say(graph: ApplicationGraph, text: str, *, room: str = "", trigger: str = "wake",
              request_id: str | None = None) -> dict:
    """Send captured audio with a deterministic transcript through the voice pipeline."""
    graph.models.stt.text = text
    event = CommandEvent(np.full(16000, .1, dtype=np.float32), room_id=room,
                         capture_complete=True, capture_trigger=trigger)
    return await graph.pipeline.process_command(event, request_id=request_id)


def put_weather(transport: TaskTransport, condition: str, attributes: dict) -> None:
    transport.states[WEATHER] = {"entity_id": WEATHER, "state": condition, "attributes": attributes}


def weather_config(tmp_path: Path, url: str) -> dict:
    config = offline_config(tmp_path, url)
    config["home_assistant"]["weather_entity"] = WEATHER
    return config


def assert_answer(graph: ApplicationGraph, result: dict, intent: str, path: str, response: str,
                  status: str) -> None:
    assert (result["intent"], result["path"], result["response"]) == (intent, path, response)
    assert result["outcome"]["status"] == status
    assert result["task_completed"] is (status == "completed")
    assert result["response_completed"]
    assert graph.models.tts.texts == [response]


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "condition", "attributes", "http_status", "response"), CURRENT_WEATHER)
async def test_current_weather(tmp_path: Path, leaves: ApplicationFactories, text: str, condition: str | None,
                               attributes: dict, http_status: int, response: str) -> None:
    transport = TaskTransport(states_status=http_status)
    transport.states["weather.other"] = {
        "entity_id": "weather.other", "state": "sunny", "attributes": {"temperature": 99},
    }
    if condition is not None:
        put_weather(transport, condition, attributes)
    async with transport.serve() as url, AsyncExitStack() as stack:
        graph = await build_application(weather_config(tmp_path, url), shutdown=stack, factories=leaves)
        before = transport.state_reads
        result = await say(graph, text)
        assert_answer(graph, result, "weather", "fast_weather", response,
                      "failed" if response == NO_CURRENT else "completed")
        observation = result["source_observation"]
        assert observation["source"] == "ha-rest-states"
        assert "measurement_time" in observation["unobserved_fields"]
        if http_status != 200:
            assert observation["observed_at"] is None
            assert observation["reason"] == ("source_denied" if http_status == 403 else "source_unavailable")
        assert transport.state_reads == before + 1
        assert transport.forecast_reads == transport.sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "entries", "http_status", "response", "status", "reason"), FORECAST)
async def test_forecast(tmp_path: Path, leaves: ApplicationFactories, monkeypatch: pytest.MonkeyPatch,
                        text: str, entries: list[dict] | None, http_status: int, response: str,
                        status: str, reason: str | None) -> None:
    transport = TaskTransport(forecast_status=http_status)
    put_weather(transport, "windy", {"temperature": 99})
    transport.forecast_payload = {"service_response": {WEATHER: {"forecast": entries}}}
    async with transport.serve() as url, AsyncExitStack() as stack:
        graph = await build_application(weather_config(tmp_path, url), shutdown=stack, factories=leaves)
        monkeypatch.setattr(graph.home_capabilities, "_clock", lambda: BEFORE_MIDNIGHT)
        before = transport.state_reads
        result = await say(graph, text)
        assert_answer(graph, result, "weather", "fast_weather", response, status)
        assert transport.sends == [] and transport.state_reads == before
        if entries is None:
            assert "source_observation" not in result and transport.forecast_reads == []
        else:
            observation = result["source_observation"]
            assert observation["source"] == "ha-weather-forecast"
            assert observation["attempted_at"] == BEFORE_MIDNIGHT.isoformat()
            assert transport.forecast_reads == [{"entity_id": WEATHER, "type": "daily"}]
            if reason is not None:
                assert observation["reason"] == reason and observation["observed_at"] is None
            else:
                assert observation["observed_at"] == BEFORE_MIDNIGHT.isoformat()
                assert observation["unobserved_fields"] == (
                    ("issued_at",) if status == "completed" else ("issued_at", "forecast_date_alignment")
                )


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "instant", "response"), TIME_DATE)
async def test_time_date(tmp_path: Path, leaves: ApplicationFactories, monkeypatch: pytest.MonkeyPatch,
                         text: str, instant: datetime | None, response: str) -> None:
    transport = TaskTransport()
    epoch = instant.timestamp() if instant is not None else float("nan")
    async with transport.serve() as url, AsyncExitStack() as stack:
        graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
        monkeypatch.setattr(graph.household_clock, "_time_source", lambda: epoch)
        before = transport.state_reads
        result = await say(graph, text)
        assert_answer(graph, result, "clock", "fast_clock", response,
                      "completed" if instant is not None else "failed")
        if instant is not None:
            assert result["outcome"]["observation_scope"] == "software_state"
        assert transport.state_reads == before and transport.forecast_reads == transport.sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("entity", "state", "role", "status"), DEVICE_STATE)
async def test_device_state(tmp_path: Path, leaves: ApplicationFactories, dashboard_auth: dict[str, str],
                            entity: str, state: str | int, role: str, status: int) -> None:
    transport = TaskTransport()
    payload = {"entity_id": entity, "state": state, "attributes": {}}
    transport.states[entity] = payload
    async with transport.serve() as url, AsyncExitStack() as stack:
        graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
        auth = dashboard_auth if role == "diagnostics" else {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=graph.dashboard.app),
                                     base_url="http://testserver", headers=auth) as client:
            if role == "home":
                authority = graph.dashboard_authority
                grant = await authority.issue_credential(authority.resolve(dashboard_auth["Authorization"]),
                    subject="synthetic-home", role=DashboardRole.HOME, expires_at=time.time() + 120)
                login = await client.post("/api/admin/auth/login", json={"token": grant.token})
                assert login.status_code == 200
            reads_before = len(transport.entity_reads)
            result = await client.get(f"/api/entities/{entity}")
        assert result.status_code == status
        expected_reads = [] if status in (401, 403) else [entity]
        assert transport.entity_reads[reads_before:] == expected_reads
        if status == 200:
            assert result.json() == payload
            assert result.headers["x-kza-source-id"] == "ha-rest-entity"
            assert result.headers["x-kza-observation-state"] == "fresh"
            assert result.headers["cache-control"] == "no-store"
            assert graph.ha_client.get_entity_observation(entity).state == payload
        else:
            assert state not in result.json().values()
            assert "x-kza-observed-at" not in result.headers
            assert graph.ha_client.get_entity_observation(entity).observed_at is None
            if status == 503:
                assert result.headers["x-kza-observation-reason"] == "source_invalid"
        assert transport.sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "parts", "status", "ha_mode", "repeat"), COMPOUND_ELLIPSIS)
async def test_compound_ellipsis(tmp_path: Path, leaves: ApplicationFactories, text: str, parts: tuple[Part, ...],
                                 status: str, ha_mode: str, repeat: bool) -> None:
    transport = TaskTransport(modes={"light.living": ha_mode})
    for part in parts:
        if part.entity is not None:
            transport.states[part.entity]["state"] = "off" if part.service == "turn_on" else "on"
    request_id = "voice:t037-compound"
    async with transport.serve() as url, AsyncExitStack() as stack:
        config = offline_config(tmp_path, url, rooms=True)
        config["rooms"].update({
            "living": {"default_light": "light.living", "ma1260_zone": 1},
            "cocina": {"default_light": "light.second"},
            "escritorio": {"default_light": "light.third"},
        })
        config["home_assistant"]["control_selection_path"] = selection(tmp_path)
        graph = await build_application(config, shutdown=stack, factories=leaves)
        result = await say(graph, text, room="living", request_id=request_id)
        assert (result["intent"], result["path"]) == ("compound", "fast_household")
        assert result["outcome"]["status"] == status
        assert result["task_completed"] is (status == "completed")
        assert result["response_completed"]
        expected_response = " ".join(f"{part.segment}: {part.speech}" for part in parts)
        assert result["response"] == expected_response
        assert graph.models.tts.texts == [expected_response]
        assert [part["status"] for part in result["outcome"]["parts"]] == [part.status for part in parts]
        assert [
            (part["index"], part["request_id"], part["verdict"], part["task_status"], part["dispatch"], part["effect"])
            for part in result["compound_parts"]
        ] == [
            (i, f"{request_id}:part{i}", part.verdict, part.status, part.dispatch, part.effect)
            for i, part in enumerate(parts, start=1)
        ]
        for part, report in zip(parts, result["compound_parts"], strict=True):
            assert report["retried"] is False
            receipt = graph.home_actions.get_receipt(actor=HomeActor.household(), request_id=report["request_id"])
            if part.receipt is None:
                assert receipt is None
            else:
                assert receipt is not None
                assert (receipt.entity_id, receipt.status, receipt.dispatch.value, receipt.effect.value) == (
                    part.entity, part.receipt, part.dispatch, part.effect)
                if part.receipt == "confirmed":
                    assert receipt.observation_scope == "ha_state"
                    assert transport.states[part.entity]["state"] == ("on" if part.service == "turn_on" else "off")
        expected_sends = [(p.entity, p.service, {"entity_id": p.entity}) for p in parts if p.entity is not None]
        assert transport.sends == expected_sends
        if repeat:
            replay = await say(graph, text, room="living", request_id=request_id)
            assert replay["compound_parts"] == [{**part, "retried": True} for part in result["compound_parts"]]
            assert replay["outcome"] == result["outcome"] and replay["response"] == result["response"]
            assert transport.sends == expected_sends


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "trigger", "route", "response", "status"), (
    pytest.param("qué temperatura hay en el living", "wake", ("indoor_temperature", "fast_household"),
                 "Todavía no puedo decirte la temperatura de adentro de la casa, solo la de afuera.",
                 "unsupported", id="C02-indoor-is-not-outdoor"),
    pytest.param("prendé la luz y apagá la luz", "wake", ("compound", "fast_household"),
                 "Decímelo de a una cosa.", "needs_input", id="C02-inadmissible-compound"),
    pytest.param("qué hora es", "follow_up", ("llm_rejected:no_router", None),
                 "", "failed", id="C02-borrowed-wake"),
))
async def test_C02_no_home_effect(tmp_path: Path, leaves: ApplicationFactories, text: str, trigger: str,
                                 route: tuple[str, str | None], response: str, status: str) -> None:
    transport = TaskTransport()
    put_weather(transport, "sunny", {"temperature": 99})
    async with transport.serve() as url, AsyncExitStack() as stack:
        graph = await build_application(weather_config(tmp_path, url), shutdown=stack, factories=leaves)
        before = transport.state_reads
        result = await say(graph, text, trigger=trigger)
        assert (result["intent"], result.get("path")) == route
        assert result["outcome"]["status"] == status and not result["task_completed"]
        assert result["response"] == response
        assert graph.models.tts.texts == ([response] if response else [])
        assert "compound_parts" not in result and "source_observation" not in result
        assert transport.state_reads == before and transport.forecast_reads == transport.sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize("text", (
    pytest.param("¿A qué hora es el partido?", id="C02-event-time-out-of-domain"),
    pytest.param("¿Qué hora es en Madrid?", id="C02-remote-clock-out-of-domain"),
))
async def test_C02_outside_local_clock_uses_the_normal_veto(
        tmp_path: Path, leaves: ApplicationFactories, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    prompts: list[str] = []
    model = "synthetic-router"
    client_factory = httpx.Client

    def provider(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "object": "list",
                "data": [{"id": model, "object": "model", "created": 0, "owned_by": "synthetic"}],
            })
        assert (request.method, request.url.path) == ("POST", "/v1/completions")
        prompts.append(json.loads(request.content)["prompt"])
        verdict = {"is_command": False, "confidence": 0.99, "rejection_reason": "incomplete"}
        return httpx.Response(200, json={
            "id": "cmpl-synthetic", "object": "text_completion", "created": 0, "model": model,
            "choices": [{"text": json.dumps(verdict), "index": 0, "finish_reason": "stop"}],
        })

    class ProviderHTTP:
        """Swap the 7B transport while keeping its SDK, classifier and veto."""

        @staticmethod
        def Client(**kwargs: object) -> httpx.Client:
            return client_factory(transport=httpx.MockTransport(provider), **kwargs)

    monkeypatch.setattr(reasoner, "httpx", ProviderHTTP())
    monkeypatch.setattr(reasoner, "_resolve_api_key", lambda *args: "synthetic-token")
    transport = TaskTransport()
    async with transport.serve() as url, AsyncExitStack() as stack:
        config = offline_config(tmp_path, url)
        config["router"] = {"enabled": True, "base_url": "http://synthetic.invalid/v1",
                            "model": model, "destination": "local"}
        config["nlu"] = {"llm_router": {"enabled": True}}
        graph = await build_application(config, shutdown=stack, factories=leaves)
        before = transport.state_reads
        result = await say(graph, text)
        # Empty or malformed provider replies fall back to "noise".
        assert result["intent"] == "llm_rejected:incomplete"
        assert result["response"] == "" and not result["task_completed"]
        assert len(prompts) == 1 and text.casefold().strip("¿?") in prompts[0].casefold()
        assert graph.models.tts.texts == []
        assert transport.state_reads == before and transport.forecast_reads == transport.sends == []


@pytest.mark.asyncio
async def test_restart_reads_changed_weather_and_device_state(
        tmp_path: Path, leaves: ApplicationFactories, dashboard_auth: dict[str, str]) -> None:
    """Rebuild on the same paths and read the new HA state without writing or replaying actions."""
    transport = TaskTransport()
    async with transport.serve() as url:
        config = weather_config(tmp_path, url)
        for condition, temperature, weather_response, light_state in (
            ("sunny", 22, "Hay 22 grados, soleado.", "off"),
            ("windy", 15, "Hay 15 grados, ventoso.", "on"),
        ):
            put_weather(transport, condition, {"temperature": temperature})
            transport.states["light.living"]["state"] = light_state
            async with AsyncExitStack() as stack:
                graph = await build_application(config, shutdown=stack, factories=leaves)
                assert_answer(graph, await say(graph, "qué tiempo hace"), "weather", "fast_weather",
                              weather_response, "completed")
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=graph.dashboard.app),
                                             base_url="http://testserver", headers=dashboard_auth) as client:
                    result = await client.get("/api/entities/light.living")
                assert result.status_code == 200 and result.json()["state"] == light_state
                assert result.headers["x-kza-observation-state"] == "fresh"
            assert transport.forecast_reads == transport.sends == []
