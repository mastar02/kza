"""Exercise the real HA recovery controller and timers against transport leaves."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import aiohttp
import pytest

import src.home_assistant.ha_client as ha_module
from src.home_assistant.ha_client import HomeAssistantClient
from src.home_assistant.state_projection import MAX_ENTITIES
from tests.integration.test_ha_transport_ownership import _compact_state

pytestmark = pytest.mark.asyncio
ENTITY = "light.synthetic"


class _Clock:
    """Skip timer waits without erasing real time spent in native CPU work."""

    def __init__(self, loop):
        self.loop, self.real = loop, loop.time
        self.origin, self.wall_origin = self.real(), time.time()
        self.offset = 0.

    def now(self):
        return self.real() + self.offset

    def epoch(self):
        return self.wall_origin + self.now() - self.origin

    def relative(self):
        return self.now() - self.origin

    async def drive(self, predicate, client, *, horizon=240.):
        wall_deadline, deadline = time.monotonic() + 12., self.now() + horizon
        while not predicate():
            assert time.monotonic() < wall_deadline, "virtual recovery driver stalled"
            assert self.now() <= deadline, "virtual recovery horizon exhausted"
            await asyncio.sleep(0)
            if predicate():
                return
            # Drain ready callbacks and actual native work before skipping time.
            # Otherwise a queued thread could falsely consume its whole deadline.
            if any(not handle.cancelled() for handle in self.loop._ready):
                continue
            if client._state_projection_work.status()["operations"]:
                continue
            pending = [handle.when() for handle in self.loop._scheduled if not handle.cancelled()]
            assert pending, "no timer or native work can advance recovery"
            self.offset += max(0., min(pending) - self.now())


class _Socket:
    def __init__(self, model, purpose, index):
        self.model, self.purpose, self.index = model, purpose, index
        self.created_at = model.clock.relative()
        # Connections admitted before restoration remain the failed old candidates.
        self.healthy_on_open = model.healthy if purpose == "events" else model.calls_available
        self.slow = purpose == "events" and index >= 4 and model.scenario != "max_backoff"
        self.slow_calls = purpose == "calls" and model.scenario == "calls_restoration"
        self.auth_delay = 4.5 if self.slow_calls else 3.5 if self.slow else .1
        self.closed, self.stage, self.subscribe_id = False, 0, None
        self.receivers = self.max_receivers = 0
        self.sent, self.stream = [], asyncio.Event()
        self.close_started = self.close_finished = None
        self.close_cancelled = False

    async def send_json(self, value):
        self.sent.append(dict(value))
        if value.get("type") == "auth":
            assert value["access_token"] == self.model.token
        elif value.get("type") == "subscribe_entities":
            assert self.purpose == "events"
            self.subscribe_id = value["id"]
        else:
            raise AssertionError("recovery dispatched an unexpected command")

    async def receive_json(self, *, loads=json.loads, **_kwargs):
        self.receivers += 1
        self.max_receivers = max(self.max_receivers, self.receivers)
        try:
            stage, self.stage = self.stage, self.stage + 1
            if stage == 0:
                await self.model.pause(self.auth_delay)
                value = {"type": "auth_required"}
            elif stage == 1:
                await self.model.pause(self.auth_delay)
                value = {"type": "auth_ok" if self.purpose == "events" or self.healthy_on_open else "auth_invalid"}
            elif stage == 2:
                assert self.purpose == "events" and self.subscribe_id is not None
                await self.model.pause(1.5 if self.slow else .1)
                value = {"id": self.subscribe_id, "type": "result", "success": True}
            elif stage == 3:
                assert self.purpose == "events"
                if not self.healthy_on_open:
                    await self.model.pause(1.5 if self.slow else .1)
                    raise aiohttp.ClientConnectionError("synthetic old candidate failed")
                await self.model.pause(1.5 if self.slow else 1.)
                self.model.initial_received_at = self.model.clock.relative()
                value = {"id": self.subscribe_id, "type": "event", "event": {
                    "a": {ENTITY: _compact_state("on"), **{
                        f"sensor.synthetic_{index}": _compact_state("1", {"padding": "x" * 600})
                        for index in range(1, self.model.entity_count)
                    }},
                }}
            else:
                await self.stream.wait()
                raise aiohttp.ClientConnectionError("synthetic stream retired")
            return loads(json.dumps(value))
        finally:
            self.receivers -= 1

    async def close(self, **_kwargs):
        if self.closed:
            return False
        # aiohttp exposes this flag before writer/peer close has finished.
        self.closed = True
        self.close_started = self.model.clock.relative()
        delay = (20. if (self.purpose == "events" and self.index == 4
                 and self.model.scenario != "max_backoff") or (self.slow_calls and self.index == 1) else 0.)
        if self.model.scenario == "prior_retirement" and self.index == 4:
            self.model.schedule_healthy(.01)
        try:
            if self.model.hold_cleanup and self.purpose == "events" and self.index == 4:
                while not self.model.retirement_release.is_set():
                    try:
                        await self.model.retirement_release.wait()
                    except asyncio.CancelledError:
                        self.close_cancelled = True
            else:
                await self.model.pause(0. if self.model.stopping else delay)
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise
        finally:
            self.close_finished = self.model.clock.relative()
        return True


class _Session:
    def __init__(self, model):
        self.model, self.closed = model, False

    async def ws_connect(self, _url, **options):
        purpose = "events" if "max_msg_size" in options else "calls"
        if purpose == "events":
            self.model.event_count += 1
            index = self.model.event_count
        else:
            self.model.calls_count += 1
            index = self.model.calls_count
        socket = _Socket(self.model, purpose, index)
        self.model.sockets.append(socket)
        if self.model.scenario == "prior_attempt" and purpose == "events" and index == 4:
            self.model.schedule_healthy(.05)
        if socket.slow_calls and index == 1:
            self.model.schedule_healthy(.05, calls=True)
        await self.model.pause(8. if socket.slow_calls else 3.5 if socket.slow else .1)
        return socket

    def get(self, url, **_kwargs):
        assert self.model.scenario == "calls_restoration"
        assert url == f"http://synthetic.invalid/api/states/{ENTITY}"
        self.model.rest_reads.append({"url": url, "at": self.model.clock.relative()})
        return _Response(self.model)

    def post(self, *_args, **_kwargs):
        raise AssertionError("state recovery attempted a service effect")

    async def close(self):
        self.closed = True


class _Response:
    status = 200

    def __init__(self, model):
        self.model = model
        self.content = self
        self.payload = json.dumps({"entity_id": ENTITY, "state": "on", "attributes": {}}).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def raise_for_status(self):
        return None

    async def read(self, size):
        if not self.payload:
            return b""
        await self.model.pause(.5)
        part, self.payload = self.payload[:size], self.payload[size:]
        return part


class _Model:
    def __init__(self, clock, scenario, entity_count):
        self.clock, self.scenario, self.entity_count = clock, scenario, entity_count
        self.token, self.event_count, self.calls_count = "synthetic-token", 0, 0
        self.calls_available = scenario != "calls_restoration"
        self.sockets, self.sessions, self.retries, self.rest_reads = [], [], [], []
        self.healthy_at = self.initial_received_at = self.healthy_handle = None
        self.stopping, self.healthy, self.beats = False, scenario == "calls_restoration", 0
        self.hold_cleanup = False
        self.retirement_release = asyncio.Event()

    async def pause(self, duration):
        await asyncio.sleep(duration)

    def session_factory(self, **_kwargs):
        session = _Session(self)
        self.sessions.append(session)
        return session

    def schedule_healthy(self, delay, *, calls=False):
        if self.healthy_at is None and self.healthy_handle is None:
            def healthy():
                if calls:
                    self.calls_available = True
                else:
                    self.healthy = True
                self.healthy_at = self.clock.relative()
            self.healthy_handle = self.clock.loop.call_later(delay, healthy)

    async def heartbeat(self):
        while not self.stopping:
            self.beats += 1
            await asyncio.sleep(.005)


async def _exercise(monkeypatch, scenario, entity_count, *, held_cleanup=False, fail_after_baseline=False):
    loop = asyncio.get_running_loop()
    clock = _Clock(loop)
    model = _Model(clock, scenario, entity_count)
    model.hold_cleanup = held_cleanup
    client = HomeAssistantClient("http://synthetic.invalid", model.token)
    start = beat = None
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", clock.now)
        patch.setattr(ha_module, "monotonic", clock.now)
        patch.setattr(ha_module, "time", SimpleNamespace(time=clock.epoch, perf_counter=clock.now))
        patch.setattr(ha_module.aiohttp, "ClientSession", model.session_factory)
        original_wait = client._state_retry_wait

        async def observed_wait(delay):
            model.retries.append({"at": clock.relative(), "delay": delay,
                                  "attempt": model.event_count})
            if scenario == "max_backoff" and model.event_count == 4:
                model.schedule_healthy(.01)
            await original_wait(delay)

        patch.setattr(client, "_state_retry_wait", observed_wait)
        try:
            beat = asyncio.create_task(model.heartbeat())
            start = asyncio.create_task(client.start_state_sync())

            def recovered():
                value = client.get_entity_observation(ENTITY)
                return (model.event_count >= 5 and value.current(clock.epoch())
                        and client.get_health_status().events.phase == "subscribed")

            if scenario == "calls_restoration":
                await clock.drive(lambda: client._state_frame.initialized
                    and client._ws_calls is not None and client._ws_calls.index >= 2
                    and client._ws_connected, client)
                point = asyncio.create_task(client.read_entity_observation(ENTITY))
                await clock.drive(point.done, client)
                value = await point
                assert value.current(clock.epoch()) and value.source == "ha-rest-entity"
                health = client.get_health_status()
                assert health.calls.required and health.calls.state is ha_module.HAConnectionState.CONNECTED
                assert health.events.phase == "subscribed"
                assert health.events.state is ha_module.HAConnectionState.CONNECTED
                assert health.rest.state is ha_module.HAConnectionState.CONNECTED
                assert len(model.rest_reads) == 1
                assert model.calls_available and model.healthy_at is not None
                result = {"scenario": scenario, "healthy_at": model.healthy_at,
                    "elapsed_s": clock.relative() - model.healthy_at,
                    "calls_attempts": model.calls_count, "events_attempts": model.event_count,
                    "fresh_point_source": value.source, "fresh_point_observed_at": value.observed_at,
                    "rest_reads": model.rest_reads,
                    "scope": "required calls auth/retirement, events baseline and real REST point; synthetic leaves"}
                print("HA_RECOVERY_BUDGET " + json.dumps(result, sort_keys=True))
                return result

            if held_cleanup:
                await clock.drive(lambda: model.healthy_at is not None
                    and clock.relative() >= model.healthy_at + 60., client)
                previous = next(socket for socket in model.sockets
                    if socket.purpose == "events" and socket.index == 4)
                owner = client._socket_close_tasks[id(previous)]
                assert previous.closed and previous.close_finished is None
                assert previous.close_cancelled and not owner.done()
                assert id(previous) in client._owned_sockets
                assert model.event_count == 4 and not recovered()
                assert not client.get_entity_observation(ENTITY).current(clock.epoch())
                assert client.get_health_status().events.phase != "subscribed"
                result = {"scenario": "cancel_resistant_cleanup", "target_met": False,
                    "healthy_at": model.healthy_at,
                    "elapsed_s": clock.relative() - model.healthy_at,
                    "owner_pending": True, "new_candidate_admitted": False,
                    "scope": "explicit modeled C20 miss; retained resource and unknown data"}
                print("HA_RECOVERY_BUDGET " + json.dumps(result, sort_keys=True))
                return result

            await clock.drive(recovered, client)
            assert start.done() and start.exception() is None
            value = client.get_entity_observation(ENTITY)
            assert value.source == "ha-ws-entities" and value.state["state"] == "on"
            assert len(client._state_frame.entries) == entity_count
            assert model.healthy and model.healthy_at is not None
            assert model.initial_received_at is not None
            assert len(model.retries) == 4
            assert [row["attempt"] for row in model.retries] == [1, 2, 3, 4]
            events = {socket.index: socket for socket in model.sockets if socket.purpose == "events"}
            for index, spacing in [(2, 10.), (3, 20.), (4, 30.)]:
                # Connection admission follows the absolute cadence; the small
                # tolerance is for real callback dispatch, not the 60-second target.
                assert events[index + 1].created_at - events[index].created_at == pytest.approx(spacing, abs=.1)
            for index in range(1, 5):
                assert events[index].close_finished is not None
                assert events[index].close_finished <= events[index + 1].created_at
            assert all(socket.max_receivers <= 1 for socket in model.sockets)
            assert model.beats > 1
            recovered_at = clock.relative()
            result = {
                "scenario": scenario, "entities": entity_count, "healthy_at": model.healthy_at,
                "recovered_at": recovered_at, "elapsed_s": recovered_at - model.healthy_at,
                "initial_received_at": model.initial_received_at,
                "retries": model.retries, "virtual_heartbeat_progress": model.beats,
                "candidate_admissions": {index: socket.created_at for index, socket in events.items()},
                "entity_observed_at": value.observed_at, "entity_expires_at": value.expires_at,
                "retirements": [{"index": socket.index, "start": socket.close_started,
                    "end": socket.close_finished, "cancelled": socket.close_cancelled}
                    for socket in model.sockets if socket.purpose == "events"],
                "scope": "real controller/timers; timed synthetic leaves; not field or C03 acceptance",
            }
            if fail_after_baseline:
                events[5].stream.set()
                await clock.drive(lambda: model.event_count >= 6 and recovered(), client)
                successor = next(socket for socket in model.sockets
                    if socket.purpose == "events" and socket.index == 6)
                assert successor.created_at - events[5].created_at == pytest.approx(5., abs=.1)
                assert len(model.retries) == 5 and model.retries[-1]["attempt"] == 5
                assert client.get_entity_observation(ENTITY).current(clock.epoch())
                result["after_baseline_failure"] = {
                    "next_candidate_at": successor.created_at,
                    "spacing_s": successor.created_at - events[5].created_at,
                    "last_wait": model.retries[-1]["delay"],
                }
            print("HA_RECOVERY_BUDGET " + json.dumps(result, sort_keys=True))
            return result
        finally:
            model.stopping = True
            close = asyncio.create_task(client.close())
            await asyncio.sleep(0)  # Seal admission before releasing a retained fault.
            model.retirement_release.set()
            if model.healthy_handle is not None:
                model.healthy_handle.cancel()
            if beat is not None:
                beat.cancel()
                await asyncio.gather(beat, return_exceptions=True)
            for socket in model.sockets:
                socket.stream.set()
            await clock.drive(close.done, client, horizon=60.)
            await close
            if start is not None:
                if not start.done():
                    start.cancel()
                await asyncio.gather(start, return_exceptions=True)
            assert all(session.closed for session in model.sessions)
            assert client._state_projection_work.status()["operations"] == 0


@pytest.mark.parametrize("entity_count", [1, MAX_ENTITIES])
async def test_healthy_after_maximum_retry_sleep_recovers_with_real_asyncio_timer(monkeypatch, entity_count):
    result = await _exercise(monkeypatch, "max_backoff", entity_count)
    assert result["elapsed_s"] <= 60.


@pytest.mark.parametrize("entity_count", [1, MAX_ENTITIES])
@pytest.mark.parametrize("scenario", ["prior_attempt", "prior_retirement"])
async def test_healthy_before_actual_retirement_uses_complete_original_budget(monkeypatch, scenario, entity_count):
    result = await _exercise(monkeypatch, scenario, entity_count)
    assert result["elapsed_s"] <= 60.


async def test_cancel_resistant_retirement_records_a_target_miss_without_false_recovery(monkeypatch):
    result = await _exercise(monkeypatch, "prior_retirement", 1, held_cleanup=True)
    assert result["elapsed_s"] >= 60.
    assert result["target_met"] is False


async def test_required_calls_restore_with_events_and_fresh_read_within_whole_budget(monkeypatch):
    result = await _exercise(monkeypatch, "calls_restoration", 1)
    assert result["elapsed_s"] <= 60.


async def test_coherent_baseline_resets_backoff_before_its_receiver_eventually_fails(monkeypatch):
    result = await _exercise(monkeypatch, "max_backoff", 1, fail_after_baseline=True)
    assert result["after_baseline_failure"]["spacing_s"] == pytest.approx(5., abs=.1)
