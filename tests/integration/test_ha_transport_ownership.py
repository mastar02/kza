"""Private T021 proposal: real client lifecycle and controlled transport leaves."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

import src.home_assistant.ha_client as ha_module
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallStatus
from src.monitoring.health_aggregator import HealthAggregator, OverallStatus

pytestmark = pytest.mark.asyncio


WIRE_TIME = 1_700_000_000.0


def _compact_state(state="off", attributes=None):
    return {
        "s": state, "a": deepcopy(attributes or {}), "c": "synthetic-context",
        "lc": WIRE_TIME, "lu": WIRE_TIME,
    }


def _rest_state(entity_id="light.synthetic", state="off", attributes=None):
    return {
        "entity_id": entity_id, "state": state,
        "attributes": deepcopy(attributes or {}),
        "context": {"id": "synthetic-context", "parent_id": None, "user_id": None},
        "last_changed": "2023-11-14T22:13:20+00:00",
        "last_updated": "2023-11-14T22:13:20+00:00",
    }


async def _reached(event: asyncio.Event) -> None:
    """Timeout guards a deadlock; all behavioral assertions use explicit gates."""
    async with asyncio.timeout(5):
        await event.wait()


def _wire_result(frame, options):
    loader = options.get("loads", json.loads)
    assert callable(loader), "JSON loader was not callable"
    raw = frame if isinstance(frame, str) else json.dumps(frame)
    return loader(raw)


class _Socket:
    def __init__(
        self, name: str, *, hold_auth=False, complete_auth_after_cancel=False,
        hold_close=False, confirmation="matching", fail_after_confirmation=False,
        hold_registry_reply=False, initial_event=None, hold_initial=False,
        complete_initial_after_cancel=False, hold_delta=False,
        complete_delta_after_cancel=False, initial_id_offset=0,
    ) -> None:
        self.name = name
        self.closed = False
        self.opened = asyncio.Event()
        self.auth_entered = asyncio.Event()
        self.auth_release = asyncio.Event()
        self.auth_cancelled = asyncio.Event()
        self.close_entered = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.subscribe_sent = asyncio.Event()
        self.confirmation_read = asyncio.Event()
        self.initial_entered = asyncio.Event()
        self.initial_release = asyncio.Event()
        self.initial_finished = asyncio.Event()
        self.stream_waiting = asyncio.Event()
        self.initial_event = deepcopy(
            initial_event if initial_event is not None
            else {"a": {"light.synthetic": _compact_state()}}
        )
        self.initial_frame = None
        self.initial_id_offset = initial_id_offset
        self.raw_loads_called = False
        self.complete_initial_after_cancel = complete_initial_after_cancel
        self.hold_delta = hold_delta
        self.complete_delta_after_cancel = complete_delta_after_cancel
        self.delta_entered = asyncio.Event()
        self.delta_release = asyncio.Event()
        self.registry_reply_entered = asyncio.Event()
        self.registry_reply_release = asyncio.Event()
        self.registry_result_ids: set[int] = set()
        self.frames: asyncio.Queue = asyncio.Queue(maxsize=16)
        self.sent: list[dict] = []
        self.subscriptions: list[int] = []
        self.close_calls = 0
        self.receive_calls = 0
        self.active_receivers = 0
        self.max_receivers = 0
        self.complete_auth_after_cancel = complete_auth_after_cancel
        self.confirmation = confirmation
        self.fail_after_confirmation = fail_after_confirmation
        if not hold_auth:
            self.auth_release.set()
        if not hold_close:
            self.close_release.set()
        if not hold_registry_reply:
            self.registry_reply_release.set()
        if not hold_initial:
            self.initial_release.set()
        if not hold_delta:
            self.delta_release.set()

    async def receive_json(self, **_kwargs):
        self.receive_calls += 1
        receive_number = self.receive_calls
        self.active_receivers += 1
        self.max_receivers = max(self.max_receivers, self.active_receivers)
        try:
            if receive_number == 1:
                return _wire_result({"type": "auth_required"}, _kwargs)
            if receive_number == 2:
                self.auth_entered.set()
                while not self.auth_release.is_set():
                    try:
                        await self.auth_release.wait()
                    except asyncio.CancelledError:
                        self.auth_cancelled.set()
                        if not self.complete_auth_after_cancel:
                            raise
                return _wire_result({"type": "auth_ok"}, _kwargs)
            if self.initial_finished.is_set():
                self.stream_waiting.set()
            frame = await self.frames.get()
            if isinstance(frame, BaseException):
                raise frame
            if isinstance(frame, str):
                self.raw_loads_called = True
                return _wire_result(frame, _kwargs)
            if frame is self.initial_frame:
                self.initial_entered.set()
                while not self.initial_release.is_set():
                    try:
                        await self.initial_release.wait()
                    except asyncio.CancelledError:
                        if not self.complete_initial_after_cancel:
                            raise
                self.initial_finished.set()
            elif frame.get("type") == "event" and self.hold_delta:
                self.delta_entered.set()
                while not self.delta_release.is_set():
                    try:
                        await self.delta_release.wait()
                    except asyncio.CancelledError:
                        if not self.complete_delta_after_cancel:
                            raise
            if (
                frame.get("type") == "result"
                and frame.get("id") in self.registry_result_ids
            ):
                self.registry_reply_entered.set()
                # Complete a late leaf response even after caller cancellation.
                # The original lifecycle epoch must still deny publication.
                while not self.registry_reply_release.is_set():
                    try:
                        await self.registry_reply_release.wait()
                    except asyncio.CancelledError:
                        pass
            if frame.get("type") == "result":
                self.confirmation_read.set()
            return _wire_result(frame, _kwargs)
        finally:
            self.active_receivers -= 1

    async def send_json(self, body, **_kwargs) -> None:
        assert not self.closed, "attempted send through a closed socket"
        self.sent.append(dict(body))
        registries = {
            "config/area_registry/list": [
                {"area_id": "synthetic-area", "name": "Synthetic"},
            ],
            "config/entity_registry/list": [
                {"entity_id": "light.synthetic", "area_id": "synthetic-area"},
            ],
            "config/device_registry/list": [],
        }
        if body.get("type") in registries:
            self.registry_result_ids.add(body["id"])
            self.frames.put_nowait({
                "type": "result", "id": body["id"], "success": True,
                "result": registries[body["type"]],
            })
            return
        if body.get("type") == "subscribe_events":
            raise AssertionError("obsolete noncoherent state subscription")
        if body.get("type") != "subscribe_entities":
            return
        subscription_id = body["id"]
        self.subscriptions.append(subscription_id)
        self.subscribe_sent.set()
        if self.confirmation == "mismatched_id":
            result = {"type": "result", "id": subscription_id + 1, "success": True}
        elif self.confirmation == "rejected":
            result = {"type": "result", "id": subscription_id, "success": False}
        else:
            result = {"type": "result", "id": subscription_id, "success": True}
        self.frames.put_nowait(result)
        if self.fail_after_confirmation:
            self.frames.put_nowait(RuntimeError("synthetic_receive_failure"))
        else:
            self.initial_frame = {
                "type": "event", "id": subscription_id + self.initial_id_offset,
                "event": deepcopy(self.initial_event),
            }
            self.frames.put_nowait(self.initial_frame)

    async def close(self, **_kwargs) -> None:
        self.close_calls += 1
        self.close_entered.set()
        await self.close_release.wait()
        self.closed = True
        self.close_finished.set()

    def state_changed(self, state: str) -> None:
        self.delta({"c": {"light.synthetic": {"+": {
            "s": state, "lc": WIRE_TIME + 1, "lu": WIRE_TIME + 1,
        }}}})

    def delta(self, changes: dict, *, id_offset=0) -> None:
        self.frames.put_nowait({
            "type": "event",
            "id": self.subscriptions[-1] + id_offset,
            "event": deepcopy(changes),
        })


class _Content:
    def __init__(self, response):
        self.response = response
        # Encode fixture data before the measured/gated worker window.
        self.raw = json.dumps(response.payload).encode("utf-8")
        self.offset = 0
        self.read_sizes = []

    async def read(self, n):
        assert type(n) is int and 0 < n <= 65536, "unbounded HTTP body read"
        self.read_sizes.append(n)
        if self.offset == 0:
            self.response.entered.set()
            while not self.response.release.is_set():
                try:
                    await self.response.release.wait()
                except asyncio.CancelledError:
                    if not self.response.complete_after_cancel:
                        raise
        part = self.raw[self.offset:self.offset + n]
        self.offset += len(part)
        if self.offset >= len(self.raw):
            self.response.finished.set()
        return part


class _Response:
    def __init__(self, payload=None, *, status=200, hold=False,
                 complete_after_cancel=False):
        self.payload = deepcopy(payload)
        self.content = _Content(self)
        self.status = status
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.complete_after_cancel = complete_after_cancel
        if not hold:
            self.release.set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise ha_module.aiohttp.ClientResponseError(
                request_info=SimpleNamespace(real_url="http://synthetic.invalid"),
                history=(), status=self.status, message="synthetic_source_failure",
            )

    async def json(self, **_kwargs):
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                if not self.complete_after_cancel:
                    raise
        self.finished.set()
        return _wire_result(self.payload, _kwargs)


class _Session:
    def __init__(self, transport: _Transport) -> None:
        self.transport = transport
        self.closed = False
        self.close_calls = 0
        self.ws_options = []

    async def ws_connect(self, _url, **_kwargs):
        assert not self.closed, "closed session admitted another WS connection"
        socket = self.transport.next_socket()
        self.ws_options.append(dict(_kwargs))
        socket.connect_options = dict(_kwargs)
        self.transport.opened.append(socket)
        socket.opened.set()
        return socket

    def get(self, url, **_kwargs):
        assert not self.closed, "closed session admitted another REST read"
        assert "/api/states" in url, "unexpected recovery HTTP endpoint"
        self.transport.state_reads += 1
        self.transport.state_urls.append(url)
        if self.transport.responses:
            return self.transport.responses.popleft()
        if url.endswith("/api/states"):
            return _Response([_rest_state()])
        return _Response(_rest_state(url.rsplit("/", 1)[-1]))

    def post(self, *_args, **_kwargs):
        self.transport.service_posts += 1
        raise AssertionError("recovery dispatched an HTTP service command")

    async def close(self) -> None:
        self.close_calls += 1
        # No implicit socket close: the client must retire its own candidate.
        # Auto-closing here would conceal a stale publication in the client.
        self.closed = True


class _Transport:
    def __init__(self) -> None:
        self.planned: deque[_Socket] = deque()
        self.all_sockets: list[_Socket] = []
        self.opened: list[_Socket] = []
        self.sessions: list[_Session] = []
        self.observers: list[asyncio.Task] = []
        self.cleanup_gates: list[asyncio.Event] = []
        self.state_reads = 0
        self.state_urls = []
        self.responses: deque[_Response] = deque()
        self.service_posts = 0

    def socket(self, name: str, **kwargs) -> _Socket:
        socket = _Socket(name, **kwargs)
        self.planned.append(socket)
        self.all_sockets.append(socket)
        return socket

    def response(self, payload, **kwargs) -> _Response:
        response = _Response(payload, **kwargs)
        self.responses.append(response)
        self.cleanup_gates.append(response.release)
        return response

    def next_socket(self) -> _Socket:
        if self.planned:
            return self.planned.popleft()
        socket = _Socket(f"unexpected-{len(self.all_sockets)}")
        self.all_sockets.append(socket)
        return socket

    def session_factory(self, **_kwargs) -> _Session:
        session = _Session(self)
        self.sessions.append(session)
        return session

    def spawn(self, awaitable) -> asyncio.Task:
        task = asyncio.create_task(awaitable)
        self.observers.append(task)
        return task

    def release_all(self) -> None:
        for gate in self.cleanup_gates:
            gate.set()
        for socket in self.all_sockets:
            socket.auth_release.set()
            socket.initial_release.set()
            socket.delta_release.set()
            socket.registry_reply_release.set()
            socket.close_release.set()

    def assert_no_effects_or_duplicate_receivers(self) -> None:
        assert self.service_posts == 0
        for socket in self.all_sockets:
            assert socket.max_receivers <= 1, socket.name
            assert len(socket.subscriptions) <= 1, socket.name
            assert all(
                frame.get("type") != "call_service"
                for frame in socket.sent
            ), socket.name


@pytest.fixture
def transport(monkeypatch) -> _Transport:
    controlled = _Transport()
    monkeypatch.setattr(ha_module.aiohttp, "ClientSession", controlled.session_factory)
    return controlled


@asynccontextmanager
async def _client(transport: _Transport):
    client = HomeAssistantClient("http://synthetic.invalid", "synthetic-token")
    try:
        yield client
    finally:
        transport.release_all()
        for observer in transport.observers:
            if not observer.done():
                observer.cancel()
        async with asyncio.timeout(5):
            await asyncio.gather(*transport.observers, return_exceptions=True)
            await client.close()


async def _entered_call(entered: asyncio.Event, call):
    entered.set()
    # Setting an Event does not yield: the observer reaches the client's first
    # suspension before the test resumes from _reached(entered).
    return await call()


async def test_terminal_close_prevents_new_service_session_or_dispatch(transport):
    async with _client(transport) as client:
        await client.close()
        sessions_before = len(transport.sessions)

        outcome = await client.call_service_once(
            "light", "turn_on", "light.synthetic", {},
        )

        assert outcome.status is ServiceCallStatus.NOT_SENT
        assert outcome.reason == "ha_session_unavailable"
        assert len(transport.sessions) == sessions_before
        assert transport.service_posts == 0
        assert transport.opened == []


async def test_calls_auth_and_health_reads_do_not_renew_rest_or_entity_observations(
    transport, monkeypatch,
):
    clock = {"now": 100.}
    monkeypatch.setattr(ha_module.time, "time", lambda: clock["now"])
    transport.socket("calls")
    async with _client(transport) as client:
        snapshot = await client.get_state_snapshot()
        rest = client.get_health_status().rest
        assert snapshot.available is True
        assert rest.observed_at == 100. and rest.expires_at == 160.

        clock["now"] = 120.
        assert await client.connect_websocket() is True
        health = client.get_health_status()
        assert health.calls.observed_at == 120.
        assert health.rest.observed_at == rest.observed_at
        assert health.rest.expires_at == rest.expires_at
        assert client._home_states is snapshot

        monitor = HealthAggregator(ha_client=client, health_clock=lambda: clock["now"])
        assert monitor._check_ha().status is OverallStatus.HEALTHY
        clock["now"] = 160.
        result = monitor._check_ha()
        assert result.status is OverallStatus.DEGRADED
        assert result.extra["channels"]["rest"]["phase"] == "expired"
        assert result.extra["channels"]["calls"]["current"] is True
        assert client._home_states is snapshot
        assert client.get_health_status().rest.observed_at == 100.
        assert client.get_health_status().rest.expires_at == 160.


async def test_concurrent_start_owns_warmup_before_its_first_await(transport):
    events = transport.socket("events", hold_auth=True)
    calls = transport.socket("calls")
    async with _client(transport) as client:
        first = transport.spawn(client.start_state_sync())
        await _reached(events.auth_entered)

        owner = client._state_subscribe_task
        assert owner is not None and not owner.done()
        assert client._ws_events is None

        second_entered = asyncio.Event()
        second = transport.spawn(
            _entered_call(second_entered, client.start_state_sync)
        )
        await _reached(second_entered)
        assert client._state_subscribe_task is owner
        assert not first.done() and not second.done()
        assert transport.opened == [events]

        events.auth_release.set()
        async with asyncio.timeout(5):
            assert await first is None
            assert await second is None
        await _reached(events.confirmation_read)

        assert client._state_subscribe_task is owner and not owner.done()
        assert client._ws_events is events
        assert client._ws_calls is calls
        assert transport.opened == [events, calls]
        assert len(events.subscriptions) == 1
        transport.assert_no_effects_or_duplicate_receivers()


async def test_cancelled_start_observer_keeps_the_admitted_warmup_owner(transport):
    events = transport.socket("events", hold_auth=True)
    calls = transport.socket("calls")
    async with _client(transport) as client:
        first = transport.spawn(client.start_state_sync())
        await _reached(events.auth_entered)
        owner = client._state_subscribe_task
        assert owner is not None and not owner.done()
        first.cancel()

        second_entered = asyncio.Event()
        second = transport.spawn(
            _entered_call(second_entered, client.start_state_sync)
        )
        await _reached(second_entered)
        assert client._state_subscribe_task is owner and not owner.done()
        assert not events.auth_cancelled.is_set()
        assert transport.opened == [events]

        events.auth_release.set()
        async with asyncio.timeout(5):
            first_result, second_result = await asyncio.gather(
                first, second, return_exceptions=True,
            )
        assert isinstance(first_result, asyncio.CancelledError)
        assert second_result is None
        await _reached(events.confirmation_read)
        assert client._state_subscribe_task is owner and not owner.done()
        assert client._ws_events is events and client._ws_calls is calls
        assert transport.opened == [events, calls]
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("shutdown", ["stop_state_sync", "close"])
async def test_stop_or_close_drains_late_authenticated_startup_candidate(
    transport, shutdown,
):
    candidate = transport.socket(
        "held-startup", hold_auth=True, complete_auth_after_cancel=True,
        hold_close=True,
    )
    async with _client(transport) as client:
        starting = transport.spawn(client.start_state_sync())
        await _reached(candidate.auth_entered)
        owner = client._state_subscribe_task
        assert owner is not None and not owner.done()

        shutdown_entered = asyncio.Event()
        stopping = transport.spawn(
            _entered_call(shutdown_entered, getattr(client, shutdown))
        )
        await _reached(shutdown_entered)
        assert not stopping.done()

        # The transport is allowed to complete authentication after a cancel
        # request. That success belongs to the sealed generation, not a restart.
        candidate.auth_release.set()
        await _reached(candidate.close_entered)
        assert not candidate.closed
        assert not stopping.done()
        assert client._ws_events is None
        assert client._ws_calls is None
        assert not candidate.subscriptions

        candidate.close_release.set()
        async with asyncio.timeout(5):
            assert await starting is None
            assert await stopping is None
        assert candidate.closed
        assert owner.done()
        assert client._state_subscribe_task is None
        assert transport.opened == [candidate]

        if shutdown == "stop_state_sync":
            fresh_events = transport.socket("restarted-events")
            fresh_calls = transport.socket("restarted-calls")
            await client.start_state_sync()
            await _reached(fresh_events.confirmation_read)
            assert client._state_subscribe_task is not owner
            assert client._ws_events is fresh_events
            assert client._ws_calls is fresh_calls
            assert transport.opened == [candidate, fresh_events, fresh_calls]
        else:
            sessions_before = len(transport.sessions)
            assert await client.connect_websocket() is False
            assert len(transport.sessions) == sessions_before
            assert all(session.closed for session in transport.sessions)
            assert client._ws_events is None and client._ws_calls is None
        transport.assert_no_effects_or_duplicate_receivers()


async def test_full_close_drains_direct_calls_handshake_before_it_can_publish(
    transport,
):
    candidate = transport.socket(
        "held-calls", hold_auth=True, complete_auth_after_cancel=True,
        hold_close=True,
    )
    async with _client(transport) as client:
        connecting = transport.spawn(client.connect_websocket())
        await _reached(candidate.auth_entered)
        shutdown_entered = asyncio.Event()
        closing = transport.spawn(_entered_call(shutdown_entered, client.close))
        await _reached(shutdown_entered)
        assert not closing.done()

        candidate.auth_release.set()
        await _reached(candidate.close_entered)
        assert client._ws_calls is None
        assert not closing.done() and not candidate.closed
        candidate.close_release.set()

        async with asyncio.timeout(5):
            assert await connecting is False
            assert await closing is None
        assert candidate.closed
        assert client._ws_calls is None and client._ws_events is None
        assert all(session.closed for session in transport.sessions)
        assert await client.connect_websocket() is False
        assert transport.opened == [candidate]
        transport.assert_no_effects_or_duplicate_receivers()


async def test_cancelled_close_observer_retains_one_actual_socket_close(transport):
    calls = transport.socket("calls", hold_close=True)
    async with _client(transport) as client:
        assert await client.connect_websocket() is True
        first = transport.spawn(client.close())
        await _reached(calls.close_entered)
        owner = client._close_task
        assert owner is not None and not owner.done()
        first.cancel()

        second_entered = asyncio.Event()
        second = transport.spawn(_entered_call(second_entered, client.close))
        await _reached(second_entered)
        assert client._close_task is owner
        assert not owner.done() and not second.done()
        assert not calls.closed
        assert calls.close_calls == 1

        calls.close_release.set()
        async with asyncio.timeout(5):
            first_result, second_result = await asyncio.gather(
                first, second, return_exceptions=True,
            )
        assert isinstance(first_result, asyncio.CancelledError)
        assert second_result is None
        assert owner.done()
        assert calls.closed and calls.close_calls == 1
        assert all(session.closed for session in transport.sessions)
        assert client._ws_calls is None and client._ws_events is None
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("failure", ["mismatched_id", "rejected", "receive"])
async def test_failed_open_events_socket_retires_before_fresh_subscribed_successor(
    transport, failure,
):
    broken = transport.socket(
        "events-A", hold_close=True,
        confirmation=failure if failure != "receive" else "matching",
        fail_after_confirmation=failure == "receive",
    )
    calls = transport.socket("calls")
    successor = transport.socket("events-B")
    retry_entered = asyncio.Event()
    retry_release = asyncio.Event()
    transport.cleanup_gates.append(retry_release)
    delays = []

    async def retry_wait(delay):
        delays.append(delay)
        retry_entered.set()
        await retry_release.wait()

    async with _client(transport) as client:
        # A scheduling seam only: retain the real warmup, subscribe, receive,
        # failure cleanup and successor logic.
        client._state_retry_wait = retry_wait
        await client.start_state_sync()
        await _reached(broken.close_entered)
        assert not broken.closed
        assert not successor.opened.is_set()
        assert transport.opened == [broken, calls]
        assert len(broken.subscriptions) == 1

        broken.close_release.set()
        await _reached(broken.close_finished)
        await _reached(retry_entered)
        assert broken.closed
        assert not successor.opened.is_set()
        assert len(delays) == 1
        # Acquisition and actual retirement consume the original attempt spacing.
        assert 0. <= delays[0] < ha_module.STATE_SYNC_BACKOFF_INITIAL_S

        changed = asyncio.Event()

        def observed(entity_id, state):
            if entity_id == "light.synthetic" and state["state"] == "on":
                changed.set()

        client.register_state_callback(observed)
        retry_release.set()
        await _reached(successor.confirmation_read)
        successor.state_changed("on")
        await _reached(changed)

        assert client._ws_events is successor
        assert client._ws_calls is calls
        assert client.get_entity_state_cached("light.synthetic")["state"] == "on"
        assert transport.opened == [broken, calls, successor]
        assert broken.close_calls == 1
        assert len(successor.subscriptions) == 1
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("phase", ["auth", "registry_reply"])
async def test_close_drains_temporary_registry_lookup_without_late_publication(
    transport, phase,
):
    candidate = transport.socket(
        "temporary-registry", hold_auth=phase == "auth",
        complete_auth_after_cancel=True, hold_close=True,
        hold_registry_reply=phase == "registry_reply",
    )
    async with _client(transport) as client:
        reading = transport.spawn(client.get_area_assignments())
        held_phase = (
            candidate.auth_entered
            if phase == "auth" else candidate.registry_reply_entered
        )
        await _reached(held_phase)
        assert not reading.done()
        assert client._ws_calls is None and client._ws_events is None

        close_entered = asyncio.Event()
        closing = transport.spawn(_entered_call(close_entered, client.close))
        await _reached(close_entered)
        assert not closing.done()

        # A late leaf response belongs to the sealed generation. Close must
        # retain that request until its actual temporary socket cleanup exits.
        candidate.auth_release.set()
        candidate.registry_reply_release.set()
        await _reached(candidate.close_entered)
        assert not candidate.closed and not closing.done()
        candidate.close_release.set()

        async with asyncio.timeout(5):
            result, close_result = await asyncio.gather(
                reading, closing, return_exceptions=True,
            )
        assert close_result is None
        # Ordinary retirement maps to ClientError -> unavailable DTO, never an
        # implicit caller cancellation. Report a different contract explicitly.
        assert isinstance(result, ha_module.HAAreaAssignments), type(result).__name__
        assert result.available is False
        assert result.observed_at is None and result.expires_at is None
        assert result.area_names == {} and result.entity_area_ids == {}
        published = client._area_assignments
        assert published is None or (
            published.available is False
            and published.observed_at is None
            and published.expires_at is None
            and published.area_names == {}
            and published.entity_area_ids == {}
        )
        assert candidate.closed and candidate.close_calls == 1
        assert transport.opened == [candidate]
        assert all(session.closed for session in transport.sessions)
        assert client._ws_calls is None and client._ws_events is None

        sessions_before = len(transport.sessions)
        again = await client.get_area_assignments()
        assert again.available is False
        assert len(transport.sessions) == sessions_before
        assert transport.opened == [candidate]
        transport.assert_no_effects_or_duplicate_receivers()


async def test_restart_queued_behind_stop_is_joined_then_sealed_by_a_second_stop(
    transport,
):
    events = transport.socket("events-A", hold_close=True)
    calls = transport.socket("calls-A")
    forbidden = transport.socket("forbidden-generation-B")
    async with _client(transport) as client:
        await client.start_state_sync()
        await _reached(events.confirmation_read)
        owner_a = client._state_subscribe_task
        assert owner_a is not None and not owner_a.done()

        stop_a_entered = asyncio.Event()
        stop_a = transport.spawn(
            _entered_call(stop_a_entered, client.stop_state_sync)
        )
        await _reached(stop_a_entered)
        await _reached(events.close_entered)
        assert not events.closed and not stop_a.done()

        start_b_entered = asyncio.Event()
        start_b = transport.spawn(
            _entered_call(start_b_entered, client.start_state_sync)
        )
        await _reached(start_b_entered)
        owner_b = client._state_subscribe_task
        ready_b = client._state_start_ready
        assert owner_b is not None and owner_b is not owner_a
        assert not owner_b.done() and not start_b.done()
        assert not forbidden.opened.is_set()

        start_b2_entered = asyncio.Event()
        start_b2 = transport.spawn(
            _entered_call(start_b2_entered, client.start_state_sync)
        )
        await _reached(start_b2_entered)
        assert client._state_subscribe_task is owner_b
        assert client._state_start_ready is ready_b
        assert not start_b2.done()

        stop_b_entered = asyncio.Event()
        stop_b = transport.spawn(
            _entered_call(stop_b_entered, client.stop_state_sync)
        )
        await _reached(stop_b_entered)
        assert client._state_sync_running is False
        assert not stop_b.done()
        assert not forbidden.opened.is_set()
        assert transport.opened == [events, calls]

        events.close_release.set()
        async with asyncio.timeout(5):
            results = await asyncio.gather(
                stop_a, start_b, start_b2, stop_b, return_exceptions=True,
            )
        assert results == [None, None, None, None]
        assert owner_a.done() and owner_b.done()
        assert client._state_subscribe_task is None
        assert client._state_sync_running is False
        assert client._ws_events is None
        assert client._ws_calls is calls and not calls.closed
        assert events.closed and events.close_calls == 1
        assert not forbidden.opened.is_set()
        assert transport.opened == [events, calls]
        assert not any(
            purpose == "events" and not task.done()
            for task, purpose in client._transport_tasks.items()
        )
        transport.assert_no_effects_or_duplicate_receivers()
