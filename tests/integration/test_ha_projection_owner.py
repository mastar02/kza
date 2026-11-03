"""Real HA CPU ownership with retained native gates and async progress probes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import threading

import pytest

from src.core.owned_work import OwnedBlockingWork
import src.home_assistant.ha_client as ha_module
from src.home_assistant.state_projection import MAX_ENTITIES
from tests.integration.test_ha_entity_stream import _baseline as _io_baseline
from tests.integration.test_ha_transport_ownership import (
    WIRE_TIME,
    _Transport,
    _client,
    _compact_state,
    _entered_call,
    _reached,
    _rest_state,
)

pytestmark = pytest.mark.asyncio
REQUEST_CONTEXT = ContextVar("ha_projection_test_request_context", default=None)


@pytest.fixture
def transport(monkeypatch):
    controlled = _Transport()
    monkeypatch.setattr(ha_module.aiohttp, "ClientSession", controlled.session_factory)
    return controlled


class _NativeGate:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Event()
        self.exited = asyncio.Event()
        self.release = threading.Event()
        self.calls = []

    def wrap(self, function, *, selected=None):
        def held(*args, **kwargs):
            if selected is not None and not selected(args, kwargs):
                return function(*args, **kwargs)
            thread = threading.get_ident()
            self.calls.append((thread, REQUEST_CONTEXT.get()))
            self.loop.call_soon_threadsafe(self.entered.set)
            try:
                assert thread != self.loop_thread, "projection ran on the event-loop thread"
                # Only a native deadlock guard. Test behavior is controlled by
                # release/probe Events, never a duration/throughput threshold.
                if not self.release.wait(5):
                    raise RuntimeError("synthetic_native_gate_not_released")
                return function(*args, **kwargs)
            finally:
                self.loop.call_soon_threadsafe(self.exited.set)
        return held


@asynccontextmanager
async def _owned_client(transport, *gates):
    async with _client(transport) as client:
        try:
            yield client
        finally:
            # Release before _client attempts actual native drain, including
            # when an assertion fails or an observer is canceled.
            for gate in gates:
                gate.release.set()


async def _probe_while_native_is_held(gate):
    assert gate.calls and not gate.release.is_set() and not gate.exited.is_set()
    progressed = asyncio.Event()

    async def probe():
        await asyncio.sleep(0)
        progressed.set()

    task = asyncio.create_task(probe())
    try:
        await _reached(progressed)
        assert not gate.release.is_set() and not gate.exited.is_set()
        assert all(thread != gate.loop_thread and context is None
                   for thread, context in gate.calls)
    finally:
        await task


def _work(client):
    work = client._state_projection_work
    assert isinstance(work, OwnedBlockingWork)
    assert work.status()["capacity"] == 1
    return work


def _pending(client, purpose):
    tasks = tuple(task for task in client._state_projection_tasks[purpose]
                  if not task.done())
    assert len(tasks) == 1
    task = tasks[0]
    return task


async def _scope_admitted(client, purpose):
    async with asyncio.timeout(5):
        while not any(not task.done() for task in client._state_projection_tasks[purpose]):
            await asyncio.sleep(0)
    return _pending(client, purpose)


async def _baseline(client, socket):
    await _io_baseline(client, socket)
    async with asyncio.timeout(5):
        while any(not task.done() for task in client._state_projection_tasks["events"]):
            await asyncio.sleep(0)
    assert client._state_frame.initialized


def _native_snapshot(work):
    assert work.status()["operations"] == 1
    native = tuple(work._tasks)
    assert len(native) == 1 and not native[0].done()
    return native[0]


def _rest_point(state="on", **kwargs):
    value = _rest_state(state=state, **kwargs)
    value["last_changed"] = "2023-11-14T22:13:21+00:00"
    value["last_updated"] = "2023-11-14T22:13:21+00:00"
    return value


@pytest.mark.parametrize("entities", [329, MAX_ENTITIES])
@pytest.mark.parametrize("phase", ["initial", "decode", "reduce"])
async def test_real_projection_owner_keeps_loop_progress_at_the_entity_cap(
    transport, monkeypatch, entities, phase,
):
    gate = _NativeGate()
    if phase == "initial":
        monkeypatch.setattr(ha_module, "apply_initial", gate.wrap(ha_module.apply_initial))
    elif phase == "decode":
        monkeypatch.setattr(ha_module, "decode_message", gate.wrap(
            ha_module.decode_message,
            selected=lambda args, _kwargs: (
                bool(args) and isinstance(args[0], str) and "owned-block" in args[0]
            ),
        ))
    else:
        monkeypatch.setattr(ha_module, "apply_delta", gate.wrap(ha_module.apply_delta))
    states = {"light.synthetic": _compact_state()}
    states.update({f"sensor.synthetic_{i}": _compact_state("1")
                   for i in range(entities - 1)})
    events = transport.socket("events", initial_event={"a": states})
    transport.socket("calls")
    async with _owned_client(transport, gate) as client:
        token = REQUEST_CONTEXT.set(object())
        try:
            if phase == "initial":
                starting = transport.spawn(client.start_state_sync())
            else:
                await _baseline(client, events)
                assert len(client._state_frame.stream_states) == entities
                events.stream_waiting.clear()
                events.delta({"c": {"light.synthetic": {"+": {
                    "s": "on", "lc": WIRE_TIME + 1,
                    "a": {"cpu_marker": "owned-block"},
                }}}})
            await _reached(gate.entered)
            work = _work(client)
            native = _native_snapshot(work)
            scope = _pending(client, "events")
            before = client.get_entity_state_cached("light.synthetic")
            if phase == "initial":
                assert before is None
            else:
                assert before["state"] == "off"
            await _probe_while_native_is_held(gate)
            assert _native_snapshot(work) is native
            assert _pending(client, "events") is scope
            gate.release.set()
            await _reached(gate.exited)
            await _reached(events.stream_waiting)
            if phase == "initial":
                assert await starting is None
        finally:
            REQUEST_CONTEXT.reset(token)
        assert work.status()["operations"] == 0
        expected = "off" if phase == "initial" else "on"
        assert client.get_entity_state_cached("light.synthetic")["state"] == expected
        assert len(client._state_frame.stream_states) == entities
        assert len(gate.calls) == 1
        transport.assert_no_effects_or_duplicate_receivers()


async def test_cancelled_rest_observer_retains_native_slot_and_loop_commit(
    transport, monkeypatch,
):
    gate = _NativeGate()
    monkeypatch.setattr(ha_module, "apply_rest", gate.wrap(ha_module.apply_rest))
    events = transport.socket("events")
    transport.socket("calls")
    async with _owned_client(transport, gate) as client:
        await _baseline(client, events)
        transport.response(_rest_point())
        token = REQUEST_CONTEXT.set(object())
        try:
            observer = transport.spawn(
                client.read_entity_observation("light.synthetic", timeout=2.),
            )
            await _reached(gate.entered)
            work = _work(client)
            native = _native_snapshot(work)
            scope = _pending(client, "rest")
            observer.cancel()
            await _probe_while_native_is_held(gate)
            assert _native_snapshot(work) is native
            assert _pending(client, "rest") is scope
            assert client.get_entity_state_cached("light.synthetic")["state"] == "off"
            gate.release.set()
            async with asyncio.timeout(5):
                result = await asyncio.gather(observer, return_exceptions=True)
                await asyncio.shield(scope)
            assert isinstance(result[0], asyncio.CancelledError)
        finally:
            REQUEST_CONTEXT.reset(token)
        assert gate.exited.is_set() and scope.done()
        assert work.status()["operations"] == 0
        assert client.get_entity_state_cached("light.synthetic")["state"] == "on"
        assert all(context is None for _, context in gate.calls)
        transport.assert_no_effects_or_duplicate_receivers()


async def test_held_rest_io_has_empty_context_without_holding_the_cpu_lock(
    transport, monkeypatch,
):
    gate = _NativeGate()
    monkeypatch.setattr(ha_module, "apply_delta", gate.wrap(ha_module.apply_delta))
    events = transport.socket("events")
    transport.socket("calls")
    async with _owned_client(transport, gate) as client:
        await _baseline(client, events)
        response = transport.response(_rest_state(state="off"), hold=True,
                                      complete_after_cancel=True)
        contexts = []
        actual_read = response.content.read

        async def read(n):
            contexts.append(REQUEST_CONTEXT.get())
            return await actual_read(n)

        monkeypatch.setattr(response.content, "read", read)
        token = REQUEST_CONTEXT.set(object())
        try:
            observer = transport.spawn(
                client.read_entity_observation("light.synthetic", timeout=2.),
            )
            await _reached(response.entered)
            rest_scope = _pending(client, "rest")
            work = _work(client)
            assert work.status()["operations"] == 0
            assert not client._state_projection_lock.locked()
            observer.cancel()

            events.stream_waiting.clear()
            events.delta({"c": {"light.synthetic": {"+": {
                "s": "on", "lc": WIRE_TIME + 2,
            }}}})
            await _reached(gate.entered)
            assert _pending(client, "rest") is rest_scope
            _pending(client, "events")
            _native_snapshot(work)
            await _probe_while_native_is_held(gate)
            gate.release.set()
            await _reached(events.stream_waiting)
            assert client.get_entity_state_cached("light.synthetic")["state"] == "on"
            assert _pending(client, "rest") is rest_scope
            response.release.set()
            async with asyncio.timeout(5):
                canceled = await asyncio.gather(observer, return_exceptions=True)
                await asyncio.shield(rest_scope)
        finally:
            REQUEST_CONTEXT.reset(token)
        assert isinstance(canceled[0], asyncio.CancelledError)
        assert contexts and all(value is None for value in contexts)
        assert all(0 < size <= 65536 for size in response.content.read_sizes)
        assert client.get_entity_state_cached("light.synthetic")["state"] == "on"
        assert work.status()["operations"] == 0
        transport.assert_no_effects_or_duplicate_receivers()


async def test_stop_and_restart_keep_the_events_native_owner_until_actual_exit(
    transport, monkeypatch,
):
    gate = _NativeGate()
    monkeypatch.setattr(ha_module, "apply_delta", gate.wrap(ha_module.apply_delta))
    first = transport.socket("events-A")
    calls = transport.socket("calls")
    next_state = _compact_state("off")
    next_state.update(lc=WIRE_TIME + 2, lu=WIRE_TIME + 2)
    second = transport.socket("events-B", hold_auth=True,
                              initial_event={"a": {"light.synthetic": next_state}})
    async with _owned_client(transport, gate) as client:
        await _baseline(client, first)
        callbacks = []
        client.register_state_callback(lambda eid, state: callbacks.append((eid, state)))
        first.delta({"c": {"light.synthetic": {"+": {
            "s": "on", "lc": WIRE_TIME + 1,
            "a": {"cpu_marker": "owned-old-event"},
        }}}})
        await _reached(gate.entered)
        work = _work(client)
        native = _native_snapshot(work)
        event_scope = _pending(client, "events")
        response = transport.response(_rest_point(attributes={"rest_marker": "actual-rest"}))
        rest_observer = transport.spawn(
            client.read_entity_observation("light.synthetic", timeout=2.),
        )
        await _reached(response.finished)
        rest_scope = await _scope_admitted(client, "rest")
        assert _native_snapshot(work) is native
        assert not rest_scope.done()

        stopping_entered = asyncio.Event()
        stopping = transport.spawn(_entered_call(stopping_entered, client.stop_state_sync))
        await _reached(stopping_entered)
        starting_entered = asyncio.Event()
        starting = transport.spawn(_entered_call(starting_entered, client.start_state_sync))
        await _reached(starting_entered)
        assert not stopping.done() and not second.opened.is_set()
        assert client.get_entity_state_cached("light.synthetic") is None
        assert _pending(client, "events") is event_scope
        await _probe_while_native_is_held(gate)
        assert _native_snapshot(work) is native

        gate.release.set()
        await _reached(gate.exited)
        async with asyncio.timeout(5):
            assert await stopping is None
            await asyncio.shield(rest_scope)
            await rest_observer
        await _reached(second.auth_entered)
        second.auth_release.set()
        async with asyncio.timeout(5):
            assert await starting is None
        await _reached(second.stream_waiting)
        assert event_scope.done()
        assert work.status()["operations"] == 0
        assert first.closed and client._ws_calls is calls
        current = client.get_entity_state_cached("light.synthetic")
        assert current["state"] == "off" and current["attributes"] == {}
        assert all(state is None or "cpu_marker" not in state["attributes"]
                   for _, state in callbacks)
        transport.assert_no_effects_or_duplicate_receivers()


async def test_terminal_close_retains_rest_cpu_and_queued_event_after_observer_cancel(
    transport, monkeypatch,
):
    gate = _NativeGate()
    monkeypatch.setattr(ha_module, "apply_rest", gate.wrap(ha_module.apply_rest))
    events = transport.socket("events")
    transport.socket("calls")
    async with _owned_client(transport, gate) as client:
        await _baseline(client, events)
        callbacks = []
        client.register_state_callback(lambda eid, state: callbacks.append((eid, state)))
        transport.response(_rest_point(attributes={"rest_marker": "owned-rest"}))
        reading = transport.spawn(client.read_entity_observation("light.synthetic", timeout=2.))
        await _reached(gate.entered)
        work = _work(client)
        native = _native_snapshot(work)
        rest_scope = _pending(client, "rest")
        events.delta({"c": {"light.synthetic": {"+": {
            "s": "on", "lc": WIRE_TIME + 1,
            "a": {"rest_marker": "owned-rest"},
        }}}})
        event_scope = await _scope_admitted(client, "events")
        assert _native_snapshot(work) is native

        entered = asyncio.Event()
        closing = transport.spawn(_entered_call(entered, client.close))
        await _reached(entered)
        close_owner = client._close_task
        assert close_owner is not None and not close_owner.done()
        closing.cancel()
        await _probe_while_native_is_held(gate)
        assert _native_snapshot(work) is native
        assert _pending(client, "rest") is rest_scope
        assert _pending(client, "events") is event_scope
        assert not close_owner.done()
        assert any(not session.closed for session in transport.sessions)
        second_close = transport.spawn(client.close())
        gate.release.set()
        async with asyncio.timeout(5):
            canceled, read_result, closed = await asyncio.gather(
                closing, reading, second_close, return_exceptions=True,
            )
        assert isinstance(canceled, asyncio.CancelledError)
        assert closed is None
        assert not isinstance(read_result, BaseException)
        assert not read_result.current(client._entity_now())
        assert gate.exited.is_set() and event_scope.done() and rest_scope.done()
        assert close_owner.done() and work.status()["operations"] == 0
        assert all(session.closed for session in transport.sessions)
        assert all(socket.closed for socket in transport.opened)
        assert callbacks == []
        history = client._state_frame.entries.get("light.synthetic")
        assert history is None or history.state["state"] == "off"
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("purpose", ["events", "rest"])
async def test_native_wait_keeps_the_original_entity_observation_deadline(
    transport, monkeypatch, purpose,
):
    clock = {"now": 100.}
    monkeypatch.setattr(ha_module.time, "time", lambda: clock["now"])
    gate = _NativeGate()
    symbol = "apply_delta" if purpose == "events" else "apply_rest"
    monkeypatch.setattr(ha_module, symbol, gate.wrap(getattr(ha_module, symbol)))
    events = transport.socket("events")
    transport.socket("calls")
    async with _owned_client(transport, gate) as client:
        await _baseline(client, events)
        original = client.get_entity_observation("light.synthetic")
        assert (original.observed_at, original.expires_at) == (100., 106.)
        callbacks = []
        client.register_state_callback(lambda eid, state: callbacks.append((eid, state)))
        if purpose == "events":
            events.delta({"c": {"light.synthetic": {"+": {
                "s": "on", "lc": WIRE_TIME + 1,
            }}}})
            observer = None
        else:
            transport.response(_rest_point())
            observer = transport.spawn(
                client.read_entity_observation("light.synthetic", timeout=2.),
            )
        await _reached(gate.entered)
        scope = _pending(client, purpose)
        native = _native_snapshot(_work(client))
        await _probe_while_native_is_held(gate)
        clock["now"] = 107.
        assert _native_snapshot(_work(client)) is native
        gate.release.set()
        async with asyncio.timeout(5):
            await asyncio.gather(asyncio.shield(scope), return_exceptions=True)
            if observer is not None:
                returned = await observer
                assert not returned.current(107.)
        assert gate.exited.is_set()
        actual = client.get_entity_observation("light.synthetic")
        assert not actual.current(107.)
        assert (actual.observed_at, actual.expires_at) == (100., 106.)
        assert client.get_entity_state_cached("light.synthetic") is None
        assert callbacks == []  # expired positive evidence is not a deletion
        assert _work(client).status()["operations"] == 0
        transport.assert_no_effects_or_duplicate_receivers()


async def test_three_real_rest_reads_enter_together_and_commit_in_different_order(
    transport, monkeypatch,
):
    identities = ["light.a", "light.b", "light.c"]
    events = transport.socket("events", initial_event={"a": {
        identity: _compact_state() for identity in identities
    }})
    transport.socket("calls")
    async with _owned_client(transport) as client:
        await _baseline(client, events)
        responses = {}
        for index, identity in enumerate(identities):
            value = _rest_state(identity, "on", {"read": index})
            value["last_changed"] = "2023-11-14T22:13:21+00:00"
            value["last_updated"] = "2023-11-14T22:13:21+00:00"
            responses[identity] = transport.response(value, hold=True)
        contexts = []

        def route_get(url, **_kwargs):
            identity = url.rsplit("/", 1)[-1]
            assert identity in responses
            contexts.append(REQUEST_CONTEXT.get())
            response = responses[identity]
            transport.responses.remove(response)
            transport.state_reads += 1
            transport.state_urls.append(url)
            return response

        monkeypatch.setattr(transport.sessions[0], "get", route_get)

        async def read(identity):
            token = REQUEST_CONTEXT.set(object())
            try:
                return await client.get_entity_state_fresh(identity, timeout=2.)
            finally:
                REQUEST_CONTEXT.reset(token)

        observers = {identity: transport.spawn(read(identity)) for identity in identities}
        for response in responses.values():
            await _reached(response.entered)
        admitted = tuple(task for task in client._state_projection_tasks["rest"]
                         if not task.done())
        assert len(admitted) == 3
        assert _work(client).status()["operations"] == 0
        assert not client._state_projection_lock.locked()
        assert all(not observer.done() for observer in observers.values())
        assert contexts == [None, None, None]

        for identity in ["light.c", "light.a", "light.b"]:
            responses[identity].release.set()
            async with asyncio.timeout(5):
                result = await observers[identity]
            assert result["entity_id"] == identity and result["state"] == "on"
            assert result["attributes"]["read"] == identities.index(identity)
            for other, response in responses.items():
                if not response.release.is_set():
                    assert not observers[other].done()
        for identity in identities:
            value = client.get_entity_observation(identity)
            assert value.current(client._entity_now()) and value.exists is True
            assert value.state["attributes"]["read"] == identities.index(identity)
        assert _work(client).status()["operations"] == 0
        assert transport.state_reads == 3
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("status", [403, 503])
async def test_failed_rest_read_uses_the_same_retained_native_owner(
    transport, monkeypatch, status,
):
    gate = _NativeGate()
    monkeypatch.setattr(ha_module, "apply_rest_failure", gate.wrap(ha_module.apply_rest_failure))
    events = transport.socket("events")
    transport.socket("calls")
    async with _owned_client(transport, gate) as client:
        await _baseline(client, events)
        previous = client.get_entity_observation("light.synthetic")
        transport.response(None, status=status)
        token = REQUEST_CONTEXT.set(object())
        try:
            observer = transport.spawn(
                client.read_entity_observation("light.synthetic", timeout=2.),
            )
            await _reached(gate.entered)
            scope = _pending(client, "rest")
            native = _native_snapshot(_work(client))
            await _probe_while_native_is_held(gate)
            assert _pending(client, "rest") is scope
            assert _native_snapshot(_work(client)) is native
            gate.release.set()
            async with asyncio.timeout(5):
                actual = await observer
        finally:
            REQUEST_CONTEXT.reset(token)
        assert gate.exited.is_set() and scope.done()
        assert actual.exists is None and not actual.current(client._entity_now())
        assert actual.reason == ("source_denied" if status == 403 else "source_unavailable")
        assert actual.state == previous.state
        assert (actual.observed_at, actual.expires_at) == (
            previous.observed_at, previous.expires_at,
        )
        assert client.get_entity_state_cached("light.synthetic") is None
        assert _work(client).status()["operations"] == 0
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("status", [403, 503])
async def test_failed_missing_entity_at_capacity_retains_original_inventory_evidence(
    transport, monkeypatch, status,
):
    states = {f"sensor.synthetic_{index}": _compact_state("1")
              for index in range(MAX_ENTITIES)}
    events = transport.socket("events", initial_event={"a": states})
    transport.socket("calls")
    async with _owned_client(transport) as client:
        monkeypatch.setattr(client, "_entity_now", lambda: 100.)
        await _baseline(client, events)
        previous = client.get_entity_observation("light.missing")
        assert previous.current(100.) and previous.exists is False
        transport.response(None, status=status)
        actual = await client.read_entity_observation("light.missing", timeout=2.)
        assert actual.exists is None and not actual.current(100.)
        assert actual.reason == ("source_denied" if status == 403 else "source_unavailable")
        assert actual.source == previous.source == "ha-ws-entities"
        assert (actual.observed_at, actual.expires_at, actual.attempted_at) == (100., 106., 100.)
        assert len(client._state_frame.entries) == MAX_ENTITIES
        assert "light.missing" not in client._state_frame.entries
        assert not client._state_frame.initialized
        transport.assert_no_effects_or_duplicate_receivers()


async def test_terminal_close_before_native_admission_returns_closed_observation(
    transport, monkeypatch,
):
    entered, release = asyncio.Event(), asyncio.Event()
    async with _owned_client(transport) as client:
        work = _work(client)
        real_run = work.run

        async def before_admission(*args, **kwargs):
            entered.set()
            await release.wait()
            return await real_run(*args, **kwargs)

        monkeypatch.setattr(work, "run", before_admission)
        transport.response(_rest_point("on"))
        observer = transport.spawn(client.read_entity_observation("light.synthetic", timeout=2.))
        try:
            await _reached(entered)
            before = client._state_frame
            close = transport.spawn(client.close())
            async with asyncio.timeout(5):
                while not client._closed:
                    await asyncio.sleep(0)
            assert not close.done() and work.status()["operations"] == 0
            release.set()
            async with asyncio.timeout(5):
                value = await observer
                await close
        finally:
            release.set()
        assert value.reason == "source_closed" and value.exists is None
        assert client._state_frame is before
        assert work.status()["operations"] == 0 and client._state_raw_bytes == 0
        assert all(session.closed for session in transport.sessions)
        transport.assert_no_effects_or_duplicate_receivers()

