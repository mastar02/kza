"""Coherent entity handoff/freshness through the real HA client and leaf I/O."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

import src.home_assistant.ha_client as ha_module
from src.home_assistant.state_projection import MAX_ENTITIES, MAX_FRAME_BYTES
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


@pytest.fixture
def transport(monkeypatch):
    controlled = _Transport()
    monkeypatch.setattr(ha_module.aiohttp, "ClientSession", controlled.session_factory)
    return controlled


@pytest.fixture
def clock(monkeypatch):
    value = {"now": 100.}
    monkeypatch.setattr(ha_module.time, "time", lambda: value["now"])
    return value


async def _baseline(client, socket):
    await client.start_state_sync()
    # The leaf has returned the initial event and the client has asked for its
    # next frame. This proves the initial frame was processed, without polling.
    await _reached(socket.stream_waiting)


async def _delta(socket, change, **kwargs):
    socket.stream_waiting.clear()
    socket.delta(change, **kwargs)
    await _reached(socket.stream_waiting)


def _hold_retry(client, transport):
    entered, release = asyncio.Event(), asyncio.Event()
    delays = []
    transport.cleanup_gates.append(release)

    async def wait(delay):
        delays.append(delay)
        entered.set()
        await release.wait()

    client._state_retry_wait = wait
    return entered, release, delays


@pytest.mark.parametrize("empty", [False, True])
async def test_first_compact_snapshot_is_atomic_nonempty_or_authoritative_empty(
    transport, clock, empty,
):
    snapshot = {"a": {} if empty else {"light.synthetic": _compact_state()}}
    events = transport.socket("events", initial_event=snapshot, hold_initial=True)
    transport.socket("calls")
    async with _client(transport) as client:
        starting = transport.spawn(client.start_state_sync())
        await _reached(events.initial_entered)
        before = client.get_entity_observation("light.synthetic")
        assert before.exists is None and not before.current(clock["now"])
        assert client.get_entity_state_cached("light.synthetic") is None
        assert client.get_health_status().events.phase != "subscribed"
        assert transport.state_reads == 0

        events.initial_release.set()
        await _reached(events.stream_waiting)
        async with asyncio.timeout(5):
            assert await starting is None
        observed = client.get_entity_observation("light.synthetic")
        assert observed.current(clock["now"])
        assert observed.exists is (not empty)
        assert observed.source == "ha-ws-entities"
        assert observed.observed_at == 100. and observed.expires_at == 106.
        assert client.get_health_status().events.phase == "subscribed"
        if empty:
            assert observed.state is None
            assert client.get_entity_state_cached("light.synthetic") is None
        else:
            assert observed.state["state"] == "off"
            assert client.get_entity_state_cached("light.synthetic")["state"] == "off"
        assert transport.state_reads == 0
        transport.assert_no_effects_or_duplicate_receivers()


async def test_compact_changes_and_deletion_keep_actual_observation_deadlines(
    transport, clock,
):
    events = transport.socket("events")
    transport.socket("calls")
    async with _client(transport) as client:
        await _baseline(client, events)
        first = client.get_entity_observation("light.synthetic")
        clock["now"] = 101.
        await _delta(events, {"c": {"light.synthetic": {"+": {
            "s": "on", "a": {"brightness": 128}, "lc": WIRE_TIME + 1,
        }}}})
        changed = client.get_entity_observation("light.synthetic")
        assert changed.current(101.) and changed.exists is True
        assert changed.state["state"] == "on"
        assert changed.state["attributes"]["brightness"] == 128
        assert (changed.observed_at, changed.expires_at) == (101., 107.)
        assert (first.observed_at, first.expires_at) == (100., 106.)

        clock["now"] = 102.
        await _delta(events, {"r": ["light.synthetic"]})
        removed = client.get_entity_observation("light.synthetic")
        assert removed.current(102.) and removed.exists is False
        assert removed.state is None
        assert (removed.observed_at, removed.expires_at) == (102., 108.)
        assert client.get_entity_state_cached("light.synthetic") is None
        assert client.is_entity_available("light.synthetic") is None
        assert transport.state_reads == 0
        transport.assert_no_effects_or_duplicate_receivers()


async def test_callbacks_and_readers_receive_detached_nested_states(
    transport, clock,
):
    events = transport.socket("events", initial_event={"a": {
        "light.synthetic": _compact_state(attributes={"nested": {"n": 1}}),
    }})
    transport.socket("calls")
    async with _client(transport) as client:
        await _baseline(client, events)
        received = []

        def mutating(_entity_id, state):
            state["attributes"]["nested"]["n"] = 999
            raise RuntimeError("synthetic_observer_failure")

        def observing(entity_id, state):
            received.append((entity_id, deepcopy(state)))

        client.register_state_callback(mutating)
        client.register_state_callback(observing)
        clock["now"] = 101.
        await _delta(events, {"c": {"light.synthetic": {"+": {"s": "on"}}}})
        assert len(received) == 1
        assert received[0][1]["attributes"]["nested"]["n"] == 1
        cached = client.get_entity_state_cached("light.synthetic")
        assert cached["state"] == "on"
        assert cached["attributes"]["nested"]["n"] == 1
        cached["attributes"]["nested"]["n"] = 888
        observed = client.get_entity_observation("light.synthetic")
        observed.state["attributes"]["nested"]["n"] = 777
        assert client.get_entity_state_cached("light.synthetic")["attributes"]["nested"]["n"] == 1
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("newer", ["delta", "deletion", "rebuild"])
async def test_held_rest_response_cannot_overwrite_a_newer_stream_observation(
    transport, clock, newer,
):
    events = transport.socket("events-A")
    transport.socket("calls")
    successor = transport.socket("events-B", initial_event={"a": {
        "light.synthetic": _compact_state("on"),
    }})
    async with _client(transport) as client:
        retry_entered, retry_release, _ = _hold_retry(client, transport)
        await _baseline(client, events)
        response = transport.response(
            _rest_state(state="old-rest"), hold=True, complete_after_cancel=True,
        )
        reading = transport.spawn(
            client.read_entity_observation("light.synthetic", timeout=2.),
        )
        await _reached(response.entered)

        clock["now"] = 101.
        if newer == "delta":
            await _delta(events, {"c": {"light.synthetic": {"+": {"s": "on"}}}})
        elif newer == "deletion":
            await _delta(events, {"r": ["light.synthetic"]})
        else:
            events.frames.put_nowait(RuntimeError("synthetic_disconnect"))
            await _reached(events.close_finished)
            await _reached(retry_entered)
            retry_release.set()
            await _reached(successor.stream_waiting)
        native = client.get_entity_observation("light.synthetic")
        assert native.current(101.)
        assert native.exists is (newer != "deletion")
        if newer != "deletion":
            assert native.state["state"] == "on"
        assert (native.observed_at, native.expires_at) == (101., 107.)

        clock["now"] = 102.
        response.release.set()
        async with asyncio.timeout(5):
            returned = await reading
        assert returned.state is None or returned.state["state"] != "old-rest"
        after = client.get_entity_observation("light.synthetic")
        assert after.exists is native.exists
        assert after.state == native.state
        assert after.source == "ha-ws-entities"
        assert (after.observed_at, after.expires_at) == (101., 107.)
        assert transport.state_reads == 1
        transport.assert_no_effects_or_duplicate_receivers()


async def test_rest_entry_does_not_replace_the_compact_reducer_baseline(
    transport, clock,
):
    events = transport.socket("events", initial_event={"a": {
        "light.synthetic": _compact_state("off", {"brightness": 1}),
    }})
    transport.socket("calls")
    async with _client(transport) as client:
        await _baseline(client, events)
        clock["now"] = 101.
        point = _rest_state(state="on", attributes={"rest_only": 9})
        point["last_changed"] = point["last_updated"] = "2023-11-14T22:13:22+00:00"
        transport.response(point)
        rest = await client.read_entity_observation("light.synthetic", timeout=2.)
        assert rest.current(101.) and rest.state["state"] == "on"
        assert rest.source == "ha-rest-entity"
        baseline = client._state_frame.stream_states["light.synthetic"]
        assert baseline["state"] == "off"
        assert dict(baseline["attributes"]) == {"brightness": 1}

        # This legal T1 contribution was buffered before the REST T2 point.
        # Its omitted state must come from the ordered stream baseline.
        clock["now"] = 102.
        await _delta(events, {"c": {"light.synthetic": {"+": {
            "a": {"brightness": 2}, "lu": WIRE_TIME + 1,
        }}}})
        baseline = client._state_frame.stream_states["light.synthetic"]
        assert baseline["state"] == "off"
        assert dict(baseline["attributes"]) == {"brightness": 2}
        selected = client.get_entity_observation("light.synthetic")
        assert selected.state == rest.state
        assert selected.source == rest.source
        assert (selected.observed_at, selected.expires_at) == (101., 107.)

        clock["now"] = 103.
        await _delta(events, {"c": {"light.synthetic": {
            "+": {"s": "on", "a": {"rest_only": 9}, "lc": WIRE_TIME + 2},
            "-": {"a": ["brightness"]},
        }}})
        clock["now"] = 104.
        await _delta(events, {"c": {"light.synthetic": {"+": {
            "a": {"humidity": 1}, "lu": WIRE_TIME + 3,
        }}}})
        streamed = client.get_entity_observation("light.synthetic")
        assert streamed.current(104.) and streamed.state["state"] == "on"
        assert streamed.state["attributes"] == {"rest_only": 9, "humidity": 1}
        assert streamed.source == "ha-ws-entities"
        assert streamed.observed_at == 104.
        assert transport.state_reads == 1
        transport.assert_no_effects_or_duplicate_receivers()


async def test_authentication_and_health_do_not_renew_entity_expiry_and_getter_reads(
    transport, clock,
):
    events = transport.socket("events")
    transport.socket("calls-A")
    async with _client(transport) as client:
        await _baseline(client, events)
        original = client.get_entity_observation("light.synthetic")
        assert (original.observed_at, original.expires_at) == (100., 106.)

        await client._retire_socket("calls")
        transport.socket("calls-B")
        clock["now"] = 105.
        assert await client.connect_websocket() is True
        client.get_health_status()
        assert client.get_entity_observation("light.synthetic").observed_at == 100.
        assert client.get_entity_observation("light.synthetic").expires_at == 106.

        clock["now"] = 106.
        expired = client.get_entity_observation("light.synthetic")
        assert not expired.current(106.)
        assert (expired.observed_at, expired.expires_at) == (100., 106.)
        assert client.get_entity_state_cached("light.synthetic") is None
        assert client.is_entity_available("light.synthetic") is None
        assert transport.state_reads == 0

        clock["now"] = 107.
        transport.response(_rest_state(state="on"))
        actual = await client.get_entity_state("light.synthetic")
        assert actual["state"] == "on"
        refreshed = client.get_entity_observation("light.synthetic")
        assert refreshed.current(107.) and refreshed.source == "ha-rest-entity"
        assert (refreshed.observed_at, refreshed.expires_at) == (107., 113.)
        assert transport.state_reads == 1
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("failure", [
    "initial_without_added", "wrong_initial_id", "partial_bad_delta",
    "wrong_delta_id", "entity_overflow", "wire_overflow",
])
async def test_bad_stream_frame_is_atomic_and_retires_before_a_fresh_successor(
    transport, clock, failure,
):
    initial = {"c": {}} if failure == "initial_without_added" else None
    broken = transport.socket(
        "events-A", initial_event=initial, hold_close=True,
        initial_id_offset=int(failure == "wrong_initial_id"),
    )
    calls = transport.socket("calls")
    successor = transport.socket("events-B")
    async with _client(transport) as client:
        retry_entered, retry_release, _ = _hold_retry(client, transport)
        changes = []
        await client.start_state_sync()
        if failure not in ("initial_without_added", "wrong_initial_id"):
            await _reached(broken.stream_waiting)
            before = client.get_entity_observation("light.synthetic")
            client.register_state_callback(lambda eid, state: changes.append((eid, state)))
            if failure == "partial_bad_delta":
                broken.delta({"c": {
                    "light.synthetic": {"+": {"s": "must-not-publish"}},
                    "light.absent": {"+": {"s": "invalid-without-baseline"}},
                }})
            elif failure == "wrong_delta_id":
                broken.delta({"c": {"light.synthetic": {"+": {"s": "on"}}}},
                             id_offset=1)
            elif failure == "entity_overflow":
                broken.delta({"a": {
                    f"sensor.extra_{i}": _compact_state("1")
                    for i in range(MAX_ENTITIES)
                }})
            else:
                broken.frames.put_nowait(json.dumps({
                    "type": "event", "id": broken.subscriptions[-1],
                    "event": {"a": {"sensor.oversized": _compact_state(
                        attributes={"oversized": "x" * (MAX_FRAME_BYTES + 1)},
                    )}},
                }))
        await _reached(broken.close_entered)
        assert not broken.closed and not successor.opened.is_set()
        assert changes == []
        rejected = client.get_entity_observation("light.synthetic")
        assert not rejected.current(clock["now"])
        if failure in ("partial_bad_delta", "wrong_delta_id", "entity_overflow", "wire_overflow"):
            assert rejected.state == before.state
            assert (rejected.observed_at, rejected.expires_at) == (100., 106.)
        if failure == "wire_overflow":
            assert broken.raw_loads_called
        assert broken.connect_options.get("max_msg_size") == MAX_FRAME_BYTES

        broken.close_release.set()
        await _reached(broken.close_finished)
        await _reached(retry_entered)
        assert not successor.opened.is_set()
        clock["now"] = 101.
        retry_release.set()
        await _reached(successor.stream_waiting)
        current = client.get_entity_observation("light.synthetic")
        assert current.current(101.) and current.state["state"] == "off"
        assert successor.connect_options.get("max_msg_size") == MAX_FRAME_BYTES
        assert current.observed_at == 101.
        assert transport.opened == [broken, calls, successor]
        assert broken.close_calls == 1
        assert transport.state_reads == 0
        transport.assert_no_effects_or_duplicate_receivers()


async def test_reconnect_has_no_current_gap_and_authoritative_empty_replaces_old(
    transport, clock,
):
    first = transport.socket("events-A")
    transport.socket("calls")
    second = transport.socket("events-B", initial_event={"a": {}}, hold_initial=True)
    async with _client(transport) as client:
        retry_entered, retry_release, _ = _hold_retry(client, transport)
        await _baseline(client, first)
        original = client.get_entity_observation("light.synthetic")
        callbacks = []
        client.register_state_callback(lambda eid, state: callbacks.append((eid, state)))
        first.frames.put_nowait(RuntimeError("synthetic_disconnect"))
        await _reached(first.close_finished)
        await _reached(retry_entered)
        clock["now"] = 101.
        retry_release.set()
        await _reached(second.initial_entered)
        rebuilding = client.get_entity_observation("light.synthetic")
        assert not rebuilding.current(101.)
        assert (rebuilding.observed_at, rebuilding.expires_at) == (
            original.observed_at, original.expires_at,
        )
        assert client.get_entity_state_cached("light.synthetic") is None
        assert client.get_health_status().events.phase != "subscribed"

        # A late packet remains on the retired socket; its old subscription
        # cannot append to the new initial set or mint an observation.
        first.delta({"c": {"light.synthetic": {"+": {"s": "late-old"}}}})
        second.initial_release.set()
        await _reached(second.stream_waiting)
        absent = client.get_entity_observation("light.synthetic")
        assert absent.current(101.) and absent.exists is False
        assert absent.state is None
        assert (absent.observed_at, absent.expires_at) == (101., 107.)
        assert all(state is None or state["state"] != "late-old"
                   for _, state in callbacks)
        assert transport.state_reads == 0
        transport.assert_no_effects_or_duplicate_receivers()


async def test_stop_rejects_a_delta_already_waiting_in_the_old_receive(
    transport, clock,
):
    first = transport.socket("events-A", hold_delta=True,
                             complete_delta_after_cancel=True)
    calls = transport.socket("calls")
    second = transport.socket("events-B", initial_event={"a": {
        "light.synthetic": _compact_state("on"),
    }})
    async with _client(transport) as client:
        await _baseline(client, first)
        callbacks = []
        client.register_state_callback(lambda eid, state: callbacks.append((eid, state)))
        first.delta({"c": {"light.synthetic": {"+": {"s": "late-old"}}}})
        await _reached(first.delta_entered)

        stop_entered = asyncio.Event()
        stopping = transport.spawn(_entered_call(stop_entered, client.stop_state_sync))
        await _reached(stop_entered)
        start_entered = asyncio.Event()
        starting = transport.spawn(_entered_call(start_entered, client.start_state_sync))
        await _reached(start_entered)
        assert not second.opened.is_set()
        retained = client.get_entity_observation("light.synthetic")
        assert not retained.current(clock["now"])
        assert client.get_entity_state_cached("light.synthetic") is None
        assert (retained.observed_at, retained.expires_at) == (100., 106.)
        clock["now"] = 101.
        first.delta_release.set()
        async with asyncio.timeout(5):
            assert await stopping is None
            assert await starting is None
        await _reached(second.stream_waiting)
        observed = client.get_entity_observation("light.synthetic")
        assert observed.current(101.) and observed.state["state"] == "on"
        assert first.closed and client._ws_calls is calls
        assert all(state is None or state["state"] != "late-old"
                   for _, state in callbacks)
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("shutdown", ["stop_state_sync", "close"])
async def test_shutdown_fences_an_initial_snapshot_retained_after_ack(
    transport, clock, shutdown,
):
    events = transport.socket(
        "events", hold_initial=True, complete_initial_after_cancel=True,
        hold_close=True,
    )
    calls = transport.socket("calls")
    async with _client(transport) as client:
        starting = transport.spawn(client.start_state_sync())
        await _reached(events.initial_entered)
        before = client.get_entity_observation("light.synthetic")
        assert before.observed_at is None and before.exists is None
        entered = asyncio.Event()
        shutting = transport.spawn(_entered_call(entered, getattr(client, shutdown)))
        await _reached(entered)
        assert not shutting.done()
        events.initial_release.set()
        await _reached(events.close_entered)
        assert not shutting.done() and not events.closed
        events.close_release.set()
        async with asyncio.timeout(5):
            assert await starting is None
            assert await shutting is None
        after = client.get_entity_observation("light.synthetic")
        assert not after.current(clock["now"])
        assert after.observed_at is None and after.expires_at is None
        assert client.get_entity_state_cached("light.synthetic") is None
        assert events.closed and events.close_calls == 1
        if shutdown == "close":
            assert calls.closed
            assert all(session.closed for session in transport.sessions)
        else:
            assert client._ws_calls is calls and not calls.closed
        transport.assert_no_effects_or_duplicate_receivers()


async def test_buffered_older_ws_advances_baseline_without_replacing_newer_rest(
    transport, clock,
):
    events = transport.socket(
        "events", hold_delta=True,
        initial_event={"a": {"light.synthetic": _compact_state(
            "off", {"brightness": 10, "anchor": "baseline"},
        )}},
    )
    transport.socket("calls")
    async with _client(transport) as client:
        await _baseline(client, events)
        callbacks = []
        client.register_state_callback(
            lambda eid, state: callbacks.append((eid, deepcopy(state)))
        )

        # HA point T1 precedes point T2, even though its socket delivery will
        # finish later. T1 removes the old attribute and changes off -> on.
        events.stream_waiting.clear()
        events.delta({"c": {"light.synthetic": {
            "+": {"s": "on", "a": {"brightness": 20}, "lc": WIRE_TIME + 1},
            "-": {"a": ["anchor"]},
        }}})
        await _reached(events.delta_entered)

        rest_point_2 = _rest_state(
            state="off", attributes={"brightness": 30, "rest_only": "point2"},
        )
        rest_point_2["last_changed"] = "2023-11-14T22:13:22+00:00"
        rest_point_2["last_updated"] = "2023-11-14T22:13:22+00:00"
        clock["now"] = 102.
        transport.response(rest_point_2)
        newer = await client.read_entity_observation("light.synthetic", timeout=2.)
        assert newer.current(102.)
        assert newer.state == rest_point_2
        assert newer.source == "ha-rest-entity"
        assert (newer.observed_at, newer.expires_at) == (102., 108.)

        clock["now"] = 103.
        events.delta_release.set()
        await _reached(events.stream_waiting)
        visible = client.get_entity_observation("light.synthetic")
        # Keep the newer observation (or explicitly mark its retained history
        # unavailable). Do not replace it with T1 or mint a later T1 deadline.
        assert visible.state == newer.state
        assert visible.source == newer.source
        assert (visible.observed_at, visible.expires_at) == (102., 108.)
        if visible.current(103.):
            assert client.get_entity_state_cached("light.synthetic") == rest_point_2
        else:
            assert client.get_entity_state_cached("light.synthetic") is None
        assert all(
            state is None or not (
                state["state"] == "on"
                and state["attributes"].get("brightness") == 20
            )
            for _, state in callbacks
        )

        # Entry selection must not drop the ordered stream contribution. It
        # has to reach T1, without leaking T2's REST-only attribute into it.
        baseline = client._state_frame.stream_states["light.synthetic"]
        assert baseline["state"] == "on"
        assert dict(baseline["attributes"]) == {"brightness": 20}
        assert baseline["last_updated"] == "2023-11-14T22:13:21+00:00"

        # Deliver the legitimate T2 stream update and then a T3 update with no
        # state field. Its result depends on the correctly advanced baseline.
        clock["now"] = 104.
        await _delta(events, {"c": {"light.synthetic": {"+": {
            "s": "off", "a": {"brightness": 30, "rest_only": "point2"},
            "lc": WIRE_TIME + 2,
        }}}})
        delayed = transport.response(rest_point_2, hold=True,
                                     complete_after_cancel=True)
        reading = transport.spawn(
            client.read_entity_observation("light.synthetic", timeout=2.)
        )
        await _reached(delayed.entered)
        clock["now"] = 105.
        await _delta(events, {"c": {"light.synthetic": {"+": {
            "a": {"brightness": 31}, "lu": WIRE_TIME + 3,
        }}}})
        current = client.get_entity_observation("light.synthetic")
        assert current.current(105.) and current.source == "ha-ws-entities"
        assert current.state["state"] == "off"
        assert current.state["attributes"] == {
            "brightness": 31, "rest_only": "point2",
        }
        assert (current.observed_at, current.expires_at) == (105., 111.)

        # The new selection rule must preserve the existing CAS protection.
        clock["now"] = 106.
        delayed.release.set()
        async with asyncio.timeout(5):
            returned = await reading
        assert not returned.current(106.) or returned.state == current.state
        after = client.get_entity_observation("light.synthetic")
        assert after.state == current.state
        assert (after.observed_at, after.expires_at) == (105., 111.)
        assert transport.state_reads == 2
        clock["now"] = 111.
        expired = client.get_entity_observation("light.synthetic")
        assert not expired.current(111.)
        assert (expired.observed_at, expired.expires_at) == (105., 111.)
        transport.assert_no_effects_or_duplicate_receivers()
