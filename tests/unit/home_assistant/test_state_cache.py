"""Tests para el prefetch cache de Home Assistant (S6).

Cubre:
- Populate inicial del cache via REST snapshot.
- Update atómico del cache al recibir el stream compacto de entidades.
- Cache miss → fallback a REST.
- Resiliencia: loop sigue corriendo después de un error.
- Borrados compactos con observación negativa fechada.
- Callbacks de observers se invocan sin bloquear si uno falla.
- start_state_sync es idempotente.
- Graceful shutdown via stop_state_sync.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.home_assistant.ha_client import HAStatesSnapshot, HomeAssistantClient
from src.home_assistant.state_projection import apply_initial, apply_rest, empty_frame, entity_observation


def _state(entity_id, state="off", attributes=None):
    return {
        "entity_id": entity_id, "state": state, "attributes": attributes or {},
        "context": {"id": "synthetic-context", "parent_id": None, "user_id": None},
        "last_changed": "2023-11-14T22:13:20+00:00",
        "last_updated": "2023-11-14T22:13:20+00:00",
    }


def _compact(state="off", attributes=None):
    return {"s": state, "a": attributes or {}, "c": "synthetic-context",
            "lc": 1_700_000_000., "lu": 1_700_000_000.}


def _seed_stream(ha, states):
    """Seed a complete, explicitly observed synthetic stream baseline."""
    ha._state_sync_running = True
    ha._entity_stream_generation += 1
    ha._state_frame = apply_initial(ha._state_frame,
        {"a": {key: _compact(value["state"], value.get("attributes"))
               for key, value in states.items()}},
        generation=ha._entity_stream_generation, observed_at=ha._entity_now(),
    )


def _seed_rest(ha, entity_id, state="off", attributes=None):
    frame = ha._state_frame
    ha._state_frame = apply_rest(
        frame, entity_id, _state(entity_id, state, attributes),
        observed_at=ha._entity_now(), ttl_s=6.,
        expected_generation=frame.generation,
        expected_revision=frame.entry_revision(entity_id),
    )


def _actual_observation(ha, entity_id, state="off"):
    frame = empty_frame()
    frame = apply_rest(
        frame, entity_id, _state(entity_id, state),
        observed_at=ha._entity_now(), ttl_s=6.,
        expected_generation=frame.generation,
        expected_revision=frame.entry_revision(entity_id),
    )
    return entity_observation(frame, entity_id, now=ha._entity_now())


def _dated_snapshot(ha, states):
    observed = datetime.fromtimestamp(ha._entity_now(), UTC)
    return HAStatesSnapshot(
        True, observed, None, tuple(states), observed + timedelta(seconds=6), observed,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def ha():
    """Own the dummy client and keep the unrelated call channel offline."""
    client = HomeAssistantClient(url="http://mock.local", token="fake-token")
    client.connect_websocket = AsyncMock(return_value=False)
    try:
        yield client
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Snapshot inicial y populate del cache
# ---------------------------------------------------------------------------


class TestCachePopulate:
    """Legacy manual REST publication remains dated and independent of stream state."""

    @pytest.mark.asyncio
    async def test_cache_populated_on_subscribe(self, ha):
        snapshot = _dated_snapshot(ha, [
            _state("light.escritorio", "on"),
            _state("light.living", "off"),
            _state("climate.dormitorio", "cool", {"temperature": 22}),
        ])
        ha.get_state_snapshot = AsyncMock(return_value=snapshot)
        await ha._refresh_full_state_snapshot()
        assert len(ha._state_cache) == 3
        assert ha.get_entity_state_cached("light.escritorio")["state"] == "on"
        assert ha.get_entity_state_cached("light.living")["state"] == "off"
        assert ha.get_entity_state_cached("climate.dormitorio")["attributes"]["temperature"] == 22
        observed = ha.get_entity_observation("light.escritorio")
        assert observed.current(ha._entity_now())
        assert observed.source == "ha-rest-states"
        assert observed.observed_at == snapshot.observed_at.timestamp()
        assert observed.expires_at == snapshot.expires_at.timestamp()

    @pytest.mark.asyncio
    async def test_empty_snapshot_does_not_wipe_cache(self, ha):
        _seed_rest(ha, "light.x", "on")
        previous = ha.get_entity_observation("light.x")
        ha.get_state_snapshot = AsyncMock(return_value=_dated_snapshot(ha, []))
        await ha._refresh_full_state_snapshot()
        assert ha.get_entity_state_cached("light.x")["state"] == "on"
        current = ha.get_entity_observation("light.x")
        assert current.observed_at == previous.observed_at
        assert current.expires_at == previous.expires_at

    @pytest.mark.asyncio
    async def test_malformed_snapshot_preserves_history_without_partial_publication(self, ha):
        _seed_rest(ha, "light.previous", "off")
        before = ha._state_frame
        original = ha.get_entity_observation("light.previous")
        ha.get_state_snapshot = AsyncMock(return_value=_dated_snapshot(ha, [
            _state("light.ok", "on"), {"state": "something"}, {},
        ]))
        await ha._refresh_full_state_snapshot()
        assert ha._state_frame is before
        assert ha.get_entity_state_cached("light.ok") is None
        retained = ha.get_entity_observation("light.previous")
        assert (retained.state, retained.observed_at, retained.expires_at) == (
            original.state, original.observed_at, original.expires_at)


# ---------------------------------------------------------------------------
# Events state_changed
# ---------------------------------------------------------------------------


class TestStateChangedEvents:
    """The real compact publisher commits changes/removals before current reads."""

    @pytest.mark.asyncio
    async def test_state_changed_event_updates_cache(self, ha):
        _seed_stream(ha, {"light.x": _state("light.x", "off")})
        await ha._publish_entities_delta({"c": {"light.x": {"+": {
            "s": "on", "a": {"brightness": 255},
        }}}})
        cached = ha.get_entity_state_cached("light.x")
        assert cached["state"] == "on"
        assert cached["attributes"]["brightness"] == 255
        assert ha.get_entity_observation("light.x").current(ha._entity_now())

    @pytest.mark.asyncio
    async def test_state_changed_new_entity_creates_cache_entry(self, ha):
        _seed_stream(ha, {})
        assert ha.get_entity_observation("sensor.nuevo").exists is False
        await ha._publish_entities_delta({"a": {"sensor.nuevo": _compact("23.4")}})
        assert ha.get_entity_state_cached("sensor.nuevo")["state"] == "23.4"

    @pytest.mark.asyncio
    async def test_state_changed_null_state_removes_entity(self, ha):
        _seed_stream(ha, {"light.gone": _state("light.gone", "on")})
        await ha._publish_entities_delta({"r": ["light.gone"]})
        assert ha.get_entity_state_cached("light.gone") is None
        removed = ha.get_entity_observation("light.gone")
        assert removed.current(ha._entity_now()) and removed.exists is False

    @pytest.mark.asyncio
    async def test_malformed_compact_delta_fails_atomically(self, ha):
        _seed_stream(ha, {"light.x": _state("light.x", "off")})
        before = ha._state_frame
        with pytest.raises(ValueError):
            await ha._publish_entities_delta({"c": {
                "light.x": {"+": {"s": "on"}},
                "light.missing": {"+": {"s": "invalid"}},
            }})
        assert ha._state_frame is before
        assert ha.get_entity_state_cached("light.x")["state"] == "off"

    @pytest.mark.asyncio
    async def test_legacy_event_is_history_and_malformed_legacy_does_not_crash(self, ha):
        ha._handle_state_changed({})
        ha._handle_state_changed({"data": {}})
        ha._handle_state_changed({"data": {"new_state": {"state": "on"}}})
        assert ha._state_cache == {}
        ha._handle_state_changed({"data": {
            "entity_id": "light.legacy", "new_state": _state("light.legacy", "on"),
        }})
        assert ha._state_cache["light.legacy"]["state"] == "on"
        assert ha.get_entity_state_cached("light.legacy") is None
        assert not ha.get_entity_observation("light.legacy").current(ha._entity_now())


# ---------------------------------------------------------------------------
# Cache miss fallback
# ---------------------------------------------------------------------------


class TestCacheMissFallback:
    """Only current dated cache entries suppress the bounded native fallback."""

    @pytest.mark.asyncio
    async def test_cache_miss_falls_back_to_rest(self, ha):
        observation = _actual_observation(ha, "light.no_cached", "on")
        ha.read_entity_observation = AsyncMock(return_value=observation)
        result = await ha.get_entity_state("light.no_cached")
        assert result["state"] == "on"
        ha.read_entity_observation.assert_awaited_once()
        assert ha.read_entity_observation.await_args.args[0] == "light.no_cached"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_rest(self, ha):
        _seed_rest(ha, "light.cached", "on")
        ha.read_entity_observation = AsyncMock()
        result = await ha.get_entity_state("light.cached")
        assert result["state"] == "on"
        ha.read_entity_observation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cached_lookup_without_io(self, ha):
        _seed_rest(ha, "light.a", "off")
        assert ha.get_entity_state_cached("light.a")["state"] == "off"
        assert ha.get_entity_state_cached("light.missing") is None
        assert ha._session is None

    @pytest.mark.asyncio
    async def test_raw_legacy_seed_is_not_a_current_observation(self, ha):
        ha._state_cache = {"light.legacy": _state("light.legacy", "on")}
        assert ha._state_cache["light.legacy"]["state"] == "on"
        assert ha.get_entity_state_cached("light.legacy") is None
        observed = ha.get_entity_observation("light.legacy")
        assert not observed.current(ha._entity_now())
        assert observed.observed_at is None and observed.expires_at is None


# ---------------------------------------------------------------------------
# Callbacks (observer pattern)
# ---------------------------------------------------------------------------


class TestStateCallbacks:
    """Callbacks observe committed current compact changes and stay isolated."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_state_change(self, ha):
        _seed_stream(ha, {})
        calls = []
        ha.register_state_callback(lambda eid, st: calls.append((eid, st)))
        await ha._publish_entities_delta({"a": {"light.x": _compact("on")}})
        assert len(calls) == 1
        assert calls[0][0] == "light.x"
        assert calls[0][1]["entity_id"] == "light.x"
        assert calls[0][1]["state"] == "on"
        assert ha.get_entity_observation("light.x").current(ha._entity_now())

    @pytest.mark.asyncio
    async def test_failing_callback_does_not_break_others(self, ha):
        _seed_stream(ha, {})
        calls = []

        def broken(eid, st):
            raise RuntimeError("intentional")

        def good(eid, st):
            calls.append(eid)

        ha.register_state_callback(broken)
        ha.register_state_callback(good)
        await ha._publish_entities_delta({"a": {"light.x": _compact("on")}})
        assert calls == ["light.x"]


# ---------------------------------------------------------------------------
# Loop lifecycle
# ---------------------------------------------------------------------------


class TestWaitForWsReady:
    """Warmup del WS events antes del background loop."""

    @pytest.mark.asyncio
    async def test_succeeds_first_attempt(self, ha):
        """Si _open_ws_authenticated funciona al toque → 1 sola llamada."""
        ws_mock = MagicMock()
        ws_mock.closed = False
        ws_mock.close = AsyncMock(side_effect=lambda: setattr(ws_mock, "closed", True))
        ha._open_ws_authenticated = AsyncMock(return_value=ws_mock)

        await ha._wait_for_ws_ready(max_attempts=3, backoff_s=0.0)

        assert ha._open_ws_authenticated.await_count == 1
        assert ha._ws_events is ws_mock

    @pytest.mark.asyncio
    async def test_retries_on_none_then_succeeds(self, ha):
        """Si la primera devuelve None, reintenta hasta success."""
        ws_mock = MagicMock()
        ws_mock.closed = False
        ws_mock.close = AsyncMock(side_effect=lambda: setattr(ws_mock, "closed", True))
        ha._open_ws_authenticated = AsyncMock(side_effect=[None, ws_mock])

        await ha._wait_for_ws_ready(max_attempts=3, backoff_s=0.0)

        assert ha._open_ws_authenticated.await_count == 2
        assert ha._ws_events is ws_mock

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self, ha):
        """Si todos los intentos fallan, propaga RuntimeError."""
        ha._open_ws_authenticated = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="ha_events_warmup_unavailable"):
            await ha._wait_for_ws_ready(max_attempts=3, backoff_s=0.0)

        assert ha._open_ws_authenticated.await_count == 3

    @pytest.mark.asyncio
    async def test_handles_exception_during_attempt(self, ha):
        """Excepción en _open_ws_authenticated cuenta como intento fallido."""
        ws_mock = MagicMock()
        ws_mock.closed = False
        ws_mock.close = AsyncMock(side_effect=lambda: setattr(ws_mock, "closed", True))
        ha._open_ws_authenticated = AsyncMock(
            side_effect=[ConnectionError("transient"), ws_mock]
        )

        await ha._wait_for_ws_ready(max_attempts=3, backoff_s=0.0)

        assert ha._open_ws_authenticated.await_count == 2
        assert ha._ws_events is ws_mock


class TestSyncLoopLifecycle:
    """Arranque, parada y resiliencia del loop de sync."""

    @pytest.mark.asyncio
    async def test_start_state_sync_is_idempotent(self, ha):
        """Llamar start_state_sync 2 veces no crea 2 tasks."""

        async def sleeping():
            await asyncio.sleep(10)

        # Stub el subscribe para evitar conexiones reales
        ha._subscribe_and_sync = sleeping
        ha._wait_for_ws_ready = AsyncMock()  # skip WS warmup en tests

        await ha.start_state_sync()
        first_task = ha._state_subscribe_task

        await ha.start_state_sync()
        second_task = ha._state_subscribe_task

        assert first_task is second_task

        await ha.stop_state_sync()

    @pytest.mark.asyncio
    async def test_stop_state_sync_cancels_task(self, ha):
        """stop_state_sync cancela el task y limpia la referencia."""

        async def sleeping():
            await asyncio.sleep(100)

        ha._subscribe_and_sync = sleeping
        ha._wait_for_ws_ready = AsyncMock()

        await ha.start_state_sync()
        assert ha._state_subscribe_task is not None
        assert not ha._state_subscribe_task.done()

        await ha.stop_state_sync()

        assert ha._state_subscribe_task is None
        assert ha._state_sync_running is False

    @pytest.mark.asyncio
    async def test_sync_loop_survives_subscribe_error(self, ha):
        """Si _subscribe_and_sync lanza, el loop reintenta tras backoff."""
        call_count = {"n": 0}

        async def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("transient")
            # Tercera llamada: quedarse colgado para que podamos parar el loop
            await asyncio.sleep(10)

        ha._subscribe_and_sync = flaky
        ha._wait_for_ws_ready = AsyncMock()
        # Patchear asyncio.sleep dentro del loop para no esperar los 5s de backoff
        original_sleep = asyncio.sleep

        async def fast_sleep(delay):
            # Consumir el delay real pero muy rápido
            await original_sleep(0)

        import src.home_assistant.ha_client as ha_module
        ha_module.asyncio.sleep = fast_sleep

        try:
            await ha.start_state_sync()
            # Darle tiempo a que reintente 3 veces
            for _ in range(50):
                await original_sleep(0)
                if call_count["n"] >= 3:
                    break
            assert call_count["n"] >= 3
        finally:
            ha_module.asyncio.sleep = original_sleep
            await ha.stop_state_sync()


# ---------------------------------------------------------------------------
# Subscribe & sync flow (end-to-end sin red real)
# ---------------------------------------------------------------------------


class TestSubscribeAndSync:
    """A single compact stream publishes its first snapshot before deltas."""

    @pytest.mark.asyncio
    async def test_subscribe_and_sync_populates_cache(self, ha):
        ha._fetch_all_states_rest = AsyncMock()
        ws_mock = MagicMock()
        ws_mock.closed = False
        ws_mock.close = AsyncMock(side_effect=lambda: setattr(ws_mock, "closed", True))
        ws_mock.send_json = AsyncMock()
        messages = [
            {"id": 2, "type": "result", "success": True},
            {"id": 2, "type": "event", "event": {"a": {"light.a": _compact("off")}}},
            {"id": 2, "type": "event", "event": {"c": {
                "light.a": {"+": {"s": "on"}},
            }}},
        ]
        received = []

        def observed(entity_id, state):
            if state is not None and state["state"] == "on":
                current = ha.get_entity_observation(entity_id)
                received.append((state, current.current(ha._entity_now())))

        ha.register_state_callback(observed)

        async def receive_json(**_kwargs):
            if messages:
                return _kwargs.get("loads", json.loads)(json.dumps(messages.pop(0)))
            ha._state_sync_running = False
            raise asyncio.TimeoutError()

        ws_mock.receive_json = AsyncMock(side_effect=receive_json)
        ha._ws_events = ws_mock
        ha._state_sync_running = True
        await ha._subscribe_and_sync()

        assert len(received) == 1
        assert received[0][0]["entity_id"] == "light.a"
        assert received[0][0]["state"] == "on" and received[0][1] is True
        ha._fetch_all_states_rest.assert_not_awaited()
        ws_mock.send_json.assert_awaited_once()
        assert ws_mock.send_json.await_args.args[0] == {
            "id": 2, "type": "subscribe_entities",
        }
        assert ws_mock.closed is True
