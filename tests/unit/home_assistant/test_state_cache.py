"""Tests para el prefetch cache de Home Assistant (S6).

Cubre:
- Populate inicial del cache via REST snapshot.
- Update del cache al recibir events state_changed via WebSocket.
- Cache miss → fallback a REST.
- Resiliencia: loop sigue corriendo después de un error.
- Limpieza de entities con new_state=None (entity removed/unavailable).
- Callbacks de observers se invocan sin bloquear si uno falla.
- start_state_sync es idempotente.
- Graceful shutdown via stop_state_sync.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.home_assistant.ha_client import HomeAssistantClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ha():
    """Cliente HA con URL dummy (no hace conexiones reales en estos tests)."""
    return HomeAssistantClient(url="http://mock.local", token="fake-token")


# ---------------------------------------------------------------------------
# Snapshot inicial y populate del cache
# ---------------------------------------------------------------------------


class TestCachePopulate:
    """Populación inicial del cache desde REST snapshot."""

    @pytest.mark.asyncio
    async def test_cache_populated_on_subscribe(self, ha):
        """El snapshot REST inicial puebla el cache con todas las entities."""
        ha._fetch_all_states_rest = AsyncMock(return_value=[
            {"entity_id": "light.escritorio", "state": "on"},
            {"entity_id": "light.living", "state": "off"},
            {"entity_id": "climate.dormitorio", "state": "cool",
             "attributes": {"temperature": 22}},
        ])

        await ha._refresh_full_state_snapshot()

        assert len(ha._state_cache) == 3
        assert ha.get_entity_state_cached("light.escritorio")["state"] == "on"
        assert ha.get_entity_state_cached("light.living")["state"] == "off"
        assert ha.get_entity_state_cached("climate.dormitorio")["attributes"]["temperature"] == 22

    @pytest.mark.asyncio
    async def test_empty_snapshot_does_not_wipe_cache(self, ha):
        """Si el REST devuelve [] (HA caído), el cache no se vacía."""
        ha._state_cache["light.x"] = {"entity_id": "light.x", "state": "on"}
        ha._fetch_all_states_rest = AsyncMock(return_value=[])

        await ha._refresh_full_state_snapshot()

        # El cache previo debe quedar intacto — no queremos perder state por un blip
        assert ha.get_entity_state_cached("light.x")["state"] == "on"

    @pytest.mark.asyncio
    async def test_snapshot_skips_entries_without_entity_id(self, ha):
        """Defensive: entries malformados (sin entity_id) no crashean el snapshot."""
        ha._fetch_all_states_rest = AsyncMock(return_value=[
            {"entity_id": "light.ok", "state": "on"},
            {"state": "something"},  # malformado — sin entity_id
            {},
        ])

        await ha._refresh_full_state_snapshot()

        assert len(ha._state_cache) == 1
        assert "light.ok" in ha._state_cache


# ---------------------------------------------------------------------------
# Events state_changed
# ---------------------------------------------------------------------------


class TestStateChangedEvents:
    """Handling de events WebSocket que actualizan el cache."""

    @pytest.mark.asyncio
    async def test_state_changed_event_updates_cache(self, ha):
        """Un event state_changed reemplaza el valor en el cache."""
        ha._state_cache["light.x"] = {"entity_id": "light.x", "state": "off"}

        event = {
            "event_type": "state_changed",
            "data": {
                "entity_id": "light.x",
                "new_state": {"entity_id": "light.x", "state": "on",
                              "attributes": {"brightness": 255}},
                "old_state": {"entity_id": "light.x", "state": "off"},
            },
        }
        ha._handle_state_changed(event)

        cached = ha.get_entity_state_cached("light.x")
        assert cached["state"] == "on"
        assert cached["attributes"]["brightness"] == 255

    @pytest.mark.asyncio
    async def test_state_changed_new_entity_creates_cache_entry(self, ha):
        """Una entity nueva aparece en el cache al primer event."""
        assert ha.get_entity_state_cached("sensor.nuevo") is None

        event = {
            "data": {
                "entity_id": "sensor.nuevo",
                "new_state": {"entity_id": "sensor.nuevo", "state": "23.4"},
            },
        }
        ha._handle_state_changed(event)

        assert ha.get_entity_state_cached("sensor.nuevo")["state"] == "23.4"

    @pytest.mark.asyncio
    async def test_state_changed_null_state_removes_entity(self, ha):
        """new_state=None (entity eliminada/unavailable) purga el cache."""
        ha._state_cache["light.gone"] = {"entity_id": "light.gone", "state": "on"}

        event = {
            "data": {"entity_id": "light.gone", "new_state": None},
        }
        ha._handle_state_changed(event)

        assert ha.get_entity_state_cached("light.gone") is None

    @pytest.mark.asyncio
    async def test_malformed_event_does_not_crash(self, ha):
        """Events sin entity_id no crashean el handler."""
        # No debe raise
        ha._handle_state_changed({})
        ha._handle_state_changed({"data": {}})
        ha._handle_state_changed({"data": {"new_state": {"state": "on"}}})

        assert len(ha._state_cache) == 0


# ---------------------------------------------------------------------------
# Cache miss fallback
# ---------------------------------------------------------------------------


class TestCacheMissFallback:
    """Cache-first con fallback a REST si no está en cache."""

    @pytest.mark.asyncio
    async def test_cache_miss_falls_back_to_rest(self, ha):
        """get_entity_state llama al REST si la entity no está en cache."""
        rest_result = {"entity_id": "light.no_cached", "state": "on"}
        ha._get_entity_state_rest = AsyncMock(return_value=rest_result)

        result = await ha.get_entity_state("light.no_cached")

        assert result == rest_result
        ha._get_entity_state_rest.assert_awaited_once_with("light.no_cached")

    @pytest.mark.asyncio
    async def test_cache_hit_skips_rest(self, ha):
        """Si la entity está en cache, no se llama al REST (ahorro de latencia)."""
        ha._state_cache["light.cached"] = {"entity_id": "light.cached", "state": "on"}
        ha._get_entity_state_rest = AsyncMock(return_value=None)

        result = await ha.get_entity_state("light.cached")

        assert result["state"] == "on"
        ha._get_entity_state_rest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cached_lookup_without_io(self, ha):
        """get_entity_state_cached es sync y no toca la network."""
        ha._state_cache["light.a"] = {"entity_id": "light.a", "state": "off"}

        # Si esto tocase network, el test colgaría o fallaría — no hay session real
        result = ha.get_entity_state_cached("light.a")
        assert result["state"] == "off"

        # Miss debe ser None, no exception
        assert ha.get_entity_state_cached("light.missing") is None


# ---------------------------------------------------------------------------
# Callbacks (observer pattern)
# ---------------------------------------------------------------------------


class TestStateCallbacks:
    """Callbacks registrados via register_state_callback."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_state_change(self, ha):
        """El callback recibe (entity_id, new_state) cuando cambia el state."""
        calls = []
        ha.register_state_callback(lambda eid, st: calls.append((eid, st)))

        event = {
            "data": {
                "entity_id": "light.x",
                "new_state": {"state": "on"},
            },
        }
        ha._handle_state_changed(event)

        assert len(calls) == 1
        assert calls[0][0] == "light.x"
        assert calls[0][1] == {"state": "on"}

    @pytest.mark.asyncio
    async def test_failing_callback_does_not_break_others(self, ha):
        """Un callback que raise no previene que otros corran (aislamiento)."""
        calls = []

        def broken(eid, st):
            raise RuntimeError("intentional")

        def good(eid, st):
            calls.append(eid)

        ha.register_state_callback(broken)
        ha.register_state_callback(good)

        event = {"data": {"entity_id": "light.x", "new_state": {"state": "on"}}}
        ha._handle_state_changed(event)  # no raise

        assert calls == ["light.x"]


# ---------------------------------------------------------------------------
# Loop lifecycle
# ---------------------------------------------------------------------------


class TestSyncLoopLifecycle:
    """Arranque, parada y resiliencia del loop de sync."""

    @pytest.mark.asyncio
    async def test_start_state_sync_is_idempotent(self, ha):
        """Llamar start_state_sync 2 veces no crea 2 tasks."""

        async def sleeping():
            await asyncio.sleep(10)

        # Stub el subscribe para evitar conexiones reales
        ha._subscribe_and_sync = sleeping

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
    """Flow completo: snapshot → subscribe → consumir events."""

    @pytest.mark.asyncio
    async def test_subscribe_and_sync_populates_cache(self, ha):
        """El flow completo puebla el cache y procesa al menos un event."""

        # Snapshot REST inicial
        ha._fetch_all_states_rest = AsyncMock(return_value=[
            {"entity_id": "light.a", "state": "off"},
        ])

        # WS stub: simula receive_json devolviendo
        # 1) confirmación del subscribe
        # 2) un event state_changed
        # 3) timeout (asyncio.TimeoutError) para salir del loop vía stop
        ha.ensure_websocket_connected = AsyncMock(return_value=True)
        ws_mock = MagicMock()
        ws_mock.closed = False
        ws_mock.send_json = AsyncMock()

        messages = [
            {"id": 2, "type": "result", "success": True},
            {"type": "event", "event": {
                "data": {"entity_id": "light.a", "new_state": {"state": "on"}},
            }},
        ]
        idx = {"i": 0}

        async def receive_json():
            i = idx["i"]
            idx["i"] += 1
            if i < len(messages):
                return messages[i]
            # Después del segundo message, parar el loop
            ha._state_sync_running = False
            raise asyncio.TimeoutError()

        ws_mock.receive_json = AsyncMock(side_effect=receive_json)
        # Seed events channel directly so _subscribe_and_sync skips the
        # _open_ws_authenticated handshake (dual-WS architecture 2026-04-24).
        ha._ws_events = ws_mock
        ha._state_sync_running = True

        await ha._subscribe_and_sync()

        # El cache debe tener el snapshot aplicado y el event aplicado
        assert ha.get_entity_state_cached("light.a")["state"] == "on"
        ws_mock.send_json.assert_awaited_once()
        sent = ws_mock.send_json.await_args[0][0]
        assert sent["type"] == "subscribe_events"
        assert sent["event_type"] == "state_changed"
