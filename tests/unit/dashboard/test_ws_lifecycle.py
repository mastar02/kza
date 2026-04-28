"""Tests del lifecycle del WS /ws/live: subscribe → frame → disconnect → unsubscribe."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.live_event_bus import LiveEventBus, LiveEvent, LiveEventType
from src.dashboard.observability import register_observability_routes


def _build_app(bus: LiveEventBus | None):
    app = FastAPI()
    register_observability_routes(app, event_bus=bus, use_mocks=True)
    return app


def test_ws_route_absent_when_no_bus():
    app = _build_app(bus=None)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/ws/live" not in paths


def test_ws_route_present_with_bus():
    bus = LiveEventBus()
    app = _build_app(bus)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/ws/live" in paths


def test_ws_handshake_and_unsubscribe_on_disconnect():
    bus = LiveEventBus()
    app = _build_app(bus)
    client = TestClient(app)
    assert bus.subscriber_count == 0
    with client.websocket_connect("/ws/live"):
        # Loop interno del TestClient debe haber registrado el subscriber
        # antes de que termine el `with` body.
        pass
    # Tras cerrar, el finally del handler debe haber unsubscribed
    assert bus.subscriber_count == 0


def test_ws_publish_round_trip():
    """Publish desde el mismo loop del server (vía un endpoint trigger)."""
    bus = LiveEventBus()
    app = _build_app(bus)

    @app.post("/test/publish")
    async def _trigger():
        await bus.publish(LiveEvent(
            type=LiveEventType.WAKE,
            payload={"zone": "sala", "confidence": 0.91},
            ts="14:32:08",
        ))
        return {"ok": True}

    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        client.post("/test/publish")
        frame = ws.receive_json()
        assert frame == {
            "type": "wake",
            "payload": {"zone": "sala", "confidence": 0.91},
            "ts": "14:32:08",
        }


def test_ws_two_subscribers_each_receive_frame():
    bus = LiveEventBus()
    app = _build_app(bus)

    @app.post("/test/publish")
    async def _trigger():
        await bus.publish(LiveEvent(type=LiveEventType.TURN, payload={"id": "t1"}))
        return {"ok": True}

    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws1:
        with client.websocket_connect("/ws/live") as ws2:
            client.post("/test/publish")
            f1 = ws1.receive_json()
            f2 = ws2.receive_json()
            assert f1["payload"]["id"] == "t1"
            assert f2["payload"]["id"] == "t1"
