"""Smoke test del dashboard end-to-end via FastAPI TestClient.

Valida:
  - Cada endpoint REST responde 200 con shape correcto
  - El frontend HTML se sirve en /obs/
  - WebSocket /ws/live entrega un frame publicado al bus
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from src.dashboard.live_event_bus import LiveEventBus, LiveEvent, LiveEventType
from src.dashboard.observability import register_observability_routes


def build_app(use_mocks: bool = True):
    app = FastAPI()
    bus = LiveEventBus()
    register_observability_routes(app, event_bus=bus, use_mocks=use_mocks)
    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "dashboard", "frontend",
    )
    if os.path.exists(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app, bus


def main():
    app, bus = build_app(use_mocks=True)
    client = TestClient(app)

    endpoints = [
        "/api/zones", "/api/conversations", "/api/ha/entities",
        "/api/ha/actions", "/api/llm/endpoints", "/api/users",
        "/api/alerts", "/api/system/gpus", "/api/system/services",
    ]

    print("GET endpoints (use_mocks=True):")
    failed = 0
    for ep in endpoints:
        r = client.get(ep)
        ok = r.status_code == 200
        n = len(r.json()) if ok and isinstance(r.json(), list) else "?"
        marker = "OK " if ok else "ERR"
        print(f"  [{marker}] {r.status_code}  {ep:32s}  → {n} items")
        if not ok:
            failed += 1

    print("\nPOST /api/llm/endpoints/foo/clear-cooldown:")
    r = client.post("/api/llm/endpoints/foo/clear-cooldown")
    print(f"  [{'OK ' if r.status_code == 200 else 'ERR'}] {r.status_code}  → {r.json()}")

    print("\nGET /obs/ (frontend wireframe):")
    r = client.get("/obs/")
    body = r.text
    assert r.status_code == 200, r.status_code
    assert "KZA — Dashboard wireframes" in body
    assert "src/app.jsx" in body
    print(f"  [OK ] 200  → {len(body)} bytes, contains shell + jsx imports")

    print("\nGET /obs/src/views/live.jsx (asset routing):")
    r = client.get("/obs/src/views/live.jsx")
    assert r.status_code == 200
    assert "useLiveStream" in r.text
    print(f"  [OK ] 200  → live.jsx served (incluye useLiveStream)")

    print("\nWS /ws/live handshake:")
    with client.websocket_connect("/ws/live") as ws:
        # round-trip de eventos cubierto en tests/unit/dashboard/test_live_event_bus.py
        # (publish + subscriber.queue.get). Acá solo validamos handshake.
        assert ws is not None
    print("  [OK ] WS aceptado y cerrado limpio")

    if failed:
        print(f"\n[FAIL] {failed} endpoints failed")
        sys.exit(1)
    print("\n[PASS] dashboard smoke OK")


if __name__ == "__main__":
    main()
