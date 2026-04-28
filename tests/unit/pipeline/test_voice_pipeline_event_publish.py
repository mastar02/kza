"""Tests para VoicePipeline._publish_turn_event — bus None, errores, payload."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncio
import numpy as np
import pytest

import src.pipeline.voice_pipeline as vp_module
from src.pipeline.voice_pipeline import VoicePipeline
from src.dashboard.live_event_bus import LiveEventBus, LiveEventType


def _make_pipeline(event_bus=None, request_router=None):
    return VoicePipeline(
        audio_loop=MagicMock(),
        command_processor=MagicMock(),
        request_router=request_router or MagicMock(),
        response_handler=MagicMock(),
        feature_manager=MagicMock(),
        event_bus=event_bus,
    )


@pytest.mark.asyncio
async def test_publish_turn_event_noop_when_bus_is_none():
    p = _make_pipeline(event_bus=None)
    await p._publish_turn_event(SimpleNamespace(room_id="sala"), {"text": "x"})


@pytest.mark.asyncio
async def test_publish_turn_event_full_payload():
    bus = LiveEventBus(queue_size=4)
    sub_id, q = await bus.subscribe()
    p = _make_pipeline(event_bus=bus)
    audio = SimpleNamespace(room_id="cocina")
    result = {"turn_id": "t1", "user": "Marco", "text": "hola",
              "intent": "light.on", "response": "Listo.",
              "latency_ms": 200, "success": True, "path": "fast"}
    await p._publish_turn_event(audio, result)
    ev = await asyncio.wait_for(q.get(), timeout=0.5)
    assert ev.type is LiveEventType.TURN
    assert ev.payload["zone"] == "cocina"
    assert ev.payload["user"] == "Marco"
    assert ev.payload["latency_ms"] == 200
    assert ev.payload["success"] is True
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_publish_turn_event_falls_back_to_zone_attr():
    bus = LiveEventBus()
    sub_id, q = await bus.subscribe()
    p = _make_pipeline(event_bus=bus)
    audio = SimpleNamespace(zone="estudio")  # no room_id
    await p._publish_turn_event(audio, {})
    ev = await asyncio.wait_for(q.get(), timeout=0.5)
    assert ev.payload["zone"] == "estudio"
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_publish_turn_event_with_ndarray_input_zone_is_none():
    bus = LiveEventBus()
    sub_id, q = await bus.subscribe()
    p = _make_pipeline(event_bus=bus)
    audio = np.zeros(16000, dtype=np.float32)
    await p._publish_turn_event(audio, {"text": "hola"})
    ev = await asyncio.wait_for(q.get(), timeout=0.5)
    assert ev.payload["zone"] is None
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_publish_turn_event_swallows_publish_exception(monkeypatch):
    # Reset module-level flag para que el primer fail loguee
    monkeypatch.setattr(vp_module, "_PUBLISH_FAILURE_LOGGED", False)
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("kaboom"))
    p = _make_pipeline(event_bus=bus)
    # No debe propagar
    await p._publish_turn_event(SimpleNamespace(room_id="sala"), {"text": "x"})
    bus.publish.assert_called_once()


@pytest.mark.asyncio
async def test_publish_turn_event_defaults_when_result_is_empty():
    bus = LiveEventBus()
    sub_id, q = await bus.subscribe()
    p = _make_pipeline(event_bus=bus)
    await p._publish_turn_event(SimpleNamespace(room_id="sala"), {})
    ev = await asyncio.wait_for(q.get(), timeout=0.5)
    assert ev.payload["success"] is False
    assert ev.payload["path"] == "fast"
    assert ev.payload["id"] is None
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_process_command_publishes_event():
    bus = LiveEventBus()
    sub_id, q = await bus.subscribe()
    request_router = MagicMock()
    request_router.process_command = AsyncMock(return_value={
        "text": "x", "success": True, "latency_ms": 150,
    })
    p = _make_pipeline(event_bus=bus, request_router=request_router)
    result = await p.process_command(SimpleNamespace(room_id="patio"))
    assert result["success"] is True
    ev = await asyncio.wait_for(q.get(), timeout=0.5)
    assert ev.payload["zone"] == "patio"
    await bus.unsubscribe(sub_id)
