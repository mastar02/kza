"""Tests para LiveEventBus."""

import asyncio
import pytest

from src.dashboard.live_event_bus import LiveEventType


@pytest.mark.asyncio
async def test_publish_no_subscribers_does_not_raise(live_graph):
    bus = live_graph.bus()
    await bus.publish(live_graph.event(type=LiveEventType.WAKE))


@pytest.mark.asyncio
async def test_subscriber_receives_event(live_graph):
    bus = live_graph.bus()
    sub_id, q = await bus.subscribe()
    ev = live_graph.event(type=LiveEventType.TURN, payload={"latency_ms": 1})
    await bus.publish(ev)
    received = await asyncio.wait_for(q.get(), timeout=0.5)
    assert received.payload == {"latency_ms": 1}
    await bus.unsubscribe(sub_id)
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_drop_oldest_when_full(live_graph):
    bus = live_graph.bus(queue_size=2, overflow_policy="drop_oldest")
    sub_id, q = await bus.subscribe()
    for i in range(5):
        await bus.publish(live_graph.event(type=LiveEventType.TURN, payload={"latency_ms": i}))
    assert q.qsize() == 2
    a = await q.get()
    b = await q.get()
    assert a.payload["latency_ms"] == 3
    assert b.payload["latency_ms"] == 4
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_drop_newest_when_full(live_graph):
    bus = live_graph.bus(queue_size=2, overflow_policy="drop_newest")
    sub_id, q = await bus.subscribe()
    for i in range(5):
        await bus.publish(live_graph.event(type=LiveEventType.TURN, payload={"latency_ms": i}))
    a = await q.get()
    b = await q.get()
    assert a.payload["latency_ms"] == 0
    assert b.payload["latency_ms"] == 1
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_publish_does_not_block_with_full_subscriber(live_graph):
    bus = live_graph.bus(queue_size=1, overflow_policy="drop_oldest")
    await bus.subscribe()
    # 1000 publishes deben terminar inmediatamente — sin esperas.
    async with asyncio.timeout(1.0):
        for i in range(1000):
            await bus.publish(live_graph.event(type=LiveEventType.WAKE, payload={"latency_ms": i}))


@pytest.mark.asyncio
async def test_to_frame_shape(live_graph):
    ev = live_graph.event(type=LiveEventType.ALERT, payload={"a": 1}, ts="14:30:00")
    assert ev.to_frame() == {"type": "alert", "payload": {"a": 1}, "ts": "14:30:00"}
