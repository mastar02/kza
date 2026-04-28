"""Tests para LiveEventBus."""

import asyncio
import pytest

from src.dashboard.live_event_bus import LiveEventBus, LiveEvent, LiveEventType


@pytest.mark.asyncio
async def test_publish_no_subscribers_does_not_raise():
    bus = LiveEventBus()
    await bus.publish(LiveEvent(type=LiveEventType.WAKE))


@pytest.mark.asyncio
async def test_subscriber_receives_event():
    bus = LiveEventBus()
    sub_id, q = await bus.subscribe()
    ev = LiveEvent(type=LiveEventType.TURN, payload={"x": 1})
    await bus.publish(ev)
    received = await asyncio.wait_for(q.get(), timeout=0.5)
    assert received.payload == {"x": 1}
    await bus.unsubscribe(sub_id)
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_drop_oldest_when_full():
    bus = LiveEventBus(queue_size=2, overflow_policy="drop_oldest")
    sub_id, q = await bus.subscribe()
    for i in range(5):
        await bus.publish(LiveEvent(type=LiveEventType.TURN, payload={"i": i}))
    assert q.qsize() == 2
    a = await q.get()
    b = await q.get()
    assert a.payload["i"] == 3
    assert b.payload["i"] == 4
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_drop_newest_when_full():
    bus = LiveEventBus(queue_size=2, overflow_policy="drop_newest")
    sub_id, q = await bus.subscribe()
    for i in range(5):
        await bus.publish(LiveEvent(type=LiveEventType.TURN, payload={"i": i}))
    a = await q.get()
    b = await q.get()
    assert a.payload["i"] == 0
    assert b.payload["i"] == 1
    await bus.unsubscribe(sub_id)


@pytest.mark.asyncio
async def test_publish_does_not_block_with_full_subscriber():
    bus = LiveEventBus(queue_size=1, overflow_policy="drop_oldest")
    await bus.subscribe()
    # 1000 publishes deben terminar inmediatamente — sin esperas.
    async with asyncio.timeout(1.0):
        for i in range(1000):
            await bus.publish(LiveEvent(type=LiveEventType.WAKE, payload={"i": i}))


@pytest.mark.asyncio
async def test_to_frame_shape():
    ev = LiveEvent(type=LiveEventType.ALERT, payload={"a": 1}, ts="14:30:00")
    assert ev.to_frame() == {"type": "alert", "payload": {"a": 1}, "ts": "14:30:00"}
