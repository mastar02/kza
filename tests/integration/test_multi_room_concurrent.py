"""
Integration test: concurrent commands from multiple rooms.
Verifies commands from different rooms process in parallel,
echo deduplication works correctly, priority queue ordering,
shared list concurrency, and missed-reminder-on-arrival wiring.
"""
import sys
from unittest.mock import MagicMock

# Mock hardware dependencies BEFORE any imports
for mod in ["sounddevice", "soundfile", "pyaudio", "torch", "torch.cuda"]:
    sys.modules.setdefault(mod, MagicMock())

import asyncio
import os
import tempfile
import numpy as np
import pytest
import time
from unittest.mock import AsyncMock

from src.pipeline.command_event import CommandEvent
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from src.orchestrator.priority_queue import (
    Priority,
    PriorityRequestQueue,
    RequestStatus,
)
from src.orchestrator.cancellation import CancellationReason


# =====================================================================
# Wake-word deduplication
# =====================================================================


@pytest.mark.asyncio
async def test_concurrent_commands_from_different_rooms():
    """Two commands from different rooms should process in parallel."""
    results = []

    async def mock_process(event: CommandEvent) -> dict:
        await asyncio.sleep(0.1)  # Simulate 100ms processing
        results.append(event.room_id)
        return {"text": f"processed_{event.room_id}", "success": True}

    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(is_active=False),
    )
    loop.on_command(mock_process)

    event1 = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="cocina",
        mic_device_index=3,
    )
    event2 = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="escritorio",
        mic_device_index=4,
    )

    start = time.time()
    await asyncio.gather(
        loop._dispatch_command(event1),
        loop._dispatch_command(event2),
    )
    total = time.time() - start

    assert len(results) == 2
    assert "cocina" in results
    assert "escritorio" in results
    # Parallel: ~100ms, not ~200ms
    assert total < 0.18, f"Commands took {total:.3f}s — should be parallel"


@pytest.mark.asyncio
async def test_dedup_prevents_echo_but_allows_concurrent():
    """Echo dedup blocks echoes but allows genuine concurrent commands."""
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(),
        dedup_window_ms=200,
    )

    now = time.time()

    # Genuine concurrent (>200ms apart)
    assert loop._should_accept_wakeword("cocina", 0.5, now) is True
    assert loop._should_accept_wakeword("escritorio", 0.5, now + 0.3) is True

    # Echo (within 200ms, weaker)
    now2 = time.time()
    assert loop._should_accept_wakeword("living", 0.8, now2) is True
    assert loop._should_accept_wakeword("hall", 0.3, now2 + 0.05) is False


@pytest.mark.asyncio
async def test_dedup_500ms_default_window():
    """Default dedup window is 500ms — echoes within that range are rejected."""
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(),
    )
    assert loop.dedup_window_ms == 500

    now = time.time()

    # First detection accepted
    assert loop._should_accept_wakeword("cocina", 0.7, now) is True

    # Within 500ms from a different room with weaker signal = echo
    assert loop._should_accept_wakeword("living", 0.3, now + 0.4) is False

    # Same room within window is accepted (continued speech)
    assert loop._should_accept_wakeword("cocina", 0.6, now + 0.2) is True

    # Stronger signal from different room within window replaces original
    assert loop._should_accept_wakeword("escritorio", 0.9, now + 0.3) is True
    # Verify the last accepted room is now escritorio
    assert loop._last_wakeword_room == "escritorio"

    # After window expires relative to LAST accepted (escritorio at now+0.3),
    # a new detection from any room is accepted (now+0.3 + 0.6 = now+0.9)
    assert loop._should_accept_wakeword("bano", 0.4, now + 0.9) is True


@pytest.mark.asyncio
async def test_post_command_callback_called_for_each():
    """Post-command callback should fire for each concurrent command."""
    post_results = []

    async def mock_process(event: CommandEvent) -> dict:
        return {"room": event.room_id}

    async def mock_post(result: dict, event: CommandEvent):
        post_results.append(event.room_id)

    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(),
    )
    loop.on_command(mock_process)
    loop.on_post_command(mock_post)

    events = [
        CommandEvent(audio=np.zeros(8000, dtype=np.float32), room_id="cocina"),
        CommandEvent(audio=np.zeros(8000, dtype=np.float32), room_id="living"),
        CommandEvent(audio=np.zeros(8000, dtype=np.float32), room_id="bano"),
    ]

    await asyncio.gather(*[loop._dispatch_command(e) for e in events])

    assert len(post_results) == 3
    assert set(post_results) == {"cocina", "living", "bano"}


# =====================================================================
# Concurrent commands stress tests
# =====================================================================


@pytest.mark.asyncio
async def test_concurrent_commands_different_rooms_three_rooms():
    """Three rooms sending commands simultaneously all complete."""
    results = []

    async def mock_process(event: CommandEvent) -> dict:
        await asyncio.sleep(0.05)
        results.append(event.room_id)
        return {"room": event.room_id, "success": True}

    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(is_active=False),
    )
    loop.on_command(mock_process)

    rooms = ["cocina", "escritorio", "recamara"]
    events = [
        CommandEvent(
            audio=np.zeros(16000, dtype=np.float32),
            room_id=room,
            mic_device_index=i,
        )
        for i, room in enumerate(rooms)
    ]

    start = time.time()
    await asyncio.gather(*[loop._dispatch_command(e) for e in events])
    total = time.time() - start

    assert set(results) == set(rooms), f"Expected all 3 rooms, got {results}"
    assert len(results) == 3
    # All 3 parallel: should be ~50ms not ~150ms
    assert total < 0.12, f"3 commands took {total:.3f}s — should be parallel"


@pytest.mark.asyncio
async def test_cancellation_interrupts_slow_request():
    """User cancellation stops a slow LLM request in the priority queue."""
    queue = PriorityRequestQueue(auto_cancel_previous=True)

    # Enqueue a slow conversational request
    slow_req = queue.enqueue(
        user_id="user_a",
        text="Explica la teoria de cuerdas en detalle",
        priority=Priority.LOW,
    )
    assert slow_req.status == RequestStatus.PENDING

    # User sends a new request — previous should be auto-cancelled
    fast_req = queue.enqueue(
        user_id="user_a",
        text="Prende la luz",
        priority=Priority.HIGH,
    )

    assert slow_req.is_cancelled, "Previous request should be auto-cancelled"
    assert slow_req.cancellation_token.reason == CancellationReason.USER_NEW_REQUEST

    # The new request should be dequeue-able
    next_req = await queue.dequeue_async(timeout=0.5)
    assert next_req is not None
    assert next_req.request_id == fast_req.request_id


@pytest.mark.asyncio
async def test_urgent_command_priority():
    """CRITICAL command jumps ahead of LOW/MEDIUM requests."""
    queue = PriorityRequestQueue(auto_cancel_previous=False)

    # Fill queue with lower-priority requests from different users
    low_req = queue.enqueue(
        user_id="user_a",
        text="Cuentame sobre el universo",
        priority=Priority.LOW,
    )
    medium_req = queue.enqueue(
        user_id="user_b",
        text="Pon una rutina de buenas noches",
        priority=Priority.MEDIUM,
    )
    critical_req = queue.enqueue(
        user_id="user_c",
        text="Alarma de seguridad activada",
        priority=Priority.CRITICAL,
    )

    # CRITICAL should be dequeued first
    first = await queue.dequeue_async(timeout=0.5)
    assert first is not None
    assert first.request_id == critical_req.request_id
    assert first.priority == Priority.CRITICAL

    # Then MEDIUM
    second = await queue.dequeue_async(timeout=0.5)
    assert second is not None
    assert second.request_id == medium_req.request_id

    # Then LOW
    third = await queue.dequeue_async(timeout=0.5)
    assert third is not None
    assert third.request_id == low_req.request_id


@pytest.mark.asyncio
async def test_interrupt_for_priority_detects_urgent():
    """interrupt_for_priority returns True when a higher-priority request exists."""
    queue = PriorityRequestQueue(auto_cancel_previous=False)

    queue.enqueue(
        user_id="user_a",
        text="Alarma de incendio",
        priority=Priority.CRITICAL,
    )

    # A LOW-priority task should see there's something more urgent
    assert queue.interrupt_for_priority(Priority.LOW) is True
    assert queue.interrupt_for_priority(Priority.MEDIUM) is True
    assert queue.interrupt_for_priority(Priority.HIGH) is True
    # CRITICAL should not be interrupted by CRITICAL
    assert queue.interrupt_for_priority(Priority.CRITICAL) is False


# =====================================================================
# Shared list concurrency
# =====================================================================


@pytest.mark.asyncio
async def test_shared_list_concurrent_adds():
    """Two users adding items to the same shared list simultaneously succeed."""
    from src.lists.list_store import ListStore, OwnerType

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "lists.db")
        store = ListStore(db_path)
        await store.initialize()

        # Create a shared list
        shared_list = await store.create_list(
            name="compras", owner_type=OwnerType.SHARED, owner_id="user_a"
        )

        # Both users add items concurrently
        async def add_item(user_id: str, text: str):
            return await store.add_item(
                list_id=shared_list.id, text=text, added_by=user_id
            )

        results = await asyncio.gather(
            add_item("user_a", "leche"),
            add_item("user_b", "pan"),
        )

        assert len(results) == 2
        assert results[0].text == "leche"
        assert results[0].added_by == "user_a"
        assert results[1].text == "pan"
        assert results[1].added_by == "user_b"

        # Verify both items persisted
        items = await store.get_items(shared_list.id)
        item_texts = {item.text for item in items}
        assert item_texts == {"leche", "pan"}, f"Expected both items, got {item_texts}"

        await store.close()


@pytest.mark.asyncio
async def test_shared_list_concurrent_add_and_remove():
    """One user adds while another removes from the same list."""
    from src.lists.list_store import ListStore, OwnerType

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "lists.db")
        store = ListStore(db_path)
        await store.initialize()

        shared_list = await store.create_list(
            name="compras", owner_type=OwnerType.SHARED, owner_id="user_a"
        )
        existing_item = await store.add_item(
            list_id=shared_list.id, text="huevos", added_by="user_a"
        )

        async def add_new():
            return await store.add_item(
                list_id=shared_list.id, text="mantequilla", added_by="user_b"
            )

        async def remove_existing():
            await store.remove_item(existing_item.id)

        await asyncio.gather(add_new(), remove_existing())

        items = await store.get_items(shared_list.id)
        item_texts = {item.text for item in items}
        assert "huevos" not in item_texts, "Removed item should be gone"
        assert "mantequilla" in item_texts, "Added item should be present"

        await store.close()


# =====================================================================
# Missed reminder on arrival (presence wiring)
# =====================================================================


@pytest.mark.asyncio
async def test_missed_reminder_on_arrival():
    """When a user arrives home, overdue reminders are delivered via TTS."""
    from src.reminders.reminder_store import ReminderStore
    from src.reminders.reminder_scheduler import ReminderScheduler

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "reminders.db")
        store = ReminderStore(db_path)
        await store.initialize()

        # Create an overdue reminder (trigger_at in the past)
        past_time = time.time() - 3600  # 1 hour ago
        await store.create(
            user_id="user_a",
            text="tomar la medicina",
            trigger_at=past_time,
        )

        # Mock TTS
        tts_mock = AsyncMock()

        # Mock presence detector (no active wiring in this test)
        presence_mock = MagicMock()
        presence_mock.on_user_arrived = MagicMock()
        presence_mock.get_user_zone = MagicMock(return_value="cocina")

        # Mock HA client
        ha_mock = AsyncMock()

        scheduler = ReminderScheduler(
            store=store,
            tts=tts_mock,
            presence_detector=presence_mock,
            ha_client=ha_mock,
        )

        # Directly call missed_reminder_on_arrival
        delivered = await scheduler.missed_reminder_on_arrival(
            user_id="user_a", zone_id="cocina"
        )

        assert delivered == 1, f"Expected 1 delivered, got {delivered}"
        tts_mock.speak.assert_called_once()
        call_args = tts_mock.speak.call_args
        assert "tomar la medicina" in call_args[0][0]
        assert call_args[1]["zone_id"] == "cocina"

        await store.close()


@pytest.mark.asyncio
async def test_missed_reminder_on_arrival_no_overdue():
    """No TTS call when there are no overdue reminders."""
    from src.reminders.reminder_store import ReminderStore
    from src.reminders.reminder_scheduler import ReminderScheduler

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "reminders.db")
        store = ReminderStore(db_path)
        await store.initialize()

        # Create a future reminder (not overdue)
        future_time = time.time() + 3600
        await store.create(
            user_id="user_a",
            text="comprar pan",
            trigger_at=future_time,
        )

        tts_mock = AsyncMock()
        presence_mock = MagicMock()
        presence_mock.on_user_arrived = MagicMock()
        ha_mock = AsyncMock()

        scheduler = ReminderScheduler(
            store=store,
            tts=tts_mock,
            presence_detector=presence_mock,
            ha_client=ha_mock,
        )

        delivered = await scheduler.missed_reminder_on_arrival(
            user_id="user_a", zone_id="cocina"
        )

        assert delivered == 0
        tts_mock.speak.assert_not_called()

        await store.close()


@pytest.mark.asyncio
async def test_presence_detector_wiring_calls_on_user_arrived():
    """ReminderScheduler registers its callback with PresenceDetector."""
    from src.reminders.reminder_scheduler import ReminderScheduler

    presence_mock = MagicMock()
    presence_mock.on_user_arrived = MagicMock()

    store_mock = AsyncMock()
    tts_mock = AsyncMock()
    ha_mock = AsyncMock()

    ReminderScheduler(
        store=store_mock,
        tts=tts_mock,
        presence_detector=presence_mock,
        ha_client=ha_mock,
    )

    # Verify on_user_arrived was called during __init__
    presence_mock.on_user_arrived.assert_called_once()
    callback = presence_mock.on_user_arrived.call_args[0][0]
    assert callable(callback), "Callback should be callable"


@pytest.mark.asyncio
async def test_missed_reminder_marks_fired_after_delivery():
    """Delivered missed reminders are marked as fired so they don't repeat."""
    from src.reminders.reminder_store import ReminderStore, ReminderState
    from src.reminders.reminder_scheduler import ReminderScheduler

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "reminders.db")
        store = ReminderStore(db_path)
        await store.initialize()

        past_time = time.time() - 600
        reminder = await store.create(
            user_id="user_a",
            text="regar las plantas",
            trigger_at=past_time,
        )

        tts_mock = AsyncMock()
        presence_mock = MagicMock()
        presence_mock.on_user_arrived = MagicMock()
        presence_mock.get_user_zone = MagicMock(return_value="living")
        ha_mock = AsyncMock()

        scheduler = ReminderScheduler(
            store=store,
            tts=tts_mock,
            presence_detector=presence_mock,
            ha_client=ha_mock,
        )

        delivered = await scheduler.missed_reminder_on_arrival(
            user_id="user_a", zone_id="living"
        )
        assert delivered == 1

        # Verify the reminder was marked as fired
        updated = await store.get_by_id(reminder.id)
        assert updated.state == ReminderState.FIRED

        # A second arrival should not re-deliver
        delivered2 = await scheduler.missed_reminder_on_arrival(
            user_id="user_a", zone_id="living"
        )
        assert delivered2 == 0

        await store.close()
