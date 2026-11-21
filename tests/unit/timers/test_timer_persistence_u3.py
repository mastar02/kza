"""T030-U3 persistence on the real SCHEMA-V2 reminder database."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import threading

import pytest

from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.reminders.reminder_store import ReminderActionStatus, ReminderStore
from src.timers.named_timers import NamedTimerManager, TimerState
from src.timers.timer_store import RETENTION_S, TimerConflict, TimerStore
from tests.unit.timers.test_timer_runtime import VirtualClock


@asynccontextmanager
async def open_timers(path, clock, output=None):
    reminders = ReminderStore(str(path))
    await reminders.initialize()
    store = TimerStore(reminders)
    manager = NamedTimerManager(tts_callback=output, clock=clock, wall_clock=clock.wall, sleep=clock.sleep)
    manager.bind_store(store)
    try:
        await manager.start()
        yield manager, store, reminders
    finally:
        await manager.stop()
        await reminders.close()


async def create(manager, seconds=5):
    return await manager.commit_command(lambda view: view.create_timer("pasta", seconds, zone_id="zone_cocina"))


async def test_recovers_running_deadline_and_frozen_paused_time(tmp_path):
    db = tmp_path / "reminders.db"
    clock = VirtualClock()
    async with open_timers(db, clock) as (manager, _, _):
        running = await create(manager, 10)
        paused = await create(manager, 20)
        await clock.advance(2)
        assert await manager.commit_command(lambda view: view.pause_timer(paused.timer_id))

    await clock.advance(5)
    spoken = asyncio.Event()
    output = []

    async def speak(text, zone):
        output.append((text, zone))
        spoken.set()

    async with open_timers(db, clock, speak) as (manager, store, _):
        recovered = manager.get_timer(running.timer_id)
        frozen = manager.get_timer(paused.timer_id)
        assert recovered.ends_at == running.ends_at
        assert recovered.remaining_seconds == 3
        assert frozen.state is TimerState.PAUSED and frozen.remaining_seconds == 18
        assert manager.get_status()["durable_active"] == 2
        await clock.advance(2.5)
        assert not spoken.is_set()
        await clock.advance(0.5)
        await asyncio.wait_for(spoken.wait(), 2)
        assert output == [("¡Terminó el temporizador de pasta!", "zone_cocina")]
        await asyncio.gather(*manager._expiry_tasks.values())
        assert (await store.get(running.timer_id)).state is TimerState.COMPLETED


async def test_timer_rings_on_time_even_if_sqlite_update_fails_at_expiry(tmp_path, caplog):
    clock = VirtualClock()
    output = []
    retry_sleep = asyncio.Queue()

    async def speak(text, zone):
        output.append((clock(), text, zone))

    async with open_timers(tmp_path / "reminders.db", clock, speak) as (manager, store, reminders):
        async def observe_sleep(delay):
            if asyncio.current_task() in manager._expiry_tasks.values():
                retry_sleep.put_nowait(delay)
            await clock.sleep(delay)

        manager._sleep = observe_sleep
        started = clock()
        timer = await create(manager)

        def reject_update(db):
            db.execute("CREATE TRIGGER reject_expiry BEFORE UPDATE ON timers "
                       "BEGIN SELECT RAISE(ABORT, 'synthetic storage failure'); END")

        await reminders._work.write(reject_update)
        await clock.advance(4.5)
        assert output == []
        await clock.advance(0.5)
        assert output == [(started + 5, "¡Terminó el temporizador de pasta!", "zone_cocina")]
        assert timer.state is TimerState.COMPLETED
        assert manager.get_status()["persistence"] == "uncertain"

        # Drive six failed retries, including more than 50 ordinary ticks.
        for delay in (1, 2, 4, 8, 16, 30):
            assert await asyncio.wait_for(retry_sleep.get(), 2) == delay
            await clock.advance(delay)
        assert await asyncio.wait_for(retry_sleep.get(), 2) == 30
        assert (await store.get(timer.timer_id)).state is TimerState.RUNNING
        assert manager.get_status()["persistence"] == "uncertain"
        warnings = [record for record in caplog.records
                    if record.name == "src.timers.named_timers" and record.levelname == "WARNING"]
        assert [record.getMessage() for record in warnings] == ["Timer store expiry unverified: IntegrityError"]

        def restore_updates(db):
            db.execute("DROP TRIGGER reject_expiry")

        await reminders._work.write(restore_updates)
        pending = manager._expiry_tasks[timer.timer_id]
        await clock.advance(30)
        await asyncio.wait_for(pending, 2)
        row = await store.get(timer.timer_id)
        assert (row.state, row.revision, row.remaining_s) == (TimerState.COMPLETED, 2, 0)
        assert manager.get_status()["persistence"] == "committed"
        assert [when for when, _, _ in output] == [started + 5, started + 7, started + 9]
        assert await reminders.get_notice_rows() == []


async def test_recover_failure_leaves_timer_manager_uncertain_without_raising(tmp_path, caplog):
    db = tmp_path / "reminders.db"
    clock = VirtualClock()
    async with open_timers(db, clock) as (manager, _, reminders):
        timer = await create(manager)

        def corrupt_expiry(connection):
            connection.execute("UPDATE timers SET ends_at = NULL WHERE id = ?", (timer.timer_id,))

        await reminders._work.write(corrupt_expiry)

    async with open_timers(db, clock) as (manager, _, _):
        assert manager.get_status()["persistence"] == "uncertain"
        assert not manager.running and manager.get_all_timers() == []
        assert "Timer recovery failed: TypeError" in caplog.messages


async def test_stop_cancels_ring_and_drains_admitted_expiry_write(tmp_path):
    clock = VirtualClock()
    played = asyncio.Event()
    admitted = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    async def speak(text, zone):
        played.set()

    def hold_write():
        loop.call_soon_threadsafe(admitted.set)
        if not release.wait(3):
            raise RuntimeError("test did not release expiry write")

    async with open_timers(tmp_path / "reminders.db", clock, speak) as (manager, store, reminders):
        timer = await create(manager)

        def install_hold(db):
            db.create_function("hold_expiry_write", 0, hold_write)
            db.execute("CREATE TRIGGER hold_expiry BEFORE UPDATE ON timers BEGIN SELECT hold_expiry_write(); END")

        await reminders._work.write(install_hold)
        stopping = None
        try:
            await clock.advance(5)
            await asyncio.wait_for(admitted.wait(), 2)
            assert played.is_set() and timer.state is TimerState.COMPLETED
            expiry = manager._expiry_tasks[timer.timer_id]
            stopping = asyncio.create_task(manager.stop())
            await clock.settle()
            assert not stopping.done() and not expiry.done()
            assert manager._update_task.done() and manager._update_task.exception() is None
            assert manager.ringing_timers() == []
        finally:
            release.set()
            if stopping is not None:
                await asyncio.wait_for(stopping, 2)
        assert expiry.done() and expiry.exception() is None
        assert manager._expiry_tasks == {}
        row = await store.get(timer.timer_id)
        assert (row.state, row.revision) == (TimerState.COMPLETED, 2)


async def test_missed_expiry_records_one_pending_notice_and_delivers_once_without_ring(tmp_path):
    db = tmp_path / "reminders.db"
    clock = VirtualClock()
    async with open_timers(db, clock) as (manager, _, _):
        timer = await create(manager)

    await clock.advance(5)
    for _ in range(2):
        async with open_timers(db, clock) as (manager, store, reminders):
            assert manager.get_all_timers() == [] and manager.ringing_timers() == []
            assert manager.get_status()["pending_notices"] == 1
            record = await store.get(timer.timer_id)
            assert (record.state, record.revision) == (TimerState.COMPLETED, 2)
            (notice,) = await reminders.get_notice_rows()
            assert notice.request_id.startswith("notice:")
            assert notice.status is ReminderActionStatus.FAILED
            assert json.loads(notice.body) == {
                "kind": "timer_expired", "timer_id": timer.timer_id, "room": "zone_cocina", "trigger_at": timer.ends_at,
            }

    output = []

    async def speak(text, zone):
        output.append((text, zone))
        return ResponseResult("notice", GenerationState.COMPLETE, DeliveryState.COMPLETE, output_started=True)

    async with open_timers(db, clock, speak) as (manager, _, reminders):
        async with asyncio.timeout(2):
            while manager.get_status()["pending_notices"]:
                await asyncio.sleep(0.001)
        (notice,) = await reminders.get_notice_rows()
        assert notice.status is ReminderActionStatus.CONFIRMED
        assert manager.ringing_timers() == []
    async with open_timers(db, clock, speak) as (_, _, reminders):
        (notice,) = await reminders.get_notice_rows()
        assert notice.status is ReminderActionStatus.CONFIRMED
    assert output == [("El temporizador de pasta terminó mientras el sistema estaba apagado.", "zone_cocina")]


async def test_two_writers_cancelling_and_expiring_converge_on_one_terminal_revision(tmp_path):
    db = tmp_path / "reminders.db"
    clock = VirtualClock()
    async with open_timers(db, clock) as (manager, store, _):
        timer = await create(manager)
        record = await store.get(timer.timer_id)
        peer = ReminderStore(str(db))
        await peer.initialize()
        try:
            other = TimerStore(peer)
            cancelled = replace(record, state=TimerState.CANCELLED, ends_at=None, remaining_s=0)
            completed = replace(record, state=TimerState.COMPLETED, remaining_s=0)
            results = await asyncio.gather(store.save([cancelled]), other.save([completed]), return_exceptions=True)
            assert sum(result is None for result in results) == 1
            assert sum(isinstance(result, TimerConflict) for result in results) == 1
            persisted = await store.get(timer.timer_id)
            expected = TimerState.CANCELLED if results[0] is None else TimerState.COMPLETED
            assert (persisted.state, persisted.revision, persisted.remaining_s) == (expected, 2, 0)
            assert persisted.ends_at == (None if expected is TimerState.CANCELLED else record.ends_at)
        finally:
            await peer.close()


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_manager_serializes_cancel_and_due_tick(tmp_path, cancel_first):
    clock = VirtualClock()
    output = []

    async def speak(text, zone):
        output.append((text, zone))

    async with open_timers(tmp_path / "reminders.db", clock, speak) as (manager, store, reminders):
        timer = await create(manager)
        await clock.settle()  # Let the normal loop sleep before manually racing its next tick.
        clock.now += 5
        cancel = manager.commit_command(lambda view: view.cancel_timer(timer.timer_id))
        expiry = manager._tick_durable()
        order = (cancel, expiry) if cancel_first else (expiry, cancel)
        results = await asyncio.gather(*order)
        assert results[order.index(cancel)] is cancel_first
        expected = TimerState.CANCELLED if cancel_first else TimerState.COMPLETED
        assert timer.state is expected
        await asyncio.gather(*manager._expiry_tasks.values())
        row = await store.get(timer.timer_id)
        assert (row.state, row.revision) == (expected, 2)
        assert len(output) == (0 if cancel_first else 1)
        assert await reminders.get_notice_rows() == []


async def test_retention_keeps_an_old_pending_notice_and_its_timer_until_settled(tmp_path):
    db = tmp_path / "reminders.db"
    clock = VirtualClock()
    async with open_timers(db, clock) as (manager, _, _):
        cancelled = await create(manager)
        assert await manager.commit_command(lambda view: view.cancel_timer(cancelled.timer_id))
        missed = await create(manager)

    await clock.advance(5)
    async with open_timers(db, clock):
        pass
    await clock.advance(RETENTION_S - 5)
    async with open_timers(db, clock) as (_, store, _):
        assert await store.get(cancelled.timer_id) is not None  # Exactly 48 h.
    await clock.advance(5)
    async with open_timers(db, clock) as (_, store, reminders):
        assert await store.get(cancelled.timer_id) is None
        assert await store.get(missed.timer_id) is not None
        clock.now += 1
        assert await reminders.purge_notice_rows(clock.wall()) == 0
        assert await store.cleanup(clock.wall()) == []
        (pending,) = await store.pending_notices()
        assert pending.id == missed.timer_id
        assert await store.claim_notice(missed.timer_id)
        assert await store.cleanup(clock.wall()) == []
        await store.finish_notice(missed.timer_id, ReminderActionStatus.CONFIRMED, None)
        assert await store.cleanup(clock.wall()) == [missed.timer_id]
        assert await store.get(missed.timer_id) is None
        assert await reminders.get_notice_rows() == []
