"""Expand reusable pause with real native actors and synthetic transport leaves.

These candidate-contract cases do not relabel older terminal-close tests as RED.
"""

import asyncio
from datetime import datetime, timezone
import json
import struct
import threading
import time

import pytest

from src.audio.direction_observer import DirectionObservationConfig, DirectionObserver
from src.audio.microphone_status import DirectionState, UsbInventory
from src.conversation.follow_up_mode import FollowUpMode
from tests.integration.test_audio_health_publication_owner import BeforeReplace, close, loop_for
from tests.integration.test_xvf_native_retirement import cleanup, make_loop, until


pytestmark = pytest.mark.asyncio


def native_loop(monkeypatch):
    loop, controller, device, discovery = make_loop(monkeypatch)
    loop.follow_up = FollowUpMode(follow_up_window=8.0)
    return loop, controller, device, discovery


async def cancel_observer(task):
    task.cancel()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)


async def pause(loop):
    loop.pause_admission()
    observer = asyncio.create_task(loop.drain_pause())
    await until(lambda: loop._pausing is not None)
    return loop._pausing, observer


async def finish(loop, controller, device, discovery, *, gates=(), observers=(),
                 directions=(), expected_failures=()):
    """Release only fixture gates, then observe the real terminal owners."""
    for gate in gates:
        gate.set()
    results = await asyncio.gather(
        cleanup(loop, controller, device, discovery, threading.Event(), None),
        return_exceptions=True,
    )
    for observer in directions:
        await asyncio.to_thread(observer.wait_stopped)
        assert observer._thread is None or not observer._thread.is_alive()
    if observers:
        await asyncio.gather(*observers, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not any(result is error for error in expected_failures):
            raise result


async def test_late_admitted_xvf_start_is_sealed_again_and_real_poller_is_joined(monkeypatch):
    loop, controller, device, discovery = native_loop(monkeypatch)
    discovery.release.set()
    entered, release = threading.Event(), threading.Event()
    returned, stopped_after_start = threading.Event(), threading.Event()
    started, threads = [], []
    original_stop = controller.request_stop

    def before_start():
        entered.set()
        assert release.wait(5), 'fixture did not release admitted start'
        started.append(controller.start())
        threads.append(controller._thread)
        returned.set()

    def request_stop():
        original_stop()
        if returned.is_set():
            stopped_after_start.set()

    monkeypatch.setattr(controller, 'request_stop', request_stop)
    native = asyncio.create_task(loop._xvf_native.run(before_start))
    pausing = None
    try:
        await until(entered.is_set)
        await cancel_observer(native)
        owner, pausing = await pause(loop)
        assert not owner.done() and controller._thread is None
        release.set()
        await until(device.read_entered.is_set)
        await until(stopped_after_start.is_set)
        assert returned.is_set() and started == [True]
        assert len(threads) == 1 and threads[0].is_alive()
        assert not owner.done() and not owner.cancelled()
        await cancel_observer(pausing)
        assert loop._pausing is owner and not owner.done()

        device.read_release.set()
        await asyncio.wait_for(asyncio.shield(loop.drain_pause()), 3)
        assert owner.done() and owner.exception() is None
        assert not threads[0].is_alive() and controller._thread is None
        assert controller.peak_since(0) is None and device.energy_reads == 1
        assert not loop._xvf_native.status()['closed']
    finally:
        await finish(loop, controller, device, discovery,
                     gates=(release, device.read_release), observers=(native,)
                     + ((pausing,) if pausing is not None else ()))


async def test_direction_pause_waits_for_real_read_and_native_thread_postlude(monkeypatch):
    import src.audio.direction_observer as direction_module

    loop, controller, device, discovery = native_loop(monkeypatch)
    read_entered, read_release = threading.Event(), threading.Event()
    read_returned = threading.Event()
    postlude_entered, postlude_release = threading.Event(), threading.Event()

    class DirectionUSB:
        bus = 5
        port_numbers = (5, 4)

        def ctrl_transfer(self, _bm, _request, value, _index, length, _timeout):
            if value == 0x80:
                assert length == 4
                return [0, 1, 2, 3]
            assert value == (0x80 | 18) and length == 5
            read_entered.set()
            try:
                assert read_release.wait(5), 'fixture did not release actual DOA read'
                return [0] + list(struct.pack('<HH', 90, 1))
            finally:
                read_returned.set()

    class PostludeThread(threading.Thread):
        def run(self):
            try:
                super().run()
            finally:
                postlude_entered.set()
                assert postlude_release.wait(5), 'fixture did not release owned Thread postlude'

    discovery.device = DirectionUSB()
    discovery.release.set()
    monkeypatch.setattr(direction_module, 'Thread', PostludeThread)
    observer = DirectionObserver(
        controller=controller, room_id='synthetic', usb_port='5-5.4', mic_id='synthetic',
        config=DirectionObservationConfig(enabled=True), calibration=None,
        monotonic=time.monotonic, utc_now=lambda: datetime.now(timezone.utc),
    )
    loop._direction_started = True
    loop._direction_observers['synthetic'] = observer
    loop._direction_keys['synthetic'] = ('synthetic',)
    pausing = None
    try:
        assert observer.start()
        await until(read_entered.is_set)
        thread = observer._thread
        owner, pausing = await pause(loop)
        assert thread.is_alive() and not owner.done()
        assert observer.snapshot().state is DirectionState.DISABLED
        read_release.set()
        await until(postlude_entered.is_set)
        assert read_returned.is_set() and thread.is_alive()
        assert not owner.done() and observer.snapshot().state is DirectionState.DISABLED

        postlude_release.set()
        await asyncio.wait_for(asyncio.shield(pausing), 3)
        assert owner.done() and owner.exception() is None and not thread.is_alive()
        assert observer not in loop._direction_retired
        assert observer not in loop._direction_observers.values()
    finally:
        await finish(loop, controller, device, discovery,
                     gates=(read_release, postlude_release),
                     observers=((pausing,) if pausing is not None else ()), directions=(observer,))


async def test_pause_retains_cancelled_sysfs_observer_until_same_native_read_returns(monkeypatch):
    loop, controller, device, discovery = native_loop(monkeypatch)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    calls, threads, publications = [], [], []
    original_publish = loop._publish_microphone_inventory

    def read_inventory(_root):
        calls.append(True)
        threads.append(threading.current_thread())
        entered.set()
        try:
            assert release.wait(5), 'fixture did not release actual sysfs reader'
            return UsbInventory(time.time(), time.monotonic(), True)
        finally:
            returned.set()

    def publish(inventory):
        publications.append(inventory)
        return original_publish(inventory)

    loop._microphone_reader = read_inventory
    monkeypatch.setattr(loop, '_publish_microphone_inventory', publish)
    refreshing = asyncio.create_task(loop._refresh_microphone_inventory())
    pausing = second = None
    try:
        await until(entered.is_set)
        worker = loop._microphone_read_task
        assert worker is not None and not worker.done()
        await cancel_observer(refreshing)
        owner, pausing = await pause(loop)
        assert not worker.done() and not owner.done() and calls == [True]
        assert threads[0] is not threading.current_thread()

        release.set()
        await asyncio.wait_for(asyncio.shield(pausing), 3)
        assert returned.is_set() and worker.done() and not worker.cancelled()
        assert worker.exception() is None and owner.exception() is None
        assert calls == [True] and publications == [] and not loop._direction_observers

        second = asyncio.create_task(loop._refresh_microphone_inventory())
        await asyncio.wait_for(asyncio.shield(second), 3)
        assert second.done() and not second.cancelled() and second.exception() is None
        assert calls == [True] and publications == []
    finally:
        release.set()
        if second is not None and not second.done():
            second.cancel()
        observers = (refreshing,) + tuple(task for task in (pausing, second) if task is not None)
        await finish(loop, controller, device, discovery, gates=(release,), observers=observers)


async def test_pause_retains_actual_audio_health_replace_after_wrapper_cancel(tmp_path, monkeypatch):
    path = tmp_path / 'audio-health.json'
    gate = BeforeReplace(monkeypatch, path)
    loop, _room = loop_for(path)
    pausing = None
    try:
        loop._publish_audio_health()
        await until(gate.entered.is_set)
        wrapper = loop._audio_health_task
        native = tuple(loop._audio_health_work._tasks)
        assert len(native) == 1 and not native[0].done()
        await cancel_observer(wrapper)
        owner, pausing = await pause(loop)
        for _ in range(3):
            loop._publish_audio_health()
        assert gate.calls == 1 and not native[0].done() and not owner.done()
        assert not path.exists() and gate.thread_ids[0] != threading.get_ident()

        gate.release.set()
        await asyncio.wait_for(asyncio.shield(pausing), 3)
        assert native[0].done() and native[0].exception() is None
        assert owner.exception() is None and path.exists() and gate.calls == 1
        assert isinstance(json.loads(path.read_text()), dict)
        assert not loop._audio_health_work.status()['closed']
        previous = loop._audio_health_task
        loop._publish_audio_health()
        assert loop._audio_health_task is previous
        assert not loop._audio_health_work._tasks and gate.calls == 1
    finally:
        gate.release.set()
        if pausing is not None:
            await asyncio.gather(pausing, return_exceptions=True)
        await close(loop)


async def test_paused_start_and_run_do_not_start_auxiliaries_and_resume_is_reusable(monkeypatch):
    loop, controller, device, discovery = native_loop(monkeypatch)
    discovery.release.set()
    calls = []
    capture_entered = asyncio.Event()
    original_capture = loop._run_capture

    def read_inventory(_root):
        calls.append(threading.current_thread())
        return UsbInventory(time.time(), time.monotonic(), True)

    async def capture():
        capture_entered.set()
        await original_capture()

    loop._microphone_reader = read_inventory
    monkeypatch.setattr(loop, '_run_capture', capture)
    running = resuming = pausing = None
    try:
        loop.pause_admission()
        await loop.start()
        await loop.drain_pause()
        assert discovery.calls == 0 and device.energy_reads == 0
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(capture_entered.wait(), 3)
        assert loop._running and not running.done()
        assert calls == [] and loop._microphone_read_task is None
        assert loop._microphone_telemetry_task is None
        assert controller._thread is None and not loop._direction_observers

        loop.resume_admission()
        readiness = loop._auxiliary_resume_task
        resuming = asyncio.create_task(loop.drain_resume())
        await until(device.read_entered.is_set)
        await asyncio.wait_for(asyncio.shield(resuming), 3)
        assert readiness is loop._auxiliary_resume_task and readiness.done()
        assert readiness.exception() is None and controller._thread.is_alive()
        assert not loop._closed and not loop._xvf_native.status()['closed']
        await until(lambda: bool(calls))
        thread = controller._thread

        owner, pausing = await pause(loop)
        assert thread.is_alive() and not owner.done()
        device.read_release.set()
        await asyncio.wait_for(asyncio.shield(pausing), 3)
        assert not thread.is_alive() and owner.exception() is None
        assert not running.done() and controller._thread is None
    finally:
        await finish(loop, controller, device, discovery, gates=(device.read_release,),
                     observers=tuple(task for task in (running, resuming, pausing) if task is not None))


async def test_failed_real_xvf_wait_keeps_pause_failed_and_cannot_resume(monkeypatch):
    loop, controller, device, discovery = native_loop(monkeypatch)
    discovery.release.set()
    entered = threading.Event()
    failure = RuntimeError('synthetic auxiliary join observer failed')
    original_wait = controller.wait_stopped
    pausing = observed_failure = None

    def wait_then_fail():
        entered.set()
        original_wait()
        raise failure

    try:
        assert await asyncio.to_thread(controller.start)
        await until(device.read_entered.is_set)
        thread = controller._thread
        with monkeypatch.context() as fault:
            fault.setattr(controller, 'wait_stopped', wait_then_fail)
            owner, pausing = await pause(loop)
            await until(entered.is_set)
            assert thread.is_alive() and not owner.done()
            device.read_release.set()
            with pytest.raises(RuntimeError) as observed:
                await asyncio.wait_for(asyncio.shield(pausing), 3)
            observed_failure = observed.value
            assert owner.done() and owner.exception() is observed_failure
            assert not thread.is_alive() and controller._thread is None
            with pytest.raises(RuntimeError):
                loop.resume_admission()
            assert loop._paused and loop._pausing is owner
            assert device.energy_reads == 1 and controller.peak_since(0) is None
    finally:
        await finish(loop, controller, device, discovery, gates=(device.read_release,),
                     observers=((pausing,) if pausing is not None else ()),
                     expected_failures=(failure, observed_failure))


async def test_first_start_after_completed_prestart_resume_runs_one_real_poller(monkeypatch):
    loop, controller, device, discovery = native_loop(monkeypatch)
    discovery.release.set()
    starting = pausing = None
    try:
        loop.pause_admission()
        await loop.drain_pause()
        loop.resume_admission()
        prestart = loop._auxiliary_resume_task
        await loop.drain_resume()
        assert prestart.done() and prestart.exception() is None
        assert controller._thread is None and discovery.calls == 0
        assert device.energy_reads == 0

        starting = asyncio.create_task(loop.start())
        await until(device.read_entered.is_set)
        await asyncio.wait_for(asyncio.shield(starting), 3)
        readiness = loop._auxiliary_resume_task
        assert readiness is not prestart and readiness.done()
        assert readiness.exception() is None
        thread = controller._thread
        assert thread is not None and thread.is_alive()
        assert discovery.calls == 1 and device.energy_reads == 1

        await loop.start()
        assert loop._auxiliary_resume_task is readiness
        assert controller._thread is thread
        assert discovery.calls == 1 and device.energy_reads == 1
        owner, pausing = await pause(loop)
        assert thread.is_alive() and not owner.done()
        device.read_release.set()
        await asyncio.wait_for(asyncio.shield(pausing), 3)
        assert not thread.is_alive() and controller._thread is None
        assert owner.exception() is None and controller.peak_since(0) is None
        assert not loop._xvf_native.status()['closed']
    finally:
        await finish(loop, controller, device, discovery, gates=(device.read_release,),
                     observers=tuple(task for task in (starting, pausing) if task is not None))
