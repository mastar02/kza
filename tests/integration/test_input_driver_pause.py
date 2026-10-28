"""Actual native driver lifetimes, late cancellation and reusable input fencing."""

import asyncio
import threading

import numpy as np
import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.producer_context import PRODUCER_CONTEXT, ProducerContext, producer_context
from tests.integration.test_capture_producer_privacy import graph, until
from tests.integration.test_input_callback_retirement import HeldInput

pytestmark = pytest.mark.asyncio


class Driver:
    def __init__(self, stage=None, *, fail=None):
        self.stage, self.fail = stage, fail
        self.entered, self.release = threading.Event(), threading.Event()
        self.started = self.closed = self.close_attempts = 0
        self.callback = None
        self.contexts = []

    def held(self, stage):
        self.contexts.append(PRODUCER_CONTEXT.get())
        if self.stage == stage:
            self.entered.set()
            assert self.release.wait(4)
        if self.fail == stage:
            self.fail = None
            raise RuntimeError('synthetic_driver_failure')

    @property
    def samplerate(self):
        self.held('configure')
        return 16000

    channels, dtype, device = 1, 'float32', 0

    def start(self):
        self.started += 1
        self.held('start')

    def stop(self):
        self.held('stop')

    def close(self):
        self.close_attempts += 1
        self.held('close')
        self.closed += 1


def portaudio(monkeypatch, drivers):
    sequence = iter(drivers)
    def factory(**kwargs):
        driver = next(sequence)
        driver.callback = kwargs['callback']
        driver.held('construct')
        return driver
    monkeypatch.setattr('src.pipeline.multi_room_audio_loop.sd.query_devices',
                        lambda index: {'max_input_channels': 1, 'name': 'synthetic'})
    monkeypatch.setattr('src.pipeline.multi_room_audio_loop.sd.InputStream', factory)


@pytest.mark.parametrize('stage', ['construct', 'configure', 'start'])
async def test_cancelled_open_stays_owned_through_actual_native_exit_and_close(tmp_path, monkeypatch, stage):
    driver = Driver(stage)
    portaudio(monkeypatch, [driver])
    async with graph(tmp_path, controllers=False) as g:
        opening = asyncio.create_task(g.loop._try_reopen_once(g.loop.room_streams['synthetic-a']))
        closing = None
        try:
            await until(driver.entered.is_set)
            opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)
            closing = g.loop.begin_close()
            await asyncio.sleep(.02)
            assert not closing.done(), 'native opening still owns the eventual driver'
        finally:
            driver.release.set()
            await asyncio.gather(opening, return_exceptions=True)
            if closing is not None:
                await closing
        assert driver.closed == 1 and not g.loop._streams
        assert driver.started == (1 if stage == 'start' else 0)
        assert not g.loop._microphone_captures['synthetic-a'].observation.opened


@pytest.mark.parametrize('stage', ['configure', 'start'])
async def test_failed_configuration_or_start_closes_the_constructed_handle(tmp_path, monkeypatch, stage):
    driver = Driver(fail=stage)
    portaudio(monkeypatch, [driver])
    async with graph(tmp_path, controllers=False) as g:
        assert not await g.loop._try_reopen_once(g.loop.room_streams['synthetic-a'])
        assert driver.closed == 1 and not g.loop._streams


async def test_pause_closes_drivers_blocks_watchdog_and_reopens_only_fresh_handles(tmp_path, monkeypatch):
    first, second = Driver(), Driver()
    portaudio(monkeypatch, [first, second])
    async with graph(tmp_path, controllers=False) as g:
        rs = g.loop.room_streams['synthetic-a']
        assert await g.loop._try_reopen_once(rs)
        first.callback(np.ones((1280, 1), dtype=np.float32), 1280, None, None)
        assert g.loop._capture_rooms[rs.room_id].resource_status()['queued'] == 1
        pause = g.loop.begin_pause_inputs()
        await pause
        assert first.closed == 1 and not g.loop._streams
        assert g.loop._inputs.resource_status()['pause_complete']
        await g.loop._recover_streams([rs.room_id])
        g.loop._schedule_reopen(rs)
        assert not await g.loop._try_reopen_once(rs)
        assert second.started == 0 and not g.loop._reopen_tasks
        first.callback(np.ones((1280, 1), dtype=np.float32), 1280, None, None)
        assert g.loop._capture_rooms[rs.room_id].resource_status()['queued'] == 0
        g.clock.now += 1
        g.loop.resume_inputs()
        assert await g.loop._try_reopen_once(rs)
        assert second.started == 1 and g.loop._streams[rs.room_id] is second
        second.callback(np.ones((1280, 1), dtype=np.float32), 1280, None, None)
        assert g.loop._capture_rooms[rs.room_id].resource_status()['queued'] == 1
        first.callback(np.ones((1280, 1), dtype=np.float32), 1280, None, None)
        assert g.loop._capture_rooms[rs.room_id].resource_status()['queued'] == 1


async def test_pause_retains_failed_native_close_and_refuses_premature_resume(tmp_path, monkeypatch):
    driver = Driver(fail='close')
    portaudio(monkeypatch, [driver])
    async with graph(tmp_path, controllers=False) as g:
        assert await g.loop._try_reopen_once(g.loop.room_streams['synthetic-a'])
        pause = g.loop.begin_pause_inputs()
        try:
            await until(lambda: driver.close_attempts == 1)
            driver.stage = 'close'
            await until(driver.entered.is_set)
            assert not pause.done() and driver.closed == 0
            assert g.loop._inputs.resource_status()['operations'] > 0
            with pytest.raises(RuntimeError, match='still_owned'):
                g.loop.resume_inputs()
        finally:
            driver.release.set()
            await pause
        assert driver.closed == 1 and driver.close_attempts == 2


async def test_driver_pause_waits_for_the_actual_callback_after_native_close(tmp_path, monkeypatch):
    driver, audio = Driver(), HeldInput('any')
    portaudio(monkeypatch, [driver])
    async with graph(tmp_path, controllers=False) as g:
        assert await g.loop._try_reopen_once(g.loop.room_streams['synthetic-a'])
        pushing = asyncio.create_task(asyncio.to_thread(driver.callback, audio, 1280, None, None))
        pause = None
        try:
            await until(audio.entered.is_set)
            pause = g.loop.begin_pause_inputs()
            await until(lambda: driver.closed == 1)
            assert not pause.done() and g.loop._inputs.resource_status()['callbacks'] == 1
            with pytest.raises(RuntimeError, match='still_owned'):
                g.loop.resume_inputs()
        finally:
            audio.release.set()
            await pushing
            if pause is not None:
                await pause
        assert g.loop._inputs.resource_status()['callbacks'] == 0


async def test_catalog_work_remains_owned_after_observer_cancel_and_blocks_pause(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def native_catalog():
        entered.set()
        assert release.wait(4)
    async with graph(tmp_path, controllers=False) as g:
        observing = asyncio.create_task(g.loop._inputs.catalog_call(native_catalog))
        pause = None
        try:
            await until(entered.is_set)
            observing.cancel()
            await asyncio.gather(observing, return_exceptions=True)
            pause = g.loop.begin_pause_inputs()
            await asyncio.sleep(.02)
            assert not pause.done() and g.loop._inputs.resource_status()['operations'] > 0
        finally:
            release.set()
            if pause is not None:
                await pause


async def test_observer_cancel_cannot_lose_a_successfully_opened_room_or_duplicate_its_open(tmp_path, monkeypatch):
    driver = Driver('start')
    portaudio(monkeypatch, [driver])
    async with graph(tmp_path, controllers=False) as g:
        rs = g.loop.room_streams['synthetic-a']
        first = asyncio.create_task(g.loop._try_reopen_once(rs))
        try:
            await until(driver.entered.is_set)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            second = asyncio.create_task(g.loop._try_reopen_once(rs))
        finally:
            driver.release.set()
        assert await second
        assert driver.started == 1 and g.loop._streams[rs.room_id] is driver
        assert rs.opened_ts > 0
        await g.loop.begin_pause_inputs()
        assert driver.closed == 1


@pytest.mark.parametrize('stage', ['construct', 'close'])
async def test_driver_native_work_keeps_the_privacy_lease_after_bounded_stop(tmp_path, monkeypatch, stage):
    driver = Driver(stage)
    portaudio(monkeypatch, [driver])
    async with graph(tmp_path, controllers=False) as g:
        opening = asyncio.create_task(g.loop._try_reopen_once(g.loop.room_streams['synthetic-a']))
        try:
            if stage == 'close':
                assert await opening
            else:
                await until(driver.entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await g.loop.stop()
            await until(driver.entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
            assert not g.loop.begin_close().done()
        finally:
            driver.release.set()
            await asyncio.gather(opening, return_exceptions=True)
            await g.privacy.begin_close()
        assert driver.closed == 1
        assert not g.privacy.resource_status()['writer_lease']
        assert not g.loop.snapshot_operations()


async def test_native_driver_and_catalog_calls_do_not_inherit_private_request_context(tmp_path, monkeypatch):
    driver, catalog_contexts = Driver(), []
    portaudio(monkeypatch, [driver])
    async with graph(tmp_path, controllers=False) as g:
        with producer_context('synthetic-private-request'):
            assert PRODUCER_CONTEXT.get().room_id == 'synthetic-private-request'
            assert await g.loop._try_reopen_once(g.loop.room_streams['synthetic-a'])
            await g.loop._inputs.catalog_call(lambda: catalog_contexts.append(PRODUCER_CONTEXT.get()))
            await g.loop.begin_pause_inputs()
        assert driver.contexts and all(context == ProducerContext() for context in driver.contexts)
        assert catalog_contexts == [ProducerContext()]
