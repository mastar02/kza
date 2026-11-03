"""Real watchdog and native-owner recovery with synthetic PortAudio drivers."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.owned_work import WorkCloseTimeout
import src.monitoring.audio_health as audio_health
import src.pipeline.multi_room_audio_loop as audio_loop
from tests.integration.test_capture_producer_privacy import graph, until
from tests.integration.test_input_driver_pause import Driver, portaudio


pytestmark = pytest.mark.asyncio
A = 'synthetic-a'
B = 'synthetic-b'


class Catalog:
    def __init__(self, held=False):
        self.held = held
        self.entered, self.release = threading.Event(), threading.Event()
        self.terminated = self.initialized = 0

    def terminate(self):
        self.terminated += 1
        if self.held:
            self.entered.set()
            assert self.release.wait(4)

    def initialize(self):
        self.initialized += 1


class Publications:
    def __init__(self, monkeypatch):
        self.values = deque(maxlen=256)
        self.lock = threading.Lock()
        write = audio_health.write_audio_health

        def publish(path, rooms, now_wall):
            write(path, rooms, now_wall=now_wall)
            # Read the actual atomically published file, not a fabricated DTO.
            value = json.loads(Path(path).read_text())
            assert value['wall'] == now_wall
            assert value['rooms'] == {key: {'age_s': now_wall-anchor, 'ever': ever}
                for key, (anchor, ever) in rooms.items()}
            with self.lock:
                self.values.append(value)

        monkeypatch.setattr(audio_health, 'write_audio_health', publish)

    def snapshot(self):
        with self.lock:
            return list(self.values)


@dataclass(repr=False)
class RecoveryCase:
    g: object
    old: Driver
    replacement: Driver
    healthy: Driver | None
    catalog: Catalog
    publications: Publications
    watcher: asyncio.Task
    entered: threading.Event
    release: threading.Event
    constructed: list


@asynccontextmanager
async def recovering(tmp_path, monkeypatch, stage, *, extra_releases=(), extra_drivers=()):
    assert stage in {'close', 'catalog', 'open'}
    old = Driver('close' if stage == 'close' else None)
    replacement = Driver('construct' if stage == 'open' else None)
    healthy = None if stage == 'catalog' else Driver()
    drivers = iter([old, *([healthy] if healthy is not None else []), replacement, *extra_drivers])
    constructed = []

    def factory(**kwargs):
        driver = next(drivers)
        constructed.append(driver)
        driver.callback = kwargs['callback']
        driver.held('construct')
        return driver

    monkeypatch.setattr(audio_loop.sd, 'query_devices',
        lambda index: {'max_input_channels': 1, 'name': 'synthetic'})
    monkeypatch.setattr(audio_loop.sd, 'InputStream', factory)
    catalog = Catalog(held=stage == 'catalog')
    monkeypatch.setattr(audio_loop.sd, '_terminate', catalog.terminate)
    monkeypatch.setattr(audio_loop.sd, '_initialize', catalog.initialize)
    # Offset only this module's clock, without changing asyncio's own clock.
    # A freshly booted CI host must not turn our stale positive timestamp into
    # the <=0 sentinel for a microphone that has never delivered.
    real_monotonic = time.monotonic
    monkeypatch.setattr(audio_loop, 'time', SimpleNamespace(
        monotonic=lambda: real_monotonic()+1_000_000., time=time.time))
    publications = Publications(monkeypatch)
    rooms = (A, B) if healthy is not None else (A,)
    async with graph(tmp_path, rooms=rooms) as g:
        watcher = feeding = None
        stop_feed = asyncio.Event()
        loop = g.loop
        loop._audio_health_path = str(tmp_path/'audio_health.json')
        loop._watchdog_check_interval_s = .004
        loop._watchdog_timeout_s = .25
        loop._watchdog_first_frame_grace_s = 60.

        async def feed_healthy():
            while not stop_feed.is_set():
                healthy.callback(np.full((1280, 1), .2, dtype=np.float32), 1280, None, None)
                await asyncio.sleep(.003)

        try:
            assert await loop._try_reopen_once(loop.room_streams[A])
            if healthy is not None:
                assert await loop._try_reopen_once(loop.room_streams[B])
                feeding = asyncio.create_task(feed_healthy())
                await until(lambda: bool(g.wakes[B].history))
            loop.room_streams[A].last_frame_ts = audio_loop.time.monotonic()-120.
            loop.room_streams[A].opened_ts = audio_loop.time.monotonic()-121.
            entered = catalog.entered if stage == 'catalog' else (
                old.entered if stage == 'close' else replacement.entered)
            release = catalog.release if stage == 'catalog' else (
                old.release if stage == 'close' else replacement.release)
            watcher = asyncio.create_task(loop._stream_watchdog())
            loop._watchdog_task = watcher
            await until(entered.is_set)
            yield RecoveryCase(g, old, replacement, healthy, catalog, publications,
                watcher, entered, release, constructed)
        finally:
            # Seal before releasing a failed assertion so teardown cannot open
            # a new handle after the test has abandoned its observer.
            closing = loop.begin_close()
            stop_feed.set()
            old.release.set()
            replacement.release.set()
            catalog.release.set()
            for release in extra_releases:
                release.set()
            if watcher is not None:
                watcher.cancel()
            await asyncio.gather(*(task for task in (watcher, feeding) if task is not None),
                return_exceptions=True)
            await closing


async def publications_while_held(case, count=3):
    assert case.entered.is_set() and not case.release.is_set()
    await until(lambda: bool(case.publications.snapshot()))
    baseline = case.publications.snapshot()[-1]['wall']
    processed = len(case.g.wakes[B].history) if case.healthy is not None else None
    await until(lambda: len([row for row in case.publications.snapshot() if row['wall']>baseline])>=count)
    values = [row for row in case.publications.snapshot() if row['wall']>baseline]
    assert case.entered.is_set() and not case.release.is_set()
    assert all(second['wall']>first['wall'] for first, second in zip(values, values[1:]))
    assert all(row['rooms'][A]['ever'] is True and row['rooms'][A]['age_s']>=119. for row in values)
    anchors = [row['wall']-row['rooms'][A]['age_s'] for row in values]
    assert max(anchors)-min(anchors)<.05, 'native recovery cannot invent a newer audio anchor'
    if case.healthy is not None:
        await until(lambda: len(case.g.wakes[B].history)>processed)
        assert all(row['rooms'][B]['ever'] is True and row['rooms'][B]['age_s']<1. for row in values)
        assert case.healthy.closed == 0 and case.healthy.started == 1
        assert case.g.loop._streams[B] is case.healthy
        assert case.catalog.terminated == case.catalog.initialized == 0
    return values


async def cancel_observer(task):
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=1.)
    assert task in done and task.cancelled(), 'observer cancellation must propagate without canceling native ownership'


@pytest.mark.parametrize('stage', ['close', 'catalog', 'open'])
async def test_actual_recovery_keeps_publishing_truthful_heartbeats_and_only_one_controller(tmp_path, monkeypatch, stage):
    async with recovering(tmp_path, monkeypatch, stage) as case:
        loop = case.g.loop
        recovery = loop._recovery_task
        assert isinstance(recovery, asyncio.Task) and not recovery.done()
        assert recovery in loop.snapshot_operations()
        await publications_while_held(case)
        assert loop._recovery_task is recovery and not recovery.done()
        assert loop._inputs.resource_status()['operations']>0
        assert case.old.close_attempts == 1
        assert case.replacement.started == 0
        if stage == 'catalog':
            assert case.old.closed == 1 and case.catalog.terminated == 1
            assert case.catalog.initialized == 0 and case.constructed == [case.old]
        elif stage == 'close':
            assert case.old.closed == 0 and case.constructed == [case.old, case.healthy]
        else:
            assert case.old.closed == 1 and case.constructed == [case.old, case.healthy, case.replacement]
        case.release.set()
        await asyncio.shield(recovery)
        await until(lambda: loop._streams.get(A) is case.replacement)
        assert case.old.closed == case.replacement.started == 1
        assert loop.room_streams[A].last_frame_ts == 0., 'opening is not evidence of real audio'
        last_wall = case.publications.snapshot()[-1]['wall']
        await until(lambda: case.publications.snapshot()[-1]['wall']>last_wall)
        assert case.publications.snapshot()[-1]['rooms'][A]['age_s']>=119.
        case.g.clock.now += 1
        case.replacement.callback(np.full((1280, 1), .3, dtype=np.float32), 1280, None, None)
        await until(lambda: bool(case.g.wakes[A].history))
        await until(lambda: case.publications.snapshot()[-1]['rooms'][A]['age_s']<1.)
        assert case.replacement.started == 1


@pytest.mark.parametrize('stage', ['close', 'catalog', 'open'])
@pytest.mark.parametrize('ending', ['observer_cancel', 'pause', 'stop'])
async def test_recovery_native_owner_survives_observer_cancel_pause_and_bounded_stop(tmp_path, monkeypatch, stage, ending):
    async with recovering(tmp_path, monkeypatch, stage) as case:
        loop = case.g.loop
        recovery = loop._recovery_task
        await publications_while_held(case)
        epoch = loop._recovery_epoch
        if ending == 'observer_cancel':
            await cancel_observer(case.watcher)
            assert loop._recovery_task is recovery and not recovery.done() and not recovery.cancelled()
            drain = loop.begin_close()
        elif ending == 'pause':
            drain = loop.begin_pause_inputs()
            with pytest.raises(RuntimeError, match='still_owned'):
                loop.resume_inputs()
        else:
            with pytest.raises(WorkCloseTimeout):
                await loop.stop()
            with pytest.raises(WorkCloseTimeout):
                await case.g.privacy.aclose()
            assert case.g.privacy.resource_status()['writer_lease']
            drain = loop.begin_close()
        await asyncio.sleep(0)
        assert loop._recovery_epoch != epoch
        assert not drain.done() and not recovery.done() and not recovery.cancelled()
        assert loop._inputs.resource_status()['operations']>0
        assert case.replacement.started == 0
        case.release.set()
        await drain
        assert recovery.done() and not recovery.cancelled()
        assert loop._inputs.resource_status()['operations'] == 0
        assert not loop._streams and not loop._reopen_tasks
        assert case.old.closed == 1 and case.replacement.started == 0
        if stage == 'open':
            assert case.replacement.closed == 1
        else:
            assert case.replacement not in case.constructed, 'a sealed recovery cannot start a late open'
        if ending == 'stop':
            await case.g.privacy.begin_close()
            assert not case.g.privacy.resource_status()['writer_lease']


async def test_initial_open_keeps_publishing_without_starting_competing_recovery(tmp_path, monkeypatch):
    driver = Driver('construct')
    portaudio(monkeypatch, [driver])
    catalog = Catalog()
    monkeypatch.setattr(audio_loop.sd, '_terminate', catalog.terminate)
    monkeypatch.setattr(audio_loop.sd, '_initialize', catalog.initialize)
    publications = Publications(monkeypatch)
    async with graph(tmp_path, controllers=False) as g:
        loop = g.loop
        loop._audio_health_path = str(tmp_path/'audio_health.json')
        loop._watchdog_enabled = True
        loop._watchdog_check_interval_s = .004
        loop._watchdog_timeout_s = .01
        loop._watchdog_first_frame_grace_s = .001
        running = asyncio.create_task(loop._run_capture())
        watcher = None
        try:
            await until(driver.entered.is_set)
            assert loop._recovery_ready is False
            watcher = loop._watchdog_task
            assert isinstance(watcher, asyncio.Task)
            await until(lambda: bool(publications.snapshot()))
            baseline = publications.snapshot()[-1]['wall']
            await until(lambda: len([row for row in publications.snapshot() if row['wall']>baseline])>=3)
            values = [row for row in publications.snapshot() if row['wall']>baseline]
            assert all(row['rooms'][A]['ever'] is False and row['rooms'][A]['age_s']>=0. for row in values)
            assert all(b['wall']>a['wall'] for a, b in zip(values, values[1:]))
            assert loop._recovery_task is None and not running.done()
            assert driver.started == driver.close_attempts == 0
            assert loop._inputs.resource_status()['operations']>0
            assert catalog.terminated == catalog.initialized == 0
        finally:
            closing = loop.begin_close()
            driver.release.set()
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            await closing
        assert driver.closed == 1 and driver.started == 0 and not loop._streams
        assert watcher is not None and watcher.done()


@pytest.mark.parametrize('stage', ['close', 'catalog', 'open'])
async def test_resume_waits_for_old_recovery_epoch_and_opens_only_fresh_drivers(tmp_path, monkeypatch, stage):
    successors = (Driver(), Driver())
    async with recovering(tmp_path, monkeypatch, stage, extra_drivers=successors) as case:
        loop = case.g.loop
        recovery = loop._recovery_task
        old_epoch = loop._recovery_epoch
        pause = loop.begin_pause_inputs()
        assert loop._recovery_epoch != old_epoch
        with pytest.raises(RuntimeError, match='still_owned'):
            loop.resume_inputs()
        assert not recovery.done() and not pause.done()
        case.release.set()
        await pause
        assert recovery.done() and not recovery.cancelled() and not loop._streams
        before = len(case.constructed)
        old_constructed = tuple(case.constructed)
        case.g.clock.now += 1
        loop.resume_inputs()
        await until(lambda: set(loop._streams) == set(loop.room_streams))
        fresh = case.constructed[before:]
        assert len(fresh) == len(loop.room_streams)
        assert set(loop._streams.values()) == set(fresh)
        assert all(driver not in old_constructed and driver.started == 1 and driver.closed == 0 for driver in fresh)
        assert case.old.closed == 1
        if case.healthy is not None:
            assert case.healthy.closed == 1
        for driver in fresh:
            driver.callback(np.full((1280, 1), .3, dtype=np.float32), 1280, None, None)
        await until(lambda: all(any(np.allclose(audio, .3) for audio in case.g.wakes[room].history)
            for room in loop.room_streams))


async def test_global_catalog_fence_excludes_retry_opens_until_actual_catalog_exit(tmp_path, monkeypatch):
    async with recovering(tmp_path, monkeypatch, 'catalog') as case:
        loop = case.g.loop
        recovery = loop._recovery_task
        assert loop._global_recovery is True
        assert not await asyncio.wait_for(loop._try_reopen_once(loop.room_streams[A]), timeout=1.)
        loop._schedule_reopen(loop.room_streams[A])
        await publications_while_held(case)
        assert loop._recovery_task is recovery and not recovery.done()
        assert case.constructed == [case.old] and not loop._reopen_tasks
        assert case.catalog.terminated == 1 and case.catalog.initialized == 0
        case.release.set()
        await asyncio.shield(recovery)
        await until(lambda: loop._streams.get(A) is case.replacement)
        assert not loop._global_recovery
        assert case.catalog.terminated == case.catalog.initialized == 1
        assert case.replacement.started == 1
