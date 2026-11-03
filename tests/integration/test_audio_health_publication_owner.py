"""Retained audio-health publication with synthetic room metadata and real I/O."""

import asyncio
import json
from pathlib import Path
import threading
import time

import pytest

import src.monitoring.audio_health as audio_health
import src.pipeline.multi_room_audio_loop as capture
from tests.unit.pipeline.test_multi_room_audio_loop import (
    _make_multi_room_loop,
    _make_room_stream,
)


pytestmark = pytest.mark.asyncio


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


class BeforeReplace:
    """Hold the first real writer after serialization, before atomic replace."""

    def __init__(self, monkeypatch, path):
        self.path = Path(path)
        self.entered, self.release = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._payloads, self._threads = [], []
        real_replace = audio_health.os.replace

        def replace(source, destination):
            if Path(destination) != self.path:
                return real_replace(source, destination)
            with Path(source).open() as stream:
                document = json.load(stream)
            with self._lock:
                self._payloads.append(document)
                self._threads.append(threading.get_ident())
                first = len(self._payloads) == 1
            if first:
                self.entered.set()
                if not self.release.wait(5):
                    raise TimeoutError('synthetic health publication barrier')
            return real_replace(source, destination)

        monkeypatch.setattr(audio_health.os, 'replace', replace)

    @property
    def calls(self):
        with self._lock:
            return len(self._payloads)

    @property
    def payloads(self):
        with self._lock:
            return list(self._payloads)

    @property
    def thread_ids(self):
        with self._lock:
            return list(self._threads)


def loop_for(path):
    room = _make_room_stream('synthetic-room')
    room.last_frame_ts = time.monotonic() - 1
    loop = _make_multi_room_loop(
        rooms={room.room_id: room}, audio_health_path=str(path), close_timeout_s=.03,
    )
    loop._loop = asyncio.get_running_loop()
    loop._running = True
    return loop, room


async def close(loop):
    await asyncio.wait_for(asyncio.shield(loop.begin_close()), 3)


async def other_native_owners_finished(loop):
    """Do not mistake unrelated input/reset cleanup for writer retention."""
    await asyncio.gather(loop._inputs.begin_close(), loop._capture_native.begin_close(),
                         *(room.begin_close() for room in loop._capture_rooms.values()))
    await asyncio.sleep(0)


async def test_publication_timeout_keeps_one_native_write_and_frozen_observation(tmp_path, monkeypatch):
    path = tmp_path / 'audio_health.json'
    gate = BeforeReplace(monkeypatch, path)
    wall = [1000.]
    monkeypatch.setattr(capture.time, 'time', lambda: wall[0])
    loop, room = loop_for(path)
    main_thread = threading.get_ident()
    try:
        loop._publish_audio_health()
        await until(gate.entered.is_set)
        admitted = loop._audio_health_task
        frozen = gate.payloads[0]
        assert frozen['wall'] == 1000. and not path.exists()
        assert frozen['rooms']['synthetic-room']['ever'] is True
        assert len(gate.thread_ids) == 1
        assert gate.thread_ids[0] != main_thread

        loop._audio_health_started -= capture.AUDIO_HEALTH_WRITE_TIMEOUT_S + 1
        anchors = []
        for observation in (1010., 1011., 1012.):
            wall[0] = observation
            room.last_frame_ts = time.monotonic()
            for _ in range(7):
                loop._publish_audio_health()
            anchors.append(loop._last_audio_wall['synthetic-room'])
        assert anchors == sorted(anchors) and len(set(anchors)) == 3
        assert loop._audio_health_timed_out
        assert loop._audio_health_task is admitted and not admitted.done()
        assert loop._audio_health_work.status() == {
            'closed': False, 'operations': 1, 'capacity': 1,
        }
        assert gate.calls == 1 and not path.exists()

        gate.release.set()
        await asyncio.wait_for(asyncio.shield(admitted), 3)
        assert loop._audio_health_work.status()['operations'] == 0
        # A late successful write reports its original observation. Waiting
        # never stamps the old room payload with the current wall clock.
        assert json.loads(path.read_text()) == frozen
        assert json.loads(path.read_text())['wall'] == 1000.

        room.last_frame_ts = time.monotonic()
        loop._publish_audio_health()
        await asyncio.wait_for(asyncio.shield(loop._audio_health_task), 3)
        current = json.loads(path.read_text())
        assert gate.calls == 2 and current['wall'] == 1012.
        assert current['rooms']['synthetic-room']['age_s'] < .1
        assert loop._audio_health_work.status()['operations'] == 0
    finally:
        gate.release.set()
        await close(loop)


@pytest.mark.parametrize('observer', ['watchdog', 'write_wrapper', 'stop'])
async def test_cancelled_observer_cannot_release_actual_publication_or_close_owner(tmp_path, monkeypatch, observer):
    path = tmp_path / 'audio_health.json'
    gate = BeforeReplace(monkeypatch, path)
    loop, _ = loop_for(path)
    watching = stopping = None
    try:
        loop._publish_audio_health()
        await until(gate.entered.is_set)
        admitted = loop._audio_health_task
        if observer == 'watchdog':
            loop._watchdog_check_interval_s = .001
            loop._recovery_ready = False
            loop._audio_health_started -= capture.AUDIO_HEALTH_WRITE_TIMEOUT_S + 1
            watching = loop._watchdog_task = asyncio.create_task(loop._stream_watchdog())
            await until(lambda: loop._audio_health_timed_out)
            watching.cancel()
            await asyncio.gather(watching, return_exceptions=True)
            assert watching.cancelled()
        elif observer == 'write_wrapper':
            admitted.cancel()
            await asyncio.gather(admitted, return_exceptions=True)
            assert admitted.cancelled()
        else:
            # Cancellation is selected by an actual admission barrier, rather
            # than racing this test against the bounded public-stop timeout.
            loop._close_timeout_s = 2.
            stopping = asyncio.create_task(loop.stop())
            await until(lambda: loop._audio_health_work.status()['closed'])
            stopping.cancel()
            await asyncio.gather(stopping, return_exceptions=True)
            assert stopping.cancelled()

        for _ in range(20):
            loop._publish_audio_health()
        assert gate.calls == 1
        assert loop._audio_health_work.status()['operations'] == 1
        closing = loop.begin_close()
        await other_native_owners_finished(loop)
        assert not closing.done() and not closing.cancelled()
        assert loop.begin_close() is closing
        assert loop._audio_health_work.status()['closed']
        assert not path.exists()

        gate.release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert path.exists() and gate.calls == 1
        assert loop._audio_health_work.status() == {
            'closed': True, 'operations': 0, 'capacity': 1,
        }
        assert not loop._audio_health_work._tasks
    finally:
        gate.release.set()
        if watching is not None:
            watching.cancel()
            await asyncio.gather(watching, return_exceptions=True)
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        await close(loop)


async def test_completed_close_prevents_old_native_replace_over_successor_snapshot(tmp_path, monkeypatch):
    path = tmp_path / 'audio_health.json'
    gate = BeforeReplace(monkeypatch, path)
    wall = [1000.]
    monkeypatch.setattr(capture.time, 'time', lambda: wall[0])
    old, _ = loop_for(path)
    successor = None
    try:
        old._publish_audio_health()
        await until(gate.entered.is_set)
        closing = old.begin_close()
        await other_native_owners_finished(old)
        assert not closing.done() and not path.exists()

        gate.release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert old._audio_health_work.status()['operations'] == 0
        assert json.loads(path.read_text())['wall'] == 1000.

        wall[0] = 2000.
        successor, _ = loop_for(path)
        successor._publish_audio_health()
        await asyncio.wait_for(asyncio.shield(successor._audio_health_task), 3)
        fresh = json.loads(path.read_text())
        assert fresh['wall'] == 2000. and gate.calls == 2
        assert successor._audio_health_work.status()['operations'] == 0

        # Closed predecessors cannot admit a late tick, and complete close
        # means no native writer remains capable of replacing this file.
        for _ in range(20):
            old._publish_audio_health()
        await close(old)
        assert not old._audio_health_work._tasks
        assert gate.calls == 2 and json.loads(path.read_text()) == fresh
    finally:
        gate.release.set()
        await close(old)
        if successor is not None:
            await close(successor)


async def test_standalone_disabled_defaults_do_not_admit_health_io(tmp_path, monkeypatch):
    path = tmp_path / 'audio_health.json'
    gate = BeforeReplace(monkeypatch, path)
    loop, _ = loop_for(path)
    try:
        assert not loop._watchdog_enabled
        assert loop._watchdog_task is None and loop._audio_health_task is None
        assert loop._audio_health_work.status() == {
            'closed': False, 'operations': 0, 'capacity': 1,
        }
        assert gate.calls == 0 and not path.exists()
    finally:
        gate.release.set()
        await close(loop)
