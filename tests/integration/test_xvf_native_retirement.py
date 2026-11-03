"""Admitted XVF work remains owned across close, without capture or models."""

import asyncio
import struct
import sys
import threading
import types

import pytest

from src.audio.xvf_controller import XvfController
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop


pytestmark = pytest.mark.asyncio


class USBLeaf:
    bus = 5
    port_numbers = (5, 4)

    def __init__(self):
        self.read_entered = threading.Event()
        self.read_release = threading.Event()
        self.write_entered = threading.Event()
        self.write_release = threading.Event()
        self.energy_reads = 0
        self.writes = 0

    def ctrl_transfer(self, bm, _request, _value, _index, payload, _timeout):
        if bm & 0x80:
            if payload == 17:
                self.energy_reads += 1
                self.read_entered.set()
                assert self.read_release.wait(5), "synthetic energy read not released"
                return [0] + list(struct.pack("<ffff", 1.0, 2.0, 3.0, 144.0))
            assert payload == 5
            return [0] + list(struct.pack("<f", 1.0))
        self.writes += 1
        self.write_entered.set()
        assert self.write_release.wait(5), "synthetic tuning write not released"
        return len(payload)


class Discovery:
    def __init__(self, monkeypatch, device):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.device = device
        core, package = types.ModuleType("usb.core"), types.ModuleType("usb")
        core.find = self.find
        package.core = core
        monkeypatch.setitem(sys.modules, "usb", package)
        monkeypatch.setitem(sys.modules, "usb.core", core)

    def find(self, **options):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(5), "synthetic discovery not released"
        assert options["custom_match"](self.device)
        return self.device


def make_loop(monkeypatch, *, tuning=False):
    device = USBLeaf()
    discovery = Discovery(monkeypatch, device)
    controller = XvfController(usb_port="5-5.4", poll_interval_s=0.001, max_retries=1)
    loop = MultiRoomAudioLoop(
        room_streams={}, follow_up=None, xvf_controllers={"synthetic": controller},
        spenergy_gate_enabled=True, close_timeout_s=0.02,
        xvf_tuning={"apply_on_start": tuning, "params": {"PP_AGCGAIN": [2.0]}},
    )
    loop._loop = asyncio.get_running_loop()
    return loop, controller, device, discovery


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def cleanup(loop, controller, device, discovery, release, observer):
    closing = loop.begin_close()
    release.set()
    discovery.release.set()
    device.read_release.set()
    device.write_release.set()
    # RED cleanup repeats the seal only after actual admitted native work exits.
    # This happens after the assertions; it cannot make a missing second seal pass.
    await loop._xvf_native.begin_close()
    controller.request_stop()
    await asyncio.to_thread(controller.wait_stopped)
    await asyncio.gather(asyncio.shield(closing), return_exceptions=True)
    if observer is not None:
        await asyncio.gather(observer, return_exceptions=True)
    await loop.stop()


async def test_cancelled_admitted_start_enters_after_first_seal_but_cannot_escape_second_seal(monkeypatch):
    loop, controller, device, discovery = make_loop(monkeypatch)
    discovery.release.set()
    entered, release = threading.Event(), threading.Event()
    started, threads = [], []

    def before_start():
        entered.set()
        assert release.wait(5), "admitted start wrapper not released"
        started.append(controller.start())
        threads.append(controller._thread)

    observer = asyncio.create_task(loop._xvf_native.run(before_start))
    try:
        await until(entered.is_set)
        assert loop._xvf_native.status()["operations"] == 1
        assert controller._thread is None and discovery.calls == 0
        observer.cancel()
        result = await asyncio.gather(observer, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        closing = loop.begin_close()
        assert loop._xvf_native.status()["closed"]
        assert not closing.done() and not closing.cancelled()
        assert controller._thread is None and discovery.calls == 0

        release.set()
        await until(device.read_entered.is_set)
        await until(lambda: loop._xvf_native.status()["operations"] == 0)
        assert started == [True] and len(threads) == 1
        thread = threads[0]
        assert thread is not None and thread.is_alive()
        assert controller._thread is thread and discovery.calls == 1
        # A live active run returns True. False here witnesses the second seal,
        # without mocking stop/wait or assuming the controller's token internals.
        await until(lambda: controller.start() is False)
        assert not closing.done() and not closing.cancelled()
        assert not loop._direction_started and not loop._direction_observers
        assert not loop._direction_opening

        device.read_release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert not thread.is_alive() and controller._thread is None
        assert controller.peak_since(0) is None and device.energy_reads == 1
        assert closing.exception() is None and loop.begin_close() is closing
        assert not loop._direction_started and not loop._direction_observers
        assert not loop._direction_opening
    finally:
        await cleanup(loop, controller, device, discovery, release, observer)


async def test_cancelled_admitted_tuning_keeps_real_usb_write_owned_until_exit(monkeypatch):
    loop, controller, device, discovery = make_loop(monkeypatch, tuning=True)
    discovery.release.set()
    entered, release = threading.Event(), threading.Event()

    def before_tuning():
        entered.set()
        assert release.wait(5), "admitted tuning wrapper not released"
        loop._apply_xvf_tuning()

    observer = asyncio.create_task(loop._xvf_native.run(before_tuning))
    try:
        await until(entered.is_set)
        assert loop._xvf_native.status()["operations"] == 1
        assert discovery.calls == 0 and device.writes == 0
        observer.cancel()
        result = await asyncio.gather(observer, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        closing = loop.begin_close()
        assert loop._xvf_native.status()["closed"] and not closing.done()

        release.set()
        await until(device.write_entered.is_set)
        assert loop._xvf_native.status()["operations"] == 1
        assert not closing.done() and not closing.cancelled()
        assert controller._thread is None
        assert not loop._direction_started and not loop._direction_observers
        assert not loop._direction_opening
        device.write_release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert loop._xvf_native.status()["operations"] == 0
        assert controller._thread is None and device.writes == 1
        assert device.energy_reads == 0 and discovery.calls == 1
        assert closing.exception() is None and loop.begin_close() is closing
        assert not loop._direction_started and not loop._direction_observers
        assert not loop._direction_opening
    finally:
        await cleanup(loop, controller, device, discovery, release, observer)


async def test_real_start_resuming_after_closed_discovery_cannot_resurrect_direction(monkeypatch):
    loop, controller, device, discovery = make_loop(monkeypatch)
    release = threading.Event()
    observer = asyncio.create_task(loop.start())
    try:
        await until(discovery.entered.is_set)
        assert loop.room_streams == {} and controller._thread is None
        assert loop._xvf_native.status()["operations"] == 1
        closing = loop.begin_close()
        assert loop._xvf_native.status()["closed"] and not closing.done()
        assert controller._thread is None and controller.start() is False
        assert discovery.calls == 1 and device.energy_reads == 0
        assert not loop._direction_started and not loop._direction_observers

        discovery.release.set()
        result = await asyncio.gather(observer, return_exceptions=True)
        assert isinstance(result[0], RuntimeError) and str(result[0]) == "capture_closed"
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert loop._xvf_native.status()["operations"] == 0
        assert controller._thread is None and device.energy_reads == 0
        assert discovery.calls == 1 and controller.peak_since(0) is None
        assert not loop._direction_started and not loop._direction_observers
        assert not loop._direction_opening and closing.exception() is None
    finally:
        await cleanup(loop, controller, device, discovery, release, observer)
