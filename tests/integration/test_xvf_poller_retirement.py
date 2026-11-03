"""Real XVF poller lifetime against a held synthetic USB transfer."""

from contextlib import contextmanager
import struct
import threading
import time
import sys
import types

from src.audio.xvf_controller import XvfController


class HeldReadDevice:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fresh_read = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def ctrl_transfer(self, *_args, **_kwargs):
        with self.lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            self.entered.set()
            self.release.wait()
        else:
            self.fresh_read.set()
        return [0] + list(struct.pack("<ffff", 0., 0., 0., 1234. if first else 2345.))


@contextmanager
def held_poller():
    device = HeldReadDevice()
    controller = XvfController(device=device, poll_interval_s=.005, max_retries=1)
    original = None
    try:
        assert controller.start()
        original = controller._thread
        assert device.entered.wait(3)
        yield controller, device, original
    finally:
        device.release.set()
        current = controller._thread
        controller.stop()
        for thread in {original, current} - {None}:
            thread.join(3)
            assert not thread.is_alive(), "synthetic poller must really exit before fixture cleanup"


def test_stop_keeps_exact_live_thread_registered_after_bounded_join():
    with held_poller() as (controller, device, original):
        controller.stop()
        assert original.is_alive()
        assert controller._thread is original
        assert controller.start() is False
        assert controller._thread is original and device.calls == 1
        device.release.set()
        controller.wait_stopped()
        assert not original.is_alive() and controller._thread is None


def test_stopped_poll_cannot_publish_its_late_usb_result_as_a_fresh_sample():
    with held_poller() as (controller, device, original):
        controller.stop()
        device.release.set()
        original.join(3)
        assert not original.is_alive()
        assert controller.peak_since(0.) is None


def test_retiring_poll_cannot_be_revived_by_a_successor_start():
    with held_poller() as (controller, device, original):
        controller.stop()
        assert controller.start() is False
        device.release.set()
        controller.wait_stopped()
        assert not original.is_alive() and controller.peak_since(0.) is None
        assert controller.start() is True
        successor = controller._thread
        assert successor is not original
        assert device.fresh_read.wait(3)
        deadline = time.monotonic() + 3
        while controller.peak_since(0.) != 2345.:
            assert time.monotonic() < deadline
            time.sleep(.001)
        controller.stop()
        controller.wait_stopped()
        assert not successor.is_alive() and controller._thread is None


@contextmanager
def held_discovery(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    device = HeldReadDevice()
    device.release.set()
    core, package = types.ModuleType("usb.core"), types.ModuleType("usb")

    def find(**_kwargs):
        entered.set()
        release.wait()
        return device

    core.find = find
    package.core = core
    monkeypatch.setitem(sys.modules, "usb", package)
    monkeypatch.setitem(sys.modules, "usb.core", core)
    controller = XvfController(poll_interval_s=.005, max_retries=1)
    results, errors = [], []

    def start():
        try:
            results.append(controller.start())
        except BaseException as error:
            errors.append(error)

    starter = threading.Thread(target=start, name="synthetic-xvf-start")
    starter.start()
    try:
        assert entered.wait(3)
        yield controller, device, release, results, errors, starter
    finally:
        release.set()
        starter.join(3)
        assert not starter.is_alive()
        poller = controller._thread
        controller.stop()
        if poller is not None:
            poller.join(3)
            assert not poller.is_alive()


def test_stop_seal_prevents_late_poll_admission_after_device_discovery(monkeypatch):
    with held_discovery(monkeypatch) as (controller, device, release, results, errors, starter):
        controller.stop()
        assert starter.is_alive()
        release.set()
        starter.join(3)
        assert not errors and results == [False]
        assert controller._thread is None and device.calls == 0
        assert controller.peak_since(0.) is None


def test_actual_drain_cannot_finish_while_start_discovery_is_pending(monkeypatch):
    with held_discovery(monkeypatch) as (controller, device, release, results, errors, starter):
        controller.request_stop()
        drained = threading.Event()

        def drain():
            controller.wait_stopped()
            drained.set()

        observer = threading.Thread(target=drain, name="synthetic-xvf-drain")
        observer.start()
        try:
            assert not drained.wait(.05)
            assert controller.start() is False
            release.set()
            starter.join(3)
            observer.join(3)
            assert drained.is_set() and not observer.is_alive()
            assert not errors and results == [False] and device.calls == 0
        finally:
            release.set()
            observer.join(3)
            assert not observer.is_alive()
