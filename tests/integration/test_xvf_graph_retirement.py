"""Real graph teardown with only an injected synthetic XVF USB read running.

These tests never start capture, warmup, graph services or model producers.
The asyncio counter is a progress probe, not a numeric C03 latency measurement.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import struct
import threading

import pytest

from src.audio.xvf_controller import XvfController
from src.core.owned_work import WorkCloseTimeout
from src.main import build_application
from tests.integration import test_application_graph as application_fixtures
from tests.mocks.application_graph import OfflineWake, offline_config, offline_home
from tests.unit.pipeline.test_multi_room_audio_loop import _make_multi_room_loop


pytestmark = pytest.mark.asyncio
leaves = application_fixtures.leaves


class HeldUSBRead:
    """The real ctrl_transfer leaf, retaining the first SPENERGY read."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.calls = 0

    def ctrl_transfer(self, bm, brequest, wvalue, windex, length, timeout):
        self.calls += 1
        self.entered.set()
        try:
            assert self.release.wait(5), "synthetic USB leaf was not released"
            return [0] + list(struct.pack("<ffff", 1.0, 2.0, 3.0, 144.0))
        finally:
            self.returned.set()


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def heartbeat(stop, ticks):
    while not stop.is_set():
        ticks[0] += 1
        await asyncio.sleep(0)


async def observe_stop(loop, entered):
    entered.set()
    await loop.stop()


async def test_multiroom_close_retains_real_xvf_read_through_repeated_observer_cancel():
    device = HeldUSBRead()
    controller = XvfController(device=device, poll_interval_s=0.001, max_retries=1)
    loop = _make_multi_room_loop(
        xvf_controllers={"cocina": controller}, close_timeout_s=0.02,
    )
    loop._loop = asyncio.get_running_loop()
    closing = None
    observers = []
    probe_stop, ticks = asyncio.Event(), [0]
    probe = asyncio.create_task(heartbeat(probe_stop, ticks))
    try:
        assert controller.start() is True
        await until(device.entered.is_set)
        thread = controller._thread
        assert thread is not None and thread.is_alive()
        closing = loop.begin_close()
        assert loop.begin_close() is closing
        assert not closing.done() and not closing.cancelled()
        assert controller._thread is thread and thread.is_alive()
        assert not device.returned.is_set()
        assert controller.start() is False

        for _ in range(2):
            entered = asyncio.Event()
            observer = asyncio.create_task(observe_stop(loop, entered))
            observers.append(observer)
            await entered.wait()
            observer.cancel()
            await asyncio.sleep(0)
            if not observer.done():
                observer.cancel()
            assert loop.begin_close() is closing
            assert not closing.done() and not closing.cancelled()
            assert controller._thread is thread and thread.is_alive()
            assert not device.returned.is_set()

        before = ticks[0]
        for _ in range(20):
            await asyncio.sleep(0)
        assert ticks[0] > before
        assert not closing.done() and not device.returned.is_set()
        with pytest.raises(WorkCloseTimeout):
            await loop.stop()
        assert loop.begin_close() is closing
        assert not closing.done() and controller._thread is thread
        assert controller.start() is False and device.calls == 1

        device.release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        results = await asyncio.gather(*observers, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert device.returned.is_set() and not thread.is_alive()
        assert controller._thread is None
        assert controller.peak_since(0) is None
        assert device.calls == 1
        assert closing.exception() is None
        assert loop.begin_close() is closing
        await loop.stop()
    finally:
        device.release.set()
        probe_stop.set()
        await probe
        controller.request_stop()
        await asyncio.to_thread(controller.wait_stopped)
        if closing is not None:
            await asyncio.gather(asyncio.shield(closing), return_exceptions=True)
        await asyncio.gather(*observers, return_exceptions=True)
        await loop.stop()


@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_factory_xvf_drain_precedes_real_pipeline_and_authority_retirement(
    leaves, tmp_path, monkeypatch, close_path,
):
    device = HeldUSBRead()
    created = []
    original_init = XvfController.__init__

    def construct(self, *args, **kwargs):
        kwargs["device"] = device
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(XvfController, "__init__", construct)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url, rooms=True)
        config["rooms"]["living"] = {
            "name": "Synthetic living", "mic_device_index": 0,
            "mic_usb_port": "5-5.4", "capture_channel": 0,
        }
        config["rooms"]["wake_word"] = {
            "spenergy_gate": {"enabled": True, "poll_interval_s": 0.001},
        }
        async with AsyncExitStack() as stack:
            graph = await build_application(
                config, shutdown=stack, factories=replace(leaves, wake_detector=OfflineWake),
            )
            loop = graph.pipeline.audio_loop
            assert len(created) == 1
            controller = created[0]
            assert type(controller) is XvfController
            assert loop._xvf_by_room == {"living": controller}
            assert controller._thread is None
            assert device.calls == 0 and not device.entered.is_set()
            assert loop._streams == {}
            assert graph.models.stt.loads == graph.models.tts.loads == 0
            assert all(room.wake_detector.loads == 0 for room in loop.room_streams.values())
            processor = graph.pipeline.command_processor
            original_close = processor.aclose
            retired = []

            async def close_processor():
                retired.append(True)
                await original_close()

            monkeypatch.setattr(processor, "aclose", close_processor)
            closing = native_close = None
            probe_stop, ticks = asyncio.Event(), [0]
            probe = asyncio.create_task(heartbeat(probe_stop, ticks))
            try:
                assert controller.start() is True
                await until(device.entered.is_set)
                thread = controller._thread
                assert thread is not None and thread.is_alive()
                if close_path == "factory_stack":
                    closing = asyncio.create_task(stack.aclose())
                else:
                    # This is the actual runtime teardown entry, without starting services.
                    closing = asyncio.create_task(graph._stop_services(None, None))
                await until(lambda: loop._closed)
                native_close = loop._closing
                assert native_close is not None and not native_close.done()
                for _ in range(2):
                    closing.cancel()
                    await asyncio.sleep(0)
                    assert not closing.done()
                    assert loop._closing is native_close and not native_close.done()
                    assert not native_close.cancelled()
                    assert controller._thread is thread and thread.is_alive()
                    assert not device.returned.is_set()
                before = ticks[0]
                for _ in range(20):
                    await asyncio.sleep(0)
                assert ticks[0] > before
                assert retired == []
                assert not graph.pipeline.request_router._closed
                assert not graph.ha_client._session.closed
                assert graph.dashboard_authority._lease_fd is not None
                assert graph.privacy.resource_status()["writer_lease"]
                assert controller.start() is False
                assert loop._streams == {} and device.calls == 1

                device.release.set()
                await asyncio.wait_for(asyncio.shield(native_close), 3)
                await asyncio.gather(closing, return_exceptions=True)
                assert native_close.exception() is None
                assert not thread.is_alive() and controller._thread is None
                assert controller.peak_since(0) is None and device.calls == 1
                assert retired and graph.pipeline.request_router._closed
                assert graph.models.stt.loads == graph.models.tts.loads == 0
                assert observations["service_calls"] == 0
            finally:
                device.release.set()
                probe_stop.set()
                await probe
                controller.request_stop()
                await asyncio.to_thread(controller.wait_stopped)
                if native_close is not None:
                    await asyncio.gather(asyncio.shield(native_close), return_exceptions=True)
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0
