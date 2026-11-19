"""T048-U1: automatic recovery of a stuck XVF3800 by firmware reboot.

Incident 2026-11-19 (usb_overrun_20261119): after a warm host reboot every
reSpeaker XVF3800 kept its isochronous endpoint stuck. PortAudio opened the
streams and no usable audio ever arrived; reopening and sysfs resets did not
help, the firmware REBOOT vendor command did. User decision
xvf_auto_recovery_20261119: "Sí, como unidad".

Every test uses fakes: no sounddevice, no pyusb, no real USB device.
"""

import sys
import asyncio
import logging
import struct
import time
import types
from unittest.mock import MagicMock, patch

sys.modules.setdefault('sounddevice', MagicMock())

import pytest

from src.audio.microphone_status import UsbInventory, UsbMicrophone, read_usb_microphones
from src.audio.owned_input import OwnedInputStream
from src.audio.xvf_controller import XvfController, XvfRebootOutcome, XvfRebootResult
from src.audio.xvf_recovery import XvfRebootPolicy
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream

LOGGER = "src.pipeline.multi_room_audio_loop"
TUNING = {"apply_on_start": True, "params": {"PP_AGCMAXGAIN": [16.0], "PP_MIN_NN": [0.4]}}


def _room(room_id: str, index: int, port: str) -> RoomStream:
    wake = MagicMock()
    wake.detect = MagicMock(return_value=None)
    echo = MagicMock()
    echo.is_safe_to_listen = True
    rs = RoomStream(room_id=room_id, device_index=index, wake_detector=wake, echo_suppressor=echo)
    rs.mic_usb_port = port
    return rs


def _follow_up():
    follow_up = MagicMock()
    follow_up.is_active = False
    return follow_up


async def _until(predicate, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.002)


class _FakeXvf:
    """Port-bound XVF controller double: records reboots and tuning writes."""

    def __init__(self, usb_port, *, outcome=XvfRebootOutcome.SENT, reenumerates=True,
                 reboot_error=None, on_reboot=None):
        self.usb_port = usb_port
        self.outcome = outcome
        self.reenumerates = reenumerates
        self.reboot_error = reboot_error
        self.on_reboot = on_reboot
        self.reboots = 0
        self.waits = []
        self.writes = []
        self.invalidations = 0

    def reboot(self):
        self.reboots += 1
        if self.reboot_error is not None:
            raise self.reboot_error
        if self.on_reboot is not None:
            self.on_reboot()
        return XvfRebootResult(self.outcome, previous_address=9)

    def wait_reenumerated(self, previous_address, timeout_s, *, abort=None, **_kwargs):
        self.waits.append((previous_address, timeout_s, abort))
        return self.reenumerates

    def invalidate_device(self):
        self.invalidations += 1

    def read_param(self, name):
        return (64.0,)

    def write_param(self, name, values):
        self.writes.append((name, list(values)))
        return True

    def start(self):
        return True

    def stop(self):
        pass

    def peak_since(self, _ts):
        return None


class _Rig:
    """Two rooms on fake sysfs + PortAudio: Living stuck, Cocina healthy.

    PortAudio lists ALSA cards as they were at the last ``_initialize``; a
    port resolves through sysfs (port -> card) and then that catalog.
    """

    LIVING_PORT, LIVING_CARD = "1-2.3", 3
    COCINA_PORT, COCINA_CARD = "7-5.3", 6

    def __init__(self, tmp_path, *, policy=None, tuning=TUNING, living_ctrl=None,
                 follow_up=None, catalog_interval_s=300.0):
        self.sysfs = tmp_path
        self.clock = [1000.0]
        self._card_dir(self.LIVING_PORT, self.LIVING_CARD)
        self._card_dir(self.COCINA_PORT, self.COCINA_CARD)
        self.catalog = self._sysfs_catalog()
        self.streams = []
        self.sd = MagicMock()
        self.sd.PortAudioError = type("PortAudioError", (Exception,), {})
        self.sd.query_devices.return_value = {"max_input_channels": 2}
        self.sd.InputStream.side_effect = self._new_stream
        self.sd._initialize.side_effect = self._initialize
        self.living = _room("living", self.index(self.LIVING_CARD), self.LIVING_PORT)
        self.cocina = _room("cocina", self.index(self.COCINA_CARD), self.COCINA_PORT)
        self.living_ctrl = living_ctrl or _FakeXvf(self.LIVING_PORT)
        self.cocina_ctrl = _FakeXvf(self.COCINA_PORT)
        self.follow_up = follow_up or _follow_up()
        self.loop = MultiRoomAudioLoop(
            room_streams={"living": self.living, "cocina": self.cocina},
            follow_up=self.follow_up,
            microphone_usb_root=tmp_path,
            microphone_monotonic=lambda: self.clock[0],
            xvf_controllers={"living": self.living_ctrl, "cocina": self.cocina_ctrl},
            xvf_tuning=tuning,
            xvf_reboot_policy=policy or XvfRebootPolicy(
                enabled=True, failures_before_reboot=2, max_per_hour=3, settle_s=1.0),
            stream_watchdog_catalog_refresh_min_interval_s=catalog_interval_s,
            audio_health_path=str(tmp_path / "audio_health.json"),
        )
        self.loop._running = True
        self.loop._watchdog_backoff_min_s = 0.001
        self.loop._watchdog_backoff_max_s = 0.002
        self.old = {"living": MagicMock(), "cocina": MagicMock()}

    @staticmethod
    def index(card: int) -> int:
        return 100 + card

    def _card_dir(self, port, card):
        (self.sysfs / f"{port}:1.0" / "sound" / f"card{card}").mkdir(parents=True)

    def _sysfs_catalog(self) -> dict[int, int]:
        cards = {}
        for path in self.sysfs.glob("*:1.0/sound/card*"):
            card = int(path.name[4:])
            cards[card] = self.index(card)
        return cards

    def _initialize(self):
        self.catalog = self._sysfs_catalog()

    def _new_stream(self, **_kwargs):
        stream = MagicMock()
        self.streams.append(stream)
        return stream

    def resolve(self, port):
        from src.rooms.room_context import usb_port_to_alsa_card
        card = usb_port_to_alsa_card(port, str(self.sysfs))
        return None if card is None else self.catalog.get(card)

    def renumber_living(self, card: int) -> None:
        """The re-enumerated device comes back as another ALSA card."""
        sound = self.sysfs / f"{self.LIVING_PORT}:1.0" / "sound"
        for path in sound.iterdir():
            path.rmdir()
        (sound / f"card{card}").mkdir()

    def patches(self):
        return (patch("src.pipeline.multi_room_audio_loop.sd", self.sd),
                patch("src.pipeline.multi_room_audio_loop.resolve_mic_usb_port", side_effect=self.resolve))

    async def start(self):
        """Both streams open; Living has been dead past its first-frame grace."""
        for key, stream in self.old.items():
            owner = OwnedInputStream(lambda stream=stream: stream,
                                     lambda key=key: self.loop._inputs._retire_key(key))
            self.loop._inputs._owners[key] = owner
            await owner.open()
            self.loop._streams[key] = stream
        now = time.monotonic()
        self.cocina.opened_ts = now - 60.0
        self.cocina.last_frame_ts = now
        self.living.opened_ts = now - 200.0
        self.living.last_frame_ts = 0.0

    def fail_recoveries(self, times: int) -> None:
        for opening in range(times):
            self.loop._xvf_reboot_budget.note_recovery("living", float(opening + 1))

    async def settle(self):
        await _until(lambda: self.loop._recovery_task is None)

    async def stop(self):
        self.loop._running = False
        await asyncio.gather(*self.loop._cancel_reopen_tasks(), return_exceptions=True)
        if self.loop._recovery_task is not None:
            await asyncio.gather(self.loop._recovery_task, return_exceptions=True)
        await self.loop._audio_health_work.drain_admitted()
        await self.loop._xvf_native.drain_admitted()


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


class TestRebootFlow:
    async def test_dead_room_is_rebooted_retuned_and_reopened_alone(self, tmp_path, caplog):
        rig = _Rig(tmp_path)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch, caplog.at_level(logging.INFO, logger=LOGGER):
            await rig.start()
            rig.fail_recoveries(2)
            rig.loop._recover_dead_streams(["living"])
            await rig.settle()
            await rig.stop()

        assert rig.living_ctrl.reboots == 1
        assert rig.living_ctrl.writes == [("PP_AGCMAXGAIN", [16.0]), ("PP_MIN_NN", [0.4])]
        # The other room is untouched: no reboot, no tuning, no capture cut.
        assert rig.cocina_ctrl.reboots == 0 and rig.cocina_ctrl.writes == []
        rig.old["cocina"].close.assert_not_called()
        assert rig.loop._streams["cocina"] is rig.old["cocina"]
        # The card kept its number, so PortAudio still lists it: no refresh.
        assert rig.sd._terminate.call_count == 0
        rig.old["living"].close.assert_called_once()
        assert rig.loop._streams["living"] in rig.streams
        assert rig.loop._xvf_reboot_budget.reboots_in_window("living") == 1
        assert rig.loop._xvf_reboot_budget.failures("living") == 0
        assert rig.loop._xvf_rebooting == set()
        messages = _messages(caplog)
        assert ("[audio-watchdog] living: isochronous capture dead after 2 reopens; "
                "XVF firmware reboot 1/3 (USB port 1-2.3)") in messages
        assert any("stream reopened after the XVF firmware reboot" in m for m in messages)
        assert not any("Audio recovery failed" in m for m in messages)

    async def test_renumbered_card_requests_the_catalog_refresh(self, tmp_path, caplog):
        rig = _Rig(tmp_path)
        rig.living_ctrl.on_reboot = lambda: rig.renumber_living(8)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch, caplog.at_level(logging.INFO, logger=LOGGER):
            await rig.start()
            rig.fail_recoveries(2)
            rig.loop._recover_dead_streams(["living"])
            await rig.settle()
            await rig.stop()

        assert rig.living_ctrl.reboots == 1
        assert rig.sd._terminate.call_count == 1 and rig.sd._initialize.call_count == 1
        assert rig.living.device_index == rig.index(8)
        assert rig.loop._streams["living"] in rig.streams
        assert rig.loop._streams["cocina"] in rig.streams  # brief cut, back again
        assert rig.living_ctrl.writes  # tuning reapplied before the refresh
        assert any("PortAudio does not list ALSA cards {'living': 8}" in m for m in _messages(caplog))

    async def test_rebooted_room_is_addressed_by_its_port_through_the_real_controller(
            self, tmp_path, monkeypatch):
        """End to end with XvfController over a fake pyusb bus: REBOOT bytes and target."""
        bus = _FakeUsbBus(monkeypatch)
        stuck = bus.add(1, (2, 3), address=9)
        other = bus.add(7, (5, 3), address=4)
        ctrl = XvfController(usb_port="1-2.3", read_timeout_ms=50, max_retries=2)
        rig = _Rig(tmp_path, living_ctrl=ctrl)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch:
            await rig.start()
            rig.fail_recoveries(2)
            rig.loop._recover_dead_streams(["living"])
            await rig.settle()
            await rig.stop()

        assert [t[:5] for t in stuck.transfers] == [(0x40, 0, 7, 48, b"\x01")]
        assert other.transfers == []
        reborn = bus.devices[(1, (2, 3))]
        assert reborn is not stuck and reborn.address == 10
        writes = [t[:5] for t in reborn.transfers if t[0] == 0x40]
        assert writes == [(0x40, 0, 11, 17, struct.pack("<f", 16.0)),
                          (0x40, 0, 22, 17, struct.pack("<f", 0.4))]
        assert ctrl._dev is reborn


class _FakeUsbDevice:
    def __init__(self, bus, ports, address, on_reboot):
        self.bus, self.port_numbers, self.address = bus, ports, address
        self.transfers = []
        self._on_reboot = on_reboot

    def ctrl_transfer(self, bm, brequest, wvalue, windex, data_or_length, timeout):
        if bm == 0xC0:
            self.transfers.append((bm, brequest, wvalue, windex, data_or_length, timeout))
            return [0] + [0] * (data_or_length - 1)
        self.transfers.append((bm, brequest, wvalue, windex, bytes(data_or_length), timeout))
        if (wvalue, windex) == (7, 48):
            self._on_reboot(self)
        return len(data_or_length)


class _FakeUsbBus:
    """usb.core double: a REBOOT makes the device come back with a new address."""

    def __init__(self, monkeypatch):
        self.devices = {}
        core = types.ModuleType("usb.core")
        core.find = self.find
        package = types.ModuleType("usb")
        package.core = core
        monkeypatch.setitem(sys.modules, "usb", package)
        monkeypatch.setitem(sys.modules, "usb.core", core)

    def add(self, bus, ports, *, address):
        device = _FakeUsbDevice(bus, ports, address, self._reenumerate)
        self.devices[(bus, ports)] = device
        return device

    def _reenumerate(self, device):
        self.add(device.bus, device.port_numbers, address=device.address + 1)

    def find(self, *, idVendor=None, idProduct=None, custom_match=None, **_kwargs):
        for device in tuple(self.devices.values()):
            if custom_match is None or custom_match(device):
                return device
        return None


class TestTriggerThreshold:
    async def test_ordinary_recovery_until_the_threshold_then_reboot(self, tmp_path, caplog):
        rig = _Rig(tmp_path)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch, caplog.at_level(logging.INFO, logger=LOGGER):
            await rig.start()
            for expected_failures in (1, 2):
                rig.loop._recover_dead_streams(["living"])
                await rig.settle()
                assert rig.living_ctrl.reboots == 0
                assert rig.loop._xvf_reboot_budget.failures("living") == expected_failures
                # The reopened stream delivers nothing again.
                assert rig.living.opened_ts > 0 and rig.living.last_frame_ts == 0.0
            rig.loop._recover_dead_streams(["living"])
            await rig.settle()
            await rig.stop()

        assert rig.living_ctrl.reboots == 1
        assert any("dead after 2 reopens; XVF firmware reboot 1/3" in m for m in _messages(caplog))

    async def test_a_recovery_owner_already_running_is_not_counted(self, tmp_path):
        rig = _Rig(tmp_path)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch:
            await rig.start()
            blocker = asyncio.Event()
            rig.loop._recovery_task = asyncio.create_task(blocker.wait())
            rig.loop._recover_dead_streams(["living"])
            rig.loop._recover_dead_streams(["living"])
            assert rig.loop._xvf_reboot_budget.failures("living") == 0
            blocker.set()
            await rig.loop._recovery_task
            rig.loop._recovery_task = None
            await rig.stop()

    async def test_audio_from_the_room_resets_the_streak_in_the_watchdog(self, tmp_path):
        rig = _Rig(tmp_path)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch:
            await rig.start()
            rig.fail_recoveries(2)
            rig.living.last_frame_ts = time.monotonic()  # Living delivers again
            rig.loop._watchdog_check_interval_s = 0.005
            rig.loop._recovery_ready = True
            watchdog = asyncio.create_task(rig.loop._stream_watchdog())
            await _until(lambda: rig.loop._xvf_reboot_budget.failures("living") == 0)
            rig.loop._running = False
            await asyncio.wait_for(watchdog, 1.0)
            await rig.stop()
        assert rig.living_ctrl.reboots == 0

    async def test_watchdog_routes_a_dead_room_past_the_threshold_to_the_reboot(self, tmp_path):
        rig = _Rig(tmp_path)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch:
            await rig.start()
            rig.fail_recoveries(2)
            rig.loop._watchdog_check_interval_s = 0.005
            rig.loop._recovery_ready = True
            watchdog = asyncio.create_task(rig.loop._stream_watchdog())
            await _until(lambda: rig.living_ctrl.reboots == 1)
            rig.loop._running = False
            await asyncio.wait_for(watchdog, 1.0)
            await rig.stop()

    async def test_disabled_policy_keeps_the_previous_behaviour(self, tmp_path):
        rig = _Rig(tmp_path, policy=XvfRebootPolicy(enabled=False))
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch:
            await rig.start()
            for _ in range(4):
                rig.loop._recover_dead_streams(["living"])
                await rig.settle()
            await rig.stop()
        assert rig.living_ctrl.reboots == 0


class TestCapAndBackoff:
    async def test_cap_stops_reboots_and_asks_for_a_replug(self, tmp_path, caplog):
        rig = _Rig(tmp_path, policy=XvfRebootPolicy(
            enabled=True, failures_before_reboot=1, max_per_hour=1, settle_s=1.0))
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch, caplog.at_level(logging.INFO, logger=LOGGER):
            await rig.start()
            reboots = []
            for _ in range(6):
                rig.loop._recover_dead_streams(["living"])
                await rig.settle()
                reboots.append(rig.living_ctrl.reboots)
            # 1 failure -> reboot; then 2 failures (backoff) -> capped.
            assert reboots == [0, 1, 1, 1, 1, 1]
            capped = [m for m in _messages(caplog) if "physically unplug and replug" in m]
            assert capped == [
                "[audio-watchdog] living: isochronous capture still dead after 1 XVF firmware "
                "reboots in the last hour (cap 1); no more reboots for now: physically unplug "
                "and replug the microphone at USB port 1-2.3"
            ]
            rig.clock[0] += 3601.0  # the reboot leaves the hourly window
            rig.loop._recover_dead_streams(["living"])
            await rig.settle()
            await rig.stop()
        assert rig.living_ctrl.reboots == 2

    async def test_each_reboot_doubles_the_failures_before_the_next(self, tmp_path):
        rig = _Rig(tmp_path, policy=XvfRebootPolicy(
            enabled=True, failures_before_reboot=1, max_per_hour=5, settle_s=1.0))
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch:
            await rig.start()
            reboots_after = []
            for _ in range(10):
                rig.loop._recover_dead_streams(["living"])
                await rig.settle()
                reboots_after.append(rig.living_ctrl.reboots)
            await rig.stop()
        # recover, REBOOT, recover x2, REBOOT, recover x4, REBOOT
        assert reboots_after == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]


class TestNeverDuringActivity:
    @pytest.mark.parametrize("blocker", ["command", "conversation", "response"])
    async def test_due_reboot_waits_for_the_activity(self, tmp_path, caplog, blocker):
        rig = _Rig(tmp_path)
        if blocker == "command":
            rig.cocina.listening = True
        elif blocker == "conversation":
            rig.follow_up.is_active = True
        else:
            handler = MagicMock()
            handler.is_responding = True
            rig.loop.attach_response_handler(handler)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch, caplog.at_level(logging.INFO, logger=LOGGER):
            await rig.start()
            rig.fail_recoveries(2)
            for _ in range(3):
                rig.loop._recover_dead_streams(["living"])
                assert rig.loop._recovery_task is None
            assert rig.living_ctrl.reboots == 0
            rig.old["living"].close.assert_not_called()
            deferred = [m for m in _messages(caplog) if "XVF firmware reboot deferred" in m]
            assert deferred == [f"[audio-watchdog] living: XVF firmware reboot deferred ({blocker})"]
            rig.cocina.listening = False
            rig.follow_up.is_active = False
            if blocker == "response":
                handler.is_responding = False
            rig.loop._recover_dead_streams(["living"])
            await rig.settle()
            await rig.stop()
        assert rig.living_ctrl.reboots == 1

    async def test_activity_starting_after_admission_cancels_the_reboot(self, tmp_path):
        rig = _Rig(tmp_path)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch:
            await rig.start()
            rig.fail_recoveries(2)
            assert rig.loop._schedule_recovery(["living"], xvf_reboot=True)
            rig.follow_up.is_active = True  # a follow-up starts before it runs
            await rig.settle()
            await rig.stop()
        assert rig.living_ctrl.reboots == 0
        assert rig.loop._xvf_reboot_budget.reboots_in_window("living") == 0
        assert rig.loop._xvf_reboot_budget.failures("living") == 2
        rig.old["living"].close.assert_not_called()


class TestFailOpen:
    @pytest.mark.parametrize("ctrl_options, warning", [
        ({"reboot_error": OSError("usb backend exploded")}, "XVF firmware reboot failed: OSError"),
        ({"outcome": XvfRebootOutcome.NO_DEVICE}, None),
        ({"outcome": XvfRebootOutcome.USB_ERROR, "reenumerates": False},
         "did not re-enumerate within 1s"),
    ])
    async def test_usb_problems_never_break_the_recovery(self, tmp_path, caplog, ctrl_options, warning):
        ctrl = _FakeXvf(_Rig.LIVING_PORT, **ctrl_options)
        rig = _Rig(tmp_path, living_ctrl=ctrl)
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch, caplog.at_level(logging.INFO, logger=LOGGER):
            await rig.start()
            rig.fail_recoveries(2)
            rig.loop._recover_dead_streams(["living"])
            await rig.settle()
            await rig.stop()
        messages = _messages(caplog)
        assert not any("Audio recovery failed" in m for m in messages)
        assert ctrl.reboots == 1
        assert ctrl.writes == []  # nothing re-enumerated: no tuning written
        assert rig.loop._xvf_reboot_budget.reboots_in_window("living") == 1  # bounded anyway
        assert rig.loop._streams["living"] in rig.streams  # the room is reopened
        assert rig.loop._xvf_rebooting == set()
        if warning is not None:
            assert any(warning in m for m in messages)

    async def test_room_without_a_reboot_capable_controller_keeps_recovering(self, tmp_path, caplog):
        rig = _Rig(tmp_path)
        rig.loop._xvf_by_room["living"] = MagicMock(spec=["read_param", "write_param"])
        sd_patch, resolve_patch = rig.patches()
        with sd_patch, resolve_patch, caplog.at_level(logging.INFO, logger=LOGGER):
            await rig.start()
            rig.fail_recoveries(2)
            for _ in range(2):
                rig.loop._recover_dead_streams(["living"])
                await rig.settle()
            await rig.stop()
        unavailable = [m for m in _messages(caplog) if "firmware reboot unavailable" in m]
        assert len(unavailable) == 1
        assert rig.loop._xvf_reboot_budget.failures("living") == 4

    async def test_native_wait_stops_when_a_pause_supersedes_the_epoch(self, tmp_path):
        rig = _Rig(tmp_path)
        epoch = rig.loop._recovery_epoch
        report = rig.loop._reboot_xvf_room(rig.living_ctrl, rig.LIVING_PORT, epoch)
        assert report.reenumerated and report.alsa_card == rig.LIVING_CARD
        _address, timeout_s, abort = rig.living_ctrl.waits[0]
        assert 0 < timeout_s <= 1.0
        assert abort() is False
        rig.loop._recovery_epoch += 1
        assert abort() is True


class TestTuningOnReenumeration:
    def _inventory(self, *devices):
        return UsbInventory(observed_at=time.time(), observed_monotonic=time.monotonic(),
                            success=True, devices=tuple(devices))

    def _living(self, devnum, card=3):
        return UsbMicrophone(usb_port=_Rig.LIVING_PORT, alsa_card=card, devnum=devnum)

    async def _publish(self, rig, inventory):
        rig.loop._microphone_reader = lambda _root: inventory
        await rig.loop._refresh_microphone_inventory()

    async def test_replug_reapplies_the_tuning_to_that_room_only(self, tmp_path, caplog):
        rig = _Rig(tmp_path)
        cocina = UsbMicrophone(usb_port=_Rig.COCINA_PORT, alsa_card=6, devnum=2)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            await self._publish(rig, self._inventory(self._living(4), cocina))
            await self._publish(rig, self._inventory(self._living(4), cocina))
            assert rig.living_ctrl.writes == []  # baseline: start already tuned
            await self._publish(rig, self._inventory(self._living(5), cocina))  # fast replug
            assert rig.living_ctrl.writes == [("PP_AGCMAXGAIN", [16.0]), ("PP_MIN_NN", [0.4])]
            assert rig.living_ctrl.invalidations == 1  # stale handle dropped first
            await self._publish(rig, self._inventory(cocina))  # unplugged: nothing to tune
            assert len(rig.living_ctrl.writes) == 2
            await self._publish(rig, self._inventory(self._living(5, card=4), cocina))
            assert len(rig.living_ctrl.writes) == 4  # plugged back
        assert rig.cocina_ctrl.writes == []
        assert sum("device re-enumerated at 1-2.3" in m for m in _messages(caplog)) == 2

    async def test_own_reboot_in_flight_is_not_tuned_twice(self, tmp_path):
        rig = _Rig(tmp_path)
        await self._publish(rig, self._inventory(self._living(4)))
        rig.loop._xvf_rebooting.add("living")
        await self._publish(rig, self._inventory(self._living(5)))
        assert rig.living_ctrl.writes == []
        rig.loop._xvf_rebooting.discard("living")
        await self._publish(rig, self._inventory(self._living(5)))
        assert rig.living_ctrl.writes == []  # the change was already recorded

    async def test_tuning_disabled_means_no_writes_on_replug(self, tmp_path):
        rig = _Rig(tmp_path, tuning={"apply_on_start": False, "params": {"PP_AGCMAXGAIN": [16.0]}})
        await self._publish(rig, self._inventory(self._living(4)))
        await self._publish(rig, self._inventory(self._living(5)))
        assert rig.living_ctrl.writes == []

    async def test_retune_failure_is_fail_open(self, tmp_path, caplog):
        rig = _Rig(tmp_path)

        def broken(_name, _values):
            raise OSError("gone")
        rig.living_ctrl.write_param = broken
        with caplog.at_level(logging.INFO, logger=LOGGER):
            await self._publish(rig, self._inventory(self._living(4)))
            await self._publish(rig, self._inventory(self._living(5)))
        assert any("PP_AGCMAXGAIN no aplicado" in m for m in _messages(caplog))

    def test_inventory_reads_the_kernel_device_number(self, tmp_path):
        device = tmp_path / "1-2.3"
        device.mkdir()
        (device / "devnum").write_text("5\n")
        (device / "serial").write_text("ABC\n")
        (tmp_path / "1-2.3:1.0" / "sound" / "card3").mkdir(parents=True)
        inventory = read_usb_microphones(tmp_path)
        assert inventory.success
        assert [(d.usb_port, d.alsa_card, d.devnum) for d in inventory.devices] == [("1-2.3", 3, 5)]


class TestConfigurationContract:
    def test_settings_enable_the_reboot_and_main_wires_it(self):
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        settings = yaml.safe_load((root / "config" / "settings.yaml").read_text())
        watchdog = settings["rooms"]["stream_watchdog"]
        policy = XvfRebootPolicy(
            enabled=watchdog["xvf_reboot_enabled"],
            failures_before_reboot=watchdog["xvf_reboot_after_failures"],
            max_per_hour=watchdog["xvf_reboot_max_per_hour"],
            settle_s=watchdog["xvf_reboot_settle_s"],
        )
        assert policy.enabled is True
        assert policy.max_per_hour <= 6 and policy.settle_s <= 30
        main_source = (root / "src" / "main.py").read_text()
        for key in ("xvf_reboot_enabled", "xvf_reboot_after_failures",
                    "xvf_reboot_max_per_hour", "xvf_reboot_settle_s"):
            assert f'watchdog_cfg.get("{key}"' in main_source
        assert "xvf_reboot_policy=xvf_reboot_policy," in main_source
        assert "reboot_enabled=xvf_reboot_policy.enabled" in main_source

    def test_reboot_needs_a_controller_per_mic_even_without_tuning(self):
        from src.main import _build_xvf_controllers

        rooms = {"living": _room("living", 4, "1-2.3"), "sin_puerto": _room("sin_puerto", 5, None)}
        assert _build_xvf_controllers(rooms, {}, {}, ()) == {}
        controllers = _build_xvf_controllers(rooms, {}, {}, (), reboot_enabled=True)
        assert list(controllers) == ["living"]
        assert controllers["living"].usb_port == "1-2.3"

    def test_invalid_policy_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            XvfRebootPolicy(enabled=True, max_per_hour=0)
