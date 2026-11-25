"""Native sd_notify: real datagrams when NOTIFY_SOCKET is set, clean no-op otherwise."""

import asyncio
import os
import pathlib
import socket

import pytest

from nexa_ml.notify import SdNotifier


def test_noop_without_notify_socket() -> None:
    notifier = SdNotifier(env={})
    assert not notifier.enabled
    assert notifier.notify("READY=1") is False
    assert notifier.watchdog_interval_s() == 15.0


def test_sends_datagrams(short_dir: pathlib.Path) -> None:
    path = short_dir / "notify.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
        receiver.bind(str(path))
        receiver.settimeout(2)
        notifier = SdNotifier(env={"NOTIFY_SOCKET": str(path)})
        assert notifier.notify("READY=1", "STATUS=ok")
        assert receiver.recv(1024) == b"READY=1\nSTATUS=ok"
        assert notifier.notify("STOPPING=1")
        assert receiver.recv(1024) == b"STOPPING=1"


def test_send_failure_is_reported_not_raised(short_dir: pathlib.Path) -> None:
    notifier = SdNotifier(env={"NOTIFY_SOCKET": str(short_dir / "missing.sock")})
    assert notifier.enabled
    assert notifier.notify("READY=1") is False


def test_watchdog_interval_from_env() -> None:
    pid = str(os.getpid())
    assert SdNotifier(env={"WATCHDOG_USEC": "30000000", "WATCHDOG_PID": pid}).watchdog_interval_s() == 15.0
    assert SdNotifier(env={"WATCHDOG_USEC": "10000000"}).watchdog_interval_s() == 5.0
    assert SdNotifier(env={"WATCHDOG_USEC": "10000000", "WATCHDOG_PID": "1"}).watchdog_interval_s() == 15.0
    assert SdNotifier(env={"WATCHDOG_USEC": "junk"}).watchdog_interval_s() == 15.0


async def test_watchdog_loop_pings(short_dir: pathlib.Path) -> None:
    path = short_dir / "wd.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
        receiver.bind(str(path))
        receiver.setblocking(False)
        task = asyncio.create_task(SdNotifier(env={"NOTIFY_SOCKET": str(path)}).watchdog_loop(0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert receiver.recv(64) == b"WATCHDOG=1"


async def test_watchdog_loop_disabled_returns() -> None:
    await asyncio.wait_for(SdNotifier(env={}).watchdog_loop(0.01), timeout=1)
