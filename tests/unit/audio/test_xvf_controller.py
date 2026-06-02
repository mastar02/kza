"""Tests: XvfController (lector SPENERGY del XVF3800, fail-open)."""
import struct

import pytest

from src.audio.xvf_controller import XvfController


class FakeDev:
    """Device USB falso: devuelve respuestas de ctrl_transfer predefinidas.

    responses: lista de (status_byte, (f0,f1,f2,f3)). Cada ctrl_transfer consume
    una. Si raise=True, ctrl_transfer levanta excepción.
    """

    def __init__(self, responses=None, raise_exc=False):
        self._responses = list(responses or [])
        self.raise_exc = raise_exc
        self.calls = 0

    def ctrl_transfer(self, bm, brequest, wvalue, windex, length, timeout):
        self.calls += 1
        if self.raise_exc:
            raise OSError("usb error")
        status, floats = self._responses.pop(0)
        return [status] + list(struct.pack("<ffff", *floats))


def _ctrl(device):
    return XvfController(device=device, max_retries=5)


def test_read_parses_four_floats():
    c = _ctrl(FakeDev([(0, (1.0, 2.0, 3.0, 4.0))]))
    vals = c.read_spenergy()
    assert vals == pytest.approx((1.0, 2.0, 3.0, 4.0))


def test_read_retries_then_success():
    dev = FakeDev([(64, (0, 0, 0, 0)), (64, (0, 0, 0, 0)), (0, (0, 0, 0, 29000.0))])
    c = _ctrl(dev)
    vals = c.read_spenergy()
    assert vals[3] == pytest.approx(29000.0)
    assert dev.calls == 3  # 2 retries + 1 success


def test_read_unknown_status_returns_none():
    c = _ctrl(FakeDev([(99, (0, 0, 0, 0))]))
    assert c.read_spenergy() is None


def test_read_exhausts_retries_returns_none():
    # siempre RETRY -> agota max_retries -> None (no cuelga)
    c = _ctrl(FakeDev([(64, (0, 0, 0, 0))] * 50))
    assert c.read_spenergy() is None


def test_read_exception_fail_open():
    c = _ctrl(FakeDev(raise_exc=True))
    assert c.read_spenergy() is None


def test_read_no_device_fail_open(monkeypatch):
    c = XvfController(device=None)
    monkeypatch.setattr(c, "open", lambda: False)
    assert c.read_spenergy() is None


def test_peak_since_returns_max_in_window():
    c = _ctrl(FakeDev())
    c._record(100.0, now=10.0)
    c._record(29000.0, now=11.0)
    c._record(50.0, now=12.0)
    assert c.peak_since(10.5) == pytest.approx(29000.0)


def test_peak_since_empty_window_is_none():
    # None = fail-open: el caller procesa el comando si no hay datos
    c = _ctrl(FakeDev())
    assert c.peak_since(0.0) is None
    c._record(500.0, now=5.0)
    assert c.peak_since(10.0) is None  # nada con ts >= 10.0


def test_record_evicts_old_samples():
    c = XvfController(device=FakeDev(), window_s=2.0)
    c._record(1.0, now=10.0)
    c._record(2.0, now=11.0)
    c._record(3.0, now=13.0)  # evicta lo anterior a 11.0
    # solo quedan ts >= 11.0
    assert c.peak_since(0.0) == pytest.approx(3.0)
    with c._lock:
        assert all(ts >= 11.0 for ts, _ in c._samples)


def test_poller_start_stop_with_fake_device():
    dev = FakeDev([(0, (0, 0, 0, 1234.0))] * 1000)
    c = XvfController(device=dev, poll_interval_s=0.005)
    assert c.start() is True
    # dar tiempo a unas pocas lecturas
    import time
    time.sleep(0.05)
    c.stop()
    assert dev.calls > 0
    assert c.peak_since(0.0) == pytest.approx(1234.0)
