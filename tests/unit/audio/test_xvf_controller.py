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


# ============================================================
# L-1 (2026-06-04): write/read genérico de parámetros por nombre.
# Protocolo verificado contra python_control/xvf_host.py oficial de Seeed:
# WRITE = ctrl_transfer(0x40, 0, wValue=cmdid SIN bit 0x80, wIndex=resid,
# payload tipado LE). Solo RAM — SAVE_CONFIGURATION/REBOOT deliberadamente
# NO expuestos (issue #8 Seeed: save puede dejar el device sin enumerar).
# ============================================================

from src.audio.xvf_controller import PARAMETERS


class FakeRWDev:
    """Device USB falso bidireccional: captura writes, sirve reads tipados."""

    def __init__(self, read_responses=None, raise_exc=False):
        self.raise_exc = raise_exc
        self.writes = []  # (bm, brequest, wvalue, windex, payload_bytes)
        self._read_responses = list(read_responses or [])

    def ctrl_transfer(self, bm, brequest, wvalue, windex, data_or_length, timeout):
        if self.raise_exc:
            raise OSError("usb error")
        if bm == 0x40:  # CTRL_OUT — write: data_or_length es el payload
            self.writes.append((bm, brequest, wvalue, windex, bytes(bytearray(data_or_length))))
            return len(data_or_length)
        status, payload = self._read_responses.pop(0)
        return [status] + list(payload)


class TestWriteParam:
    def test_write_float_protocol_exact(self):
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("PP_AGCMAXGAIN", [8.0]) is True
        assert len(dev.writes) == 1
        bm, breq, wvalue, windex, payload = dev.writes[0]
        assert bm == 0x40          # CTRL_OUT|VENDOR|DEVICE
        assert breq == 0
        assert wvalue == 11        # cmdid de PP_AGCMAXGAIN, SIN bit 0x80
        assert windex == 17        # resid del post-processor
        assert payload == struct.pack("<f", 8.0)

    def test_write_int32_protocol(self):
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("PP_AGCONOFF", [0]) is True
        _, _, wvalue, windex, payload = dev.writes[0]
        assert (windex, wvalue) == (17, 10)
        assert payload == struct.pack("<i", 0)

    def test_write_uint8_pair_protocol(self):
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("AUDIO_MGR_OP_L", [0, 1]) is True
        _, _, wvalue, windex, payload = dev.writes[0]
        assert (windex, wvalue) == (35, 15)
        assert payload == b"\x00\x01"

    def test_write_radians_pair(self):
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("AEC_FIXEDBEAMSAZIMUTH_VALUES", [1.5, 4.6]) is True
        _, _, wvalue, windex, payload = dev.writes[0]
        assert (windex, wvalue) == (33, 81)
        assert payload == struct.pack("<ff", 1.5, 4.6)

    def test_write_readonly_raises(self):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="read-only"):
            c.write_param("AEC_SPENERGY_VALUES", [0.0, 0.0, 0.0, 0.0])

    def test_write_unknown_param_raises(self):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="desconocido"):
            c.write_param("NO_EXISTE", [1])

    def test_write_wrong_count_raises(self):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="espera 1"):
            c.write_param("PP_AGCMAXGAIN", [1.0, 2.0])

    def test_write_usb_error_fail_open_false(self):
        # Error de hardware/USB → False sin excepción (fail-open operativo;
        # los ValueError de arriba son errores de PROGRAMACIÓN y sí explotan).
        c = XvfController(device=FakeRWDev(raise_exc=True))
        assert c.write_param("PP_AGCMAXGAIN", [8.0]) is False

    def test_write_no_device_fail_open_false(self, monkeypatch):
        c = XvfController(device=None)
        monkeypatch.setattr(c, "open", lambda: False)
        assert c.write_param("PP_AGCMAXGAIN", [8.0]) is False


class TestDangerousParamsNotExposed:
    """Regla de oro del tuning: SOLO RAM. Nada que persista o rebootee."""

    @pytest.mark.parametrize("name", [
        "SAVE_CONFIGURATION", "CLEAR_CONFIGURATION", "REBOOT",
        "TEST_CORE_BURN", "USB_BIT_DEPTH", "TEST_AEC_DISABLE_CONTROL",
    ])
    def test_param_not_in_map(self, name):
        assert name not in PARAMETERS


class TestReadParam:
    def test_read_float_param(self):
        dev = FakeRWDev(read_responses=[(0, struct.pack("<f", 64.0))])
        c = XvfController(device=dev)
        vals = c.read_param("PP_AGCMAXGAIN")
        assert vals == pytest.approx((64.0,))

    def test_read_uint8_version(self):
        dev = FakeRWDev(read_responses=[(0, bytes([3, 2, 1]))])
        c = XvfController(device=dev)
        assert c.read_param("VERSION") == (3, 2, 1)

    def test_read_int32_param(self):
        dev = FakeRWDev(read_responses=[(0, struct.pack("<i", 1))])
        c = XvfController(device=dev)
        assert c.read_param("PP_AGCONOFF") == (1,)

    def test_read_param_usb_error_fail_open(self):
        c = XvfController(device=FakeRWDev(raise_exc=True))
        assert c.read_param("PP_AGCMAXGAIN") is None

    def test_read_param_unknown_raises(self):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="desconocido"):
            c.read_param("NO_EXISTE")

    def test_read_spenergy_delegates_consistently(self):
        # read_spenergy debe seguir funcionando igual (mismo protocolo/retry)
        dev = FakeRWDev(read_responses=[(0, struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))])
        c = XvfController(device=dev)
        assert c.read_spenergy() == pytest.approx((1.0, 2.0, 3.0, 4.0))


class TestWriteParamRangeValidation:
    """Fix review Fase 1: validar rangos oficiales antes de escribir a RAM.

    Los rangos vienen de las descripciones del xvf_host.py oficial (p.ej.
    PP_AGCMAXGAIN 'Valid range: [1.0 .. 1000.0]'). Un fat-finger (0, negativo)
    NO debe llegar al DSP de producción."""

    def test_write_below_range_raises(self):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="rango"):
            c.write_param("PP_AGCMAXGAIN", [0.0])  # oficial: [1.0 .. 1000.0]

    def test_write_above_range_raises(self):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="rango"):
            c.write_param("PP_AGCMAXGAIN", [1001.0])

    def test_write_onoff_only_binary(self):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="rango"):
            c.write_param("PP_AGCONOFF", [2])

    def test_write_in_range_passes(self):
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("PP_AGCMAXGAIN", [16.0]) is True
        assert len(dev.writes) == 1

    def test_write_param_without_known_range_not_blocked(self):
        # Parámetros sin rango oficial declarado (azimuth) no se validan.
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("AEC_FIXEDBEAMSAZIMUTH_VALUES", [1.5, 4.6]) is True

    def test_usb_lock_exists_and_wraps_transfers(self):
        # Lock dedicado para serializar ctrl_transfer (poller vs write).
        c = XvfController(device=FakeRWDev())
        assert hasattr(c, "_usb_lock")
        assert c._usb_lock is not c._lock  # NO reusar el lock del deque
