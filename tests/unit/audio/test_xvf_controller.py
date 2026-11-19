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


class TestFase3TuningParams:
    """Params anti-TV agregados 2026-07-16 (IDs del xvf_host.py oficial)."""

    @pytest.mark.parametrize("name,resid,cmdid", [
        ("AUDIO_MGR_MIC_GAIN", 35, 0),
        ("PP_MIN_NS", 17, 21),
        ("PP_MIN_NN", 17, 22),
        ("PP_FMIN_SPEINDEX", 17, 30),
        ("PP_ATTNS_NOMINAL", 17, 33),
        ("PP_ATTNS_SLOPE", 17, 34),
    ])
    def test_write_float_ids_exactos(self, name, resid, cmdid):
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param(name, [0.5]) is True
        _, _, wvalue, windex, payload = dev.writes[0]
        assert (windex, wvalue) == (resid, cmdid)
        assert payload == struct.pack("<f", 0.5)

    def test_write_attns_mode_int32(self):
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("PP_ATTNS_MODE", [1]) is True
        _, _, wvalue, windex, payload = dev.writes[0]
        assert (windex, wvalue) == (17, 32)
        assert payload == struct.pack("<i", 1)

    @pytest.mark.parametrize("name,bad", [
        ("PP_MIN_NN", 1.5),         # oficial [0.0 .. 1.0]
        ("PP_ATTNS_MODE", 2),        # oficial 0,1
        ("PP_ATTNS_SLOPE", 5.1),     # oficial [0.0 .. 5.0]
        ("PP_FMIN_SPEINDEX", 8000.0),  # oficial [0.0 .. 7999.0]
    ])
    def test_write_fuera_de_rango_oficial_raises(self, name, bad):
        c = XvfController(device=FakeRWDev())
        with pytest.raises(ValueError, match="rango"):
            c.write_param(name, [bad])

    def test_mic_gain_sin_rango_oficial_no_valida(self):
        # AUDIO_MGR_MIC_GAIN no declara rango en el oficial → la regla del
        # dict es no inventar validación (0.0 silencia el mic: cuidado
        # operativo, no de programación).
        dev = FakeRWDev()
        c = XvfController(device=dev)
        assert c.write_param("AUDIO_MGR_MIC_GAIN", [90.0]) is True


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


# ============================================================
# Binding por puerto USB (2026-07-25): con DOS XVF3800 conectados,
# usb.core.find() devuelve el PRIMERO que matchee VID/PID. Sin binding, el
# gate SPENERGY de una room puede quedar leyendo el mic de OTRA habitación.
# Formato de puerto = el mismo de `mic_usb_port` en RoomConfig:
# "5-5.4" → bus 5, cadena de puertos (5, 4).
# ============================================================

import sys
import types

from src.audio.xvf_controller import parse_usb_port


class FakeUsbDev:
    """Device pyusb falso: expone bus/port_numbers como los reales."""

    def __init__(self, bus, port_numbers):
        self.bus = bus
        self.port_numbers = port_numbers


def _install_fake_usb(monkeypatch, devices):
    """Instala un usb.core falso que replica la semántica de find()."""
    core = types.ModuleType("usb.core")

    def find(*, idVendor=None, idProduct=None, custom_match=None, **kw):
        for d in devices:
            if custom_match is None or custom_match(d):
                return d
        return None

    core.find = find
    pkg = types.ModuleType("usb")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "usb", pkg)
    monkeypatch.setitem(sys.modules, "usb.core", core)


class TestUsbPortBinding:
    def test_parse_usb_port_splits_bus_and_port_chain(self):
        assert parse_usb_port("5-5.4") == (5, (5, 4))

    def test_parse_usb_port_single_port(self):
        assert parse_usb_port("3-1") == (3, (1,))

    def test_parse_usb_port_invalid_returns_none(self):
        assert parse_usb_port("no-es-un-puerto") is None
        assert parse_usb_port("") is None
        assert parse_usb_port(None) is None

    def test_open_selects_device_matching_usb_port(self, monkeypatch):
        escritorio = FakeUsbDev(3, (1, 4))
        cocina = FakeUsbDev(5, (5, 4))
        _install_fake_usb(monkeypatch, [escritorio, cocina])
        c = XvfController(usb_port="5-5.4")
        assert c.open() is True
        assert c._dev is cocina  # NO el primero de la lista

    def test_open_without_usb_port_takes_first_device(self, monkeypatch):
        # Compat: sin puerto declarado, comportamiento actual.
        escritorio = FakeUsbDev(3, (1, 4))
        cocina = FakeUsbDev(5, (5, 4))
        _install_fake_usb(monkeypatch, [escritorio, cocina])
        c = XvfController()
        assert c.open() is True
        assert c._dev is escritorio

    def test_open_fails_when_usb_port_matches_no_device(self, monkeypatch):
        # Fail-open: mejor SIN gate que gateando con el mic de otra habitación.
        escritorio = FakeUsbDev(3, (1, 4))
        _install_fake_usb(monkeypatch, [escritorio])
        c = XvfController(usb_port="5-5.4")
        assert c.open() is False
        assert c._dev is None


# ============================================================
# Firmware REBOOT (T048-U1, incidente 2026-11-19): un XVF3800 con el endpoint
# isócrono trabado solo se recuperó con el comando REBOOT del firmware. Es un
# método deliberado, separado de write_param, y sigue fuera de PARAMETERS.
# ============================================================

import time

from src.audio.xvf_controller import XvfRebootOutcome, XvfRebootResult


class FakeBusDev(FakeUsbDev):
    """Device pyusb falso con dirección USB y control transfers registrados."""

    def __init__(self, bus, port_numbers, address, raise_exc=None):
        super().__init__(bus, port_numbers)
        self.address = address
        self.raise_exc = raise_exc
        self.transfers = []

    def ctrl_transfer(self, bm, brequest, wvalue, windex, data_or_length, timeout):
        self.transfers.append((bm, brequest, wvalue, windex, bytes(data_or_length), timeout))
        if self.raise_exc is not None:
            raise self.raise_exc
        return len(data_or_length)


class _Bus:
    """usb.core falso cuyo contenido cambia entre llamadas a find()."""

    def __init__(self, monkeypatch, states):
        self.states = list(states)  # una lista de devices por find()
        self.calls = 0
        core = types.ModuleType("usb.core")
        core.find = self.find
        pkg = types.ModuleType("usb")
        pkg.core = core
        monkeypatch.setitem(sys.modules, "usb", pkg)
        monkeypatch.setitem(sys.modules, "usb.core", core)

    def find(self, *, idVendor=None, idProduct=None, custom_match=None, **kw):
        devices = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        for d in devices:
            if custom_match is None or custom_match(d):
                return d
        return None


class TestFirmwareReboot:
    def test_reboot_payload_and_addressing_are_exact(self, monkeypatch):
        other = FakeBusDev(3, (1, 4), address=4)
        target = FakeBusDev(7, (5, 3), address=9)
        _Bus(monkeypatch, [[other, target]])
        c = XvfController(usb_port="7-5.3")
        result = c.reboot()
        assert result == XvfRebootResult(XvfRebootOutcome.SENT, previous_address=9)
        # vendor OUT (0x40), bRequest 0, wValue = cmdid 7, wIndex = resid 48,
        # payload = un uint8 con valor 1.
        assert [t[:5] for t in target.transfers] == [(0x40, 0, 7, 48, b"\x01")]
        assert other.transfers == []  # solo el device de ESTA room
        assert c._dev is None  # el handle viejo se retira

    def test_reboot_is_not_reachable_through_parameters(self):
        for name in ("REBOOT", "SAVE_CONFIGURATION", "CLEAR_CONFIGURATION"):
            assert name not in PARAMETERS
            with pytest.raises(ValueError):
                XvfController(device=FakeRWDev()).write_param(name, [1])

    def test_reboot_ignores_a_stale_cached_handle(self, monkeypatch):
        stale = FakeBusDev(7, (5, 3), address=9, raise_exc=OSError("No such device"))
        fresh = FakeBusDev(7, (5, 3), address=12)
        _Bus(monkeypatch, [[fresh]])
        c = XvfController(usb_port="7-5.3", device=stale)
        assert c.reboot().outcome is XvfRebootOutcome.SENT
        assert stale.transfers == [] and len(fresh.transfers) == 1

    def test_reboot_without_device_fails_open(self, monkeypatch):
        _Bus(monkeypatch, [[FakeBusDev(3, (1, 4), address=4)]])
        c = XvfController(usb_port="7-5.3")
        assert c.reboot() == XvfRebootResult(XvfRebootOutcome.NO_DEVICE)

    @pytest.mark.parametrize("usb_port", [None, "", "not-a-port"])
    def test_unbound_controller_never_reboots_the_first_xvf_on_the_bus(self, monkeypatch, usb_port):
        """T048-U1-RV1 F2: sin puerto propio no se rebootea el mic de OTRA room."""
        other_room = FakeBusDev(7, (5, 3), address=4)
        bus = _Bus(monkeypatch, [[other_room]])
        c = XvfController(usb_port=usb_port)
        assert c.reboot() == XvfRebootResult(XvfRebootOutcome.NO_DEVICE)
        assert other_room.transfers == []
        assert bus.calls == 0  # ni siquiera se busca
        assert c.wait_reenumerated(4, 5.0, sleep=lambda _s: None) is False
        assert c._dev is None and bus.calls == 0

    def test_reboot_without_pyusb_fails_open(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "usb.core", None)
        c = XvfController(usb_port="7-5.3")
        assert c.reboot().outcome is XvfRebootOutcome.NO_DEVICE

    def test_reboot_transfer_error_fails_open_and_retires_the_handle(self, monkeypatch):
        dev = FakeBusDev(7, (5, 3), address=9, raise_exc=OSError("pipe error"))
        _Bus(monkeypatch, [[dev]])
        c = XvfController(usb_port="7-5.3")
        c._record(5.0, time.time(), c._device_generation)
        result = c.reboot()
        assert result == XvfRebootResult(XvfRebootOutcome.USB_ERROR, previous_address=9)
        assert c._dev is None
        assert c.peak_since(0) is None  # muestras del device viejo descartadas

    def test_wait_adopts_the_device_with_a_new_address(self, monkeypatch):
        old = FakeBusDev(7, (5, 3), address=9)
        new = FakeBusDev(7, (5, 3), address=10)
        bus = _Bus(monkeypatch, [[old], [], [new]])
        c = XvfController(usb_port="7-5.3")
        sleeps = []
        assert c.wait_reenumerated(9, 5.0, sleep=sleeps.append) is True
        assert c._dev is new and bus.calls == 3 and len(sleeps) == 2

    def test_wait_with_unknown_address_needs_the_device_to_leave_first(self, monkeypatch):
        old = FakeBusDev(7, (5, 3), address=9)
        _Bus(monkeypatch, [[old], [old], [], [old]])
        c = XvfController(usb_port="7-5.3")
        assert c.wait_reenumerated(None, 5.0, sleep=lambda _s: None) is True
        assert c._dev is old

    def test_wait_times_out_when_the_device_never_comes_back(self, monkeypatch):
        _Bus(monkeypatch, [[]])
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        c = XvfController(usb_port="7-5.3")
        assert c.wait_reenumerated(9, 1.0, poll_interval_s=0.25, sleep=sleep,
                                   monotonic=lambda: clock[0]) is False
        assert c._dev is None and clock[0] == pytest.approx(1.0)

    def test_wait_stops_when_the_owner_aborts(self, monkeypatch):
        bus = _Bus(monkeypatch, [[]])
        c = XvfController(usb_port="7-5.3")
        assert c.wait_reenumerated(9, 60.0, abort=lambda: True, sleep=lambda _s: None) is False
        assert bus.calls == 0
