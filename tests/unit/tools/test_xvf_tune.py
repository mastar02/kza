"""Tests: conversión de valores de la CLI de tuning del XVF3800."""
import pytest

from tools.xvf_tune import parse_values


def test_parse_float_param():
    assert parse_values("PP_AGCMAXGAIN", ["16.0"]) == [16.0]


def test_parse_radians_pair():
    vals = parse_values("AEC_FIXEDBEAMSAZIMUTH_VALUES", ["1.5", "4.6"])
    assert vals == [1.5, 4.6]


def test_parse_int_param():
    vals = parse_values("PP_AGCONOFF", ["0"])
    assert vals == [0]
    assert isinstance(vals[0], int)


def test_parse_uint8_pair():
    assert parse_values("AUDIO_MGR_OP_L", ["0", "1"]) == [0, 1]


def test_parse_unknown_param_raises():
    with pytest.raises(ValueError, match="desconocido"):
        parse_values("NO_EXISTE", ["1"])


def test_parse_wrong_count_raises():
    with pytest.raises(ValueError, match="espera 1"):
        parse_values("PP_AGCMAXGAIN", ["1.0", "2.0"])


def test_cli_write_unknown_param_clean_exit():
    # Typo del operador → mensaje limpio + exit 2, nunca traceback crudo.
    # La validación corre ANTES de abrir el device (no requiere hardware).
    from tools.xvf_tune import main
    assert main(["--write", "NO_EXISTE", "1"]) == 2


def test_cli_write_bad_value_clean_exit():
    from tools.xvf_tune import main
    assert main(["--write", "PP_AGCMAXGAIN", "banana"]) == 2


def test_cli_write_out_of_range_clean_exit():
    from tools.xvf_tune import main
    assert main(["--write", "PP_AGCMAXGAIN", "0.0"]) == 2


# --port (2026-07-25): con DOS XVF3800 conectados, sin puerto la CLI escribe
# sobre el que enumere primero — es decir, sobre el mic de otra habitación.

def _fake_controller(captured):
    class FakeCtrl:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def open(self):
            return False  # corta antes de tocar USB

    return FakeCtrl


def test_cli_passes_port_to_controller(monkeypatch):
    import tools.xvf_tune as xvf_tune

    captured = {}
    monkeypatch.setattr(xvf_tune, "XvfController", _fake_controller(captured))
    rc = xvf_tune.main(["--port", "5-5.4", "--read", "PP_AGCMAXGAIN"])
    assert captured.get("usb_port") == "5-5.4"
    assert rc == 1  # open() False → error limpio, sin traceback


def test_cli_without_port_leaves_controller_unbound(monkeypatch):
    import tools.xvf_tune as xvf_tune

    captured = {}
    monkeypatch.setattr(xvf_tune, "XvfController", _fake_controller(captured))
    xvf_tune.main(["--read", "PP_AGCMAXGAIN"])
    assert captured.get("usb_port") is None
