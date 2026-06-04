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
