"""La config de rooms apunta a entidades reales (verificado contra /api/states 2026-05-31)."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _rooms():
    cfg = yaml.safe_load((ROOT / "config/settings.yaml").read_text())
    return cfg["rooms"]


def test_default_light_uses_grupo():
    r = _rooms()
    assert r["living"]["default_light"] == "light.grupo_living"
    assert r["cocina"]["default_light"] == "light.grupo_cocina"
    assert r["bano"]["default_light"] == "light.grupo_bano"
    assert r["escritorio"]["default_light"] == "light.grupo_escritorio"
    assert r["hall"]["default_light"] == "light.grupo_pasillo"  # hall≈pasillo


def test_escritorio_real_sensors():
    e = _rooms()["escritorio"]
    assert e["motion_sensor"] == "binary_sensor.escritorio_motion"
    assert e["temperature_sensor"] == "sensor.blink_escritorio_temperature"


def test_escritorio_no_phantom_devices():
    # climate/media/tts inexistentes en HA → no deben estar activos (comentados).
    e = _rooms()["escritorio"]
    assert "default_climate" not in e
    assert "default_media_player" not in e
    assert "tts_speaker" not in e
