"""The deafness signal must distinguish a silent room from a dead mic."""
import json
import os

import pytest

import src.monitoring.audio_health as audio_health
from src.monitoring.audio_health import evaluate_health, write_audio_health


def test_room_delivering_audio_is_healthy():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 2.0, "ever": True}}}
    assert evaluate_health(snap, now_wall=1002.0, deaf_after_s=120.0) == []


def test_room_silent_past_threshold_is_deaf():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 300.0, "ever": True}}}
    assert evaluate_health(snap, now_wall=1002.0, deaf_after_s=120.0) == ["cocina"]


def test_stale_snapshot_counts_as_deaf():
    """If the writer stopped writing, the process itself is wedged."""
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 1.0, "ever": True}}}
    # el snapshot tiene 10 minutos: nadie lo actualiza
    assert evaluate_health(snap, now_wall=1600.0, deaf_after_s=120.0) == ["cocina"]


def test_room_that_never_delivered_is_not_deaf_within_grace():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 30.0, "ever": False}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=120.0, first_frame_grace_s=180.0
    ) == []


def test_room_that_never_delivered_past_grace_is_deaf():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 200.0, "ever": False}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=120.0, first_frame_grace_s=180.0
    ) == ["cocina"]


# Los cuatro tests de arriba usan age_s=30 (por debajo de deaf_after_s=120 Y
# de grace=180) o age_s=200 (por encima de ambos), así que nunca ejercitan
# la banda 120-180 donde el threshold correcto depende de `ever`. Una
# mutación que fije `threshold = deaf_after_s` siempre, o que lo fije a
# `first_frame_grace_s` siempre, pasa los cuatro de arriba sin que nada la
# note. Los dos de acá caen justo en esa banda (age_s=150) y sí distinguen.


def test_never_delivered_inside_grace_band_is_not_deaf():
    """age_s=150 está por encima de deaf_after_s=120 pero por debajo de
    grace=180. Una room que nunca entregó frame usa el umbral de gracia, así
    que 150 < 180 => sana. Si una mutación forzara threshold=deaf_after_s,
    150 > 120 daría sorda por error — este test lo detecta."""
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 150.0, "ever": False}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=120.0, first_frame_grace_s=180.0
    ) == []


def test_already_delivered_inside_grace_band_is_deaf():
    """Misma edad (150s), pero la room YA entregó audio antes (`ever=True`),
    así que el umbral normal (120s) aplica, no la gracia de primer frame
    (180s): 150 > 120 => sorda. Si una mutación forzara
    threshold=first_frame_grace_s siempre, 150 < 180 daría sana por error —
    este test lo detecta."""
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 150.0, "ever": True}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=120.0, first_frame_grace_s=180.0
    ) == ["cocina"]


# El default de producción del poller es --deaf-after-s 300, mayor que la
# gracia (180) — el régimen INVERSO al de los tests de arriba
# (deaf_after_s=120 < grace=180). Sin esto, nunca se prueba que la gracia
# siga aplicando (más corta que el umbral normal) cuando deaf_after_s crece.


def test_never_delivered_past_grace_is_deaf_under_production_regime():
    """deaf_after_s=300 > grace=180 (default real del poller). age_s=240 cae
    después de la gracia pero bien antes del umbral normal: si la gracia no
    se respetara (p.ej. se usara siempre deaf_after_s=300), esto reportaría
    sana durante 60s más de lo debido tras un arranque que no prende."""
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 240.0, "ever": False}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=300.0, first_frame_grace_s=180.0
    ) == ["cocina"]


def test_already_delivered_before_deaf_after_s_is_not_deaf_under_production_regime():
    """Mismo régimen de producción y misma edad (240s), pero la room YA
    entregó audio (`ever=True`): el umbral normal de 300s aplica, no la
    gracia de 180s. 240 < 300 => sana. Si una mutación usara siempre la
    gracia, esto reportaría sorda por error después de solo 180s de
    silencio normal, mucho antes del umbral real de producción."""
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 240.0, "ever": True}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=300.0, first_frame_grace_s=180.0
    ) == []


def test_write_audio_health_is_atomic_and_readable(tmp_path):
    path = tmp_path / "audio_health.json"
    write_audio_health(
        str(path),
        rooms={"cocina": (980.0, True), "escritorio": (900.0, False)},
        now_wall=1000.0,
    )
    data = json.loads(path.read_text())
    assert data["wall"] == 1000.0
    assert data["rooms"]["cocina"]["age_s"] == 20.0
    assert data["rooms"]["cocina"]["ever"] is True
    assert data["rooms"]["escritorio"]["ever"] is False
    assert not list(tmp_path.glob("*.tmp"))  # sin temporales huérfanos


def test_write_audio_health_uses_tempfile_and_atomic_replace(tmp_path, monkeypatch):
    """El test de arriba solo mira el resultado final, así que no distingue
    escribir atómicamente (mkstemp + os.replace) de escribir directo al
    destino con open(path, "w") — ambos producen el mismo JSON al final si
    nada falla a mitad de camino. Este test espía os.replace() para
    confirmar que el mecanismo real es el atómico: el destino nunca se toca
    directo, se escribe a un .tmp aparte y se renombra al final."""
    path = tmp_path / "audio_health.json"
    replace_calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replace_calls.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(audio_health.os, "replace", spy_replace)

    write_audio_health(
        str(path), rooms={"cocina": (980.0, True)}, now_wall=1000.0
    )

    assert len(replace_calls) == 1
    src, dst = replace_calls[0]
    assert dst == str(path)
    assert src != str(path)
    assert src.endswith(".tmp")


def test_write_audio_health_cleans_up_tmp_on_write_error(tmp_path, monkeypatch):
    """Si algo falla mientras se escribe el temporal (disco lleno, lo que
    sea), no debe quedar un .tmp huérfano ni tocarse el destino — y el error
    tiene que propagar, no tragarse. Fuerzo la falla en json.dump(), que
    corre DESPUÉS de crear el temporal pero ANTES del replace, para
    ejercitar específicamente la rama except/cleanup."""
    path = tmp_path / "audio_health.json"

    def raise_on_dump(*_args, **_kwargs):
        raise ValueError("fallo de escritura simulado")

    monkeypatch.setattr(audio_health.json, "dump", raise_on_dump)

    with pytest.raises(ValueError):
        write_audio_health(
            str(path), rooms={"cocina": (980.0, True)}, now_wall=1000.0
        )

    assert not list(tmp_path.glob("*.tmp"))  # el temporal se limpió, no quedó huérfano
    assert not path.exists()  # el destino nunca se tocó


def test_write_audio_health_file_is_readable_by_other_users(tmp_path):
    """mkstemp crea el temporal en 0600 (solo el dueño). El poller externo
    (tools/audio_watchdog_alert.py) está diseñado para correr bajo OTRO
    usuario — si el archivo final queda 0600, ese poller recibe
    PermissionError y lo reporta como sordera/anomalía para siempre. El
    modo final debe permitir lectura a otros usuarios."""
    path = tmp_path / "audio_health.json"
    write_audio_health(
        str(path), rooms={"cocina": (980.0, True)}, now_wall=1000.0
    )
    mode = os.stat(path).st_mode & 0o777
    assert mode & 0o044 == 0o044  # legible por group y other, no solo por el dueño
