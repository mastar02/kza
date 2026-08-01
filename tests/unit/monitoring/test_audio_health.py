"""The deafness signal must distinguish a silent room from a dead mic."""
import json

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


def test_write_audio_health_is_atomic_and_readable(tmp_path):
    path = tmp_path / "audio_health.json"
    write_audio_health(
        str(path),
        rooms={"cocina": (500.0, 100.0), "escritorio": (0.0, 100.0)},
        now_wall=1000.0,
        now_mono=520.0,
    )
    data = json.loads(path.read_text())
    assert data["wall"] == 1000.0
    assert data["rooms"]["cocina"]["age_s"] == 20.0
    assert data["rooms"]["cocina"]["ever"] is True
    assert data["rooms"]["escritorio"]["ever"] is False
    assert not list(tmp_path.glob("*.tmp"))  # sin temporales huérfanos
