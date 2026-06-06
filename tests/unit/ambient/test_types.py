"""Tests: DTOs del ambient path (spec 2026-06-06)."""
import numpy as np

from src.ambient.types import AmbientUtterance, RawSegment, SOURCE_VALUES


def test_ambient_utterance_defaults():
    u = AmbientUtterance(room_id="escritorio", t0=100.0, t1=103.5)
    assert u.text == ""
    assert u.speaker == "unknown"
    assert u.source == "unknown"
    assert u.azimuth is None
    assert u.during_tts is False
    assert u.distilled is False


def test_source_values_cerrados():
    # El clasificador y el store validan contra este set — 'self' incluido
    # (desviación 1 del plan: audio durante TTS propio no va al RAG).
    assert SOURCE_VALUES == {"live", "tv", "self", "unknown"}


def test_raw_segment_holds_multichannel_audio():
    audio = np.zeros((16000, 6), dtype=np.float32)
    seg = RawSegment(t0=1.0, t1=2.0, audio=audio, during_tts=True)
    assert seg.audio.shape == (16000, 6)
    assert seg.during_tts is True
    assert seg.duration_s == 1.0


def test_raw_segment_defaults_and_no_eq_crash():
    audio = np.zeros((100, 6), dtype=np.float32)
    a = RawSegment(t0=1.0, t1=2.0, audio=audio)
    b = RawSegment(t0=1.0, t1=2.0, audio=audio)
    assert a.during_tts is False
    # eq=False: comparar no debe lanzar ValueError por el ndarray (identidad)
    assert (a == b) is False
    assert (a == a) is True
