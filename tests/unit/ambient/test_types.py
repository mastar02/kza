"""Tests: DTOs del ambient path (spec 2026-06-06)."""
import numpy as np
import pytest

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


def test_text_empty_con_texto_es_un_bug_de_construccion():
    """text_empty es DERIVADO de text (review PR #14): una fila 'vacía' con
    texto destilable —o la inversa— solo puede salir de un call-site nuevo
    mal escrito. Mejor un ValueError en el constructor que una fila
    fantasma en ambient.db."""
    with pytest.raises(ValueError):
        AmbientUtterance(room_id="cocina", t0=0.0, t1=1.0,
                         text="prendé la luz", text_empty=True)


def test_text_vacio_sin_flag_sigue_siendo_legal():
    # Las filas legacy (pre-migración) tienen text="" y text_empty=False;
    # el invariante solo prohíbe la dirección peligrosa.
    u = AmbientUtterance(room_id="cocina", t0=0.0, t1=1.0, text="")
    assert u.text_empty is False
