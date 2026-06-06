"""Tests: SourceClassifier — matriz de reglas live|tv|self|unknown."""
import math

from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig


def _clf(tv_azimuth=1.0, **kw):
    cfg = SourceClassifierConfig(
        tv_azimuth=tv_azimuth,
        tv_tolerance_rad=kw.pop("tv_tolerance_rad", 0.35),
        min_stability=kw.pop("min_stability", 0.6),
        require_known_speaker_for_live=kw.pop("require_known_speaker_for_live", False),
    )
    return SourceClassifier(cfg)


def test_during_tts_is_self_regardless_of_everything():
    c = _clf()
    assert c.classify(speaker="gabriel", azimuth=1.0, stability=1.0, during_tts=True) == "self"


def test_stable_tv_direction_unknown_speaker_is_tv():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=1.1, stability=0.9, during_tts=False) == "tv"


def test_known_speaker_is_live_even_from_tv_direction():
    # El usuario sentado al lado de la TV no es la TV.
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="gabriel", azimuth=1.0, stability=0.9, during_tts=False) == "live"


def test_stable_non_tv_direction_is_live():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=-2.0, stability=0.9, during_tts=False) == "live"


def test_unstable_doa_is_unknown():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=1.0, stability=0.2, during_tts=False) == "unknown"


def test_no_doa_is_unknown():
    c = _clf(tv_azimuth=1.0)
    assert c.classify(speaker="unknown", azimuth=None, stability=0.0, during_tts=False) == "unknown"


def test_without_calibrated_tv_azimuth_never_tv():
    c = _clf(tv_azimuth=None)
    assert c.classify(speaker="unknown", azimuth=1.0, stability=0.95, during_tts=False) == "live"


def test_wraparound_angular_distance():
    # azimut pi y -pi son la misma dirección
    c = _clf(tv_azimuth=math.pi)
    assert c.classify(speaker="unknown", azimuth=-math.pi + 0.1, stability=0.9, during_tts=False) == "tv"


def test_require_known_speaker_for_live():
    c = _clf(tv_azimuth=1.0, require_known_speaker_for_live=True)
    assert c.classify(speaker="unknown", azimuth=-2.0, stability=0.9, during_tts=False) == "unknown"
    assert c.classify(speaker="gabriel", azimuth=-2.0, stability=0.9, during_tts=False) == "live"
