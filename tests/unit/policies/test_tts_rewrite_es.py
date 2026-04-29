"""Tests for tts_rewrite_es policy."""

from src.hooks import TtsCall, RewriteResult


def _tts(text):
    return TtsCall(text=text, voice=None, lang="es", user_id=None, zone_id=None)


def test_pesos_substitution():
    from src.policies.tts_rewrite_es import numeros_a_palabras

    result = numeros_a_palabras(_tts("son $1000"))
    assert isinstance(result, RewriteResult)
    assert result.modified.text == "son 1000 pesos"


def test_no_match_returns_none():
    from src.policies.tts_rewrite_es import numeros_a_palabras

    result = numeros_a_palabras(_tts("hola, qué tal"))
    assert result is None


def test_multiple_pesos_in_text():
    from src.policies.tts_rewrite_es import numeros_a_palabras

    result = numeros_a_palabras(_tts("entre $100 y $500"))
    assert result.modified.text == "entre 100 pesos y 500 pesos"
