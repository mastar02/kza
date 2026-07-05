"""Tests: language_quality — regla de calidad/idioma del ambient path.

Filtro de "español conservable" para marcar utterances al persistir (flag, no
drop). El garble code-switch (español far-field que Parakeet colapsa a inglés
fonético) se descarta; el rioplatense que langid manda a pt/gl/ca se rescata
por acentos/stopwords.
"""
from src.ambient.language_quality import (
    has_spanish_markers,
    is_spanish_keepable,
    make_quality_fn,
)


def test_accents_are_spanish_markers():
    assert has_spanish_markers("está todo bien")
    assert has_spanish_markers("el niño juega")
    assert has_spanish_markers("¿qué pasa acá?")


def test_two_distinct_stopwords_are_markers():
    # "pero", "esto", "que" → ≥2 stopwords distintas
    assert has_spanish_markers("pero esto que digo")


def test_no_markers_in_english():
    assert not has_spanish_markers("It's a beautiful day")
    assert not has_spanish_markers("the dog runs fast")


def test_single_ambiguous_word_is_not_a_marker():
    # un solo "no" (ambiguo es/en) no debe marcar una frase inglesa
    assert not has_spanish_markers("no I will not go there")


def test_keepable_spanish_with_accents():
    assert is_spanish_keepable(
        "Ah loquita, vos te fuiste de vacaciones", "es", 0.99, 0.83
    )


def test_drops_short_text():
    assert not is_spanish_keepable("Oh", "es", 0.5, 0.8)  # len < 8


def test_drops_low_vad():
    assert not is_spanish_keepable("hola que tal todo bien", "es", 0.9, 0.20)


def test_none_vad_does_not_block():
    # vad NULL (filas viejas / sin señal) no debe descartar por vad
    assert is_spanish_keepable("hola que tal todo bien acá", "es", 0.9, None)


def test_drops_english_garble_without_markers():
    # español mal transcrito a inglés-fonético, sin acentos/stopwords → drop
    assert not is_spanish_keepable(
        "It's joven when it descanses over the adoquines", "en", 0.95, 0.77
    )


def test_rescues_spanish_misdetected_as_portuguese():
    # rioplatense con acento que langid manda a pt → rescatado por markers
    assert is_spanish_keepable("Está raro esto, no sé qué pasó acá", "pt", 0.8, 0.7)


def test_keeps_plain_spanish_detected_as_es():
    # sin acentos ni 2 stopwords, pero langid=es y vad/len ok → keep por lang
    assert is_spanish_keepable("bueno dale entonces vamos rapido", "es", 0.9, 0.6)


def test_drops_real_english_detected_as_en():
    assert not is_spanish_keepable("And then I'm gonna go home", "en", 0.97, 0.8)


def test_make_quality_fn_composes_detect_and_rule():
    def fake_detect(text):
        return ("es", 0.9)

    qf = make_quality_fn(fake_detect, min_len=8, min_vad=0.40)
    lang, prob, ok = qf("Ah, vos sabés que está todo bien", 0.8)
    assert lang == "es"
    assert prob == 0.9
    assert ok is True


def test_make_quality_fn_flags_garble_false():
    def fake_detect(text):
        return ("en", 0.95)

    qf = make_quality_fn(fake_detect, min_len=8, min_vad=0.40)
    lang, prob, ok = qf("It's joven when it descanses over the adoquines", 0.77)
    assert lang == "en"
    assert ok is False
