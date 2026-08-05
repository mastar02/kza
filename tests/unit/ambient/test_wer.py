"""Tests: core de WER para medir fidelidad del ambient."""
from src.ambient.wer import (
    UNINTELLIGIBLE, bucket_of, is_excluded, normalize_words, score,
)


def test_identidad_da_wer_cero():
    r = score("prendé la luz del escritorio", "prendé la luz del escritorio")
    assert r.wer == 0.0
    assert (r.subs, r.ins, r.dels) == (0, 0, 0)


def test_una_sustitucion_en_cuatro_palabras():
    r = score("prendé la luz roja", "prendé la luz azul")
    assert r.subs == 1
    assert r.ref_words == 4
    assert r.wer == 0.25


def test_delecion_se_cuenta_como_delecion():
    r = score("prendé la luz del escritorio", "prendé la luz")
    assert r.dels == 2
    assert r.ins == 0


def test_insercion_se_cuenta_como_insercion():
    r = score("prendé la luz", "prendé la luz del escritorio")
    assert r.ins == 2
    assert r.dels == 0


def test_hipotesis_vacia_es_delecion_total():
    r = score("prendé la luz", "")
    assert r.dels == 3
    assert r.wer == 1.0


def test_referencia_vacia_con_hipotesis_es_alucinacion():
    r = score("", "¡Gracias por ver el video!")
    assert r.ref_words == 0
    assert r.ins == 5
    assert r.wer == 1.0     # convención: ref vacía + hyp no vacía = 1.0


def test_ambas_vacias_es_cero():
    assert score("", "").wer == 0.0


def test_normalizacion_conserva_acentos_y_enie():
    # "apaga" != "apagá": el acento es señal real de calidad en español
    assert normalize_words("¡Apagá!") == ["apagá"]
    assert score("apagá", "apaga").wer == 1.0
    assert normalize_words("el año") == ["el", "año"]


def test_normalizacion_baja_caso_y_saca_puntuacion_de_borde():
    assert normalize_words("  Hola,  QUE tal.  ") == ["hola", "que", "tal"]


def test_buckets_cubren_el_rango():
    assert bucket_of(0.05) == "0.00-0.20"
    assert bucket_of(0.20) == "0.20-0.35"
    assert bucket_of(0.99) == "0.80-1.00"
    assert bucket_of(None) == "sin_vad"


def test_marcadores_excluidos_del_wer():
    assert is_excluded(UNINTELLIGIBLE)
    assert is_excluded("[tv]")
    assert not is_excluded("prendé la luz")
