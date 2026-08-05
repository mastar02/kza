"""Tests: reporte de WER por bucket con agregado re-ponderado."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.ambient_wer import build_report


def test_wer_por_bucket():
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "prendé la luz",
         "hypothesis": "prendé la luz"},
        {"id": 2, "vad_prob": 0.10, "reference": "prendé la luz",
         "hypothesis": "so we got a border"},
    ]
    rep = build_report(pairs, volumes={"0.80-1.00": 100, "0.00-0.20": 100})
    assert rep["buckets"]["0.80-1.00"]["wer"] == 0.0
    assert rep["buckets"]["0.00-0.20"]["wer"] > 1.0   # 3 dels + 5 ins sobre 3 ref


def test_agregado_reponderado_por_volumen_real():
    """El set tiene asignación igual por bucket; el agregado debe pesar por
    el volumen real de la DB, si no sobre-representa los buckets altos."""
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b c d", "hypothesis": "a b c d"},
        {"id": 2, "vad_prob": 0.10, "reference": "a b c d", "hypothesis": "x y z w"},
    ]
    # el bucket malo es 9x más frecuente en la DB real
    rep = build_report(pairs, volumes={"0.80-1.00": 100, "0.00-0.20": 900})
    assert rep["wer_simple"] == 0.5                 # promedio plano
    assert rep["wer_reponderado"] == 0.9            # 0.0*0.1 + 1.0*0.9


def test_delecion_e_alucinacion_se_reportan_aparte():
    # ojo: vad=0.45 cae en "0.35-0.50"; vad=0.50 caería en "0.50-0.65"
    pairs = [
        {"id": 1, "vad_prob": 0.45, "reference": "hay habla real", "hypothesis": ""},
        {"id": 2, "vad_prob": 0.45, "reference": "", "hypothesis": "¡Gracias!"},
    ]
    rep = build_report(pairs, volumes={"0.35-0.50": 10})
    assert rep["deleciones_totales"] == 1
    assert rep["alucinaciones"] == 1


def test_marcadores_se_excluyen_del_wer():
    pairs = [
        {"id": 1, "vad_prob": 0.45, "reference": "[ininteligible]", "hypothesis": "algo"},
        {"id": 2, "vad_prob": 0.45, "reference": "hola", "hypothesis": "hola"},
    ]
    rep = build_report(pairs, volumes={"0.35-0.50": 10})
    assert rep["excluidas"] == 1
    assert rep["buckets"]["0.35-0.50"]["n"] == 1
    assert rep["buckets"]["0.35-0.50"]["wer"] == 0.0
