"""Tests: reporte de WER por bucket con agregado re-ponderado."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import tools.ambient_wer as wer_mod
from tools.ambient_wer import COBERTURA_MINIMA, build_report, load_snapshot


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


def test_reponderado_falla_ruidosamente_si_volumes_esta_vacio():
    """volumes={} no puede traducirse en 'todos los buckets pesan 0' — eso
    da un wer_reponderado=0.0 falso y creíble."""
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "x y"},
    ]
    rep = build_report(pairs, volumes={})
    assert rep["wer_reponderado"] is None
    assert rep["confiable"] is False
    assert rep["buckets_sin_volumen"] == ["0.80-1.00"]


def test_reponderado_falla_ruidosamente_si_falta_un_bucket_presente():
    """Un bucket evaluado que no aparece como clave en volumes es una
    desalineación upstream, no un peso legítimo de cero: si se tratara como
    cero, el bucket malo desaparecería silenciosamente del agregado."""
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b c d", "hypothesis": "a b c d"},
        {"id": 2, "vad_prob": 0.10, "reference": "a b c d", "hypothesis": "x y z w"},
    ]
    rep = build_report(pairs, volumes={"0.80-1.00": 100})  # falta "0.00-0.20"
    assert rep["wer_reponderado"] is None
    assert rep["confiable"] is False
    assert rep["buckets_sin_volumen"] == ["0.00-0.20"]


def test_bucket_con_volumen_real_cero_es_legitimo_no_una_falla():
    """Distinto del caso anterior: acá el bucket SÍ está en volumes, solo
    que su volumen real en la DB es 0. Eso es información válida (no
    inconsistencia) y el reporte debe seguir siendo confiable."""
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"},
        {"id": 2, "vad_prob": 0.10, "reference": "a b", "hypothesis": "x y"},
    ]
    rep = build_report(pairs, volumes={"0.80-1.00": 100, "0.00-0.20": 0})
    assert rep["confiable"] is True
    assert rep["buckets_sin_volumen"] == []
    assert rep["wer_reponderado"] == 0.0  # el bucket con volumen 0 no aporta peso


def test_un_bucket_con_volumen_real_sin_pares_hunde_el_reporte():
    """C1 — el agregado NO puede re-normalizarse sobre los sobrevivientes.

    Un bucket con volumen real grande pero sin pares evaluados desaparece del
    numerador y del denominador a la vez: el promedio queda perfecto y habla
    por el 4,6% del corpus con la autoridad del total.
    """
    pairs = [
        {"id": i, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"}
        for i in range(7)
    ]
    volumes = {"0.00-0.20": 221, "0.20-0.35": 1238, "0.35-0.50": 1462,
               "0.50-0.65": 1551, "0.65-0.80": 1368, "0.80-1.00": 283}
    rep = build_report(pairs, volumes)
    assert rep["confiable"] is False
    assert rep["wer_reponderado"] is None
    assert rep["buckets_sin_pares"] == [
        "0.00-0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", "0.65-0.80"]
    assert rep["cobertura_volumen"] == 283 / 6123
    assert any("SIN pares evaluados" in m for m in rep["motivos"])


def test_un_bucket_chico_sin_pares_tambien_hunde_el_reporte():
    """El chequeo por bucket no es redundante con el piso de cobertura: acá el
    bucket que falta es el 0,1% del volumen, la cobertura queda en 99,9% (muy
    por encima del piso) y aun así el agregado estaría hablando de un bucket
    del que no midió ni un caso."""
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"},
        {"id": 2, "vad_prob": 0.10, "reference": "a b", "hypothesis": "x y"},
    ]
    volumes = {"0.80-1.00": 5000, "0.00-0.20": 4990, "0.35-0.50": 10}
    rep = build_report(pairs, volumes)
    assert rep["cobertura_volumen"] > COBERTURA_MINIMA
    assert rep["buckets_sin_pares"] == ["0.35-0.50"]
    assert rep["confiable"] is False
    assert rep["wer_reponderado"] is None


def test_no_se_emite_agregado_cuando_el_reporte_no_es_confiable():
    """El número re-ponderado y la bandera `confiable` no pueden divergir: un
    `wer_reponderado` presente se lee como usable, mire quien mire el JSON."""
    pairs = [{"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"}]
    rep = build_report(pairs, {"0.80-1.00": 283, "0.00-0.20": 5656})
    assert rep["confiable"] is False
    assert rep["wer_reponderado"] is None


def test_el_bucket_borrado_por_ininteligible_es_el_caso_real():
    """Reproducción del escenario del review: 7 pares `[ininteligible]` en el
    bucket 0.00-0.20 (86,9% garble: ES el bucket donde el humano se rinde) y 7
    limpios en 0.80-1.00. `is_excluded` saca los primeros ANTES de que exista
    la clave, y antes del fix eso daba `confiable: True, WER: 0.0`.
    """
    pairs = [
        {"id": i, "vad_prob": 0.05, "reference": "[ininteligible]",
         "hypothesis": "so we got a border"} for i in range(7)
    ] + [
        {"id": 100 + i, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"}
        for i in range(7)
    ]
    volumes = {"0.00-0.20": 221, "0.20-0.35": 1238, "0.35-0.50": 1462,
               "0.50-0.65": 1551, "0.65-0.80": 1368, "0.80-1.00": 283}
    rep = build_report(pairs, volumes)
    assert rep["excluidas"] == 7
    assert rep["confiable"] is False
    assert rep["wer_reponderado"] is None
    assert "0.00-0.20" in rep["buckets_sin_pares"]


def test_la_cobertura_debajo_del_piso_es_motivo_propio():
    """La cobertura no es solo informativa: es el piso que decide. Sin ella el
    reporte no diría *cuánto* del corpus quedó afuera."""
    pairs = [{"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"}]
    rep = build_report(pairs, {"0.80-1.00": 283, "0.00-0.20": 5656})
    assert rep["cobertura_volumen"] == 283 / 5939
    assert any("piso" in m for m in rep["motivos"])
    assert rep["cobertura_minima"] == COBERTURA_MINIMA


def test_cobertura_total_con_todos_los_buckets_representados():
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"},
        {"id": 2, "vad_prob": 0.10, "reference": "a b", "hypothesis": "x y"},
    ]
    rep = build_report(pairs, {"0.80-1.00": 100, "0.00-0.20": 900})
    assert rep["cobertura_volumen"] == 1.0
    assert rep["confiable"] is True


def test_wer_simple_sin_datos_es_none_no_cero():
    """I2 — el mismo bug de clase que ya se arregló en wer_reponderado: 0.0
    dicho sin datos se lee como "perfecto"."""
    rep = build_report([], volumes={"0.80-1.00": 10})
    assert rep["wer_simple"] is None
    assert rep["wer_reponderado"] is None
    assert rep["confiable"] is False


def test_sin_pares_el_motivo_no_habla_de_volumen_cero():
    """Con `pairs=[]` el mensaje viejo decía "los buckets evaluados declaran
    volumen real 0" — no hay buckets evaluados de los que hablar."""
    rep = build_report([], volumes={})
    assert any("ningún par evaluado" in m for m in rep["motivos"])
    assert not any("volumen real 0" in m for m in rep["motivos"])


def test_todos_los_buckets_con_volumen_cero_dice_lo_suyo():
    pairs = [{"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"}]
    rep = build_report(pairs, volumes={"0.80-1.00": 0})
    assert rep["confiable"] is False
    assert rep["wer_reponderado"] is None
    assert any("0/0" in m for m in rep["motivos"])


def test_pares_faltantes_hunden_el_reporte():
    """C4 — si la DB purgó las hipótesis, se miden menos pares de los que el
    ground truth trae. Perderlos en silencio es cómo un set incompleto produce
    un número que parece bueno."""
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b", "hypothesis": "a b"},
        {"id": 2, "vad_prob": 0.10, "reference": "a b", "hypothesis": "x y"},
    ]
    rep = build_report(pairs, {"0.80-1.00": 100, "0.00-0.20": 900}, esperados=42)
    assert rep["confiable"] is False
    assert rep["pares_medidos"] == 2
    assert rep["pares_esperados"] == 42
    assert any("faltan 40" in m for m in rep["motivos"])


def test_load_snapshot_lee_hipotesis_y_pesos(tmp_path):
    p = tmp_path / "hypotheses.json"
    p.write_text(json.dumps({
        "utterances": {"1": {"text": "hola", "vad_prob": 0.9, "room_id": "x"}},
        "volumes": {"0.80-1.00": 3},
    }))
    utts, vols, error = load_snapshot(p)
    assert utts["1"]["text"] == "hola"
    assert vols == {"0.80-1.00": 3}
    assert error is None


def test_load_snapshot_ausente_no_lanza(tmp_path):
    assert load_snapshot(tmp_path / "no_existe.json") == ({}, {}, None)


def test_load_snapshot_corrupto_devuelve_error_no_fallback_silencioso(tmp_path):
    """Un hypotheses.json que EXISTE pero no parsea no es lo mismo que uno
    ausente: el snapshot de una campaña se escribe una sola vez, y tratar
    'corrupto' como 'ausente' manda al operador a la DB (purga 48h) en vez
    de al backup del archivo."""
    p = tmp_path / "hypotheses.json"
    p.write_text("{trunc", encoding="utf-8")
    utts, vols, error = load_snapshot(p)
    assert utts == {}
    assert vols == {}
    assert error is not None
    assert "corrupto" in error


def test_los_pesos_salen_del_mismo_universo_que_la_muestra():
    """I3 — si los pesos se calcularan sobre otra población (con 'self', o con
    filas pre-campaña sin audio), re-ponderarían la muestra por una
    distribución que la muestra jamás pudo representar."""
    from tools.ambient_groundtruth import SAMPLEABLE_WHERE
    assert wer_mod.SAMPLEABLE_WHERE is SAMPLEABLE_WHERE
    assert "audio_path IS NOT NULL" in SAMPLEABLE_WHERE
    assert "source NOT IN ('self','tv')" in SAMPLEABLE_WHERE


def test_docstring_documenta_los_exit_codes():
    assert "Exit codes:" in wer_mod.__doc__


def test_main_exit_1_si_no_hay_ni_snapshot_ni_db(tmp_path, monkeypatch):
    """El docstring promete exit 1 cuando no hay de dónde sacar las
    hipótesis (ni snapshot ni DB). Sin este chequeo, ``main()`` sigue de
    largo, arma un reporte ``confiable=False`` (fail-loud en stdout/stderr)
    pero devuelve el mismo exit 0 que una corrida exitosa — un
    `--validate && ambient_wer && …` encadenado lo lee como éxito, que es
    justo el contrato que los exit codes existen para fijar.
    """
    gt_path = tmp_path / "groundtruth.json"
    gt_path.write_text(json.dumps({"1": "prendé la luz"}), encoding="utf-8")
    db_path = tmp_path / "no_existe.db"  # nunca se crea: la apertura falla

    monkeypatch.setattr(sys, "argv", [
        "ambient_wer.py",
        "--groundtruth", str(gt_path),
        "--db", str(db_path),
        "--out", str(tmp_path / "reporte.json"),
    ])
    with pytest.raises(SystemExit) as exc:
        wer_mod.main()
    assert exc.value.code == 1
    # y no dejó un reporte a medio armar detrás: falló ANTES de escribirlo
    assert not (tmp_path / "reporte.json").exists()


def test_reporte_no_confiable_sale_con_exit_2(tmp_path, monkeypatch):
    from src.ambient.wer import bucket_of
    b_alto, b_bajo = bucket_of(0.9), bucket_of(0.1)
    (tmp_path / "groundtruth.json").write_text(
        json.dumps({"1": "hola che"}), encoding="utf-8")
    (tmp_path / "hypotheses.json").write_text(json.dumps({
        "utterances": {"1": {"text": "hola che", "vad_prob": 0.9}},
        # Volumen real concentrado en un bucket sin ningún par evaluado →
        # cobertura muy por debajo del piso → confiable=False.
        "volumes": {b_alto: 1, b_bajo: 99},
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "ambient_wer.py", "--groundtruth", str(tmp_path / "groundtruth.json"),
        "--out", str(tmp_path / "rep.json"),
    ])
    with pytest.raises(SystemExit) as ex:
        wer_mod.main()
    assert ex.value.code == 2


def test_snapshot_corrupto_es_error_duro_no_fallback_silencioso(
    tmp_path, monkeypatch, capsys
):
    """OJO con este test: no basta con --db apuntando a un archivo que no
    existe, porque el código VIEJO (que trataba corrupto como ausente) ya
    salía con exit 1 por ese lado ("ni snapshot ni DB") — un falso positivo.
    Y ``tmp_path`` se nombra a partir del nombre de ESTA función, que
    contiene la palabra "corrupto"; si el assert buscara esa palabra en el
    path impreso, pasaría por coincidencia sin que el fix hiciera nada. Por
    eso el assert busca "NO se cae a la DB", frase que SOLO el mensaje
    nuevo de main() emite (el AVISO viejo de "sin snapshot" dice "se cae a
    la DB", sin el "NO")."""
    (tmp_path / "groundtruth.json").write_text(
        json.dumps({"1": "hola"}), encoding="utf-8")
    (tmp_path / "hypotheses.json").write_text("{trunc", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "ambient_wer.py", "--groundtruth", str(tmp_path / "groundtruth.json"),
        "--db", str(tmp_path / "nonexistent.db"),
    ])
    with pytest.raises(SystemExit) as ex:
        wer_mod.main()
    assert ex.value.code == 1
    assert "NO se cae a la DB" in capsys.readouterr().err


def test_snapshot_corrupto_no_cae_a_una_db_poblada(tmp_path, monkeypatch, capsys):
    """El caso que la regresión real hubiera producido: snapshot corrupto Y
    una ambient.db real, abierta y con una fila usable para el mismo id.
    Antes del fix, ``load_snapshot`` tragaba el ``JSONDecodeError`` y
    devolvía ``({}, {})`` — main() lo leía como "sin snapshot" y caía
    derecho a esta DB poblada, armando un reporte con datos de una corrida
    vieja o purgada a medias, exit 0, sin avisar nada. Acá la DB está sana
    a propósito: si el fix se rompe, este test vería un reporte armado con
    exit 0 en vez de SystemExit(1), y la DB SÍ se habría tocado."""
    db_path = tmp_path / "ambient.db"
    db = sqlite3.connect(db_path)
    db.execute(
        "CREATE TABLE utterances (id INTEGER PRIMARY KEY, room_id TEXT, "
        "vad_prob REAL, text TEXT, audio_path TEXT, source TEXT)"
    )
    db.execute(
        "INSERT INTO utterances (id, room_id, vad_prob, text, audio_path, "
        "source) VALUES (1, 'escritorio', 0.9, 'hola', '/tmp/1.flac', 'mic')"
    )
    db.commit()
    db.close()

    (tmp_path / "groundtruth.json").write_text(
        json.dumps({"1": "hola"}), encoding="utf-8")
    (tmp_path / "hypotheses.json").write_text("{trunc", encoding="utf-8")

    connect_calls = []
    real_connect = sqlite3.connect

    def spy_connect(*a, **kw):
        connect_calls.append((a, kw))
        return real_connect(*a, **kw)

    monkeypatch.setattr(sqlite3, "connect", spy_connect)
    monkeypatch.setattr(sys, "argv", [
        "ambient_wer.py", "--groundtruth", str(tmp_path / "groundtruth.json"),
        "--db", str(db_path),
        "--out", str(tmp_path / "rep.json"),
    ])
    with pytest.raises(SystemExit) as ex:
        wer_mod.main()
    assert ex.value.code == 1
    assert connect_calls == []                    # la DB poblada NUNCA se tocó
    assert not (tmp_path / "rep.json").exists()    # no quedó un reporte a medias
    assert "NO se cae a la DB" in capsys.readouterr().err
