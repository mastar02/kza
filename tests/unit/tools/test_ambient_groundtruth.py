"""Tests: armado del set de ground truth ciego."""
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import tools.ambient_groundtruth as gt_mod
from tools.ambient_groundtruth import (
    SIN_VAD, _export, _validate, bucket_volumes, render_html, sample_stratified,
)


def _make_db(
    tmp_path: Path,
    audio_paths: list[str | None],
    vads: list[float] | None = None,
    sources: list[str] | None = None,
) -> Path:
    """DB sqlite mínima de ``utterances`` con los ``audio_path`` dados."""
    db_path = tmp_path / "ambient.db"
    db = sqlite3.connect(db_path)
    db.execute(
        "CREATE TABLE utterances (id INTEGER PRIMARY KEY, room_id TEXT, "
        "vad_prob REAL, text TEXT, audio_path TEXT, source TEXT)"
    )
    for i, audio_path in enumerate(audio_paths):
        db.execute(
            "INSERT INTO utterances (room_id, vad_prob, text, audio_path, source) "
            "VALUES (?, ?, ?, ?, ?)",
            ("escritorio", vads[i] if vads else 0.1, f"texto {i}", audio_path,
             sources[i] if sources else "unknown"),
        )
    db.commit()
    db.close()
    return db_path


def _rows(n_per_bucket: int = 20):
    """Filas sintéticas repartidas por bucket de vad."""
    out, uid = [], 0
    for centro in (0.10, 0.27, 0.42, 0.57, 0.72, 0.90):
        for _ in range(n_per_bucket):
            uid += 1
            out.append({
                "id": uid, "room_id": "escritorio", "vad_prob": centro,
                "text": f"texto {uid}", "audio_path": f"/tmp/{uid}.flac",
            })
    return out


def _ids_en_html(html: str) -> list[int]:
    """Ids en el orden en que el HTML los presenta."""
    return [int(m) for m in re.findall(r'<b>#(\d+)</b>', html)]


def test_asignacion_igual_por_bucket_no_proporcional():
    # 6 buckets con volúmenes MUY distintos: el bucket alto tiene 3 filas
    rows = [r for r in _rows(50) if r["vad_prob"] != 0.90]
    rows += [r for r in _rows(3) if r["vad_prob"] == 0.90]
    sel = sample_stratified(rows, per_bucket=7, seed=1)
    from src.ambient.wer import bucket_of
    from collections import Counter
    got = Counter(bucket_of(r["vad_prob"]) for r in sel)
    assert got["0.00-0.20"] == 7
    # el bucket escaso aporta lo que tiene, sin romper
    assert got["0.80-1.00"] == 3


def test_es_reproducible_con_la_misma_semilla():
    rows = _rows()
    a = [r["id"] for r in sample_stratified(rows, per_bucket=5, seed=42)]
    b = [r["id"] for r in sample_stratified(rows, per_bucket=5, seed=42)]
    c = [r["id"] for r in sample_stratified(rows, per_bucket=5, seed=43)]
    assert a == b
    assert a != c


def test_ignora_filas_sin_audio():
    rows = _rows(5)
    for r in rows:
        r["audio_path"] = None
    assert sample_stratified(rows, per_bucket=7, seed=1) == []


def test_no_muestrea_el_estrato_sin_vad():
    """``bucket_of(None)`` crea un 7º estrato que el spec no contempla (6×7=42)
    y que después recibiría peso en el agregado sin significar nada: el eje
    del estudio ES vad_prob, y una fila sin vad no cae en ningún punto de él."""
    rows = _rows(3) + [
        {"id": 9000 + i, "room_id": "cocina", "vad_prob": None,
         "text": "sin vad", "audio_path": f"/tmp/{9000 + i}.flac"}
        for i in range(20)
    ]
    sel = sample_stratified(rows, per_bucket=7, seed=1)
    assert sel, "el resto de los buckets sí se muestrea"
    assert all(r["vad_prob"] is not None for r in sel)
    # y tampoco recibe peso en el agregado
    assert SIN_VAD not in bucket_volumes(rows)


def test_volumenes_por_bucket_cuentan_el_universo_muestreable():
    vols = bucket_volumes(_rows(4))
    assert vols == {b: 4 for b in (
        "0.00-0.20", "0.20-0.35", "0.35-0.50",
        "0.50-0.65", "0.65-0.80", "0.80-1.00")}


def test_el_html_NO_filtra_la_salida_del_modelo():
    """Ciego por diseño: ver la hipótesis ancla al transcriptor humano."""
    items = [{"id": 1, "room_id": "escritorio", "vad_prob": 0.5,
              "audio": "1.flac", "text": "GRACIAS POR VER EL VIDEO"}]
    html = render_html(items)
    assert "GRACIAS POR VER EL VIDEO" not in html
    assert "1.flac" in html


def test_el_html_NO_filtra_vad_prob_ni_habitacion():
    """`vad_prob` es LA variable que el estudio quiere validar. Mostrarla
    junto al clip le dice al anotador dónde esforzarse y cuándo rendirse con
    `[ininteligible]`; `room_id` es el mismo sesgo por otra vía (la cocina
    está más lejos de donde se habla)."""
    items = [{"id": 1, "room_id": "cocina", "vad_prob": 0.07,
              "audio": "1.flac", "text": "hola"}]
    html = render_html(items)
    assert "cocina" not in html
    assert "0.07" not in html
    assert "vad" not in html.lower()


def test_el_html_baraja_el_orden_de_presentacion():
    """Presentados por bucket ascendente, los primeros clips son basura y los
    últimos limpios — un patrón aprendible en cinco escuchas."""
    items = [{"id": i, "audio": f"{i}.flac"} for i in range(1, 25)]
    entrada = [it["id"] for it in items]
    a = _ids_en_html(render_html(items, seed=7))
    b = _ids_en_html(render_html(items, seed=7))
    c = _ids_en_html(render_html(items, seed=8))
    assert sorted(a) == sorted(entrada)   # no se pierde ni se duplica ninguno
    assert a != entrada                   # no es el orden por bucket
    assert a == b                         # determinista con la misma semilla
    assert a != c


def test_el_html_ofrece_sin_habla_como_estado_explicito():
    """Un textarea que el humano no tocó no puede guardarse como "no había
    habla": cada clip no visitado contaría como alucinación con WER 1.0."""
    html = render_html([{"id": 1, "audio": "1.flac"}])
    assert 'type="checkbox"' in html
    assert 'data-empty="1"' in html
    # el JS de guardado solo escribe la clave si el clip fue resuelto: la
    # asignación de "" cuelga del checkbox, y lo no visitado se cuenta aparte
    save = html.split("function save()", 1)[1]
    assert "if (box.checked)" in save
    assert "faltan++" in save
    assert save.index("if (box.checked)") < save.index("out[id] = ''")
    # El cuerpo del `else` FINAL (no visitado) no puede escribir en `out`:
    # ``.index`` de arriba solo encuentra la PRIMERA ocurrencia de
    # `out[id] = ''` (la de la rama `if (box.checked)`) y no ve nada de lo
    # que pasa en la rama `else` — un `out[id] = ''` reintroducido ahí
    # sobrevive esa aserción sin que se note. Aislar el cuerpo del último
    # `else` (el que sigue al `} else {` que abre la rama de "no visitado")
    # y verificar que no toca `out` en absoluto es lo único que detecta esa
    # mutación.
    else_final = save.partition("} else {")[2].split("}", 1)[0]
    assert "faltan++" in else_final
    assert "out[" not in else_final


def test_export_falla_si_ninguna_copia_tuvo_exito(tmp_path):
    """Proxy mentiroso: la query devuelve filas pero ningún audio existe.

    ``sel`` (lo seleccionado) no está vacío — todas apuntan a un
    ``audio_path`` con nombre no vacío — pero ningún archivo existe en
    disco, así que ``items`` (lo realmente copiado) sí queda vacío. Tiene
    que fallar ruidosamente, no reportar éxito sobre un directorio inútil.
    """
    audio_paths = [str(tmp_path / f"no_existe_{i}.flac") for i in range(3)]
    db_path = _make_db(tmp_path, audio_paths)
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit) as exc:
        _export(str(db_path), str(out_dir), per_bucket=7, seed=1)
    assert exc.value.code != 0
    # y no deja un index.html invitando a transcribir la nada
    assert not (out_dir / "index.html").exists()
    assert not (out_dir / "meta.json").exists()


def test_export_reporta_exito_parcial_con_exit_distinto_de_cero(tmp_path):
    """Si se seleccionaron 3 y se copió solo 1, no es un éxito limpio."""
    real_audio = tmp_path / "real.flac"
    real_audio.write_bytes(b"fake-flac-bytes")
    audio_paths = [str(real_audio)] + [
        str(tmp_path / f"no_existe_{i}.flac") for i in range(2)
    ]
    db_path = _make_db(tmp_path, audio_paths)
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit) as exc:
        _export(str(db_path), str(out_dir), per_bucket=7, seed=1)
    assert exc.value.code == 2
    # el que sí se pudo copiar queda disponible igual
    assert (out_dir / "index.html").exists()


def test_export_excluye_self_y_tv(tmp_path):
    """'self' es nuestro propio TTS: audio limpísimo, vad alto, aterriza en el
    bucket 0.80-1.00 e infla justo el bucket que existe para responder
    "¿ya estamos en ~95%?"."""
    real = tmp_path / "real.flac"
    real.write_bytes(b"fake-flac-bytes")
    db_path = _make_db(
        tmp_path, [str(real)] * 3, vads=[0.9, 0.9, 0.9],
        sources=["unknown", "self", "tv"],
    )
    out_dir = tmp_path / "out"

    _export(str(db_path), str(out_dir), per_bucket=7, seed=1)

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["ids"] == [1]          # solo la fila 'unknown'
    hyp = json.loads((out_dir / "hypotheses.json").read_text())
    assert hyp["volumes"] == {"0.80-1.00": 1}


def test_export_snapshotea_las_hipotesis_fuera_de_la_db(tmp_path):
    """La DB purga a las 48 h y el muestreo es uniforme sobre la ventana: la
    mitad del set expira dentro del día siguiente al export. El audio
    sobrevive (se copió); sin snapshot, el texto contra el que compararlo no."""
    real = tmp_path / "real.flac"
    real.write_bytes(b"fake-flac-bytes")
    db_path = _make_db(tmp_path, [str(real)], vads=[0.9])
    out_dir = tmp_path / "out"

    _export(str(db_path), str(out_dir), per_bucket=7, seed=1)

    hyp = json.loads((out_dir / "hypotheses.json").read_text())
    assert hyp["utterances"]["1"]["text"] == "texto 0"
    assert hyp["utterances"]["1"]["vad_prob"] == 0.9
    assert hyp["volumes"]["0.80-1.00"] == 1


def test_el_snapshot_de_hipotesis_no_se_filtra_al_html(tmp_path):
    """El ciego se mantiene: el index.html no referencia hypotheses.json ni
    contiene el texto del modelo."""
    real = tmp_path / "real.flac"
    real.write_bytes(b"fake-flac-bytes")
    db_path = _make_db(tmp_path, [str(real)], vads=[0.9])
    out_dir = tmp_path / "out"

    _export(str(db_path), str(out_dir), per_bucket=7, seed=1)

    html = (out_dir / "index.html").read_text()
    assert "hypotheses" not in html
    assert "texto 0" not in html


def test_export_crea_el_directorio_con_permisos_0700(tmp_path):
    """Es una SEGUNDA copia permanente de audio de la casa, fuera del TTL."""
    real = tmp_path / "real.flac"
    real.write_bytes(b"fake-flac-bytes")
    db_path = _make_db(tmp_path, [str(real)])
    out_dir = tmp_path / "out"

    _export(str(db_path), str(out_dir), per_bucket=7, seed=1)

    assert (out_dir.stat().st_mode & 0o777) == 0o700


def _escribir_set(tmp_path: Path, data: dict, ids: list[int] | None = None):
    """groundtruth.json + meta.json hermanos, como los deja el export."""
    (tmp_path / "groundtruth.json").write_text(json.dumps(data))
    (tmp_path / "meta.json").write_text(json.dumps(
        {"seed": 1, "per_bucket": 7, "db": "x",
         "ids": ids if ids is not None else [int(k) for k in data]}))
    return str(tmp_path / "groundtruth.json")


def test_validate_acepta_un_set_completo(tmp_path):
    path = _escribir_set(tmp_path, {"1": "hola", "2": "", "3": "[ininteligible]"})
    _validate(path)          # no lanza


def test_validate_falla_si_faltan_clips(tmp_path):
    """El modo de falla real: una sesión a medias. Si el validador dice OK,
    los clips no visitados se miden como "no había habla"."""
    path = _escribir_set(tmp_path, {"1": "hola"}, ids=[1, 2, 3])
    with pytest.raises(SystemExit) as exc:
        _validate(path)
    assert exc.value.code == 1


def test_validate_falla_con_claves_ajenas_al_export(tmp_path):
    path = _escribir_set(tmp_path, {"1": "hola", "77": "de otro set"}, ids=[1])
    with pytest.raises(SystemExit) as exc:
        _validate(path)
    assert exc.value.code == 1


def test_validate_falla_si_una_referencia_no_es_texto(tmp_path):
    path = _escribir_set(tmp_path, {"1": "hola", "2": None})
    with pytest.raises(SystemExit) as exc:
        _validate(path)
    assert exc.value.code == 1


def test_validate_falla_si_no_encuentra_el_meta(tmp_path):
    """Sin los ids del export no hay forma de saber si el set está completo:
    un validador que no puede chequear no puede decir OK."""
    (tmp_path / "groundtruth.json").write_text(json.dumps({"1": "hola"}))
    with pytest.raises(SystemExit) as exc:
        _validate(str(tmp_path / "groundtruth.json"))
    assert exc.value.code == 1


def test_validate_avisa_marcador_inline(tmp_path, capsys):
    """Referencia con "[ininteligible]" INCRUSTADO en una frase más larga:
    el scorer lo bracket-stripea y lo puntúa como la palabra 'ininteligible'
    (sustitución/deleción falsa). _validate debe avisar nombrando el id,
    SIN invalidar el set (sigue siendo usable, solo distorsiona ese par)."""
    path = _escribir_set(tmp_path, {
        "1": "hola",
        "2": "dijo algo [ininteligible] y se fue",
    })
    _validate(path)          # no lanza
    assert "INLINE" in capsys.readouterr().out


def test_validate_no_avisa_marcador_completo(tmp_path, capsys):
    """M13 (review PR #15): caso negativo del AVISO inline — una referencia
    que ES exactamente '[ininteligible]' o '[tv]' completa (sin texto
    alrededor) es justo lo que is_excluded() sabe manejar; no debe
    contarse como INLINE ni generar el aviso."""
    path = _escribir_set(tmp_path, {
        "1": "[ininteligible]",
        "2": "[tv]",
    })
    _validate(path)          # no lanza
    assert "INLINE" not in capsys.readouterr().out


def test_validate_falla_con_json_malformado(tmp_path):
    (tmp_path / "groundtruth.json").write_text("{no es json")
    with pytest.raises(SystemExit) as exc:
        _validate(str(tmp_path / "groundtruth.json"))
    assert exc.value.code == 1


def test_docstring_documenta_los_exit_codes():
    """La corre a mano el dueño del proyecto durante la campaña; los códigos
    tienen que estar en --help, no en el código fuente."""
    assert "Exit codes:" in gt_mod.__doc__
