"""Tests: armado del set de ground truth ciego."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.ambient_groundtruth import _export, render_html, sample_stratified


def _make_db(tmp_path: Path, audio_paths: list[str | None]) -> Path:
    """DB sqlite mínima de ``utterances`` con los ``audio_path`` dados."""
    db_path = tmp_path / "ambient.db"
    db = sqlite3.connect(db_path)
    db.execute(
        "CREATE TABLE utterances (id INTEGER PRIMARY KEY, room_id TEXT, "
        "vad_prob REAL, text TEXT, audio_path TEXT)"
    )
    for i, audio_path in enumerate(audio_paths):
        db.execute(
            "INSERT INTO utterances (room_id, vad_prob, text, audio_path) "
            "VALUES (?, ?, ?, ?)",
            ("escritorio", 0.1, f"texto {i}", audio_path),
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


def test_el_html_NO_filtra_la_salida_del_modelo():
    """Ciego por diseño: ver la hipótesis ancla al transcriptor humano."""
    items = [{"id": 1, "room_id": "escritorio", "vad_prob": 0.5,
              "audio": "1.flac", "text": "GRACIAS POR VER EL VIDEO"}]
    html = render_html(items)
    assert "GRACIAS POR VER EL VIDEO" not in html
    assert "1.flac" in html


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
