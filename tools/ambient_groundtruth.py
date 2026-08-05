#!/usr/bin/env python3
"""Arma el set de ground truth para medir la fidelidad del ambient.

CIEGO POR DISEÑO: el HTML de transcripción muestra el audio y un campo vacío,
nunca lo que transcribió el modelo. Ver la hipótesis ancla al transcriptor
humano y contamina la referencia — es la lección directa del eval de clima,
donde medir sobre el mismo set con el que se construyó la solución dio 95,5%
y el held-out limpio reveló el fallo que canceló el proyecto.

Uso:
    # 1. exportar el set desde la DB (en el server, o sobre una copia)
    PYTHONPATH=. python3 tools/ambient_groundtruth.py --export \\
        --db data/ambient.db --out /tmp/gt --per-bucket 7

    # 2. abrir /tmp/gt/index.html, escuchar y transcribir, guardar el JSON

    # 3. validar el JSON completado
    PYTHONPATH=. python3 tools/ambient_groundtruth.py --validate /tmp/gt/groundtruth.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from src.ambient.wer import UNINTELLIGIBLE, bucket_of


def sample_stratified(
    rows: list[dict], per_bucket: int, seed: int
) -> list[dict]:
    """Muestrear ``per_bucket`` filas de CADA bucket de vad_prob.

    Asignación IGUAL, no proporcional: una proporcional le daría al bucket
    0.80-1.00 apenas 2 de 42 muestras (283 de 5.939 filas en prod), que es
    justo el bucket donde queremos saber si ya estamos en ~95%.

    Args:
        rows: Filas de utterances (necesitan ``vad_prob`` y ``audio_path``).
        per_bucket: Cuántas tomar de cada bucket (si hay menos, toma todas).
        seed: Semilla del muestreo, para reproducibilidad.

    Returns:
        Filas seleccionadas, ordenadas por bucket y luego por id.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("audio_path"):
            continue          # sin audio no se puede transcribir a mano
        by_bucket[bucket_of(r.get("vad_prob"))].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for bucket in sorted(by_bucket):
        pool = sorted(by_bucket[bucket], key=lambda r: r["id"])
        out.extend(rng.sample(pool, min(per_bucket, len(pool))))
    return out


def render_html(items: list[dict]) -> str:
    """HTML autocontenido para transcribir a ciegas.

    NO incluye el campo ``text`` de los items: el transcriptor no debe ver la
    hipótesis del modelo.

    Args:
        items: Dicts con ``id``, ``room_id``, ``vad_prob`` y ``audio``
            (nombre de archivo relativo al HTML).

    Returns:
        Documento HTML completo, sin red ni dependencias.
    """
    rows = "\n".join(
        f'<li><b>#{it["id"]}</b> <small>{it["room_id"]} · vad '
        f'{it["vad_prob"]:.2f}</small><br>'
        f'<audio controls src="{it["audio"]}"></audio><br>'
        f'<textarea data-id="{it["id"]}" rows="2" cols="70" '
        f'placeholder="lo que escuchás; vacío si no hay habla"></textarea></li>'
        for it in items
    )
    return f"""<!doctype html><meta charset="utf-8">
<title>Ground truth ambient</title>
<style>body{{font:15px system-ui;max-width:60rem;margin:2rem auto;padding:0 1rem}}
li{{margin-bottom:1.5rem}} textarea{{font:14px ui-monospace,monospace}}</style>
<h1>Transcripción de referencia</h1>
<p>Escuchá y escribí <b>exactamente</b> lo que se dice. Dejá vacío si no hay
habla. Usá <code>{UNINTELLIGIBLE}</code> si hay voz pero no se entiende, y
<code>[tv]</code> si la fuente es un parlante.</p>
<p><b>No vas a ver lo que transcribió el modelo</b> — verlo anclaría tu
referencia y arruinaría la medición.</p>
<ol>{rows}</ol>
<button onclick="save()">Guardar groundtruth.json</button>
<script>
function save() {{
  const out = {{}};
  document.querySelectorAll('textarea').forEach(t => out[t.dataset.id] = t.value);
  const b = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'groundtruth.json'; a.click();
}}
</script>"""


def _export(db_path: str, out_dir: str, per_bucket: int, seed: int) -> None:
    """Exportar el set: copia los FLAC y genera index.html + metadata."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        "SELECT id, room_id, vad_prob, text, audio_path FROM utterances "
        "WHERE audio_path IS NOT NULL"
    )]
    sel = sample_stratified(rows, per_bucket, seed)
    if not sel:
        print("ERROR: ninguna utterance tiene audio archivado. "
              "¿Está prendido ambient.keep_audio?", file=sys.stderr)
        raise SystemExit(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for r in sel:
        name = f'{r["id"]}.flac'
        try:
            shutil.copy(r["audio_path"], out / name)
        except OSError as e:
            print(f"  aviso: no se pudo copiar {r['audio_path']}: {e}", file=sys.stderr)
            continue
        items.append({"id": r["id"], "room_id": r["room_id"],
                      "vad_prob": r["vad_prob"] or 0.0, "audio": name})
    (out / "index.html").write_text(render_html(items), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(
        {"seed": seed, "per_bucket": per_bucket, "db": db_path,
         "ids": [i["id"] for i in items]}, indent=2), encoding="utf-8")
    print(f"{len(items)} utterances exportadas a {out}/")
    print(f"Abrí {out}/index.html, transcribí, y guardá el groundtruth.json ahí mismo.")


def _validate(path: str) -> None:
    """Verificar que el JSON completado sea usable."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vacias = [k for k, v in data.items() if v.strip() == ""]
    print(f"{len(data)} referencias; {len(vacias)} marcadas como sin habla")
    print("OK — usable por tools/ambient_wer.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--db", default="data/ambient.db")
    ap.add_argument("--out", default="/tmp/gt")
    ap.add_argument("--per-bucket", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--validate", metavar="JSON")
    args = ap.parse_args()
    if args.export:
        _export(args.db, args.out, args.per_bucket, args.seed)
    elif args.validate:
        _validate(args.validate)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
