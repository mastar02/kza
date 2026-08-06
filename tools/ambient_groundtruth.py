#!/usr/bin/env python3
"""Arma el set de ground truth para medir la fidelidad del ambient.

CIEGO POR DISEÑO: el HTML de transcripción muestra el audio y un campo vacío,
nunca lo que transcribió el modelo, ni el `vad_prob`, ni la habitación. Ver la
hipótesis ancla al transcriptor humano y contamina la referencia; ver el
`vad_prob` es peor todavía, porque `vad_prob` ES la variable que el estudio
quiere validar — mostrarla le dice al anotador dónde esforzarse y dónde
rendirse. Es la lección directa del eval de clima, donde medir sobre el mismo
set con el que se construyó la solución dio 95,5% y el held-out limpio reveló
el fallo que canceló el proyecto.

Uso:
    # 1. exportar el set desde la DB (en el server, o sobre una copia)
    PYTHONPATH=. python3 tools/ambient_groundtruth.py --export \\
        --db data/ambient.db --out data/gt --per-bucket 7

    # 2. abrir data/gt/index.html, escuchar y transcribir, guardar el JSON
    #    ahí mismo (data/gt/groundtruth.json)

    # 3. validar el JSON completado ANTES de medir
    PYTHONPATH=. python3 tools/ambient_groundtruth.py --validate data/gt/groundtruth.json

Exit codes:
    0   Éxito. En --export: todos los audios seleccionados se copiaron. En
        --validate: el set está completo y es usable por tools/ambient_wer.py.
    1   Fallo total. En --export: no hay nada que exportar (¿keep_audio
        apagado?) o ningún audio se pudo copiar — no queda directorio útil.
        En --validate: el JSON está incompleto, malformado, o no se pudo
        cruzar contra meta.json — NO usarlo para medir.
    2   Éxito parcial del export: se copiaron algunos audios pero no todos.
        El set quedó incompleto y puede no alcanzar para medir (p. ej. un
        bucket entero sin muestras). Revisar los avisos antes de usarlo.
        (argparse usa también el código 2 para errores de uso de la CLI.)

Privacidad: el directorio de salida es una SEGUNDA copia del audio de la casa,
fuera del TTL de la DB. Nada lo borra solo. Se crea con permisos 0700 y se
borra a mano al terminar la campaña (paso 6 del runbook del plan).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from src.ambient.wer import UNINTELLIGIBLE, bucket_of, is_excluded

# Universo muestreable. UNA sola constante a propósito: es la población de la
# que sale la muestra Y la población sobre la que se calculan los pesos del
# agregado re-ponderado (`hypotheses.json` → `volumes`). Si los pesos se
# calcularan sobre otra población (p. ej. `SELECT vad_prob FROM utterances` a
# secas), el agregado re-ponderaría la muestra por una distribución que la
# muestra no puede representar.
#   - ``audio_path IS NOT NULL``: sin audio no hay nada que transcribir a
#     mano. Las filas pre-campaña no tienen audio y jamás son muestreables.
#   - ``vad_prob IS NOT NULL``: el eje del estudio es vad_prob. Una fila sin
#     vad no cae en ningún bucket, no puede recibir peso, y crearía un 7º
#     estrato ("sin_vad") que el spec no contempla (6 buckets × 7 = 42) y que
#     después pesaría en el agregado sin significar nada.
#   - ``source NOT IN ('self','tv')``: 'self' es nuestro propio TTS — audio
#     limpísimo, vad alto — y aterrizaría en el bucket 0.80-1.00, justo el que
#     existe para responder "¿ya estamos en ~95%?". El resto del sistema lo
#     excluye por lo mismo (ver `undistilled_live` en src/ambient/store.py).
SAMPLEABLE_WHERE = (
    "audio_path IS NOT NULL AND vad_prob IS NOT NULL "
    "AND source NOT IN ('self','tv')"
)

# Etiqueta del estrato degenerado, derivada de la fuente de verdad en vez de
# hardcodeada: si src.ambient.wer la renombra, esto la sigue.
SIN_VAD = bucket_of(None)


def bucket_volumes(rows: list[dict]) -> dict[str, int]:
    """Contar cuántas filas del universo muestreable caen en cada bucket.

    Estos son los pesos del agregado re-ponderado. Se calculan acá, en el
    export, y viajan en ``hypotheses.json``: calcularlos después contra la DB
    los mediría sobre una ventana distinta, porque la purga por TTL corre cada
    48 h y se lleva filas del medio.

    Args:
        rows: Filas ya filtradas por :data:`SAMPLEABLE_WHERE`.

    Returns:
        ``{etiqueta_de_bucket: n}``, sin la entrada ``sin_vad``.
    """
    vols: dict[str, int] = defaultdict(int)
    for r in rows:
        b = bucket_of(r.get("vad_prob"))
        if b == SIN_VAD:
            continue
        vols[b] += 1
    return dict(vols)


def sample_stratified(
    rows: list[dict], per_bucket: int, seed: int
) -> list[dict]:
    """Muestrear ``per_bucket`` filas de CADA bucket de vad_prob.

    Asignación IGUAL, no proporcional: una proporcional le daría al bucket
    0.80-1.00 apenas 2 de 42 muestras (283 de 5.939 filas en prod), que es
    justo el bucket donde queremos saber si ya estamos en ~95%.

    Las filas sin ``vad_prob`` se descartan: no pertenecen a ningún bucket del
    eje que el estudio valida, y admitirlas crearía un 7º estrato con 7 clips
    más que el spec no contempla y que después recibiría peso en el agregado.

    Args:
        rows: Filas de utterances (necesitan ``vad_prob`` y ``audio_path``).
        per_bucket: Cuántas tomar de cada bucket (si hay menos, toma todas).
        seed: Semilla del muestreo, para reproducibilidad.

    Returns:
        Filas seleccionadas, agrupadas por bucket en orden ascendente
        (``0.00-0.20`` primero). Dentro de cada bucket NO están ordenadas
        por id: ``random.Random.sample`` devuelve en orden de selección,
        no en el orden del pool de entrada. El orden de **presentación** al
        humano lo baraja :func:`render_html`, que es donde importa.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("audio_path"):
            continue          # sin audio no se puede transcribir a mano
        bucket = bucket_of(r.get("vad_prob"))
        if bucket == SIN_VAD:
            continue          # fuera del eje del estudio (ver docstring)
        by_bucket[bucket].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for bucket in sorted(by_bucket):
        pool = sorted(by_bucket[bucket], key=lambda r: r["id"])
        out.extend(rng.sample(pool, min(per_bucket, len(pool))))
    return out


def render_html(items: list[dict], seed: int = 0) -> str:
    """HTML autocontenido para transcribir a ciegas.

    Muestra el audio y un campo vacío, nada más. NO incluye:

    - el campo ``text`` de los items (la hipótesis del modelo);
    - el ``vad_prob`` — es *la* variable que el estudio quiere validar
      ("¿vad_prob predice el WER?"); mostrárselo al anotador antes de que
      escriba sesga dónde pone esfuerzo y cuándo se rinde con
      ``[ininteligible]``;
    - el ``room_id`` — proxy directo del mismo sesgo (la cocina está más lejos
      de donde se habla: 24,2% de español contra 49,6% del escritorio).

    Por la misma razón el orden se baraja: presentados por bucket ascendente,
    los primeros clips son basura y los últimos limpios, y ese patrón es
    aprendible en las primeras cinco escuchas.

    "Sin habla" es un estado EXPLÍCITO (checkbox), no la ausencia de texto: un
    textarea que el humano nunca visitó no puede volcarse al JSON como
    "acá no había voz" — eso convertiría cada clip no visitado en una
    alucinación con WER 1.0.

    Args:
        items: Dicts con ``id`` y ``audio`` (nombre de archivo relativo al
            HTML). Cualquier otra clave se ignora.
        seed: Semilla del barajado de presentación (la misma del muestreo).

    Returns:
        Documento HTML completo, sin red ni dependencias.
    """
    orden = list(items)
    random.Random(seed).shuffle(orden)
    rows = "\n".join(
        f'<li><b>#{it["id"]}</b><br>'
        f'<audio controls src="{it["audio"]}"></audio><br>'
        f'<textarea data-id="{it["id"]}" rows="2" cols="70" '
        f'placeholder="lo que escuchás"></textarea><br>'
        f'<label><input type="checkbox" data-empty="{it["id"]}"> '
        f'<b>sin habla</b> — lo escuché y no hay voz</label></li>'
        for it in orden
    )
    total = len(orden)
    return f"""<!doctype html><meta charset="utf-8">
<title>Ground truth ambient</title>
<style>body{{font:15px system-ui;max-width:60rem;margin:2rem auto;padding:0 1rem}}
li{{margin-bottom:1.5rem}} textarea{{font:14px ui-monospace,monospace}}
#estado{{position:sticky;top:0;background:#fff;padding:.5rem 0;font-weight:600}}
label{{font-size:14px;color:#444}}</style>
<h1>Transcripción de referencia</h1>
<p>Escuchá y escribí <b>exactamente</b> lo que se dice. Usá
<code>{UNINTELLIGIBLE}</code> si hay voz pero no se entiende, y
<code>[tv]</code> si la fuente es un parlante.</p>
<p>Si escuchaste el clip y <b>no hay voz</b>, marcá la casilla
"sin habla". <b>No alcanza con dejarlo vacío</b>: vacío significa
"todavía no lo escuché", y no es lo mismo. Un clip sin resolver no entra al
JSON.</p>
<p><b>No vas a ver lo que transcribió el modelo, ni la señal de calidad que
midió el sistema, ni la habitación</b> — verlos anclaría tu referencia y
arruinaría la medición. El orden está barajado a propósito.</p>
<p id="estado"></p>
<ol>{rows}</ol>
<button onclick="save()">Guardar groundtruth.json</button>
<script>
const TOTAL = {total};
function resueltos() {{
  let n = 0;
  document.querySelectorAll('textarea').forEach(t => {{
    const box = document.querySelector('[data-empty="' + t.dataset.id + '"]');
    if (box.checked || t.value.trim() !== '') n++;
  }});
  return n;
}}
function pintar() {{
  const n = resueltos();
  document.getElementById('estado').textContent =
    n + ' de ' + TOTAL + ' resueltos — faltan ' + (TOTAL - n);
}}
document.addEventListener('input', pintar);
document.addEventListener('change', pintar);
pintar();
function save() {{
  const out = {{}};
  let faltan = 0;
  document.querySelectorAll('textarea').forEach(t => {{
    const id = t.dataset.id;
    const box = document.querySelector('[data-empty="' + id + '"]');
    const txt = t.value.trim();
    if (box.checked) {{
      out[id] = '';            // sin habla, explícito
    }} else if (txt !== '') {{
      out[id] = txt;
    }} else {{
      faltan++;                // no visitado: NO entra al JSON
    }}
  }});
  if (faltan > 0 && !confirm(
      faltan + ' clip(s) sin resolver quedan FUERA del JSON. ' +
      'El validador va a marcar el set como incompleto. ¿Guardar igual?')) {{
    return;
  }}
  const b = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'groundtruth.json'; a.click();
}}
</script>"""


def _export(db_path: str, out_dir: str, per_bucket: int, seed: int) -> None:
    """Exportar el set: copia los FLAC y genera index.html + metadata.

    Args:
        db_path: Ruta a ``ambient.db``.
        out_dir: Directorio de trabajo (se crea con permisos 0700).
        per_bucket: Muestras por bucket de vad_prob.
        seed: Semilla del muestreo y del barajado de presentación.

    Raises:
        SystemExit: 1 si no hay nada que exportar o no se copió ni un audio;
            2 si el export fue parcial.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        "SELECT id, room_id, vad_prob, text, audio_path FROM utterances "
        f"WHERE {SAMPLEABLE_WHERE}"  # nosec B608 -- constante del módulo, sin datos externos
    )]
    db.close()
    volumes = bucket_volumes(rows)
    sel = sample_stratified(rows, per_bucket, seed)
    if not sel:
        print("ERROR: ninguna utterance del universo muestreable tiene audio "
              "archivado. ¿Está prendido ambient.keep_audio?", file=sys.stderr)
        raise SystemExit(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Segunda copia permanente de audio de la casa, fuera del TTL de la DB:
    # 0700 explícito (mkdir(mode=) queda a merced del umask).
    try:
        os.chmod(out, 0o700)
    except OSError as e:
        print(f"  aviso: no se pudo aplicar chmod 0700 a {out}: {e}", file=sys.stderr)
    items = []
    fallidas = 0
    for r in sel:
        name = f'{r["id"]}.flac'
        try:
            shutil.copy(r["audio_path"], out / name)
        except OSError as e:
            print(f"  aviso: no se pudo copiar {r['audio_path']}: {e}", file=sys.stderr)
            fallidas += 1
            continue
        items.append({"id": r["id"], "room_id": r["room_id"],
                      "vad_prob": r["vad_prob"], "text": r["text"],
                      "audio": name})

    # El guard de "sel vacío" de arriba mide un proxy barato (la query
    # devolvió filas); lo que importa es lo que REALMENTE se copió a disco.
    # Si audio_path apunta a archivos que ya no existen, sel no está vacío
    # pero items sí, y sin este chequeo el export "termina bien" sobre un
    # directorio inútil — el mismo patrón de proxy mentiroso que ya rompió
    # el watchdog de audio (exit 0 sin haber capturado nada). Las escrituras
    # van DESPUÉS del guard: un fallo total no debe dejar un index.html que
    # invite a transcribir la nada.
    if not items:
        print(f"ERROR: {len(sel)} utterances seleccionadas, 0 copiadas "
              f"({fallidas} fallaron). El export quedó vacío pese a que la "
              f"query trajo filas — revisá si audio_path apunta a archivos "
              f"borrados del disco.", file=sys.stderr)
        try:
            out.rmdir()          # solo si quedó vacío; si no, OSError y se deja
        except OSError:
            pass
        raise SystemExit(1)

    (out / "index.html").write_text(render_html(items, seed), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(
        {"seed": seed, "per_bucket": per_bucket, "db": db_path,
         "ids": [i["id"] for i in items], "fallidas": fallidas},
        indent=2), encoding="utf-8")
    # Snapshot de la hipótesis y de los pesos. La DB purga a las 48 h y el
    # muestreo es uniforme sobre la ventana de retención: la edad esperada de
    # un clip es ~24 h, o sea que la mitad del set expira dentro del día
    # siguiente al export. El audio sobrevive (se copió); sin este archivo, el
    # texto contra el que hay que compararlo no. El index.html NO lo
    # referencia: el ciego se mantiene intacto.
    (out / "hypotheses.json").write_text(json.dumps(
        {"generado_por": "tools/ambient_groundtruth.py --export",
         "db": db_path, "seed": seed,
         "utterances": {
             str(i["id"]): {"text": i["text"], "vad_prob": i["vad_prob"],
                            "room_id": i["room_id"]}
             for i in items
         },
         "volumes": volumes},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(items)} utterances exportadas a {out}/ "
          f"({len(sel)} seleccionadas, {fallidas} fallidas)")
    print(f"  hipótesis y pesos snapshoteados en {out}/hypotheses.json "
          f"(NO lo abras: contiene lo que transcribió el modelo)")
    if fallidas:
        # Éxito parcial: el set quedó incompleto y puede no alcanzar para
        # medir bien (p.ej. un bucket entero sin muestras). Ruidoso en
        # stdout/stderr y con exit code propio para que scripts que
        # encadenen este export no lo confundan con un éxito limpio.
        print(f"AVISO: {fallidas} de {len(sel)} seleccionadas NO se "
              f"copiaron — revisá los avisos de arriba antes de usar este "
              f"set para medir.", file=sys.stderr)
        print(f"Abrí {out}/index.html, transcribí, y guardá el groundtruth.json ahí mismo.")
        raise SystemExit(2)
    print(f"Abrí {out}/index.html, transcribí, y guardá el groundtruth.json ahí mismo.")


def _validate(path: str, meta_path: str | None = None) -> None:
    """Verificar que el JSON completado esté COMPLETO y sea usable.

    Un validador que siempre dice OK es peor que no tener validador: la
    campaña se mide una sola vez y un set incompleto produce números que
    parecen buenos. Se cruza contra los ids de ``meta.json``, se cuentan
    visitados y faltantes, y se validan los tipos.

    Args:
        path: Ruta al ``groundtruth.json`` que devolvió el HTML.
        meta_path: Ruta al ``meta.json`` del export; por default, el hermano
            del ``groundtruth.json``.

    Raises:
        SystemExit: 1 si el set está incompleto, malformado, o no se pudo
            cruzar contra ``meta.json``.
    """
    gt_path = Path(path)
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: no se pudo leer {gt_path}: {e}", file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(data, dict):
        print(f"ERROR: {gt_path} no es un objeto JSON "
              f"(es {type(data).__name__}).", file=sys.stderr)
        raise SystemExit(1)

    problemas: list[str] = []

    no_str = sorted(k for k, v in data.items() if not isinstance(v, str))
    if no_str:
        problemas.append(
            f"{len(no_str)} referencia(s) no son texto: {no_str[:10]}"
        )

    meta_file = Path(meta_path) if meta_path else gt_path.parent / "meta.json"
    esperados: set[str] | None = None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        esperados = {str(i) for i in meta.get("ids", [])}
    except (OSError, json.JSONDecodeError) as e:
        problemas.append(
            f"no se pudo leer {meta_file} ({e}) — sin los ids del export es "
            f"imposible saber si el set está completo"
        )

    faltantes: list[str] = []
    sobrantes: list[str] = []
    if esperados is not None:
        if not esperados:
            problemas.append(f"{meta_file} no declara ningún id ('ids' vacío)")
        presentes = set(data)
        faltantes = sorted(esperados - presentes, key=lambda s: int(s))
        sobrantes = sorted(presentes - esperados)
        if faltantes:
            problemas.append(
                f"{len(faltantes)} de {len(esperados)} clips sin resolver: "
                f"{faltantes[:10]}{' …' if len(faltantes) > 10 else ''}"
            )
        if sobrantes:
            problemas.append(
                f"{len(sobrantes)} clave(s) que no salieron de este export: "
                f"{sobrantes[:10]}"
            )

    textos = [v for v in data.values() if isinstance(v, str)]
    sin_habla = sum(1 for v in textos if v.strip() == "")
    marcadores = sum(1 for v in textos if is_excluded(v))
    con_texto = len(textos) - sin_habla - marcadores
    total = len(esperados) if esperados else len(data)
    print(f"{len(data)} de {total} clips resueltos")
    print(f"  con transcripción      : {con_texto}")
    print(f"  sin habla (explícito)  : {sin_habla}")
    print(f"  marcadores excluidos   : {marcadores}")

    inline = [uid for uid, v in data.items()
              if isinstance(v, str) and "[" in v and not is_excluded(v)]
    if inline:
        print(f"  AVISO: {len(inline)} referencia(s) con marcador INLINE — el "
              f"scorer las puntúa como palabras; un marcador vale solo como "
              f"referencia COMPLETA. Ids: {inline[:10]}")

    if problemas:
        print("\n" + "!" * 60, file=sys.stderr)
        print("SET NO USABLE — no medir con esto:", file=sys.stderr)
        for p in problemas:
            print(f"  - {p}", file=sys.stderr)
        print("!" * 60, file=sys.stderr)
        raise SystemExit(1)
    print("\nOK — set completo, usable por tools/ambient_wer.py")


def main() -> None:
    """CLI: exportar el set o validar el JSON completado."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mutuamente excluyentes: son fases distintas de la campaña y encadenarlas
    # en una sola invocación no tiene sentido (el JSON a validar todavía no
    # existe cuando corre el export). Antes el `elif` ignoraba la segunda en
    # silencio.
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument(
        "--export", action="store_true",
        help="armar el set: copia los FLAC y genera index.html, meta.json y "
             "hypotheses.json en --out",
    )
    modo.add_argument(
        "--validate", metavar="JSON",
        help="verificar que el groundtruth.json esté completo y bien formado "
             "(cruza contra el meta.json hermano; exit 1 si falta algo)",
    )
    ap.add_argument(
        "--db", default="data/ambient.db",
        help="ruta a ambient.db, o a una copia (default: %(default)s)",
    )
    ap.add_argument(
        "--out", default="data/gt",
        help="directorio de trabajo del export; se crea con permisos 0700 y "
             "hay que borrarlo al terminar la campaña (default: %(default)s)",
    )
    ap.add_argument(
        "--per-bucket", type=int, default=7,
        help="muestras por bucket de vad_prob; asignación IGUAL, no "
             "proporcional (default: %(default)s → 6×7=42)",
    )
    ap.add_argument(
        "--seed", type=int, default=20260805,
        help="semilla del muestreo y del barajado de presentación; queda "
             "registrada en meta.json (default: %(default)s)",
    )
    ap.add_argument(
        "--meta", default=None,
        help="meta.json contra el que validar (default: el hermano del JSON)",
    )
    args = ap.parse_args()
    if args.export:
        _export(args.db, args.out, args.per_bucket, args.seed)
    elif args.validate:
        _validate(args.validate, args.meta)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
