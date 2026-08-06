#!/usr/bin/env python3
"""Mide el WER del ambient contra el ground truth humano.

Reporta SIEMPRE por bucket de vad_prob, y el agregado de dos formas: plano y
re-ponderado por el volumen real de cada bucket. El plano NO es el WER del
sistema — el set tiene asignación igual por bucket y sobre-representa los
buckets altos a propósito.

El agregado re-ponderado solo se emite si el set **cubre** el corpus: si un
bucket con volumen real quedó sin pares evaluados, el número se calcularía
sobre los buckets sobrevivientes y hablaría por una fracción del corpus con la
autoridad del total. El caso no es hipotético: el bucket 0.00-0.20 es 86,9%
garble, o sea el bucket donde el humano escribe `[ininteligible]` — y esas
referencias se excluyen del WER, borrando el bucket entero.

Uso:
    PYTHONPATH=. python3 tools/ambient_wer.py \\
        --db data/ambient.db --groundtruth data/gt/groundtruth.json

    # las hipótesis y los pesos salen de data/gt/hypotheses.json si existe
    # (lo escribe el --export); la DB es solo el fallback.

Exit codes:
    0   Reporte construido y agregado confiable.
    1   No se pudo leer el ground truth, no hay de dónde sacar las
        hipótesis (ni snapshot ni DB), o el snapshot de hipótesis existe
        pero está corrupto (no se cae a la DB en silencio: ver `load_snapshot`).
    2   Reporte construido pero NO CONFIABLE (mirá `motivos` en el JSON) —
        no publicar el agregado.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from src.ambient.wer import bucket_of, is_excluded, score
from tools.ambient_groundtruth import SAMPLEABLE_WHERE

# Piso de cobertura: qué fracción del volumen real del corpus tiene que estar
# representada por buckets con pares evaluados para que el agregado
# re-ponderado hable en nombre del sistema. 0.95 es deliberadamente alto —
# el diseño del set ya apunta a cubrir el 100% (6 buckets × 7), así que
# cualquier pérdida real es un bucket entero desaparecido, no ruido de
# muestreo. Existe como guarda independiente del chequeo por bucket: ese
# responde "¿falta alguno?" y este "¿cuánto del corpus me quedó afuera?", que
# es el número que un humano necesita para decidir si vale la pena rehacer el
# set o ampliarlo.
COBERTURA_MINIMA = 0.95


def build_report(
    pairs: list[dict],
    volumes: dict[str, int],
    esperados: int | None = None,
) -> dict:
    """Construir el reporte de WER a partir de los pares referencia/hipótesis.

    ``buckets[b]["wer"]`` es un promedio MACRO (media del WER por utterance),
    no micro (errores totales / palabras de referencia totales). Con ``n>1``
    pesa igual a un utterance de 2 palabras que a uno de 20, y una
    alucinación (referencia vacía, wer=1.0 por convención) entra con el
    mismo peso aunque no aporte ninguna palabra de referencia real. Quien
    lea ``wer_simple``/``wer_reponderado`` tiene que saber que están
    construidos sobre ese promedio macro.

    El agregado re-ponderado **no se re-normaliza sobre los sobrevivientes**.
    Un bucket con volumen real pero sin pares evaluados desaparecería del
    numerador y del denominador a la vez, y el resultado sería un WER creíble
    calculado sobre un pedazo del corpus. Eso marca el reporte no confiable.

    Args:
        pairs: Dicts con ``id``, ``vad_prob``, ``reference``, ``hypothesis``.
        volumes: Conteo real de utterances por bucket en el universo
            muestreable, para el agregado re-ponderado. Una clave **ausente**
            para un bucket que sí tiene pares evaluados es una inconsistencia
            (desalineación upstream), NO un peso legítimo de cero — se
            distingue de un bucket presente con volumen real 0.
        esperados: Cuántos pares se esperaba medir (típicamente, cuántas
            referencias trajo el ground truth). Si se midieron menos, el
            reporte no es confiable: perder pares en silencio es exactamente
            cómo un set incompleto produce un número que parece bueno.

    Returns:
        Dict con ``buckets``, ``wer_simple`` y ``wer_reponderado`` (``None``
        si no hay datos / el reporte no es confiable), ``confiable`` (bool),
        ``motivos`` (por qué no lo es), ``buckets_sin_volumen``,
        ``buckets_sin_pares``, ``cobertura_volumen``, ``cobertura_minima``,
        ``volumen_total``, ``volumen_evaluado``, ``pares_medidos``,
        ``pares_esperados``, ``deleciones_totales``, ``alucinaciones`` y
        ``excluidas``.
    """
    per_bucket: dict[str, list[float]] = defaultdict(list)
    dels_totales = alucinaciones = excluidas = 0
    for p in pairs:
        ref, hyp = p["reference"], p["hypothesis"]
        if is_excluded(ref):
            excluidas += 1
            continue
        r = score(ref, hyp)
        per_bucket[bucket_of(p.get("vad_prob"))].append(r.wer)
        if ref.strip() and not hyp.strip():
            dels_totales += 1
        if not ref.strip() and hyp.strip():
            alucinaciones += 1
    buckets = {
        b: {"n": len(v), "wer": sum(v) / len(v)}
        for b, v in sorted(per_bucket.items()) if v
    }
    todas = [w for v in per_bucket.values() for w in v]

    # Un bucket con pares evaluados pero SIN clave en `volumes` es una
    # inconsistencia de datos, no un peso cero: si se tratara como cero
    # (como hacía la versión anterior con `.get(b, 0)`), un bucket con
    # WER alto podría desaparecer silenciosamente del agregado y producir
    # un `wer_reponderado` bajo y creíble pero falso. Se distingue del caso
    # legítimo de un bucket presente en `volumes` con valor 0 (existe en la
    # DB pero de verdad no tiene utterances ahí).
    buckets_sin_volumen = sorted(b for b in buckets if b not in volumes)
    # El espejo del anterior, y el que de verdad muerde en producción: un
    # bucket con volumen real que NO tiene ni un par evaluado. Se cae del
    # numerador y del denominador a la vez, y el promedio se re-normaliza en
    # silencio sobre los que quedaron.
    buckets_sin_pares = sorted(
        b for b, n in volumes.items() if n > 0 and b not in buckets
    )

    volumen_total = sum(volumes.values())
    volumen_evaluado = sum(volumes[b] for b in buckets if b in volumes)
    cobertura = (volumen_evaluado / volumen_total) if volumen_total else None

    motivos: list[str] = []
    if not buckets:
        motivos.append(
            "no hay ningún par evaluado (¿el ground truth vino vacío, o todas "
            "las referencias son marcadores excluidos?)"
        )
    if buckets_sin_volumen:
        motivos.append(
            "bucket(s) con pares evaluados pero sin volumen real declarado: "
            + ", ".join(buckets_sin_volumen)
            + " — tratarlo como peso 0 lo haría desaparecer del agregado en "
              "vez de pesar como corresponde"
        )
    if buckets_sin_pares:
        motivos.append(
            "bucket(s) con volumen real pero SIN pares evaluados: "
            + ", ".join(f"{b} (n={volumes[b]})" for b in buckets_sin_pares)
            + " — el agregado se re-normalizaría sobre los buckets que "
              "sobrevivieron y hablaría por una fracción del corpus"
        )
    if buckets and volumen_total == 0:
        motivos.append(
            "todos los buckets declaran volumen real 0: el agregado "
            "re-ponderado no está definido (0/0), no es 0.0"
        )
    if esperados is not None and len(pairs) < esperados:
        motivos.append(
            f"se esperaban {esperados} pares y se midieron {len(pairs)}: "
            f"faltan {esperados - len(pairs)} hipótesis (¿purgadas de la DB "
            f"sin snapshot?)"
        )
    if cobertura is not None and cobertura < COBERTURA_MINIMA:
        motivos.append(
            f"cobertura {cobertura:.1%} del volumen real, debajo del piso "
            f"{COBERTURA_MINIMA:.0%}"
        )

    confiable = not motivos
    reponderado: float | None = None
    if confiable and volumen_evaluado > 0:
        reponderado = sum(
            st["wer"] * volumes[b] for b, st in buckets.items()
        ) / volumen_evaluado

    return {
        "buckets": buckets,
        "wer_simple": (sum(todas) / len(todas)) if todas else None,
        "wer_reponderado": reponderado,
        "confiable": confiable,
        "motivos": motivos,
        "buckets_sin_volumen": buckets_sin_volumen,
        "buckets_sin_pares": buckets_sin_pares,
        "cobertura_volumen": cobertura,
        "cobertura_minima": COBERTURA_MINIMA,
        "volumen_total": volumen_total,
        "volumen_evaluado": volumen_evaluado,
        "pares_medidos": len(pairs),
        "pares_esperados": esperados,
        "deleciones_totales": dels_totales,
        "alucinaciones": alucinaciones,
        "excluidas": excluidas,
    }


def load_snapshot(path: Path) -> tuple[dict[str, dict], dict[str, int], str | None]:
    """Leer el ``hypotheses.json`` que escribió el export.

    Returns:
        ``(utterances, volumes, error)``. ``error`` es None si el archivo no
        existe (fallback a DB esperado) o si se leyó bien; si el archivo
        EXISTE pero no parsea, ``error`` describe el problema — el snapshot
        de una campaña se escribe una sola vez, y confundir "corrupto" con
        "ausente" manda al operador a una DB que purga a las 48h en vez de
        al backup del archivo.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}, {}, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, {}, f"snapshot corrupto: {e}"
    utts = data.get("utterances") or {}
    vols = {k: int(v) for k, v in (data.get("volumes") or {}).items()}
    return utts, vols, None


def _fmt(x: float | None) -> str:
    """Formatear una métrica que puede no estar definida."""
    return "n/d" if x is None else f"{x:.3f}"


def main() -> None:
    """CLI: cruzar el ground truth con las hipótesis y reportar el WER."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--db", default="data/ambient.db",
        help="ambient.db, usada como FALLBACK si falta el snapshot de "
             "hipótesis (default: %(default)s)",
    )
    ap.add_argument(
        "--groundtruth", required=True,
        help="groundtruth.json con la transcripción humana (obligatorio)",
    )
    ap.add_argument(
        "--hypotheses", default=None,
        help="hypotheses.json del export, con lo que dijo el modelo y los "
             "pesos por bucket (default: el hermano del groundtruth)",
    )
    ap.add_argument(
        "--out", default=None,
        help="dónde guardar el reporte JSON "
             "(default: data/wer_report_<nombre del groundtruth>.json)",
    )
    args = ap.parse_args()

    gt_path = Path(args.groundtruth)
    try:
        refs = json.loads(gt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: no se pudo leer {gt_path}: {e}", file=sys.stderr)
        raise SystemExit(1)

    snap_path = Path(args.hypotheses) if args.hypotheses else gt_path.parent / "hypotheses.json"
    snap_utts, snap_vols, snap_error = load_snapshot(snap_path)
    if snap_error:
        print(f"ERROR: {snap_path} — {snap_error}. NO se cae a la DB (purga "
              f"48h, ventana corrida): recuperá el snapshot del backup.",
              file=sys.stderr)
        raise SystemExit(1)
    if snap_utts:
        print(f"hipótesis desde el snapshot {snap_path} ({len(snap_utts)} filas)")
    else:
        print(f"AVISO: sin snapshot usable en {snap_path} — se cae a la DB, "
              f"que purga a las 48 h y pudo haberse llevado la mitad del set",
              file=sys.stderr)

    db: sqlite3.Connection | None = None
    if not snap_utts or not snap_vols:
        try:
            db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
            db.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print(f"AVISO: no se pudo abrir {args.db} ({e})", file=sys.stderr)

    if not snap_utts and db is None:
        # Ni snapshot (hypotheses.json) ni DB: no hay de dónde sacar UNA
        # sola hipótesis, así que no hay nada que medir. El reporte de más
        # abajo lo marcaría igual como NO CONFIABLE — fail-loud para un
        # humano leyendo la consola — pero eso solo, sin exit 1, deja pasar
        # un `--validate && ambient_wer && …` encadenado: exit 0 ES el
        # contrato que ese encadenado lee como éxito.
        print("ERROR: no hay de dónde sacar las hipótesis (ni snapshot ni "
              "DB) — nada que medir.", file=sys.stderr)
        raise SystemExit(1)

    pairs = []
    perdidas = []
    for uid, reference in refs.items():
        if not isinstance(reference, str):
            print(f"  aviso: la referencia de {uid} no es texto "
                  f"({type(reference).__name__}) — se ignora", file=sys.stderr)
            perdidas.append(str(uid))
            continue
        snap = snap_utts.get(str(uid))
        if snap is not None:
            pairs.append({"id": int(uid), "vad_prob": snap.get("vad_prob"),
                          "reference": reference,
                          "hypothesis": snap.get("text") or ""})
            continue
        row = db.execute(
            "SELECT text, vad_prob FROM utterances WHERE id=?", (int(uid),)
        ).fetchone() if db is not None else None
        if row is None:
            print(f"  aviso: sin hipótesis para la utterance {uid} "
                  f"(purgada de la DB y ausente del snapshot)", file=sys.stderr)
            perdidas.append(str(uid))
            continue
        pairs.append({"id": int(uid), "vad_prob": row["vad_prob"],
                      "reference": reference, "hypothesis": row["text"]})

    volumes: dict[str, int] = dict(snap_vols)
    if not volumes and db is not None:
        # Fallback: mismo universo que el export, si no los pesos describen
        # una población de la que la muestra jamás pudo salir.
        acc: dict[str, int] = defaultdict(int)
        for row in db.execute(
            f"SELECT vad_prob FROM utterances WHERE {SAMPLEABLE_WHERE}"  # nosec B608 -- constante del módulo, sin datos externos
        ):
            acc[bucket_of(row["vad_prob"])] += 1
        volumes = dict(acc)
        print("AVISO: pesos calculados contra la DB de hoy, no contra el "
              "snapshot del export — la ventana pudo cambiar", file=sys.stderr)
    if db is not None:
        db.close()

    rep = build_report(pairs, volumes, esperados=len(refs))
    print(f"\npares evaluados: {len(pairs)} de {len(refs)} esperados "
          f"(excluidas: {rep['excluidas']})\n")
    print(f"  {'bucket':<14}{'n':>4}{'WER':>9}{'vol DB':>9}")
    for b in sorted(set(rep["buckets"]) | set(volumes)):
        st = rep["buckets"].get(b)
        n = f"{st['n']:>4}" if st else "   -"
        wer = f"{st['wer']:>9.3f}" if st else f"{'sin pares':>9}"
        print(f"  {b:<14}{n}{wer}{volumes.get(b, 0):>9}")
    print(
        "  (WER por bucket: promedio MACRO por utterance, no micro sobre "
        "palabras totales — no compara utterances de distinto largo con el "
        "mismo peso automáticamente)"
    )
    cob = rep["cobertura_volumen"]
    print(f"\n  cobertura: {'n/d' if cob is None else f'{cob:.1%}'} del "
          f"volumen real ({rep['volumen_evaluado']} de "
          f"{rep['volumen_total']}), piso {COBERTURA_MINIMA:.0%}")

    print()
    if rep["confiable"]:
        print("  " + "=" * 60)
        print(f"  WER DEL SISTEMA (re-ponderado por volumen real): "
              f"{_fmt(rep['wer_reponderado'])}")
        print("  " + "=" * 60)
    else:
        print("  " + "!" * 60)
        print("  WER DEL SISTEMA (re-ponderado): NO CONFIABLE — NO USAR")
        for motivo in rep["motivos"]:
            print(f"  - {motivo}")
        print("  " + "!" * 60)
    print(f"  (dato secundario, NO representa al sistema) WER plano del set: "
          f"{_fmt(rep['wer_simple'])}")
    print(f"\n  deleciones (habla → vacío)   : {rep['deleciones_totales']}")
    print(f"  alucinaciones (vacío → texto): {rep['alucinaciones']}")
    if perdidas:
        print(f"  referencias sin hipótesis   : {len(perdidas)} {perdidas[:10]}")

    out = args.out or f"data/wer_report_{gt_path.stem}.json"
    Path(out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreporte guardado en {out}")

    if not rep["confiable"]:
        # exit 0 acá sería el mismo proxy mentiroso que el export ya cerró:
        # un `--validate && ambient_wer && publicar` encadenado leería un
        # agregado inusable como éxito.
        raise SystemExit(2)


if __name__ == "__main__":
    main()
