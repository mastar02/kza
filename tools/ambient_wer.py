#!/usr/bin/env python3
"""Mide el WER del ambient contra el ground truth humano.

Reporta SIEMPRE por bucket de vad_prob, y el agregado de dos formas: plano y
re-ponderado por el volumen real de cada bucket en la DB. El plano NO es el WER
del sistema — el set tiene asignación igual por bucket y sobre-representa los
buckets altos a propósito.

Uso:
    PYTHONPATH=. python3 tools/ambient_wer.py \\
        --db data/ambient.db --groundtruth /tmp/gt/groundtruth.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from src.ambient.wer import bucket_of, is_excluded, score


def build_report(pairs: list[dict], volumes: dict[str, int]) -> dict:
    """Construir el reporte de WER a partir de los pares referencia/hipótesis.

    Args:
        pairs: Dicts con ``id``, ``vad_prob``, ``reference``, ``hypothesis``.
        volumes: Conteo real de utterances por bucket en la DB, para el
            agregado re-ponderado.

    Returns:
        Dict con ``buckets``, ``wer_simple``, ``wer_reponderado``,
        ``deleciones_totales``, ``alucinaciones`` y ``excluidas``.
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
    total_vol = sum(volumes.get(b, 0) for b in buckets) or 1
    reponderado = sum(
        st["wer"] * volumes.get(b, 0) for b, st in buckets.items()
    ) / total_vol
    return {
        "buckets": buckets,
        "wer_simple": (sum(todas) / len(todas)) if todas else 0.0,
        "wer_reponderado": reponderado,
        "deleciones_totales": dels_totales,
        "alucinaciones": alucinaciones,
        "excluidas": excluidas,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/ambient.db")
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    refs = json.loads(Path(args.groundtruth).read_text(encoding="utf-8"))
    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    pairs = []
    for uid, reference in refs.items():
        row = db.execute(
            "SELECT text, vad_prob FROM utterances WHERE id=?", (int(uid),)
        ).fetchone()
        if row is None:
            print(f"  aviso: la utterance {uid} ya no está en la DB (purgada)")
            continue
        pairs.append({"id": int(uid), "vad_prob": row["vad_prob"],
                      "reference": reference, "hypothesis": row["text"]})

    volumes: dict[str, int] = defaultdict(int)
    for row in db.execute("SELECT vad_prob FROM utterances"):
        volumes[bucket_of(row["vad_prob"])] += 1

    rep = build_report(pairs, dict(volumes))
    print(f"\npares evaluados: {len(pairs)}  (excluidas: {rep['excluidas']})\n")
    print(f"  {'bucket':<14}{'n':>4}{'WER':>9}{'vol DB':>9}")
    for b, st in rep["buckets"].items():
        print(f"  {b:<14}{st['n']:>4}{st['wer']:>9.3f}{volumes.get(b, 0):>9}")
    print(f"\n  WER plano (del set)          : {rep['wer_simple']:.3f}")
    print(f"  WER re-ponderado (sistema)   : {rep['wer_reponderado']:.3f}")
    print(f"  deleciones (habla → vacío)   : {rep['deleciones_totales']}")
    print(f"  alucinaciones (vacío → texto): {rep['alucinaciones']}")

    out = args.out or f"data/wer_report_{Path(args.groundtruth).stem}.json"
    Path(out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreporte guardado en {out}")


if __name__ == "__main__":
    main()
