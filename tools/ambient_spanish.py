#!/usr/bin/env python3
"""Tasa de español conservable en ambient.db — métrica de producto del 24/7.

El ambient no está para destilar charla: está para saber si Parakeet detecta
bien el rioplatense. ``lang_ok=1`` = utterance conservable (marcadores o lang=es).

Uso:
    PYTHONPATH=. python3 tools/ambient_spanish.py
    PYTHONPATH=. python3 tools/ambient_spanish.py --db data/ambient.db --hours 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.ambient.store import AmbientStore


async def _run(db: Path, hours: float | None) -> dict:
    store = AmbientStore(db_path=str(db))
    await store.init()
    since = None if hours is None else time.time() - hours * 3600
    report = await store.spanish_keep_rate(since_s=since)
    await store.close()
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("data/ambient.db"))
    p.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Ventana hacia atrás (default: toda la DB viva, TTL 48h en prod)",
    )
    args = p.parse_args()
    if not args.db.exists():
        print(f"no existe {args.db}", flush=True)
        return 1
    report = asyncio.run(_run(args.db, args.hours))
    rate = report["rate"]
    rate_s = "n/a" if rate is None else f"{rate:.1%}"
    print(
        f"español conservable: {report['ok']}/{report['ok'] + report['not_ok']} "
        f"clasificadas ({rate_s}); "
        f"n={report['n']} unclassified={report['unclassified']}"
    )
    for room, stats in sorted(report["by_room"].items()):
        r = stats["rate"]
        r_s = "n/a" if r is None else f"{r:.1%}"
        print(f"  {room}: {stats['ok']}/{stats['ok'] + stats['not_ok']} ({r_s})")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
