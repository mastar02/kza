#!/usr/bin/env python3
"""Rehearse the authorized legacy selection in a new, private quarantine copy."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.legacy_quarantine import stage_legacy_quarantine  # noqa: E402


async def main(argv: list[str] | None = None) -> int:
    """Copy or verify only an explicitly selected private rehearsal directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview', help='Vista privada de H0 con los IDs exactos a conservar')
    mode.add_argument('--resume', action='store_true', help='Verificar y terminar la misma copia sin leer el origen')
    parser.add_argument('--output-dir', required=True, help='Directorio privado nuevo; no se sobrescribe')
    args = parser.parse_args(argv)
    try:
        result = await stage_legacy_quarantine(args.preview, args.output_dir, resume=args.resume)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError, TypeError):
        print('No se pudo preparar o verificar la copia privada; los originales no se migraron.', file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
