#!/usr/bin/env python3
"""Búsqueda semántica en el code-index del server (para agentes).

Uso:
    python tools/code_search.py "cómo se maneja el timeout de HA al boot"
    python tools/code_search.py "reconexión websocket" --top-k 5

Standalone a propósito (solo stdlib): corre sin venv ni PYTHONPATH.
Los resultados ⚠ STALE difieren del índice (rama local sin deployar):
leé el archivo real en vez de confiar en el snippet.
Si el servicio no responde: fallback a Grep/Glob de siempre (exit 1).
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://192.168.1.2:9510"
SNIPPET_PRINT_CHARS = 1200


def git_blob_hash(data: bytes) -> str:
    """SHA1 estilo `git hash-object` (duplicado de src a propósito: standalone)."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def format_result(res: dict, repo_root: Path, seen_cards: set) -> str:
    """Formatear un resultado con marca de drift y card (una vez por path)."""
    local = repo_root / res["path"]
    if not local.exists():
        drift = "  ⚠ LOCAL MISSING"
    elif git_blob_hash(local.read_bytes()) != res["blob_hash"]:
        drift = "  ⚠ STALE (difiere del índice — leer el archivo real)"
    else:
        drift = ""

    lines = [
        f"== {res['path']}:{res['start_line']}-{res['end_line']}  "
        f"{res['name']} [{res['kind']}] score={res['score']:.3f}{drift}"
    ]
    if res.get("card") and res["path"] not in seen_cards:
        seen_cards.add(res["path"])
        lines += ["--- card ---", res["card"], "------------"]
    lines.append(res["snippet"][:SNIPPET_PRINT_CHARS])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    payload = json.dumps({"query": args.query, "top_k": args.top_k}).encode()
    req = urllib.request.Request(
        f"{args.url}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        print(
            f"code-index no disponible ({e}). Fallback: usar Grep/Glob.",
            file=sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    seen_cards: set = set()
    for res in data.get("results", []):
        print(format_result(res, repo_root, seen_cards))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
