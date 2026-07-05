#!/usr/bin/env python3
"""Cosecha candidatas a la blocklist de alucinaciones desde el journal.

Detecta textos que el CommandGate ACEPTÓ pero el LLMRouter DESCARTÓ
(is_command=False) — la firma de una alucinación de Whisper que pasó el gate.
SOLO PROPONE: un humano decide qué entra a _NOISE_PHRASES (criterio del fix
2026-06-02: jamás substrings plausibles de comandos válidos).

Uso (en el server):
    journalctl --user -u kza-voice --since '7 days ago' --output=cat \\
        | python3 tools/harvest_hallucinations.py --min-count 3
    python3 tools/harvest_hallucinations.py --file journal_export.txt

Standalone stdlib a propósito; el import de src.nlu.command_gate es opcional
(excluye ya-bloqueadas si está disponible).
"""

import argparse
import ast
import re
import sys
import unicodedata
from dataclasses import dataclass

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ROUTER_REJECT_RE = re.compile(
    r"\[LLMRouter [^\]]*\] is_command=False .*?"
    r"text=(?P<raw>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
)
_TS_RE = re.compile(r"^(?P<ts>\d{2}:\d{2}:\d{2}\.\d+)")


def strip_ansi(line: str) -> str:
    """Sacar códigos de color ANSI del journal."""
    return _ANSI_RE.sub("", line)


def normalize(text: str) -> str:
    """Lowercase sin acentos ni puntuación (duplicado de command_gate a
    propósito: el script corre standalone en el server)."""
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


@dataclass
class Candidate:
    """Texto candidato a blocklist con conteo y rango temporal."""

    text: str        # normalizado (clave de agrupación)
    count: int
    first_seen: str  # timestamp de la primera línea (HH:MM:SS.mmm)
    last_seen: str


def parse_candidates(lines) -> dict:
    """Agrupar por texto normalizado los rejects del router.

    Un reject del router implica que el gate aceptó antes (precondición del
    pipeline), así que la línea is_command=False alcanza como firma.
    """
    candidates: dict[str, Candidate] = {}
    for raw in lines:
        line = strip_ansi(raw)
        m = _ROUTER_REJECT_RE.search(line)
        if not m:
            continue
        raw_repr = m.group("raw")
        try:
            text = ast.literal_eval(raw_repr)
        except (ValueError, SyntaxError):
            # repr malformado (línea truncada del journal) — mejor saltear
            # que contar basura.
            continue
        norm = normalize(text)
        if not norm:
            continue
        ts_match = _TS_RE.match(line.strip())
        ts = ts_match.group("ts") if ts_match else ""
        if norm in candidates:
            c = candidates[norm]
            c.count += 1
            c.last_seen = ts or c.last_seen
        else:
            candidates[norm] = Candidate(norm, 1, ts, ts)
    return candidates


def filter_candidates(candidates, min_count, blocked_phrases, filler_words):
    """Filtrar por umbral y exclusiones; ordenar por count desc."""
    out = []
    for c in candidates.values():
        if c.count < min_count:
            continue
        if any(p in c.text for p in blocked_phrases):
            continue
        if c.text in filler_words:
            continue
        out.append(c)
    return sorted(out, key=lambda c: -c.count)


def _load_existing_blocklist():
    """Import opcional de la lista vigente (para excluir ya-bloqueadas)."""
    try:
        from src.nlu.command_gate import _FILLER_WORDS, _NOISE_PHRASES

        return tuple(_NOISE_PHRASES), frozenset(_FILLER_WORDS)
    except Exception:
        print(
            "(aviso: src.nlu.command_gate no importable — no se excluyen "
            "frases ya bloqueadas)",
            file=sys.stderr,
        )
        return (), frozenset()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--file", help="export del journal (default: stdin)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            lines = f
    else:
        sys.stdin.reconfigure(errors="replace")
        lines = sys.stdin

    blocked, fillers = _load_existing_blocklist()
    candidates = filter_candidates(
        parse_candidates(lines), args.min_count, blocked, fillers
    )

    if not candidates:
        print("Sin candidatas nuevas (todo bloqueado o bajo el umbral).")
        return 0

    print(f"{'count':>5}  {'primera':<12} {'última':<12} texto")
    for c in candidates:
        print(f"{c.count:>5}  {c.first_seen:<12} {c.last_seen:<12} {c.text!r}")
    print(
        "\nRevisar a mano antes de agregar a _NOISE_PHRASES "
        "(command_gate.py): nada que sea substring de un comando válido."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
