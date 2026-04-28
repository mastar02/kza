"""Normalización de texto antes de aplicar regex.

Decisiones explícitas:
- LOWERCASE: sí, simplifica patterns.
- ACENTOS: SE PRESERVAN. El voseo argentino distingue "apagá" (imperativo)
  de "apaga" (3ra persona indicativa) por el acento. Quitarlos rompería
  la asimetría que justamente queremos explotar.
- WHITESPACE: colapsado a espacios simples, trim al inicio y al final.
- WAKE WORD: stripeada al inicio (con coma o espacio que la siga).
- PUNTUACIÓN: signos de pregunta/admiración inicial/final se preservan
  (los blockers de pregunta los necesitan).
"""
from __future__ import annotations

import re

from .vocab import WAKE_WORDS

# Wake word al inicio: "nexa" / "nexa," / "nexa." / "nexa: " etc.
# La construimos dinámicamente desde la lista de wake words.
_WAKE_PREFIX = re.compile(
    r"^\s*(?:" + r"|".join(re.escape(w) for w in WAKE_WORDS) + r")\b[\s,.\-:]*",
    re.IGNORECASE,
)

# Whitespace múltiple a colapsar.
_WS = re.compile(r"\s+")

# Trailing puntuación que no aporta semántica (puntos, comas finales repetidas)
# pero MANTIENE "?" y "¿" para que los blockers de pregunta operen.
_TRAILING_FILLER = re.compile(r"[\s,.;]+$")


def normalize(text: str) -> str:
    """Devuelve texto listo para blockers + extracción.

    Pasos:
    1. Lowercase.
    2. Strip wake word inicial (puede estar repetida — strip iterativo).
    3. Colapsar whitespace.
    4. Eliminar puntuación trailing irrelevante.
    """
    if not text:
        return ""
    out = text.lower().strip()
    # Strip wake word — puede aparecer 1-3 veces ("nexa, nexa, nexa apagá...")
    for _ in range(3):
        new = _WAKE_PREFIX.sub("", out, count=1)
        if new == out:
            break
        out = new.strip()
    out = _WS.sub(" ", out)
    out = _TRAILING_FILLER.sub("", out)
    return out
