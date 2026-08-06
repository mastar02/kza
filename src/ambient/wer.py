"""Core de WER para medir la fidelidad de la transcripción ambient.

Sin dependencias externas: la distancia de edición a nivel palabra son 25
líneas y el proyecto no incorpora jiwer para eso.

La normalización define el número que se reporta, así que está acotada y
testeada: minúsculas, colapso de espacios, puntuación de borde fuera.
**Los acentos y la ñ se conservan** — son señal real de calidad en español y
quitarlos inflaría artificialmente el resultado.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Marcadores que el humano puede escribir en la referencia. Se excluyen del
# WER y se reportan aparte: son el techo real del audio, no error del modelo.
UNINTELLIGIBLE = "[ininteligible]"
MEDIA_MARKERS = frozenset({"[tv]", "[media]"})

# Buckets de vad_prob — los mismos del análisis del 2026-08-04 sobre ambient.db.
VAD_BUCKETS: list[tuple[float, float]] = [
    (0.00, 0.20), (0.20, 0.35), (0.35, 0.50),
    (0.50, 0.65), (0.65, 0.80), (0.80, 1.01),
]

_PUNCT = "¡!¿?.,;:…\"'()[]{}—–-«»"
_WS_RE = re.compile(r"\s+")


def bucket_of(vad: float | None) -> str:
    """Etiqueta del bucket de vad_prob al que pertenece un valor."""
    if vad is None:
        return "sin_vad"
    for lo, hi in VAD_BUCKETS:
        if lo <= vad < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return "sin_vad"


def normalize_words(text: str) -> list[str]:
    """Tokenizar para comparar: minúsculas, sin puntuación de borde.

    Conserva acentos y ñ a propósito.

    Args:
        text: Texto crudo (referencia o hipótesis).

    Returns:
        Lista de palabras normalizadas; [] si no queda nada.
    """
    out = []
    for raw in _WS_RE.split(text.strip().lower()):
        w = raw.strip(_PUNCT)
        if w:
            out.append(w)
    return out


def is_excluded(reference: str) -> bool:
    """¿La referencia es un marcador que no se puntúa (ininteligible/media)?"""
    r = reference.strip().lower()
    return r == UNINTELLIGIBLE or r in MEDIA_MARKERS


@dataclass
class WerResult:
    """Descomposición del error de una comparación referencia/hipótesis."""

    subs: int
    ins: int
    dels: int
    ref_words: int
    wer: float


def score(reference: str, hypothesis: str) -> WerResult:
    """Comparar una hipótesis contra su referencia humana.

    Convención para el caso degenerado: si la referencia está vacía y la
    hipótesis no, el WER es 1.0 (alucinación total). Si ambas están vacías,
    0.0. Sin esto, dividir por cero al normalizar.

    Args:
        reference: Lo que se dijo realmente (transcripción humana).
        hypothesis: Lo que produjo el modelo.

    Returns:
        WerResult con sustituciones, inserciones, deleciones y WER.
    """
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    subs, ins, dels = _edit_ops(ref, hyp)
    if not ref:
        return WerResult(subs, ins, dels, 0, 0.0 if not hyp else 1.0)
    return WerResult(subs, ins, dels, len(ref), (subs + ins + dels) / len(ref))


def _edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein a nivel palabra con backtrace de operaciones.

    Returns:
        (sustituciones, inserciones, deleciones) del alineamiento óptimo.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = costo; la operación se reconstruye en el backtrace comparando costos
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(
                    d[i - 1][j - 1],   # sustitución
                    d[i][j - 1],       # inserción
                    d[i - 1][j],       # deleción
                )
    subs = ins = dels = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1
    return subs, ins, dels
