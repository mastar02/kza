"""Calidad/idioma del ambient path — marca utterances al persistir.

Parakeet-TDT auto-detecta idioma por-utterance (el hint ``language='es'`` es un
no-op en esa arquitectura, verificado en onnx_asr 0.11.0): el español far-field
o débil colapsa a "inglés" fonético-garble que ningún umbral de ``vad_prob``
separa, porque el discriminante es el IDIOMA DEL TEXTO, no la energía.

Este módulo provee la regla "español conservable" para marcar cada utterance
(flag, NO drop — la señal far-field se preserva para re-análisis/re-entrenamiento,
igual que WakeClipWriter). La política de FILTRADO (qué consumidor respeta el
flag) vive en el store/distiller, no acá.
"""
from __future__ import annotations

import re
from typing import Callable

# Marcadores léxicos de español rioplatense. Se excluyen palabras ambiguas con
# inglés ("no", "a", "y", "son") para no rescatar frases inglesas. Se exige ≥2
# distintas (un solo acierto no alcanza para marcar texto mayormente inglés).
SPANISH_STOPWORDS = frozenset({
    "que", "los", "las", "una", "por", "para", "con", "pero", "esto", "esta",
    "como", "porque", "cuando", "muy", "más", "vos", "che", "eso", "esa", "ese",
    "del", "tu", "su", "mi", "te", "se", "le", "lo", "yo", "ya", "hay", "fue",
    "todo", "toda", "nada", "algo", "bien", "acá", "allá", "aquí", "ahí",
    "entonces", "tenés", "tengo", "tiene", "hacer", "decir", "dale", "boludo",
    "sabés", "está", "están", "estoy", "sí", "quiero", "porque",
})

# Acentos, ñ/ü y signos de apertura — inequívocamente español.
_ACCENT_RE = re.compile(r"[áéíóúüñ¿¡]", re.IGNORECASE)
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def has_spanish_markers(text: str) -> bool:
    """True si el texto tiene señales léxicas inequívocas de español.

    Acento/ñ/¿¡ → True directo. Si no, ≥2 stopwords españolas distintas.
    Rescata el rioplatense que py3langid manda a pt/gl/ca.
    """
    if _ACCENT_RE.search(text):
        return True
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return len(words & SPANISH_STOPWORDS) >= 2


def is_spanish_keepable(
    text: str,
    lang: str | None,
    lang_prob: float | None,
    vad_prob: float | None,
    *,
    min_len: int = 8,
    min_vad: float = 0.40,
) -> bool:
    """¿Vale la pena conservar esta utterance como español útil?

    Descarta: texto corto (``min_len``), señal débil (``vad_prob < min_vad``),
    y texto sin marcadores españoles cuyo idioma detectado no es ``es`` (garble
    code-switch o inglés real de TV). Conserva el resto.

    Args:
        text: Texto transcripto.
        lang: Código ISO-639-1 de py3langid (o None si no se detectó).
        lang_prob: Confianza de la detección (no usado hoy; reservado).
        vad_prob: Mean de Silero del segmento (None = sin señal, no bloquea).
        min_len: Longitud mínima de texto (strip) para considerarlo.
        min_vad: Umbral inferior de vad_prob.
    """
    if len(text.strip()) < min_len:
        return False
    if vad_prob is not None and vad_prob < min_vad:
        return False
    if has_spanish_markers(text):
        return True
    return lang == "es"


def make_quality_fn(
    detect_fn: Callable[[str], tuple[str, float]],
    *,
    min_len: int = 8,
    min_vad: float = 0.40,
) -> Callable[[str, float | None], tuple[str | None, float | None, bool]]:
    """Componer detector de idioma + regla en una función para el transcriber.

    Args:
        detect_fn: ``text -> (lang, prob)`` (p.ej. make_langid_fn()).
        min_len, min_vad: parámetros de is_spanish_keepable.

    Returns:
        ``(text, vad_prob) -> (lang, lang_prob, lang_ok)``.
    """
    def quality(text: str, vad_prob: float | None) -> tuple[str | None, float | None, bool]:
        lang, prob = detect_fn(text)
        ok = is_spanish_keepable(text, lang, prob, vad_prob, min_len=min_len, min_vad=min_vad)
        return lang, prob, ok

    return quality
