"""
Slot extractor y intent classifier para comandos de voz en español rioplatense.

Hybrid retrieval: el vector search recupera "qué comando" (capability + entity);
los slots del usuario (números, colores, adjetivos) sobrescriben los defaults
del metadata de Chroma con lo que realmente dijo.

Ver memoria: feedback_dense_retrieval_antonyms.md — motivación del intent léxico.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Nombres de slots (constantes para uso externo)
SLOT_BRIGHTNESS = "brightness_pct"
SLOT_COLOR = "rgb_color"
SLOT_COLOR_TEMP = "color_temp_kelvin"
SLOT_EFFECT = "effect"


# ============================================================
# Intent classifier (turn_on / turn_off)
# ============================================================
_RE_TURN_OFF = re.compile(
    r"\b(apag[aá]r?|apaguen|apago|cort[aá]r?|cortar|corte[nm]?|desactiv[aá]r?)\b",
    re.IGNORECASE,
)
_RE_TURN_ON = re.compile(
    r"\b(prend[eé]r?|prendan|prendo|encend[eé]r?|enciendan?|enciendo|ilumin[aá]r?|activ[aá]r?|pon[eéga][a-z]*|ponele|cambi[aá]r?|pint[aá]r?)\b",
    re.IGNORECASE,
)


def classify_intent(text: str) -> str | None:
    """turn_on | turn_off | None (si no hay verbo reconocible)."""
    if _RE_TURN_OFF.search(text):
        return "turn_off"
    if _RE_TURN_ON.search(text):
        return "turn_on"
    return None


# ============================================================
# Brightness
# ============================================================
_RE_PCT = re.compile(r"\b(\d{1,3})\s*(?:%|por\s*ciento|porciento)\b", re.IGNORECASE)
_RE_AL_NUM = re.compile(r"\bal\s+(\d{1,3})\b", re.IGNORECASE)
BRIGHTNESS_WORDS = {
    # Literalidad → valor canónico
    "máximo": 100, "maximo": 100, "máxima": 100, "maxima": 100, "full": 100, "todo": 100,
    "fuerte": 90, "potente": 90,
    "medio": 50, "media": 50, "mitad": 50,
    "suave": 30, "bajo": 30, "baja": 30,
    "tenue": 15, "mínimo": 10, "minimo": 10, "mínima": 10, "minima": 10,
}


def extract_brightness(text: str) -> int | None:
    """Extrae brightness_pct (0-100) de la query. None si no hay señal."""
    t = text.lower()
    # 1. Porcentaje explícito: "al 60%", "60 por ciento"
    for pat in (_RE_PCT, _RE_AL_NUM):
        m = pat.search(t)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v
    # 2. Palabras
    for word, val in BRIGHTNESS_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return val
    return None


# ============================================================
# Color (nombre → RGB)
# ============================================================
COLOR_MAP = {
    "rojo": [255, 0, 0], "roja": [255, 0, 0],
    "verde": [0, 255, 0],
    "azul": [0, 0, 255],
    "amarillo": [255, 255, 0], "amarilla": [255, 255, 0],
    "naranja": [255, 128, 0], "naranjo": [255, 128, 0],
    "rosa": [255, 105, 180], "rosado": [255, 105, 180], "rosada": [255, 105, 180],
    "violeta": [148, 0, 211], "morado": [148, 0, 211], "morada": [148, 0, 211],
    "púrpura": [148, 0, 211], "purpura": [148, 0, 211],
    "celeste": [135, 206, 235],
    "turquesa": [64, 224, 208], "cian": [0, 255, 255],
    "blanco": [255, 255, 255], "blanca": [255, 255, 255],
    "dorado": [255, 215, 0], "dorada": [255, 215, 0],
    "plateado": [192, 192, 192], "plateada": [192, 192, 192],
    "negro": [0, 0, 0], "negra": [0, 0, 0],
    "magenta": [255, 0, 255], "fucsia": [255, 0, 255],
}


def extract_color(text: str) -> list[int] | None:
    """Detecta nombre de color en la query y devuelve [R,G,B]."""
    t = text.lower()
    for name, rgb in COLOR_MAP.items():
        if re.search(rf"\b{re.escape(name)}\b", t):
            return rgb
    return None


# ============================================================
# Color temperature (cálida/neutra/fría + Kelvin)
# ============================================================
_RE_KELVIN = re.compile(r"\b(\d{4})\s*(?:K|kelvin)\b", re.IGNORECASE)
TEMP_WORDS_K = {
    "cálida": 2700, "calida": 2700, "cálido": 2700, "calido": 2700,
    "amarilla": 2700, "amarillenta": 2700,
    "neutra": 4000, "neutro": 4000, "natural": 4000,
    "fría": 6500, "fria": 6500, "frío": 6500, "frio": 6500,
    "azulada": 6500, "fluorescente": 5500,
}


def extract_color_temp(text: str) -> int | None:
    """Devuelve kelvin si la query menciona temperatura de color."""
    t = text.lower()
    m = _RE_KELVIN.search(t)
    if m:
        v = int(m.group(1))
        if 1500 <= v <= 9000:
            return v
    for word, k in TEMP_WORDS_K.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return k
    return None


# ============================================================
# Slot extraction (agregador)
# ============================================================
def extract_slots(text: str) -> dict:
    """
    Extrae todos los slots detectados en la query.
    Retorna dict con keys SLOT_* y valores (o ausente si no hay señal).
    """
    slots: dict = {}
    b = extract_brightness(text)
    if b is not None:
        slots[SLOT_BRIGHTNESS] = b
    c = extract_color(text)
    if c is not None:
        slots[SLOT_COLOR] = c
    k = extract_color_temp(text)
    if k is not None:
        slots[SLOT_COLOR_TEMP] = k
    return slots


def merge_service_data(metadata_service_data: dict, query_slots: dict) -> dict:
    """
    Combina el service_data default (del metadata de Chroma — valor canónico del preset)
    con los slots extraídos de la query del usuario. Los slots del usuario ganan.

    Ej: metadata dice {"brightness_pct": 50} (preset "al 50%"), pero el usuario dijo
    "al 73%" → slots = {"brightness_pct": 73} → resultado = {"brightness_pct": 73}.
    """
    merged = dict(metadata_service_data or {})
    # Conflictos mutuamente excluyentes para light:
    #   rgb_color vs color_temp_kelvin — el usuario sólo pide uno a la vez.
    # Si el usuario explicita color, quitar color_temp del default y viceversa.
    if SLOT_COLOR in query_slots:
        merged.pop(SLOT_COLOR_TEMP, None)
    if SLOT_COLOR_TEMP in query_slots:
        merged.pop(SLOT_COLOR, None)
    merged.update(query_slots)
    return merged
