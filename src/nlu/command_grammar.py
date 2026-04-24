"""
Command grammar parser — streaming-friendly.

Reconoce comandos de domótica estructurados: <wake> <acción> <entidad> [<lugar>]
[<modificador>]. Diseñado para consumir transcripción PARCIAL y emitir un
PartialCommand en cualquier momento, sin esperar al final de la utterance.

Uso típico:
    pc = parse_partial_command("nexa apagá la luz del escrit")
    if pc.ready_to_dispatch():
        # ejecutar sin esperar más audio
        ...

Separa NLU determinístico (esto) del vector search (Chroma): acá sacamos lo que
es extraíble por regex/léxico; el vector search resuelve la entidad concreta
(`light.escritorio` vs `light.escritorio_led`) si hay ambigüedad.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from src.nlu.slot_extractor import (
    classify_intent,
    extract_slots,
)

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ============================================================
# Entity lexicon — mapa de palabra → dominio HA.
# ============================================================
ENTITY_TERMS: dict[str, list[str]] = {
    "light": [
        "luz", "luces", "lampara", "lamparas", "foco", "focos", "bombilla",
        "bombillas", "led", "leds", "lampara de pie", "lampara de mesa",
        "velador", "veladores",
    ],
    "climate": [
        "aire", "aire acondicionado", "ac", "climatizador", "calefaccion",
        "calefactor", "estufa", "termotanque", "termostato",
    ],
    "cover": [
        "persiana", "persianas", "cortina", "cortinas", "blackout",
        "toldo", "toldos",
    ],
    "fan": [
        "ventilador", "ventiladores", "extractor", "extractores",
    ],
    "media_player": [
        "tele", "tv", "television", "televisor", "musica", "parlante",
        "parlantes", "bocina", "bocinas", "altavoz", "altavoces", "radio",
    ],
}


def extract_entity(text: str) -> str | None:
    """Devuelve el dominio HA (light/climate/cover/fan/media_player) o None."""
    t = _norm(text)
    # Preferir matches multi-palabra primero.
    candidates: list[tuple[int, str]] = []
    for domain, terms in ENTITY_TERMS.items():
        for term in terms:
            pat = rf"\b{re.escape(term)}\b"
            if re.search(pat, t):
                candidates.append((len(term), domain))
    if not candidates:
        return None
    # El término más largo gana (ej: "aire acondicionado" > "aire").
    candidates.sort(reverse=True)
    return candidates[0][1]


# ============================================================
# Room lexicon — mapa de alias → room_id canónico.
# ============================================================
# IMPORTANTE: mantenerlo alineado con config/settings.yaml rooms.* aliases.
ROOM_ALIASES: dict[str, str] = {
    "escritorio": "escritorio",
    "oficina": "escritorio",
    "estudio": "escritorio",
    "living": "living",
    "sala": "living",
    "salon": "living",
    "hall": "hall",
    "pasillo": "hall",
    "entrada": "hall",
    "cocina": "cocina",
    "kitchen": "cocina",
    "bano": "bano",
    "banio": "bano",
    "bathroom": "bano",
    "cuarto": "cuarto",
    "dormitorio": "cuarto",
    "habitacion": "cuarto",
}

# Preposiciones que introducen el lugar — "del escritorio", "de la cocina",
# "en el living", "en la cocina". La lista ayuda a desambiguar (ej "la luz
# del cuarto" vs "un cuarto de luz").
_RE_ROOM_PREP = re.compile(
    r"\b(?:del|de\s+la|de\s+el|en\s+el|en\s+la|al|a\s+la)\s+(\w+)",
    re.IGNORECASE,
)


def extract_room(text: str) -> str | None:
    """Devuelve el room_id canónico si la query menciona un lugar; None si no."""
    t = _norm(text)
    # 1. Buscar con preposición (preferido, más confiable).
    for m in _RE_ROOM_PREP.finditer(t):
        word = m.group(1)
        if word in ROOM_ALIASES:
            return ROOM_ALIASES[word]
    # 2. Fallback: room name suelto (menos confiable, pero útil para comandos cortos).
    for alias, canonical in ROOM_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return canonical
    return None


# ============================================================
# Wake word detection (streaming-friendly).
# ============================================================
WAKE_TERMS = {"nexa", "kaza", "alexa", "nexia", "neza", "nexta", "nesa",
              "next", "necks"}


def has_wake_word(text: str) -> bool:
    """True si el texto contiene alguna forma reconocible de la wake word."""
    t = _norm(text)
    words = t.split()[:5]  # solo mirar primeras 5 palabras
    return any(w in WAKE_TERMS for w in words)


# ============================================================
# PartialCommand
# ============================================================
@dataclass
class PartialCommand:
    """Resultado del parser. Se va actualizando a medida que llega más texto."""
    intent: str | None = None
    entity: str | None = None
    room: str | None = None
    slots: dict = field(default_factory=dict)
    raw_text: str = ""
    has_wake: bool = False
    confidence: float = 0.0

    def ready_to_dispatch(self) -> bool:
        """
        ¿Hay suficiente señal para ejecutar sin esperar más audio?

        Mínimo: intent (verbo) + entity (dominio). El room/slots son opcionales
        — si no vienen usamos el room actual (default) y el service_data default
        del vector search.
        """
        return self.intent is not None and self.entity is not None

    def is_empty(self) -> bool:
        return not (self.intent or self.entity or self.room or self.slots)

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """True si la confidence del comando supera el umbral dado."""
        return self.confidence >= threshold


def _compute_confidence(pc: PartialCommand) -> float:
    """
    Heurística de confidence para un PartialCommand.

    Regla:
      - Si falta intent o entity → 0.0 (el parser no puede dispatchar).
      - Base = 0.7 con intent + entity presentes.
      - +0.15 si hay wake word detectada.
      - +0.10 si hay room detectado.
      - +0.05 si hay slots extraídos.
      - Clamp a 1.0.

    Los scores NO son probabilidades calibradas — son señales para decidir
    cuándo pedir confirmación a acciones sensibles. Ver plan S4.
    """
    if pc.intent is None or pc.entity is None:
        return 0.0
    score = 0.7
    if pc.has_wake:
        score += 0.15
    if pc.room is not None:
        score += 0.10
    if pc.slots:
        score += 0.05
    return min(score, 1.0)


def parse_partial_command(text: str) -> PartialCommand:
    """
    Parsea una transcripción (potencialmente parcial) y devuelve lo que se
    puede extraer hasta ahora. Idempotente — llamadas sucesivas con texto más
    largo devuelven mismo-o-más.
    """
    if not text:
        return PartialCommand()
    pc = PartialCommand(raw_text=text)
    pc.has_wake = has_wake_word(text)
    pc.intent = classify_intent(text)
    pc.entity = extract_entity(text)
    pc.room = extract_room(text)
    pc.slots = extract_slots(text)
    pc.confidence = _compute_confidence(pc)
    return pc
