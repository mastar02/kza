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
# IntentRule — tabla declarativa de intents.
# ============================================================

@dataclass(frozen=True)
class IntentRule:
    """Regla declarativa de intent. Datos, no código."""
    intent: str
    verb_patterns: tuple[str, ...]   # regex (con \b alrededor al compilar)
    domains: frozenset[str]          # dominios HA donde aplica
    target: str = "domotics"         # "domotics" | "music"
    requires_slot: str | None = None # "any" | "volume" | None


# Orden = prioridad cuando varias reglas matchean el verbo. turn_on/off antes
# que open/close para que 'subí la luz' (light) caiga en turn_on, no open.
INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule("turn_on",  (r"prend\w*", r"encend\w*", r"ilumin\w*", r"activ\w*", r"enciend\w*", r"sub\w*"),
               frozenset({"light", "fan", "climate", "switch"})),
    IntentRule("turn_off", (r"apag\w*", r"cort\w*", r"desactiv\w*", r"apaguen"),
               frozenset({"light", "fan", "climate", "switch"})),
    IntentRule("set",      (),  frozenset({"light"}), requires_slot="any"),
    IntentRule("open",     (r"sub\w*", r"abr\w*", r"levant\w*"),
               frozenset({"cover"})),
    IntentRule("close",    (r"baj\w*", r"cerr\w*"),
               frozenset({"cover"})),
    IntentRule("media_play",  (r"pon\w*", r"reproduc\w*", r"dale", r"segu\w*"),
               frozenset({"media_player"}), target="music"),
    IntentRule("media_pause", (r"paus\w*", r"par\w*", r"fren\w*", r"callate", r"silenci\w*"),
               frozenset({"media_player"}), target="music"),
    IntentRule("media_next",  (r"siguiente", r"proxim\w*", r"cambi\w*", r"salt\w*"),
               frozenset({"media_player"}), target="music"),
    IntentRule("volume_set",  (r"volumen", r"fuerte", r"bajito"),
               frozenset({"media_player"}), target="music", requires_slot="volume"),
)


def _rule_verb_matches(rule: IntentRule, norm_text: str) -> bool:
    for pat in rule.verb_patterns:
        if re.search(rf"\b{pat}\b", norm_text):
            return True
    return False


def match_intent_rules(text: str, domain: str | None) -> IntentRule | None:
    """Devuelve la primera IntentRule cuyo verbo matchea Y es compatible con
    el dominio. Si domain es None, no se puede validar compat → None salvo que
    la regla aplique a cualquier dominio (no hay tales reglas hoy)."""
    if domain is None:
        return None
    t = _norm(text)
    for rule in INTENT_RULES:
        if domain not in rule.domains:
            continue
        if rule.intent == "set":
            # 'set' no tiene verbo; lo decide el motor por presencia de slots.
            continue
        if _rule_verb_matches(rule, t):
            return rule
    return None


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
        "volumen",
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


# ============================================================
# ParsedCommand — motor autoritativo con quality/target.
# ============================================================

@dataclass
class ParsedCommand:
    """Resultado del parser autoritativo. Determinístico, idempotente, sin I/O.

    A diferencia de PartialCommand (orientado a streaming), ParsedCommand produce
    intent + domain + target + quality cuando el utterance está completo.
    """
    intent: str | None = None
    domain: str | None = None
    room: str | None = None
    slots: dict = field(default_factory=dict)
    target: str = "domotics"
    confidence: float = 0.0
    quality: str = "none"        # "full" | "partial" | "none"
    raw_text: str = ""
    has_wake: bool = False

    def ready_to_dispatch(self) -> bool:
        """True si intent + domain están presentes (quality == 'full')."""
        return self.quality == "full"

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """True si la confidence del comando supera el umbral dado."""
        return self.confidence >= threshold


def _parsed_confidence(pc: "ParsedCommand") -> float:
    """Heurística de confidence para un ParsedCommand.

    Scores son señales orientativas, no probabilidades calibradas.
    """
    if pc.quality != "full":
        return 0.0
    score = 0.7
    if pc.has_wake:
        score += 0.15
    if pc.room is not None:
        score += 0.10
    if pc.slots:
        score += 0.05
    return min(score, 1.0)


def _has_any_onoff_verb(text: str) -> bool:
    """True si el texto contiene algún verbo de encendido/apagado."""
    t = _norm(text)
    for rule in INTENT_RULES:
        if rule.intent in ("turn_on", "turn_off") and _rule_verb_matches(rule, t):
            return True
    return False


def parse_command(text: str) -> ParsedCommand:
    """Parser autoritativo para domótica simple. Determinístico, idempotente,
    sin I/O. Produce intent + domain + target + quality.

    Casos cubiertos:
    - Comandos directos: "prendé la luz" → turn_on / light / domotics / full
    - Comandos de media: "pausá la música" → media_pause / media_player / music / full
    - Set implícito: "poné la luz al 70%" → set / light / domotics / full
    - Incompatibles: "abrí la luz" → None / light / domotics / partial
    - Sin domótica: "hola qué tal" → None / None / domotics / none
    """
    if not text:
        return ParsedCommand()
    pc = ParsedCommand(raw_text=text)
    pc.has_wake = has_wake_word(text)
    pc.domain = extract_entity(text)
    pc.room = extract_room(text)
    pc.slots = extract_slots(text)

    rule = match_intent_rules(text, pc.domain)
    if rule is not None:
        pc.intent = rule.intent
        pc.target = rule.target
    elif pc.domain == "light" and pc.slots and not _has_any_onoff_verb(text):
        # 'set' implícito: slots de luz (brillo/color/temp) sin verbo on/off.
        pc.intent = "set"
        pc.target = "domotics"

    if pc.intent is not None and pc.domain is not None:
        pc.quality = "full"
    elif pc.intent is not None or pc.domain is not None:
        pc.quality = "partial"
    else:
        pc.quality = "none"

    pc.confidence = _parsed_confidence(pc)
    return pc
