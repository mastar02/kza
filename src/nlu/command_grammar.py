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
    extract_slots,
    has_invalid_brightness_percentage,
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


# Orden = prioridad cuando varias reglas matchean el verbo. turn_on/off antes
# que open/close para que 'subí la luz' (light) caiga en turn_on, no open.
INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule("turn_on",  (r"prend\w*", r"encend\w*", r"ilumin\w*", r"activ\w*", r"enciend\w*", r"sub\w*"),
               frozenset({"light", "fan", "climate", "switch"})),
    IntentRule("turn_off", (r"apag\w*", r"cort\w*", r"desactiv\w*", r"apaguen"),
               frozenset({"light", "fan", "climate", "switch"})),
    IntentRule("set",      (),  frozenset({"light"})),
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
               frozenset({"media_player"}), target="music"),
)


_ACTION_PARTICIPLE = re.compile(
    r"(?:(?:apag|cort|desactiv|activ|ilumin|baj|levant|cerr|paus|par|fren|silenci|cambi|salt)ad[oa]s?"
    r"|(?:prend|encend|sub)id[oa]s?|abiert[oa]s?|puest[oa]s?)"
)


def _rule_verb_matches(rule: IntentRule, norm_text: str) -> bool:
    for pat in rule.verb_patterns:
        for match in re.finditer(rf"\b{pat}\b", norm_text):
            # A described state cannot supply an action verb. In particular,
            # "apagá la luz que está prendida" must not become turn_on merely
            # because turn_on rules have priority over turn_off.
            if not _ACTION_PARTICIPLE.fullmatch(match.group()):
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
        "volumen", "cancion", "canción", "tema", "temas", "playlist", "lista",
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
    # Destrozos far-field de "living" (2026-07-30). Es la única habitación con
    # nombre en inglés y la única que el STT rompe: medido sobre 4 días de log,
    # "del living" 12 veces contra "del libby" 9 — ~43% de los pedidos al living
    # se transcribían mal, perdían el locativo y terminaban en la habitación del
    # micrófono. "cocina" y "escritorio" nunca aparecieron destrozados.
    # "libby" es el observado; los otros dos son truncamientos a 1 edición.
    "libby": "living",
    "livin": "living",
    "libin": "living",
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


def llm_intent_evidenced(intent: str | None, text: str) -> bool:
    """Valida que un intent binario afirmado por el LLM tenga verbo en el texto.

    Guard determinístico (2026-06-06): el STT garblea el verbo far-field
    ('apagá'→'pero a', 'prendé'→'haba') y el 7B ADIVINA turn_on/turn_off con
    confianza alta auto-reportada sobre texto sin verbo — en vivo invirtió una
    acción (pidió apagar, prendió). La auto-confianza del modelo no es señal:
    le puso 0.95 a 'haba la luz'. Regla: si el LLM afirma turn_on o turn_off,
    el texto tiene que contener un verbo de ESA acción (stems de INTENT_RULES).
    El bloqueo de pasado, negaciones y pedidos no inmediatos se aplica a todos
    los intents. Para los demás no se exige evidencia léxica adicional.

    Args:
        intent: Intent afirmado por el LLMCommandRouter (puede ser None).
        text: Texto transcripto del comando.

    Returns:
        True si el texto autoriza un pedido inmediato y el intent está
        evidenciado o no requiere evidencia léxica adicional.
    """
    if non_immediate_action_reason(text):
        return False
    if intent not in ("turn_on", "turn_off"):
        return True
    t = _norm(text)
    for rule in INTENT_RULES:
        if rule.intent == intent and _rule_verb_matches(rule, t):
            return True
    return False


def _has_any_action_verb(text: str) -> bool:
    """True si el texto contiene cualquier verbo de acción (on/off/set/media/etc).

    Usado como guardia para evitar que slots sueltos en frases conversacionales
    (ej: 'gracias por todo' → brightness_pct=100 vía 'todo') disparen
    clasificaciones espurias.  Solo verbos con verb_patterns no vacíos.
    """
    t = _norm(text)
    for rule in INTENT_RULES:
        if rule.verb_patterns and _rule_verb_matches(rule, t):
            return True
    return False


# Unlike the extractor's past blocker, this preserves formal imperatives such
# as "apague" and the voseo imperatives "subí"/"abrí". Accents are retained by
# regex.normalize, so "apagué" can be distinguished from "apague".
_PAST_ACTION = re.compile(
    r"\b(?:\w+(?:aste|iste|aron|ieron|asteis|isteis)|"
    r"apagué|prendí|encendí|bajé|cerré|cambié|pausé|frené|puse|puso|pusimos|"
    r"apag[oó]|prendi[oó]|encendi[oó]|subi[oó]|bajó|cerró|abri[oó]|"
    r"activó|desactivó|reprodujo|"
    r"apagaba[ns]?|prend[ií]a[ns]?|encend[ií]a[ns]?|sub[ií]a[ns]?|"
    r"bajaba[ns]?|cerraba[ns]?|abr[ií]a[ns]?|pon[ií]a[ns]?)\b",
    re.IGNORECASE,
)
_PERFECT_ACTION = re.compile(
    r"\b(?:he|has|ha|hemos|han|hab[ií]a[ns]?|hab[ií]amos|hubo|habr[aá]s?)\s+"
    r"(?:\w+\s+){0,2}(?:apagado|prendido|encendido|subido|bajado|"
    r"cerrado|abierto|puesto|activado|desactivado|cambiado)\b",
    re.IGNORECASE,
)
_PAST_CONTEXT = re.compile(
    r"\b(?:ayer|anteayer|anoche|reci[eé]n|antes)\b"
    r"|\b(?:ya|yo)\s+(?:sub[ií]|abr[ií])\b",
    re.IGNORECASE,
)
_FUTURE_ACTION = re.compile(
    r"\b(?:apagar|prender|encender|subir|bajar|cerrar|abrir|activar|desactivar|poner)"
    r"(?:[eé]|[aá]s?|emos|[aá]n)\b"
    r"|\b(?:voy|vas|va|vamos|van)\s+a\s+"
    r"(?:apagar|prender|encender|subir|bajar|cerrar|abrir|activar|desactivar|poner)\b",
    re.IGNORECASE,
)
_INFORMATIONAL_ACTION = re.compile(
    r"^(?:[¿?¡!]\s*)?(?:(?:decime|dime|contame|cu[eé]ntame|explicame|expl[ií]came)\s+)?"
    r"(?:qu[eé]|c[oó]mo|cu[aá]ndo|d[oó]nde|qui[eé]n|por\s+qu[eé]|para\s+qu[eé])\b",
    re.IGNORECASE,
)
_STATE_DESCRIPTION = re.compile(
    r"^(?:[¿?¡!]\s*)?(?:(?:la\s+luz|las\s+luces|el\s+aire|las?\s+persianas?)"
    r"(?:\s+\w+){0,4}\s+)?"
    r"(?:est[aá]s?|est[aá]n|sigue[n]?|qued[oó]|quedaron|tengo|ten[eé]s)\b",
    re.IGNORECASE,
)


def non_immediate_action_reason(text: str) -> str | None:
    """Identify speech that must not authorize an immediate device action.

    Reuse negation, conditions and reported speech from the Spanish extractor.
    Check tense and question purpose separately so affirmative requests and
    formal/voseo imperatives keep working without treating states as commands.
    """
    from src.nlu.regex.blockers import BLOCKERS
    from src.nlu.regex.normalize import normalize

    light_request = extract_entity(text) == "light" or re.search(r"\bbrillo\b", _norm(text))
    if light_request and has_invalid_brightness_percentage(text):
        return "invalid_brightness_percentage"
    if not _has_any_action_verb(text) and extract_entity(text) is None and not light_request:
        return None
    normalized = normalize(text)
    blocked = {"negation", "subjunctive_cond", "report_of_speech"}
    for name, pattern in BLOCKERS:
        if name in blocked and pattern.search(normalized):
            return f"non_immediate:{name}"
    if _PAST_ACTION.search(normalized) or _PERFECT_ACTION.search(normalized) or _PAST_CONTEXT.search(normalized):
        return "non_immediate:past_action"
    if _FUTURE_ACTION.search(normalized):
        return "non_immediate:future_action"
    if _INFORMATIONAL_ACTION.search(normalized) or _STATE_DESCRIPTION.search(normalized):
        return "non_immediate:information_request"
    if re.search(
        r"\b(?:ma[ñn]ana|m[aá]s\s+tarde|despu[eé]s|luego|"
        r"en\s+(?:[\w]+\s+){1,5}(?:segundos?|minutos?|horas?)|a\s+las?\s+(?:\d+|una?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce))\b",
        normalized,
    ) and (extract_entity(text) is not None or light_request):
        return "non_immediate:deferred"
    return None

def _is_default_light_brightness_phrase(text: str) -> bool:
    """Recognize one complete room-light percentage request without guessing."""
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    # Keep signs and percent symbols so negative values cannot become positive.
    norm = re.sub(r"[^\w\s%+-]", " ", norm)
    words = norm.split()
    while words and words[0] in WAKE_TERMS:
        words.pop(0)
    norm = " ".join(words)
    rooms = "|".join(re.escape(alias) for alias in ROOM_ALIASES)
    pattern = (
        r"(?:(?:pone|pon|poner|deja|dejar|ajusta|ajustar|subi|sube|subir|baja|bajar)\s+)?"
        r"(?:(?:la|las|el)\s+)?(?:luz|luces|brillo(?:\s+de\s+(?:la\s+luz|las\s+luces))?)"
        rf"(?:\s+(?:del|de la|de el|en el|en la|de|en)\s+(?:{rooms}))?"
        r"\s+(?:al|a|el|del)\s+(?:\d{1,3}|[a-z]+(?:\s+y\s+[a-z]+)?)"
        r"(?:\s*(?:%|por\s+ciento))?(?:\s+por favor)?"
    )
    return re.fullmatch(pattern, norm) is not None


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
    if not text or non_immediate_action_reason(text):
        return ParsedCommand()
    pc = ParsedCommand(raw_text=text)
    pc.has_wake = has_wake_word(text)
    pc.domain = extract_entity(text)
    pc.room = extract_room(text)
    pc.slots = extract_slots(text)

    if pc.slots.keys() == {"brightness_pct"} and _is_default_light_brightness_phrase(text):
        pc.domain = "light"
        pc.intent = "set"
        pc.quality = "full"
        pc.confidence = _parsed_confidence(pc)
        return pc

    # Inferencia segura: brillo/color/temperatura son slots exclusivos de luz.
    # Si no hubo dominio explícito pero hay un slot de luz (y NO de volumen),
    # asumimos light → habilita 'ponela cálida', 'subí el brillo al 50', etc.
    # Guardia: requerimos al menos un verbo de acción para evitar frases
    # conversacionales con 'todo'='100%' disparando clasificaciones espurias
    # (ej: 'gracias por todo' → brightness_pct=100 por 'todo' en BRIGHTNESS_WORDS).
    _LIGHT_ONLY_SLOTS = {"brightness_pct", "rgb_color", "color_temp_kelvin"}
    # Guardia de longitud (2026-06-05): la inferencia slot→light es para
    # comandos implícitos CORTOS ('ponela cálida', 'subí el brillo al 50').
    # La charla larga colisiona: 'Tengo los coches de negro, van a bajar a…'
    # ('negro'→rgb + 'bajar'→verbo) parseó full y EJECUTÓ light.turn_off
    # (acción fantasma real 2026-06-04 19:49 con TV de fondo).
    _MAX_IMPLICIT_WORDS = 6
    if pc.domain is None and "volume_pct" not in pc.slots \
            and any(k in pc.slots for k in _LIGHT_ONLY_SLOTS) \
            and _has_any_action_verb(text) \
            and len(text.split()) <= _MAX_IMPLICIT_WORDS:
        pc.domain = "light"

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


def parse_default_light_target(text: str) -> ParsedCommand | None:
    """Match an entire on/off or brightness phrase for a room's default light.

    Specific fixtures, unknown rooms, global and multiple targets do not match.
    """
    pc = parse_command(text)
    if (
        pc.domain == "light" and pc.intent == "set"
        and pc.slots.keys() == {"brightness_pct"}
        and _is_default_light_brightness_phrase(text)
    ):
        return pc
    if pc.domain != "light" or pc.intent not in ("turn_on", "turn_off") or pc.slots:
        return None
    norm = _norm(text)
    words = norm.split()
    while words and words[0] in WAKE_TERMS:
        words.pop(0)
    norm = " ".join(words)
    verbs = "|".join(pat for rule in INTENT_RULES if rule.intent == pc.intent for pat in rule.verb_patterns)
    rooms = "|".join(re.escape(alias) for alias in ROOM_ALIASES)
    pattern = (
        rf"(?:{verbs})\s+(?:(?:la|las)\s+)?(?:luz|luces)"
        rf"(?:\s+(?:del|de la|de el|en el|en la|de|en)\s+(?:{rooms}))?"
        r"(?:\s+por favor)?"
    )
    return pc if re.fullmatch(pattern, norm) else None


def has_unresolved_light_location(text: str) -> bool:
    """Reject explicit light locations that cannot be resolved exactly."""
    if parse_default_light_target(text) is not None:
        return False
    norm = _norm(text)
    match = re.search(r"\b(?:luz|luces)\b.*?\b(?:del|de la|de el|en el|en la|en|de)\s+(.+)", norm)
    if match is None:
        return bool(re.search(r"\b(?:ambiente|habitacion)\b", norm) and extract_room(text) is None)
    location = re.sub(r"(?:\s+(?:por favor|ahora))+$", "", match.group(1)).strip()
    if location in {"hogar", "toda la casa", "todo el hogar", "la casa entera"}:
        return False
    rooms = "|".join(re.escape(alias) for alias in ROOM_ALIASES)
    return re.fullmatch(rf"(?:{rooms})(?:\s+(?:al|a|con)\s+.+)?", location) is None


# ============================================================
# PartialCommand — compat shim para early_dispatch y tests.
# ============================================================

class PartialCommand(ParsedCommand):
    """Shim de compatibilidad sobre ParsedCommand.

    Expone ``.entity`` como alias bidireccional de ``.domain`` para que el
    early_dispatch (multi_room_audio_loop, request_router) y los tests
    existentes sigan funcionando sin cambios.  Toda la lógica real vive en
    ParsedCommand / parse_command.

    Se puede construir con keyword ``entity=`` como alias de ``domain=``:
        PartialCommand(intent="turn_on", entity="light")
    """

    def __init__(
        self,
        intent: str | None = None,
        domain: str | None = None,
        room: str | None = None,
        slots: dict | None = None,
        target: str = "domotics",
        confidence: float = 0.0,
        quality: str = "none",
        raw_text: str = "",
        has_wake: bool = False,
        *,
        entity: str | None = None,
    ) -> None:
        # ``entity`` es el alias legacy de ``domain``.
        resolved_domain = domain if domain is not None else entity
        # Recalcular quality si no fue pasada explícitamente (construcción
        # directa desde test: PartialCommand(intent="turn_on", entity="light")).
        if quality == "none" and intent is not None and resolved_domain is not None:
            quality = "full"
        elif quality == "none" and (intent is not None or resolved_domain is not None):
            quality = "partial"
        super().__init__(
            intent=intent,
            domain=resolved_domain,
            room=room,
            slots=slots if slots is not None else {},
            target=target,
            confidence=confidence,
            quality=quality,
            raw_text=raw_text,
            has_wake=has_wake,
        )

    @property
    def entity(self) -> str | None:
        """Alias de domain para compat con consumidores del early_dispatch."""
        return self.domain

    @entity.setter
    def entity(self, value: str | None) -> None:
        self.domain = value

    def is_empty(self) -> bool:
        """True si no hay ninguna señal extraída."""
        return not (self.intent or self.domain or self.room or self.slots)


def _compute_confidence(pc: "PartialCommand") -> float:
    """Compat wrapper de _parsed_confidence para consumidores legacy.

    El cálculo es idéntico al de ParsedCommand — usa ``quality`` en lugar de
    comprobar ``entity`` directamente, lo que mantiene la misma semántica:
    ambos tienen que estar presentes para que la confidence sea > 0.
    """
    return _parsed_confidence(pc)


def parse_partial_command(text: str) -> PartialCommand:
    """Compat wrapper sobre parse_command para el early_dispatch streaming.

    Devuelve un PartialCommand (subclase de ParsedCommand) con .entity como
    alias de .domain, más los métodos is_empty() / is_high_confidence().
    """
    pc = parse_command(text)
    return PartialCommand(
        intent=pc.intent,
        domain=pc.domain,
        room=pc.room,
        slots=pc.slots,
        target=pc.target,
        confidence=pc.confidence,
        quality=pc.quality,
        raw_text=pc.raw_text,
        has_wake=pc.has_wake,
    )
