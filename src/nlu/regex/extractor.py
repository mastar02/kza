"""RegexExtractor — extracción determinística de comandos domóticos.

Contrato:
- Entrada: texto crudo post-Whisper (puede traer wake word, mayúsculas, etc.)
- Salida: list[RegexMatch] — vacía si no hay match limpio.
- Si el texto activa cualquier blocker, devuelve [].
- Si no se reconoce el verbo o la entidad como válidos, devuelve [].

El extractor NUNCA retorna matches con baja confianza. Si hay duda, devuelve
[] y el caller lo manda al LLM gate / reasoner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .blockers import detect as detect_blocker
from .normalize import normalize
from .vocab import (
    ALIAS_INDEX,
    ALL_VOSEO_VERBS,
    COLOR_RGB_WORDS,
    COLOR_TEMP_WORDS,
    EntityAlias,
    IMPERATIVES_VOSEO,
    POLISEMIC_VERBS,
    parse_cardinal,
)


# ─── Tipos ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegexMatch:
    """Resultado limpio de extracción para un único sub-comando."""
    intent: str                       # turn_on, turn_off, set_brightness, etc.
    entity_canonical: str | None      # "escritorio", "living"... o None si se resuelve por contexto
    entity_id: str | None             # entity_id de HA si está mapeado, sino None
    domain: str                       # light, climate, media_player, cover
    slots: dict = field(default_factory=dict)
    matched_verb: str = ""            # verbo voseo que disparó (telemetry)
    raw_segment: str = ""              # fragmento del texto que se atribuye a este match
    confidence: float = 1.0


# ─── Patrones compilados a nivel módulo ─────────────────────────────────────

# Verbos voseo válidos al INICIO del segmento, opcionalmente con clíticos
# (-lo, -la, -le, -los, -las, -lo, -nos, -me — los reales en uso).
# Capturamos: verb_base (ej "apagá") y clitic ("lo"|"" si no hay).
_VERB_RE = re.compile(
    r"^\s*(" + r"|".join(re.escape(v) for v in ALL_VOSEO_VERBS) + r")"
    r"(la|las|le|les|lo|los|nos|me)?\b\s*",
    re.IGNORECASE,
)

# Conector multi-intent: " y ", " también ", " y luego ", ", " (cuando separa
# verbos imperativos). NO splittea cuando "y" une atributos del mismo comando
# ("cálida y suave"): la heurística es que el siguiente token DEBE ser un
# verbo voseo. Si no lo es, el "y" se considera coordinación intra-comando.
_CONNECTORS_RE = re.compile(
    r"\s+(?:y\s+(?:luego\s+|despu[eé]s\s+)?|tambi[eé]n\s+|adem[aá]s\s+)",
    re.IGNORECASE,
)

# Slot brightness/volume/temperature absoluto:  "al N (por ciento)?"  /  "a N (por ciento)?"
# Capturamos N como dígitos o palabra cardinal.
_SLOT_PCT_RE = re.compile(
    r"\b(?:al|a)\s+([\w\s]+?)(?:\s+por\s+ciento|\s*%|\s*$|\s*[,.])",
    re.IGNORECASE,
)

# Slot temperatura: "a N grados", "en N grados"
_SLOT_TEMP_RE = re.compile(
    r"\b(?:a|en)\s+([\w\s]+?)\s+grados?\b",
    re.IGNORECASE,
)

# Color de luz: cálida/fría/etc. Buscamos UNO de los términos.
_COLOR_TEMP_RE = re.compile(
    r"\b(c[aá]lida|tibia|neutra|neutral|blanca|fr[ií]a)\b",
    re.IGNORECASE,
)

_COLOR_RGB_RE = re.compile(
    r"\b(roj[ao]|azul|verde|amarill[ao]|naranja|rosa|violeta|morad[ao])\b",
    re.IGNORECASE,
)

# Entidad: armamos OR con todos los aliases conocidos, ordenados por longitud
# descendente (importante: "luz del living" debe matchear antes que "living").
def _build_entity_pattern() -> re.Pattern:
    aliases_sorted = sorted(ALIAS_INDEX.keys(), key=len, reverse=True)
    return re.compile(
        r"\b(?:la|el|los|las)?\s*"
        r"(?:luz\s+(?:del|de\s+la|de\s+los|de\s+las)\s+)?"
        r"(" + r"|".join(re.escape(a) for a in aliases_sorted) + r")\b",
        re.IGNORECASE,
    )


_ENTITY_RE = _build_entity_pattern()


# ─── Resolución verbo → intent ──────────────────────────────────────────────

def _resolve_intent(verb: str, entity: EntityAlias | None,
                    has_pct_slot: bool, has_temp_slot: bool) -> str | None:
    """Mapea verbo + contexto a uno de los intents canónicos.

    Para verbos polisémicos (bajá/subí/poné), la entidad o el slot determina
    qué intent específico aplicar. Si no hay info suficiente, devuelve None
    y el caller cae al LLM.
    """
    verb = verb.lower()

    # Verbos no polisémicos — mapeo directo.
    for intent, verbs in IMPERATIVES_VOSEO.items():
        if verb in verbs and verb not in POLISEMIC_VERBS:
            # Eliminar sufijos de granularidad para volver al intent canónico.
            return intent.replace("_down", "").replace("_up", "").replace("_abs", "") \
                if intent.startswith(("set_brightness", "set_volume", "set_temperature")) \
                else intent

    # Verbos polisémicos — necesitamos entidad o slot.
    if verb not in POLISEMIC_VERBS:
        return None

    # Caso 1: bajá/subí + dominio media_player → set_volume
    if entity and entity.domain == "media_player":
        return "set_volume"

    # Caso 2: bajá/subí + dominio climate → set_temperature
    if entity and entity.domain == "climate":
        return "set_temperature"

    # Caso 3: bajá/subí + dominio light → set_brightness
    if entity and entity.domain == "light":
        return "set_brightness"

    # Caso 4: poné/ponele con slot temperatura → set_temperature
    if verb in {"poné", "ponele"} and has_temp_slot:
        return "set_temperature"

    # Caso 5: poné/ponele con slot pct → set_brightness por defecto
    # (la mayoría de los logs muestran "poné la luz al N por ciento")
    if verb in {"poné", "ponele"} and has_pct_slot:
        return "set_brightness"

    # Caso 6: bajá/subí con slot pct sin entidad → asumir set_brightness.
    # Razón: en uso real "bajá la luz al 50" es lo más común; sin pista
    # de entidad el slot pct sugiere brillo, no volumen ni temperatura.
    if verb in {"bajá", "subí", "aumentá", "reducí", "levantá"} and has_pct_slot:
        return "set_brightness"

    # No se pudo desambiguar → caer al LLM.
    return None


def _verb_direction(verb: str) -> str | None:
    """Devuelve 'up'/'down' si el verbo lo implica, sino None."""
    v = verb.lower()
    if v in {"bajá", "reducí"}:
        return "down"
    if v in {"subí", "aumentá", "levantá"}:
        return "up"
    return None


# ─── Extractor principal ────────────────────────────────────────────────────

class RegexExtractor:
    """Extractor determinístico de comandos domóticos en voseo argentino.

    Uso:
        extractor = RegexExtractor()
        matches = extractor.extract("Nexa apagá la luz del escritorio")
        # matches == [RegexMatch(intent='turn_off', entity_canonical='escritorio', ...)]
    """

    def extract(self, text: str) -> list[RegexMatch]:
        """Extrae N matches del texto. Devuelve [] si nada matchea limpio."""
        norm = normalize(text)
        if not norm:
            return []

        # Blockers globales (negación, pasado, pregunta...) sobre el texto entero.
        if detect_blocker(norm):
            return []

        # Split por conectores multi-intent. Si tras el connector NO hay verbo
        # voseo, ese connector se considera coordinación intra-comando ("cálida
        # y suave") y NO se splittea.
        segments = self._split_segments(norm)

        results: list[RegexMatch] = []
        for seg in segments:
            m = self._extract_single(seg)
            if m is None:
                # Si UN segmento no matchea limpio, abortamos todo y caemos al
                # LLM. Política conservadora: mejor enviar al gate que ejecutar
                # parcialmente.
                return []
            results.append(m)
        return results

    # ── helpers ──

    def _split_segments(self, norm: str) -> list[str]:
        """Divide por conectores SOLO si después del connector hay verbo voseo."""
        # Encontramos todas las posiciones de conector y validamos lookahead.
        cuts: list[int] = []
        for m in _CONNECTORS_RE.finditer(norm):
            after = norm[m.end():].lstrip()
            # ¿Hay verbo voseo justo después?
            if _VERB_RE.match(after):
                cuts.append(m.start())
        if not cuts:
            return [norm]
        # Construir segmentos.
        segments: list[str] = []
        prev = 0
        for cut in cuts:
            segments.append(norm[prev:cut].strip(" ,.;"))
            # Saltar el connector buscando el inicio del siguiente segmento.
            connector_match = _CONNECTORS_RE.match(norm, cut)
            prev = connector_match.end() if connector_match else cut
        segments.append(norm[prev:].strip(" ,.;"))
        return [s for s in segments if s]

    def _extract_single(self, segment: str) -> RegexMatch | None:
        """Extrae un único intent + entity + slots de un segmento."""
        # 1) Verbo voseo al inicio.
        verb_match = _VERB_RE.match(segment)
        if not verb_match:
            return None
        verb_base = verb_match.group(1).lower()
        clitic = (verb_match.group(2) or "").lower()
        rest = segment[verb_match.end():]

        # 2) Slots numéricos (pct / temp).
        pct_value = self._extract_pct_slot(rest)
        temp_value = self._extract_temp_slot(rest)

        # 3) Entidad — buscar la primera alias conocida.
        entity = self._extract_entity(rest, segment)

        # 4) Resolver intent (puede requerir entidad O slot).
        intent = _resolve_intent(
            verb_base, entity,
            has_pct_slot=pct_value is not None,
            has_temp_slot=temp_value is not None,
        )
        if intent is None:
            return None

        # 5) Si el intent requiere entidad y no tenemos, depende del verbo:
        #    - turn_off/turn_on: entidad opcional (el dispatcher resuelve por
        #      contexto de room).
        #    - bajá/subí sin entidad ni slot: ambiguo → caer al LLM.
        #    - open/close: SIEMPRE requieren entidad de domain=cover (persiana,
        #      cortina). Si no hay, "cerrá la boca" matchea por accidente.
        if entity is None and intent in {"set_brightness", "set_volume", "set_temperature"}:
            if pct_value is None and temp_value is None:
                return None  # ambigüedad real
        if intent in {"open", "close"} and (entity is None or entity.domain != "cover"):
            return None  # protege contra "cerrá la boca", "abrí los ojos", etc.

        # 6) Construir slots dict.
        slots: dict = {}
        direction = _verb_direction(verb_base)
        if direction:
            slots["direction"] = direction
        if pct_value is not None and intent in {"set_brightness", "set_volume"}:
            slots["brightness_pct" if intent == "set_brightness" else "value"] = pct_value
        if temp_value is not None and intent == "set_temperature":
            slots["value"] = temp_value
        # Color slots (sólo si intent es turn_on / set_color / set_color_temp)
        ct_match = _COLOR_TEMP_RE.search(rest)
        if ct_match:
            color_word = ct_match.group(1).lower()
            kelvin = COLOR_TEMP_WORDS.get(color_word)
            if kelvin:
                slots["color_temp_kelvin"] = kelvin
        rgb_match = _COLOR_RGB_RE.search(rest)
        if rgb_match:
            color_word = rgb_match.group(1).lower()
            rgb = COLOR_RGB_WORDS.get(color_word)
            if rgb:
                slots["rgb_color"] = list(rgb)

        return RegexMatch(
            intent=intent,
            entity_canonical=entity.canonical if entity else None,
            entity_id=entity.ha_entity_id if entity else None,
            domain=entity.domain if entity else "light",  # default light
            slots=slots,
            matched_verb=verb_base + clitic,
            raw_segment=segment,
            confidence=1.0,
        )

    def _extract_pct_slot(self, rest: str) -> int | None:
        m = _SLOT_PCT_RE.search(rest)
        if not m:
            return None
        return parse_cardinal(m.group(1).strip())

    def _extract_temp_slot(self, rest: str) -> int | None:
        m = _SLOT_TEMP_RE.search(rest)
        if not m:
            return None
        return parse_cardinal(m.group(1).strip())

    def _extract_entity(self, rest: str, full_segment: str) -> EntityAlias | None:
        """Busca la primera entidad whitelisted en el segmento."""
        for m in _ENTITY_RE.finditer(rest):
            alias = m.group(1).lower()
            if alias in ALIAS_INDEX:
                return ALIAS_INDEX[alias]
        return None
