"""Detectores de contexto que descartan un texto ANTES de probar match.

Si cualquier blocker hace match, el RegexExtractor devuelve None y el caller
delega al LLM gate / reasoner. Esto es la primera línea de defensa contra
falsos positivos.

Cada blocker:
1. Es un patrón compilado (constante de módulo, no se recompila por call).
2. Tiene un nombre estable que se loguea para telemetría/debug.
3. Es ortogonal a los otros (no overlapping, fácil de razonar).

Anclado al estilo: TODOS los blockers operan sobre texto YA NORMALIZADO
(lowercase, sin wake-word, sin signos al inicio/fin, whitespace colapsado).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockerHit:
    """Resultado de un match de blocker.

    Atributos:
        name: identificador del blocker (telemetry).
        snippet: fragmento del texto que disparó el match (debug).
    """
    name: str
    snippet: str


# ─── Negación ────────────────────────────────────────────────────────────────
# "no apagués", "no prendas", "nunca prendas", "jamás cortés"
# El verbo viene después de la negación, en cualquier persona/tiempo.
_NEGATION = re.compile(
    r"\b(?:no|nunca|jam[áa]s|tampoco)\s+\w+",
    re.IGNORECASE,
)


# ─── Pasado ──────────────────────────────────────────────────────────────────
# Detecta pasados perfectivos comunes: -é, -í, -aste, -iste, -ó, -ió, -ado, -ido
# y marcadores temporales que indican acción ya ocurrida.
#
# Importante: el lookbehind \b(ya|recién|antes|hace|...) cubre el caso
# adverbial ("ya apagué"). El segundo patrón cubre cuando el verbo solo está
# conjugado en pasado sin adverbio ("apagaste la luz?" → 2da persona pasado).

_PAST_ADVERB = re.compile(
    r"\b(?:ya|reci[eé]n|antes|anteayer|ayer|hace\s+(?:un|una|unos|unas)?\s*\w+)\b",
    re.IGNORECASE,
)

# Pretérito 2da persona: -aste, -iste, -aron, -ieron
# Pretérito 1ra persona: -é (terminaciones específicas para evitar
# falsos positivos con voseo).
#
# IMPORTANTE: hay verbos cuya 1ra persona pretérito es IDÉNTICA al voseo
# imperativo: "subí", "abrí". NO los incluimos como blocker pretérito —
# el imperativo gana por defecto. Si era pasado real, el adverbio temporal
# ("ya", "recién") lo bloquea via _PAST_ADVERB.
_PAST_VERB = re.compile(
    r"\b\w+(?:aste|iste|aron|ieron|asteis|isteis)\b"
    r"|\b(?:apagu[eé]|prend[ií]|encend[ií]|baj[eé]|pus[eo]|"
    r"cerr[eé]|cambi[eé]|pause[eé]|frene[eé]|"
    r"reproduj[eo]|toqu[eé])\b",
    re.IGNORECASE,
)


# ─── Subjuntivo / condicional ────────────────────────────────────────────────
# "ojalá apagaras", "si prendés", "cuando bajés", "si querés"
_SUBJUNCTIVE_CONDITIONAL = re.compile(
    r"\b(?:ojal[áa]|si|cuando|aunque|por\s+si|en\s+caso\s+de\s+que)\b\s+\w+",
    re.IGNORECASE,
)


# ─── Pregunta directa ────────────────────────────────────────────────────────
# Inicia con interrogativo, contiene "?" o "¿"
_QUESTION = re.compile(
    r"^\s*(?:[¿?])"
    r"|[?¿]\s*$"
    r"|^\s*(?:qu[eé]|c[oó]mo|cu[aá]ndo|d[oó]nde|cu[aá]l|qui[eé]n|por\s+qu[eé]|para\s+qu[eé])\b",
    re.IGNORECASE,
)


# ─── Reporte de habla ────────────────────────────────────────────────────────
# "le dije que apagara", "Lucas comentó", "preguntó si"
# Verbos de comunicación + complemento que introduce subordinada.
_REPORT_OF_SPEECH = re.compile(
    r"\b(?:dij[eo]s?|dije|dijiste|dijeron|"
    r"coment[óo]|coment[eé]|comentaste|comentaron|"
    r"pregunt[óo]|pregunt[eé]|preguntaste|preguntaron|"
    r"avis[óo]|avis[eé]|"
    r"explic[óo]|explic[eé])\b",
    re.IGNORECASE,
)


# ─── Infinitivo (forma no-imperativa que Whisper a veces emite) ─────────────
# El usuario en imperativo voseo dice "apagá" / "prendé".
# Whisper a veces produce el infinitivo: "apagar" / "prender" / "encender".
# Si el verbo principal está en infinitivo, NO es un comando.
#
# Heurística: "(verbo en infinitivo) (sustantivo)" sin imperativo previo →
# bloquear. La detección es estricta: solo verbos de nuestro dominio.
_INFINITIVE_DOMAIN_VERB = re.compile(
    r"^\s*(?:apagar|prender|encender|cortar|desactivar|activar|"
    r"bajar|subir|aumentar|reducir|levantar|poner|"
    r"abrir|cerrar|cambiar|pausar|reproducir|tocar)\b",
    re.IGNORECASE,
)


# ─── Tercera persona indicativa con sujeto explícito ────────────────────────
# "ella apaga", "Lucas prendió", "la casa apaga"
# Detecta artículo/pronombre/nombre propio + verbo en 3ra persona.
_THIRD_PERSON = re.compile(
    r"\b(?:[ée]l|ella|ellos|ellas|"
    r"[Ll]ucas|[Mm]ar[íi]a|[Jj]uan|"
    r"la\s+casa|el\s+chico|la\s+chica)\s+"
    r"(?:apaga|prende|enciende|baja|sube|pone|cierra|abre|"
    r"apag[óo]|prendi[óo]|baj[óo]|subi[óo])\b",
    re.IGNORECASE,
)


# ─── Wake word repetido sin contenido ───────────────────────────────────────
# "nexa", "nexa nexa nexa", "nexa." — sin verbo después.
# Esto se chequea al final, post-strip de wake. Si tras quitar la wake word
# queda <2 palabras o solo puntuación → incomplete.
_INCOMPLETE = re.compile(
    r"^\s*(?:[.,!?¿¡]*)?\s*(?:\w{1,3})?\s*[.,!?¿¡]*\s*$",
)


# ─── Trailing question mark / muletilla final ───────────────────────────────
# El usuario duda al final ("...al cincuenta, ¿no?"). Política conservadora:
# si hay duda explícita, mejor lo valida el LLM gate.
_TRAILING_DOUBT = re.compile(
    r"[,\s]+(?:[¿?]+\s*)?(?:no|verdad|ok|okay|s[íi])\s*[?¿]+\s*$"
    r"|,\s*¿no\??\s*$",
    re.IGNORECASE,
)


# Lista ordenada de blockers. El orden importa: detectores más específicos
# primero para diagnósticos más útiles en logs.
BLOCKERS: tuple[tuple[str, re.Pattern], ...] = (
    ("question",          _QUESTION),
    ("trailing_doubt",    _TRAILING_DOUBT),
    ("negation",          _NEGATION),
    ("subjunctive_cond",  _SUBJUNCTIVE_CONDITIONAL),
    ("report_of_speech",  _REPORT_OF_SPEECH),
    ("third_person",      _THIRD_PERSON),
    ("past_adverb",       _PAST_ADVERB),
    ("past_verb",         _PAST_VERB),
    ("infinitive",        _INFINITIVE_DOMAIN_VERB),
    ("incomplete",        _INCOMPLETE),
)


def detect(normalized_text: str) -> BlockerHit | None:
    """Devuelve el primer blocker que matchea, o None.

    Args:
        normalized_text: texto YA normalizado (lowercase, wake stripped, etc.)

    Returns:
        BlockerHit con name + snippet si algún blocker matchea.
        None si el texto pasa todos los filtros.
    """
    for name, pattern in BLOCKERS:
        m = pattern.search(normalized_text)
        if m:
            return BlockerHit(name=name, snippet=m.group(0)[:80])
    return None
