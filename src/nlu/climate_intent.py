"""Binary intent classifier for the climate/AC ambiguity.

⚠️ NO-GO (2026-08-04) — CÓDIGO MUERTO, NO ADOPTAR SIN RE-EVALUAR. ⚠️

Este módulo NO está cableado a nada: fuera de su test y su benchmark, nadie lo
importa. El eval contra el set B held-out dio 94% de acierto pero produjo 2
casos consulta→acción (el asistente OPERA el aire cuando el usuario solo estaba
comentando el tiempo), y ese error es asimétrico: el criterio de go/no-go
—fijado antes de ver el resultado— lo declaraba inaceptable. La clase que
rompe es la negación usada como cláusula de necesidad ("no hace falta prender"),
y le rompe IGUAL al modelo y a la gramática de reglas: cualquier enfoque nuevo
tiene que atacar eso, no "clima" en general.

Se conserva como instrumento reusable (junto al set B y su runner), no como
diseño vivo. Antes de tocarlo leer el resultado completo en
`docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md`.

Lo que sigue describe el diseño tal como se evaluó:

In Rioplatense Spanish "clima" means both the weather and the air conditioner:

    "prendé el clima"      -> turn the AC on   (command)
    "está el clima lindo"  -> what's it like   (question)

Three rounds of substring rules failed to separate these (see
docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md): every rule
was a proxy for "is this an order or an observation", and that proxy leaks in
spoken Spanish without reliable punctuation.

This module asks the local 7B instead, and only when the text actually contains
contested vocabulary. It can only ever answer ACTION or QUERY; anything else --
timeout, endpoint down, unparseable output -- returns None, which means "let the
rules decide". The caller keeps its rule-based answer as the default.
"""

import asyncio
import hashlib
import logging
import re
import unicodedata
from enum import Enum

logger = logging.getLogger(__name__)


class ClimateIntent(Enum):
    """What the speaker wants to happen."""

    ACTION = "ACCION"   # operate the AC / thermostat / heating
    QUERY = "CONSULTA"  # asking or remarking about the weather outside


# Vocabulary that is genuinely contested between the two readings. Mirrors
# _CLIMATE_DOMAIN_NOUNS in src/orchestrator/dispatcher.py -- keep in sync. No
# new vocabulary is introduced here on purpose: the gate is a presence test,
# not an intent guess.
_CONTESTED_NOUNS: tuple[str, ...] = (
    "clima", "temperatura", "termostato", "calefaccion", "aire", "grados",
)
_CONTESTED_RE = re.compile(
    r"\b(?:" + "|".join(_CONTESTED_NOUNS) + r")\b"
)

# Labels must NOT be named after the ambiguous word. Measured 2026-08-04: with
# "clima" as an option label the 7B answered "clima" for "prendé el clima" --
# it inherits the exact ambiguity we are trying to resolve. Unambiguous label
# names took the same model from failing both hard cases to 21/22.
_LABEL_ACTION = "ACCION_AIRE"
_LABEL_QUERY = "PREGUNTA_TIEMPO"

# The four few-shot examples are part of the contract, not decoration: they
# carry measured accuracy. Editing this string trips test_prompt_fingerprint_is_pinned,
# which is deliberate -- re-run benchmarks/router/climate_eval.py before changing it.
CLIMATE_PROMPT = f"""Sos el router de un asistente de hogar. Decidí qué hace el usuario.

{_LABEL_ACTION}     = ordena encender, apagar o ajustar el aire / termostato / calefacción.
{_LABEL_QUERY} = pregunta o comenta cómo está el tiempo afuera.

En rioplatense "clima" significa las dos cosas: el aparato y el tiempo.
Decidí por lo que el usuario QUIERE que pase, no por la palabra.

Texto: apagá el aire
Etiqueta: {_LABEL_ACTION}
Texto: ¿va a llover mañana?
Etiqueta: {_LABEL_QUERY}
Texto: está lindo el día, no prendas nada
Etiqueta: {_LABEL_QUERY}
Texto: poné el aire en 22 que hace calor
Etiqueta: {_LABEL_ACTION}
Texto: {{text}}
Etiqueta:"""

PROMPT_FINGERPRINT = "4bace36a9c08a904"

_STOP = ["\n", "Texto:", "Etiqueta:"]


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def has_contested_vocabulary(text: str) -> bool:
    """True if the text mentions a noun that means both the device and the weather.

    Deliberately trivial: this asks whether contested WORDS are present, never
    what the speaker intends. That is what makes it safe -- when it is wrong it
    costs a model call (latency), it cannot misroute anything.

    Args:
        text: User text, post-STT. Case and accents are normalised here.

    Returns:
        True if any contested noun appears as a whole word.
    """
    if not text:
        return False
    return _CONTESTED_RE.search(_strip_accents(text.lower())) is not None


class ClimateIntentClassifier:
    """Resolves climate/AC ambiguity via the local 7B on :8101.

    Args:
        router: Object exposing `async complete(prompt, max_tokens, temperature,
            stop) -> str`. Typically FastRouter. None disables the classifier.
        timeout_s: Hard budget for the call. Measured p95 is 118ms; the default
            sits just above it so the tail is cut without touching the typical
            case. Moving this means re-checking the eval's p95 threshold.
    """

    def __init__(self, router, timeout_s: float = 0.15):
        self.router = router
        self.timeout_s = timeout_s

    async def classify(self, text: str) -> ClimateIntent | None:
        """Classify text as an AC command or a weather question.

        Returns:
            ClimateIntent.ACTION, ClimateIntent.QUERY, or None. None means the
            classifier abstained -- no router, timeout, transport error, or
            output that is not exactly one of the two labels. The caller must
            fall back to its rule-based decision on None.
        """
        if self.router is None:
            return None

        try:
            raw = await asyncio.wait_for(
                self.router.complete(
                    CLIMATE_PROMPT.format(text=text),
                    max_tokens=10,
                    temperature=0.0,
                    stop=_STOP,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "climate_intent: timeout after %.0fms, falling back to rules",
                self.timeout_s * 1000,
            )
            return None
        except Exception as exc:
            logger.warning("climate_intent: classifier unavailable (%s), falling back to rules", exc)
            return None

        return self._parse(raw)

    @staticmethod
    def _parse(raw) -> ClimateIntent | None:
        """Strict label parsing. Anything unexpected abstains.

        No fuzzy matching on purpose: a substring match here would reintroduce
        the failure mode this module exists to remove.

        `raw` is typed loosely because a misconfigured caller can hand us
        something that is not a string: LLMRouter.complete() returns a
        RouterResult, not str. That mistake would otherwise surface as a
        silent AttributeError swallowed by classify()'s except block, i.e. an
        abstention on every single call with nothing but a debug log to show
        for it. It gets its own WARNING instead.
        """
        if not isinstance(raw, str):
            if raw is not None:
                logger.warning(
                    "climate_intent: router returned %s, expected str -- wire a "
                    "FastRouter, not an LLMRouter (which returns RouterResult). "
                    "Abstaining on every call until fixed.",
                    type(raw).__name__,
                )
            return None
        if not raw:
            return None
        head = raw.strip().upper()
        if head.startswith(_LABEL_ACTION):
            return ClimateIntent.ACTION
        if head.startswith(_LABEL_QUERY):
            return ClimateIntent.QUERY
        logger.debug("climate_intent: unrecognised label %r, abstaining", raw[:40])
        return None
