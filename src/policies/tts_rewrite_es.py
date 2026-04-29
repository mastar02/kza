"""TTS rewriting: $N → N pesos para que Kokoro suene natural en español-AR.

Plan #3 OpenClaw — use case 4. Más patrones (dólares, abreviaturas,
unidades) se pueden agregar en handlers separados con la misma firma.
"""

import re
from dataclasses import replace

from src.hooks import before_tts_speak, RewriteResult


# Pattern: $123 → 123 pesos. AR-flavored default. Si querés US-style ($1.50),
# duplicar este handler con otro regex/rule_name; el chain encadena los
# rewrites así que no hay conflicto.
_PESOS = re.compile(r"\$(\d+)")


@before_tts_speak(priority=10)
def numeros_a_palabras(call):
    """Rewrite "$N" sequences to "N pesos" for natural Spanish-AR TTS.

    Returns:
        RewriteResult with modified TtsCall if any "$\\d+" matched;
        None otherwise (pass-through, no Kokoro overhead).
    """
    new_text = _PESOS.sub(r"\1 pesos", call.text)
    if new_text != call.text:
        return RewriteResult(
            modified=replace(call, text=new_text),
            rule_name="numeros_a_palabras",
        )
    return None
