"""TTS rewriting: pesos / dolares / abreviaturas para que Kokoro suene natural.

Plan #3 OpenClaw — use case 4.
"""

import re
from dataclasses import replace

from src.hooks import before_tts_speak, RewriteResult


# Pattern: $123 → 123 pesos (Spanish-ar default)
_PESOS = re.compile(r"\$(\d+)")


@before_tts_speak(priority=10)
def numeros_a_palabras(call):
    new_text = _PESOS.sub(r"\1 pesos", call.text)
    if new_text != call.text:
        return RewriteResult(
            modified=replace(call, text=new_text),
            rule_name="numeros_a_palabras",
        )
    return None
