"""Regresión con utterances reales del journal de kza-voice (2026-05).

Origen: `journalctl --user -u kza-voice.service | grep "detectado en:"` y los
`Text=` del command_processor. Verifican que el motor determinístico:
  1. parsea los comandos legítimos (verbos rioplatenses + garbles leves que el
     STT produce de forma recurrente, ej "prendela", "prender") a full.
  2. NO fuerza match en garbles severos sin entidad recuperable ni en
     no-comandos (alucinaciones tipo "Gracias por ver el video") — esos caen
     a quality="none" y van al fallback, en vez de disparar acciones erróneas.

NOTA: los verbos ya están cubiertos por los patrones `prend\\w*` / `apag\\w*`
de INTENT_RULES; este archivo NO requirió extender el léxico — fija el
comportamiento observado para que no se regresione.
"""
import pytest

from src.nlu.command_grammar import parse_command


# --- Comandos legítimos (deben resolver a full con el intent/dominio correcto) ---
@pytest.mark.parametrize("text,intent,domain", [
    ("Nexa, prende la luz del escritorio.", "turn_on", "light"),
    ("Nexa, prender la luz.", "turn_on", "light"),
    ("nexa prendé la luz", "turn_on", "light"),
    ("nexa apagá la luz del cuarto", "turn_off", "light"),
])
def test_real_legit_commands_full(text, intent, domain):
    pc = parse_command(text)
    assert pc.quality == "full"
    assert pc.intent == intent
    assert pc.domain == domain


def test_real_command_with_room():
    pc = parse_command("Nexa, prende la luz del escritorio.")
    assert pc.room == "escritorio"


# --- Garbles severos / no-comandos: NO deben disparar acción (quality != full) ---
@pytest.mark.parametrize("text", [
    "Nexa, apagarendo directo.",        # STT garble sin entidad recuperable
    "Nexa, apagalelu, el victorio.",    # STT garble ("la luz"→"lelu", "escritorio"→"victorio")
    "Aprender el uso del escritorio.",  # no es comando (verbo no reconocido)
    "¡Gracias por ver el video!",       # alucinación de Whisper sobre silencio
    "Gracias.",                          # alucinación
])
def test_real_garbles_and_noise_not_dispatchable(text):
    pc = parse_command(text)
    assert pc.quality != "full", f"{text!r} no debería ser dispatchable (es garble/ruido)"
    assert pc.ready_to_dispatch() is False
