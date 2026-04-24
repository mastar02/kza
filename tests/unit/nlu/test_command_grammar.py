"""
Tests del parser streaming de comandos de domótica.

Simulamos cómo va llegando la transcripción (parcial → completa) y verificamos
que el parser emite PartialCommand correcto y marca ready_to_dispatch() en
el momento correcto.
"""
from __future__ import annotations

import pytest

from src.nlu.command_grammar import (
    PartialCommand,
    extract_entity,
    extract_room,
    has_wake_word,
    parse_partial_command,
)


# -----------------------------------------------------------------
# Entity extraction
# -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("apagá la luz", "light"),
    ("prendé las luces", "light"),
    ("prendé el foco del living", "light"),
    ("poné la lámpara al 50%", "light"),
    ("apagá el aire", "climate"),
    ("bajá la temperatura del aire acondicionado", "climate"),
    ("subí las persianas", "cover"),
    ("cerrá la cortina", "cover"),
    ("prendé el ventilador", "fan"),
    ("pausá la música", "media_player"),
    ("apagá la tele", "media_player"),
    ("hola como estás", None),
    ("", None),
])
def test_extract_entity(text, expected):
    assert extract_entity(text) == expected


def test_extract_entity_prefers_multi_word():
    """'aire acondicionado' debe ganar sobre 'aire' solo."""
    assert extract_entity("prendé el aire acondicionado del cuarto") == "climate"


# -----------------------------------------------------------------
# Room extraction
# -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("apagá la luz del escritorio", "escritorio"),
    ("prendé la luz de la cocina", "cocina"),
    ("apagá las luces del living", "living"),
    ("apagá las luces de la sala", "living"),
    ("subí las persianas en el cuarto", "cuarto"),
    ("prendé el aire en la oficina", "escritorio"),
    ("apagá la luz del baño", "bano"),
    ("apagá la luz", None),  # sin room
    ("", None),
])
def test_extract_room(text, expected):
    assert extract_room(text) == expected


def test_extract_room_fallback_no_preposition():
    """Si no hay preposición, agarra la mención suelta."""
    assert extract_room("cocina luces") == "cocina"


# -----------------------------------------------------------------
# Wake detection helper
# -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("nexa apagá la luz", True),
    ("alexa prendé la luz", True),
    ("next up apagá la luz", True),
    ("hola como estás", False),
    ("hola amigos como están todos nexa", False),  # nexa en pos 6 > 5
    ("", False),
])
def test_has_wake_word(text, expected):
    assert has_wake_word(text) == expected


# -----------------------------------------------------------------
# parse_partial_command — streaming simulation
# -----------------------------------------------------------------

def test_partial_empty():
    pc = parse_partial_command("")
    assert pc.is_empty()
    assert not pc.ready_to_dispatch()


def test_partial_only_wake():
    pc = parse_partial_command("nexa")
    assert pc.has_wake is True
    assert pc.intent is None
    assert pc.entity is None
    assert not pc.ready_to_dispatch()


def test_partial_wake_plus_verb():
    pc = parse_partial_command("nexa apagá")
    assert pc.has_wake is True
    assert pc.intent == "turn_off"
    assert pc.entity is None
    # Intent solo no alcanza — necesitamos entity.
    assert not pc.ready_to_dispatch()


def test_partial_wake_verb_entity_ready():
    """Este es el caso crítico: apenas hay verbo+entidad, podemos dispatchar."""
    pc = parse_partial_command("nexa apagá la luz")
    assert pc.has_wake is True
    assert pc.intent == "turn_off"
    assert pc.entity == "light"
    assert pc.room is None
    assert pc.ready_to_dispatch()  # ← dispatch ya, usar default_room


def test_partial_full_command():
    pc = parse_partial_command("nexa apagá la luz del escritorio")
    assert pc.has_wake is True
    assert pc.intent == "turn_off"
    assert pc.entity == "light"
    assert pc.room == "escritorio"
    assert pc.ready_to_dispatch()


def test_partial_with_slot():
    pc = parse_partial_command("nexa poné la luz del living al 50 por ciento")
    assert pc.intent == "turn_on"
    assert pc.entity == "light"
    assert pc.room == "living"
    assert pc.slots.get("brightness_pct") == 50
    assert pc.ready_to_dispatch()


def test_partial_with_color():
    pc = parse_partial_command("nexa poné la luz del cuarto en azul")
    assert pc.intent == "turn_on"
    assert pc.entity == "light"
    assert pc.room == "cuarto"
    assert pc.slots.get("rgb_color") == [0, 0, 255]


def test_streaming_progressive_growth():
    """Simula cómo va llegando la transcripción cada 150ms."""
    timeline = [
        ("nexa", False),
        ("nexa apa", False),
        ("nexa apagá", False),
        ("nexa apagá la", False),
        ("nexa apagá la luz", True),  # ← ready aquí
        ("nexa apagá la luz del", True),
        ("nexa apagá la luz del escri", True),
        ("nexa apagá la luz del escritorio", True),
    ]
    first_ready_idx = None
    for i, (text, expected_ready) in enumerate(timeline):
        pc = parse_partial_command(text)
        assert pc.ready_to_dispatch() == expected_ready, (
            f"Step {i} '{text}': expected ready={expected_ready}, got {pc.ready_to_dispatch()}"
        )
        if pc.ready_to_dispatch() and first_ready_idx is None:
            first_ready_idx = i
    # El 5to paso ya alcanza para dispatchar — antes del "escritorio".
    assert first_ready_idx == 4


def test_whisper_variant_alexa_still_works():
    """Si Whisper transcribe 'alexa' en vez de 'nexa', el parser sigue captando."""
    pc = parse_partial_command("alexa apagá la luz del escritorio")
    assert pc.has_wake is True
    assert pc.ready_to_dispatch()
