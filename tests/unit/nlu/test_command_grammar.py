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
    # Con parse_command, el intent requiere dominio para resolverse (match_intent_rules
    # necesita domain). Sin dominio, intent queda en None — comportamiento correcto:
    # no hay suficiente señal para dispatchar.
    pc = parse_partial_command("nexa apagá")
    assert pc.has_wake is True
    assert pc.intent is None   # sin domain no se puede resolver el intent
    assert pc.entity is None
    # Sin intent ni entity, no se puede dispatchar.
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
    # "poné la luz al 50%" tiene slot de brillo → parse_command lo clasifica
    # correctamente como 'set' (no 'turn_on'). Es semánticamente más preciso.
    pc = parse_partial_command("nexa poné la luz del living al 50 por ciento")
    assert pc.intent == "set"
    assert pc.entity == "light"
    assert pc.room == "living"
    assert pc.slots.get("brightness_pct") == 50
    assert pc.ready_to_dispatch()


def test_partial_with_color():
    # "poné la luz en azul" tiene slot de color → 'set', no 'turn_on'.
    pc = parse_partial_command("nexa poné la luz del cuarto en azul")
    assert pc.intent == "set"
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


# -----------------------------------------------------------------
# IntentRule + INTENT_RULES + match_intent_rules
# -----------------------------------------------------------------

from src.nlu.command_grammar import IntentRule, INTENT_RULES, match_intent_rules


def test_intent_rules_cover_expected_intents():
    intents = {r.intent for r in INTENT_RULES}
    assert intents == {
        "turn_on", "turn_off", "set", "open", "close",
        "media_play", "media_pause", "media_next", "volume_set",
    }


@pytest.mark.parametrize("text,domain,expected_intent", [
    ("prendé la luz", "light", "turn_on"),
    ("apagá la luz", "light", "turn_off"),
    ("subí la persiana", "cover", "open"),
    ("bajá la persiana", "cover", "close"),
    ("subí el volumen", "media_player", "volume_set"),
    ("pausá la música", "media_player", "media_pause"),
    ("poné música", "media_player", "media_play"),
    ("abrí la luz", "light", None),       # open no aplica a light → sin match
    ("subí la luz", "light", "turn_on"),  # 'subí' con light → turn_on gana por compat
])
def test_match_intent_rules_respects_domain(text, domain, expected_intent):
    rule = match_intent_rules(text, domain)
    assert (rule.intent if rule else None) == expected_intent


# -----------------------------------------------------------------
# ParsedCommand + parse_command
# -----------------------------------------------------------------

from src.nlu.command_grammar import ParsedCommand, parse_command


@pytest.mark.parametrize("text,intent,domain,target,quality", [
    ("nexa prendé la luz del escritorio", "turn_on", "light", "domotics", "full"),
    ("apagá la luz", "turn_off", "light", "domotics", "full"),
    ("subí la persiana del cuarto", "open", "cover", "domotics", "full"),
    ("subí el volumen", "volume_set", "media_player", "music", "full"),
    ("pausá la música", "media_pause", "media_player", "music", "full"),
    ("poné la luz al 70%", "set", "light", "domotics", "full"),       # set por slot, sin on/off
    ("abrí la luz", None, "light", "domotics", "partial"),            # incompat → no full
    ("hola qué tal", None, None, "domotics", "none"),                 # no domótica
    ("ponela cálida", "set", "light", "domotics", "full"),            # slot-inferred domain
])
def test_parse_command(text, intent, domain, target, quality):
    pc = parse_command(text)
    assert pc.intent == intent
    assert pc.domain == domain
    assert pc.target == target
    assert pc.quality == quality


def test_parse_command_set_includes_slot():
    pc = parse_command("poné la luz al 70%")
    assert pc.intent == "set"
    assert pc.slots.get("brightness_pct") == 70


def test_parse_command_ready_to_dispatch():
    assert parse_command("prendé la luz").ready_to_dispatch() is True
    assert parse_command("abrí la luz").ready_to_dispatch() is False


def test_parse_command_infers_light_from_slots():
    # 'cálida' es color_temp → slot exclusivo de luz → domain inferido
    pc = parse_command("ponela cálida")
    assert pc.domain == "light"
    assert pc.intent == "set"
    assert pc.slots.get("color_temp_kelvin") == 2700


def test_parse_command_no_false_light_inference():
    # sin slots de luz, no se infiere nada
    assert parse_command("dale").domain is None


# -----------------------------------------------------------------
# parse_partial_command compat wrapper (Task 4)
# -----------------------------------------------------------------

def test_partial_command_compat_shape():
    pc = parse_partial_command("nexa prendé la luz del escritorio")
    assert pc.intent == "turn_on"
    assert pc.entity == "light"          # alias de domain para compat
    assert pc.room == "escritorio"
    assert pc.ready_to_dispatch() is True
    assert pc.confidence >= 0.75


def test_partial_command_not_ready():
    pc = parse_partial_command("nexa")
    assert pc.ready_to_dispatch() is False


class TestSlotDomainInferenceLengthGuard:
    """Acción fantasma real 2026-06-04 19:49: 'Tengo los coches de negro, van
    a bajar a...' parseó full conf 0.75-0.90 → light.turn_off ejecutado.
    'negro'→rgb_color + 'bajar'→verbo de acción + inferencia domain=light
    desde el slot = comando fantasma. La inferencia slot→light es para
    comandos implícitos CORTOS ('ponela cálida'); la charla es larga."""

    def test_long_chatter_with_color_and_verb_not_full(self):
        from src.nlu.command_grammar import parse_command
        pc = parse_command("Tengo los coches de negro, van a bajar a...")
        assert pc.quality != "full"

    def test_short_implicit_set_still_full(self):
        from src.nlu.command_grammar import parse_command
        # ('bajá la luz' ya era partial antes de este fix — no es control)
        for t in ("ponela cálida", "subí el brillo al 50"):
            pc = parse_command(t)
            assert pc.quality == "full", f"{t!r} dejó de parsear full"
