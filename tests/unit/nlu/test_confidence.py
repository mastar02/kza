"""
Tests del confidence score en PartialCommand y del registry de acciones
sensibles (S4 — Confidence-based confirmation).

La heurística es:
    - intent + entity ausentes → 0.0
    - base 0.7 con intent+entity
    - +0.15 si wake presente
    - +0.10 si room presente
    - +0.05 si hay slots
    - clamp a 1.0
"""
from __future__ import annotations

import pytest

from src.nlu import is_sensitive
from src.nlu.command_grammar import (
    PartialCommand,
    _compute_confidence,
    parse_partial_command,
)


# ============================================================
# Confidence scoring
# ============================================================

def test_confidence_full_command_high():
    """Comando con wake+intent+entity+room+slots → confidence == 1.0."""
    pc = parse_partial_command("nexa apagá la luz del escritorio al 50 por ciento")
    assert pc.intent == "turn_off"
    assert pc.entity == "light"
    assert pc.room == "escritorio"
    assert pc.slots.get("brightness_pct") == 50
    assert pc.confidence >= 0.95
    assert pc.is_high_confidence()


def test_confidence_no_room_lower():
    """Sin room pero con wake+intent+entity → 0.85 (entre 0.8 y 0.95)."""
    pc = parse_partial_command("nexa apagá la luz")
    assert pc.intent == "turn_off"
    assert pc.entity == "light"
    assert pc.room is None
    # 0.7 base + 0.15 wake = 0.85
    assert 0.8 <= pc.confidence < 0.95


def test_confidence_zero_when_incomplete():
    """Sin intent (solo wake) → confidence == 0.0."""
    pc = parse_partial_command("nexa")
    assert pc.intent is None
    assert pc.entity is None
    assert pc.confidence == 0.0
    assert not pc.is_high_confidence()


def test_confidence_zero_when_no_entity():
    """Con intent pero sin entity reconocible → confidence == 0.0."""
    pc = parse_partial_command("nexa apagá")
    assert pc.intent == "turn_off"
    assert pc.entity is None
    assert pc.confidence == 0.0


def test_confidence_no_wake_base():
    """Sin wake pero con intent+entity → 0.7 base."""
    pc = parse_partial_command("apagá la luz")
    assert pc.has_wake is False
    assert pc.intent == "turn_off"
    assert pc.entity == "light"
    # 0.7 base sin wake ni room ni slots
    assert pc.confidence == pytest.approx(0.7)


def test_confidence_no_wake_with_room():
    """Sin wake, con room → 0.7 base + 0.10 room = 0.80."""
    pc = parse_partial_command("apagá la luz del living")
    assert pc.has_wake is False
    assert pc.room == "living"
    assert pc.confidence == pytest.approx(0.8)


def test_confidence_clamped_at_one():
    """Todos los boosts → clamp en 1.0 (no se pasa)."""
    pc = parse_partial_command("nexa prendé la luz del cuarto en azul al 80%")
    # 0.7 + 0.15 + 0.10 + 0.05 = 1.0
    assert pc.confidence == pytest.approx(1.0)
    assert pc.confidence <= 1.0


def test_compute_confidence_direct():
    """Unit test del helper privado _compute_confidence."""
    # Falta intent → 0
    pc = PartialCommand(entity="light", has_wake=True)
    assert _compute_confidence(pc) == 0.0
    # Falta entity → 0
    pc = PartialCommand(intent="turn_on", has_wake=True)
    assert _compute_confidence(pc) == 0.0
    # Mínimo: intent + entity → 0.7
    pc = PartialCommand(intent="turn_on", entity="light")
    assert _compute_confidence(pc) == pytest.approx(0.7)
    # Full
    pc = PartialCommand(
        intent="turn_on", entity="light", room="living",
        has_wake=True, slots={"brightness_pct": 50},
    )
    assert _compute_confidence(pc) == pytest.approx(1.0)


def test_is_high_confidence_custom_threshold():
    """is_high_confidence acepta threshold custom."""
    pc = parse_partial_command("apagá la luz")  # 0.70
    assert pc.is_high_confidence(threshold=0.65)
    assert not pc.is_high_confidence(threshold=0.80)


# ============================================================
# Sensitive actions registry
# ============================================================

@pytest.mark.parametrize("intent,entity,expected", [
    # Combos sensibles registrados
    ("turn_off", "climate", True),
    ("set_cover_position", "cover", True),
    ("turn_off", "media_player", True),
    # Reversibles / safe
    ("turn_on", "light", False),
    ("turn_off", "light", False),
    ("turn_on", "climate", False),   # prender aire no es sensible
    ("turn_on", "media_player", False),
    ("turn_on", "fan", False),
    # Inputs inválidos
    (None, "light", False),
    ("turn_on", None, False),
    (None, None, False),
    ("", "light", False),
    ("turn_off", "", False),
])
def test_sensitive_actions(intent, entity, expected):
    assert is_sensitive(intent, entity) == expected


def test_sensitive_combos_contents():
    """Sanity check del contenido del registry — aseguramos los 3 del plan."""
    from src.nlu.sensitive_actions import SENSITIVE_COMBOS
    assert ("turn_off", "climate") in SENSITIVE_COMBOS
    assert ("set_cover_position", "cover") in SENSITIVE_COMBOS
    assert ("turn_off", "media_player") in SENSITIVE_COMBOS
