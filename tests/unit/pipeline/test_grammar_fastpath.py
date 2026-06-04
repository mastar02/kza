"""
Tests for _grammar_fastpath_classification helper in RequestRouter.

Verifies that clear domotics commands with high grammar confidence
bypass the unreliable LLMCommandRouter, and that non-commands return None.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy system-level modules before any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest

from src.pipeline.request_router import _grammar_fastpath_classification


class TestGrammarFastpathClassification:
    """Unit tests for the grammar fast-path helper."""

    def test_grammar_fastpath_accepts_clear_domotics(self):
        """Clear domotics command with high confidence returns is_command=True."""
        c = _grammar_fastpath_classification("Nexa, prende la luz del escritorio.")
        assert c is not None
        assert c.is_command is True
        assert c.intent is not None

    def test_grammar_fastpath_carries_correct_intent(self):
        """Grammar fast-path preserves the extracted intent."""
        c = _grammar_fastpath_classification("Nexa, prende la luz del escritorio.")
        assert c is not None
        assert c.intent == "turn_on"

    def test_grammar_fastpath_carries_entity_hint(self):
        """Grammar fast-path preserves the extracted entity in entity_hint."""
        c = _grammar_fastpath_classification("Nexa, prende la luz del escritorio.")
        assert c is not None
        assert c.entity_hint == "light"

    def test_grammar_fastpath_none_for_non_command(self):
        """Pure conversational text with no intent/entity returns None."""
        assert _grammar_fastpath_classification("gracias por todo") is None

    def test_grammar_fastpath_none_below_threshold(self):
        """Returns None when confidence is below the given threshold."""
        # Force a threshold that the 0.95-confidence command cannot beat
        result = _grammar_fastpath_classification(
            "Nexa, prende la luz del escritorio.", confidence_threshold=0.99
        )
        assert result is None

    def test_grammar_fastpath_raw_response_marks_grammar(self):
        """raw_response must signal grammar authority for debuggability."""
        c = _grammar_fastpath_classification("Nexa, prende la luz del escritorio.")
        assert c is not None
        assert "grammar" in c.raw_response

    def test_grammar_fastpath_elapsed_ms_zero(self):
        """elapsed_ms is set to 0 (grammar is synchronous / negligible)."""
        c = _grammar_fastpath_classification("Nexa, prende la luz del escritorio.")
        assert c is not None
        assert c.elapsed_ms == 0.0


class TestWakeAcousticallyConfirmed:
    """Fix 2026-06-04: con openwakeword el wake se confirma ACÚSTICAMENTE y el
    STT a veces no transcribe 'Nexa' — 'Prende la luz.' daba conf=0.7 < 0.75 y
    caía al LLMRouter (que lo rechazó como noise). El bonus de wake (+0.15)
    corresponde también cuando el wake fue confirmado fuera del texto."""

    def test_no_wake_in_text_but_confirmed_passes(self):
        # Caso real de prod 18:24:05: wake openwakeword 0.84, STT limpio sin 'Nexa'
        c = _grammar_fastpath_classification(
            "Prende la luz.", wake_confirmed=True
        )
        assert c is not None
        assert c.intent == "turn_on"

    def test_no_wake_in_text_not_confirmed_still_falls_through(self):
        # Engine whisper (wake-as-STT): sin wake en texto = sin confirmación
        c = _grammar_fastpath_classification("Prende la luz.", wake_confirmed=False)
        assert c is None

    def test_wake_in_text_unaffected(self):
        c = _grammar_fastpath_classification(
            "Nexa, prende la luz.", wake_confirmed=False
        )
        assert c is not None

    def test_confirmed_does_not_rescue_low_quality(self):
        # El bonus no convierte basura en comando (quality != full → None)
        c = _grammar_fastpath_classification("la vida es bella", wake_confirmed=True)
        assert c is None
