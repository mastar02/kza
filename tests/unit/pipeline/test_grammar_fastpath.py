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
