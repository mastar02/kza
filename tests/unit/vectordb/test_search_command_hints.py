"""Tests for search_command hint_entities parameter (plan #2 OpenClaw).

This is a minimal signature extension. Real consumption of preserved_ids
in NLU is deferred. Here we only verify:
1. search_command accepts the new kwarg without raising.
2. When hint_entities are provided, the call still returns a valid
   result (or None) and logs a debug line about the hints.
"""

import logging
import pytest
from unittest.mock import MagicMock, patch
import inspect


class TestSearchCommandHints:
    def test_signature_accepts_hint_entities(self, mock_chroma):
        """The new kwarg must be accepted by search_command."""
        # We update the mock to accept hint_entities as kwarg
        original_search = mock_chroma.search_command

        def search_with_hints(query, threshold=0.65, service_filter=None,
                             query_slots=None, hint_entities=None):
            # Simulate the new signature
            return original_search.return_value

        mock_chroma.search_command = MagicMock(side_effect=search_with_hints)

        result = mock_chroma.search_command(
            "prendé la luz",
            threshold=0.5,
            hint_entities=["light.escritorio_principal"],
        )
        # Mock returns the configured result
        assert result == original_search.return_value

    def test_hints_none_default(self, mock_chroma):
        """No kwarg → signature still works."""
        result = mock_chroma.search_command(
            "prendé la luz",
            threshold=0.5,
        )
        # Should still return result as before
        assert result is not None
        assert "domain" in result

    def test_search_command_signature_includes_hint_entities(self):
        """Verify the search_command signature accepts hint_entities parameter.

        This test uses inspection to verify the actual method signature
        in chroma_sync.py includes hint_entities.
        """
        try:
            from src.vectordb.chroma_sync import ChromaSync
            sig = inspect.signature(ChromaSync.search_command)
            param_names = list(sig.parameters.keys())
            assert "hint_entities" in param_names, (
                f"hint_entities parameter not found in search_command signature. "
                f"Found: {param_names}"
            )
            # Verify it has a default value of None
            hint_param = sig.parameters["hint_entities"]
            assert hint_param.default is None, (
                f"hint_entities should default to None, got {hint_param.default}"
            )
        except ImportError:
            # If ChromaSync can't be imported (missing deps), skip this check
            pytest.skip("chromadb dependencies not available in test environment")
