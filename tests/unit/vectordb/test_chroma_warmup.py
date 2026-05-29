"""
Tests for ChromaSync.warmup_embedder() — ensures lazy embedder init is forced.

NOTE: If chromadb is not installed in the test environment, this test will be
skipped at collection time. The implementation should still be verified by
reading the code carefully.
"""

import pytest
from unittest.mock import MagicMock

# Skip entire module if chromadb not available
pytest.importorskip("chromadb")

from src.vectordb.chroma_sync import ChromaSync


def test_warmup_embedder_forces_lazy_init_and_encodes(monkeypatch):
    """Verify warmup_embedder() forces embedder materialization and encodes dummy.

    This test ensures that the warmup doesn't get skipped due to lazy init.
    The bug was: the guard `getattr(chroma, "_embedder", None) is not None`
    evaluated False on first call, so the warmup was silently skipped and the
    first command paid ~48ms cold start. The fix uses the `embedder` property
    (which calls initialize() if needed) instead of checking _embedder directly.
    """
    # Create a ChromaSync without calling __init__ (to avoid real chroma setup)
    chroma = ChromaSync.__new__(ChromaSync)
    chroma._embedder = None

    # Mock embedder and track if initialize was called
    fake_embedder = MagicMock()
    init_called = {"n": 0}

    def _fake_initialize():
        init_called["n"] += 1
        chroma._embedder = fake_embedder

    monkeypatch.setattr(chroma, "initialize", _fake_initialize)

    # Call warmup_embedder — should force initialize() and encode
    result = chroma.warmup_embedder()

    # Verify initialize was called exactly once
    assert init_called["n"] == 1, "warmup_embedder did not materialize the lazy embedder"

    # Verify the embedder's encode was called (with ["warmup"] as argument)
    fake_embedder.encode.assert_called_once()
    call_args = fake_embedder.encode.call_args
    assert call_args[0][0] == ["warmup"], "encode should be called with ['warmup']"

    # Verify the return value is a float (timing in ms)
    assert isinstance(result, float), "warmup_embedder should return timing in ms"
    assert result >= 0, "timing should be non-negative"
