"""Tests for sync→async adapter clients."""

import asyncio
import pytest
from unittest.mock import MagicMock

from src.llm.adapters import FastRouterAdapter, HttpReasonerAdapter


class TestFastRouterAdapter:
    @pytest.mark.asyncio
    async def test_complete_calls_generate_in_thread(self):
        # FastRouter.generate retorna list[str]
        fr = MagicMock()
        fr.generate = MagicMock(return_value=["respuesta-fast"])
        adapter = FastRouterAdapter(fr)

        result = await adapter.complete("hola", max_tokens=10)

        assert result == "respuesta-fast"
        fr.generate.assert_called_once_with(["hola"], max_tokens=10, temperature=0.3)

    @pytest.mark.asyncio
    async def test_complete_temperature_passes_through(self):
        fr = MagicMock()
        fr.generate = MagicMock(return_value=["x"])
        adapter = FastRouterAdapter(fr)

        await adapter.complete("hola", max_tokens=20, temperature=0.7)
        fr.generate.assert_called_once_with(["hola"], max_tokens=20, temperature=0.7)

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self):
        fr = MagicMock()
        fr.generate = MagicMock(side_effect=RuntimeError("boom"))
        adapter = FastRouterAdapter(fr)

        with pytest.raises(RuntimeError, match="boom"):
            await adapter.complete("hola")


class TestHttpReasonerAdapter:
    @pytest.mark.asyncio
    async def test_complete_extracts_text(self):
        # HttpReasoner.__call__ retorna {choices:[{text:...}], usage:{...}}
        hr = MagicMock()
        hr.return_value = {
            "choices": [{"text": "respuesta-deep"}],
            "usage": {"completion_tokens": 5},
        }
        adapter = HttpReasonerAdapter(hr)

        result = await adapter.complete("explica X", max_tokens=128)

        assert result == "respuesta-deep"
        hr.assert_called_once_with("explica X", max_tokens=128, temperature=0.7)
