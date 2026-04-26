"""
Adapters sync→async para los clientes existentes (FastRouter, HttpReasoner).

LLMRouter espera `await client.complete(prompt, **kw) -> str`. Los clientes
actuales tienen interfaces propias (sync). Estos adapters traducen sin tocar
los clientes (mínima invasión).
"""

from __future__ import annotations

import asyncio
from typing import Any


class FastRouterAdapter:
    """Adapter para FastRouter (vLLM compartido :8100)."""

    def __init__(self, fast_router: Any):
        self._client = fast_router

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **_kw,
    ) -> str:
        results = await asyncio.to_thread(
            self._client.generate, [prompt], max_tokens=max_tokens, temperature=temperature
        )
        return results[0]


class HttpReasonerAdapter:
    """Adapter para HttpReasoner (llama-cpp 72B :8200)."""

    def __init__(self, http_reasoner: Any):
        self._client = http_reasoner

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **_kw,
    ) -> str:
        result = await asyncio.to_thread(
            self._client, prompt, max_tokens=max_tokens, temperature=temperature
        )
        return result["choices"][0]["text"]
