"""FastRouter.complete() must forward stop sequences to the endpoint.

Regression guard: the signature ends in **_ignored, so a `stop` kwarg used to
be swallowed silently -- the caller believed it had stop tokens and did not.
"""

from unittest.mock import MagicMock

import pytest

from src.llm.reasoner import FastRouter


@pytest.fixture
def router():
    r = FastRouter(base_url="http://127.0.0.1:8101/v1", model="test-model")
    r._client = MagicMock()
    r._client.completions.create.return_value = MagicMock(
        choices=[MagicMock(text="ACCION_AIRE")], usage=None
    )
    return r


@pytest.mark.asyncio
async def test_complete_forwards_stop_sequences(router):
    await router.complete("prompt", max_tokens=10, stop=["\n", "Texto:"])

    kwargs = router._client.completions.create.call_args.kwargs
    assert kwargs["stop"] == ["\n", "Texto:"]


@pytest.mark.asyncio
async def test_complete_omits_stop_when_not_given(router):
    await router.complete("prompt", max_tokens=10)

    kwargs = router._client.completions.create.call_args.kwargs
    assert "stop" not in kwargs


@pytest.mark.asyncio
async def test_complete_omits_stop_when_empty(router):
    await router.complete("prompt", max_tokens=10, stop=[])

    kwargs = router._client.completions.create.call_args.kwargs
    assert "stop" not in kwargs
