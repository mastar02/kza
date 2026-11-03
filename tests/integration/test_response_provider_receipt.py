"""Provider-ready receipts exclude later consumer scheduling from its budget.

This isolates a verified accounting edge. It does not identify the cause of
the separate full-suite scheduling interval recorded during T022-J.
"""

import asyncio

import numpy as np
import pytest

from tests.integration.test_response_deadlines import budgets
from tests.mocks.llm_http import event
from tests.mocks.voice_graph import voice_graph


@pytest.mark.asyncio
async def test_completed_provider_read_does_not_charge_later_consumer_resume(monkeypatch):
    phrases = ["Primera oración suficientemente larga.", "Segunda oración suficientemente larga."]
    owner, engine, backend, endpoint, _ = voice_graph(
        monkeypatch,
        # Two ordinary cooperative turns leave this pull pending when an
        # already-expired timeout resumes its waiter; no slow provider is used.
        parts=[event(phrases[0] + " "), 0.0, 0.0, event(phrases[1] + " "), event(finish="stop")],
        response_budgets=budgets(generation_s=.3),
    )
    completed, release = asyncio.Event(), asyncio.Event()
    actual_wait = asyncio.wait
    held = False
    receipt = None

    async def hold_consumer_after_ready(fs, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        nonlocal held, receipt
        tasks = set(fs)
        done, pending = await actual_wait(tasks, timeout=timeout, return_when=return_when)
        caller = asyncio.current_task()
        code = getattr(caller.get_coro(), "cr_code", None) if caller is not None else None
        if not held and code is not None and code.co_name == "produce" and len(tasks) == 1 and done:
            held = True
            receipt = next(iter(done))
            # The actual GenerationStream/HTTP reader is already terminal;
            # only its consumer's continuation is held below.
            assert receipt.done() and not receipt.cancelled()
            # HttpReasoner's thought-filter holdback may split this sentence.
            token = receipt.result()["token"]
            assert type(token) is str and token and (phrases[0] + " ").startswith(token)
            completed.set()
            await release.wait()
        return done, pending

    monkeypatch.setattr(asyncio, "wait", hold_consumer_after_ready)
    response = asyncio.create_task(owner.arespond("synthetic", zone_id="zone_origin"))
    try:
        async with asyncio.timeout(3):
            await completed.wait()
        assert receipt is not None and receipt.done()
        await asyncio.sleep(.4)  # Same delay/budget as the existing backpressure contract.
        assert not response.done() and engine.calls == []
        release.set()
        result = await asyncio.wait_for(response, 2)
        assert result.success and engine.calls == phrases
        assert endpoint.body.closed and endpoint.client.is_closed()
        np.testing.assert_array_equal(
            backend.heard()[:, 0], np.concatenate([engine.audio(text) for text in phrases]),
        )
    finally:
        release.set()
        await asyncio.gather(response, return_exceptions=True)
        await owner.aclose()
