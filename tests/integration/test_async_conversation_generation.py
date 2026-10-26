"""Real queue/dispatcher/worker/context with native SDK and memory-only HTTP."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from src.llm.reasoner import HttpReasoner
from src.orchestrator.cancellation import CancellationReason
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.priority_queue import Priority, Request, RequestStatus
from tests.mocks.llm_http import Endpoint, SSEBody, event

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(monkeypatch, parts):
    endpoint = Endpoint(SSEBody(parts))
    reasoner = HttpReasoner(model="configured-model", api_style="chat", cloud_consent=True)
    monkeypatch.setattr(reasoner, "_new_async_client", endpoint.build)
    orchestrator = MultiUserOrchestrator(None, None, None, llm=reasoner)
    assert orchestrator.dispatcher.queue is orchestrator.queue
    orchestrator.context_manager.get_or_create("household", zone_id="origin")
    orchestrator._running = True
    processor = asyncio.create_task(orchestrator._process_queue())
    orchestrator._processor_task = processor
    try:
        yield orchestrator, endpoint
    finally:
        orchestrator._running = False
        processor.cancel()
        await asyncio.gather(processor, return_exceptions=True)


async def dispatch(orchestrator, *, timeout=2):
    return await orchestrator.dispatcher._slow_path(
        text="consulta original", user_id="household", user_name=None,
        zone_id="origin", priority=Priority.LOW, on_response=None, timeout=timeout,
    )


async def test_complete_response_and_context_without_blocking_capture(monkeypatch, record_property):
    all_lags = []
    loop = asyncio.get_running_loop()
    for _ in range(10):
        async with graph(monkeypatch, [0.2, event("respuesta completa"), event(finish="stop")]) as (owner, endpoint):
            task = asyncio.create_task(dispatch(owner))
            deadline = loop.time() + 0.005
            lags = []
            while not task.done():
                await asyncio.sleep(max(0, deadline - loop.time()))
                lags.append(max(0, loop.time() - deadline))
                deadline = loop.time() + 0.005
            result = await task
            assert len(lags) >= 20
            all_lags.extend(lags)
            assert result.success and result.response == "respuesta completa"
            history = owner.context_manager.get("household").conversation_history
            assert [(turn.role, turn.content) for turn in history] == [
                ("user", "consulta original"), ("assistant", "respuesta completa"),
            ]
            assert endpoint.client.is_closed() and endpoint.body.closed
            assert owner.queue._current is None
    ordered = sorted(all_lags)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    record_property("heartbeat_ms", json.dumps({
        "repetitions": 10, "stall_ms": 200, "interval_ms": 5, "samples": len(ordered),
        "p95": p95 * 1000, "max": max(ordered) * 1000,
    }))
    assert p95 <= 0.025 and max(ordered) <= 0.075


@pytest.mark.parametrize("parts", [
    [event("partial"), event(finish="length")],
    [event("partial")],
    [event(finish="stop")],
])
async def test_incomplete_answer_fails_and_never_records_success(monkeypatch, parts):
    async with graph(monkeypatch, parts) as (owner, endpoint):
        result = await dispatch(owner)
        assert not result.success and result.intent == "error"
        assert owner.context_manager.get("household").conversation_history == []
        assert endpoint.body.closed and endpoint.client.is_closed()


async def test_dispatch_timeout_closes_pending_generation_and_keeps_context_empty(monkeypatch):
    async with graph(monkeypatch, [event("pending tail"), asyncio.Event()]) as (owner, endpoint):
        result = await dispatch(owner, timeout=0.05)
        assert result.intent == "timeout"
        request = owner.queue._user_requests["household"]
        assert request.cancellation_token.reason == CancellationReason.TIMEOUT
        async with asyncio.timeout(1):
            while not endpoint.body.closed or not endpoint.client.is_closed():
                await asyncio.sleep(0.001)
        assert endpoint.client.is_closed()
        assert request.status == RequestStatus.CANCELLED and request.result is None
        assert owner.context_manager.get("household").conversation_history == []


async def test_cancelled_dispatch_cancels_work_instead_of_leaving_request_running(monkeypatch):
    async with graph(monkeypatch, [event("pending tail"), asyncio.Event()]) as (owner, endpoint):
        task = asyncio.create_task(dispatch(owner))
        await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(1):
            while not endpoint.body.closed or not endpoint.client.is_closed():
                await asyncio.sleep(0.001)
        assert owner.queue._user_requests["household"].is_cancelled
        assert owner.context_manager.get("household").conversation_history == []


async def test_direct_token_cancellation_also_notifies_waiter(monkeypatch):
    async with graph(monkeypatch, [event("pending tail"), asyncio.Event()]) as (owner, endpoint):
        task = asyncio.create_task(dispatch(owner))
        await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
        request = owner.queue._current
        request.cancellation_token.cancel(CancellationReason.USER_EXPLICIT)
        result = await asyncio.wait_for(task, 0.1)
        assert result.intent == "cancelled" and result.response == ""
        assert request.status == RequestStatus.CANCELLED
        async with asyncio.timeout(1):
            while not endpoint.body.closed or not endpoint.client.is_closed():
                await asyncio.sleep(0.001)
        assert endpoint.client.is_closed()
        assert owner.context_manager.get("household").conversation_history == []


async def test_cancellation_latency_and_cleanup_ten_runs(monkeypatch, record_property):
    controls, cleanups = [], []
    loop = asyncio.get_running_loop()
    for _ in range(10):
        async with graph(monkeypatch, [event("pending tail"), asyncio.Event()]) as (owner, endpoint):
            task = asyncio.create_task(dispatch(owner))
            await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
            started = loop.time()
            owner.queue._current.cancel(CancellationReason.USER_EXPLICIT)
            controls.append((loop.time() - started) * 1000)
            assert (await asyncio.wait_for(task, 0.1)).intent == "cancelled"
            async with asyncio.timeout(1):
                while not endpoint.body.closed or not endpoint.client.is_closed():
                    await asyncio.sleep(0.001)
            cleanups.append((loop.time() - started) * 1000)
            assert owner.context_manager.get("household").conversation_history == []

    def stats(samples):
        ordered = sorted(samples)
        return {"p50": ordered[4], "p95": ordered[9], "max": ordered[-1]}

    record_property("cancellation_ms", json.dumps({
        "repetitions": 10, "request_control": stats(controls), "stream_and_client_cleanup": stats(cleanups),
    }))
    assert max(controls) <= 100 and max(cleanups) <= 1000


@pytest.mark.parametrize("same_user", [True, False])
async def test_new_request_interrupts_waiting_read_without_old_context(monkeypatch, same_user):
    async with graph(monkeypatch, [event("pending tail"), asyncio.Event()]) as (owner, endpoint):
        old_body = endpoint.body
        task = asyncio.create_task(dispatch(owner))
        await asyncio.wait_for(old_body.waiting.wait(), 1)
        old = owner.queue._current
        endpoint.body = SSEBody([event("respuesta nueva"), event(finish="stop")])
        user = "household" if same_user else "other"
        owner.context_manager.get_or_create(user)
        new = owner.queue.enqueue(user, "consulta nueva", Priority.LOW if same_user else Priority.HIGH)
        result = await asyncio.wait_for(task, 1)
        async with asyncio.timeout(1):
            while new.status != RequestStatus.COMPLETED:
                await asyncio.sleep(0.001)
        assert not result.success and result.intent == "cancelled" and result.response == ""
        assert old.status == RequestStatus.CANCELLED and old.result is None and old_body.closed
        expected = CancellationReason.USER_NEW_REQUEST if same_user else CancellationReason.HIGHER_PRIORITY
        assert old.cancellation_token.reason == expected
        for context in (owner.context_manager.get("household"), owner.context_manager.get(user)):
            assert all(turn.content != "consulta original" and "pending" not in turn.content
                       for turn in context.conversation_history)
        assert new.result == "respuesta nueva"


async def test_shutdown_cancels_stream_and_notifies_request_waiter(monkeypatch):
    async with graph(monkeypatch, [event("pending tail"), asyncio.Event()]) as (owner, endpoint):
        task = asyncio.create_task(dispatch(owner))
        await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
        request = owner.queue._current
        await owner.stop()
        result = await asyncio.wait_for(task, 1)
        assert result.intent == "cancelled" and not result.success
        assert request.cancellation_token.reason == CancellationReason.SYSTEM_SHUTDOWN
        assert endpoint.body.closed and endpoint.client.is_closed()
        assert owner.context_manager.get("household").conversation_history == []


@pytest.mark.parametrize("first", ["cancel", "complete", "fail"])
async def test_terminal_request_cannot_be_resurrected_or_notified_twice(first):
    notifications = []
    request = Request(priority=Priority.LOW,
                      on_cancel=lambda _: notifications.append("cancel"),
                      on_complete=lambda _: notifications.append("complete"),
                      on_fail=lambda _: notifications.append("fail"))
    if first == "cancel":
        request.cancel()
    elif first == "complete":
        request.complete("first")
    else:
        request.fail("first")
    snapshot = (request.status, request.result, request.error, request.completed_at)
    request.complete("late")
    request.fail("late")
    request.cancel()
    assert (request.status, request.result, request.error, request.completed_at) == snapshot
    assert notifications == [first]
