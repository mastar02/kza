"""Tests: notificación de fallo del slow path (Issue C, 2026-05-29).

Antes, `Request.fail()` no invocaba callback → el waiter de `_slow_path`
nunca despertaba y colgaba el timeout completo (5s), devolviendo un
"tardé demasiado" FALSO ante cualquier fallo del reasoner. Ahora `fail()`
invoca `on_fail` (o `on_complete`) y `_slow_path` distingue FAILED del
timeout real, respondiendo un error accionable de inmediato.
"""

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.orchestrator.dispatcher import RequestDispatcher
from src.orchestrator.priority_queue import Request, Priority, PriorityRequestQueue
from src.orchestrator.context_manager import ContextManager


class TestRequestFailNotifies:
    """Unit: Request.fail() despierta al llamador."""

    def test_fail_invokes_on_fail(self):
        called = {}
        req = Request(priority=Priority.LOW, on_fail=lambda r: called.update(r=r))
        req.fail("boom")
        assert called.get("r") is req
        assert req.error == "boom"
        assert req.status.name == "FAILED"

    def test_fail_falls_back_to_on_complete(self):
        # Si no hay on_fail, usa on_complete para no dejar al waiter colgado.
        seen = {}
        req = Request(priority=Priority.LOW, on_complete=lambda r: seen.update(r=r))
        req.fail("boom")
        assert seen.get("r") is req

    def test_fail_without_callbacks_does_not_raise(self):
        req = Request(priority=Priority.LOW)
        req.fail("boom")  # no debe explotar
        assert req.status.name == "FAILED"

    def test_enqueue_propagates_on_fail(self):
        q = PriorityRequestQueue()
        marker = {}
        req = q.enqueue("u1", "hola", Priority.LOW, on_fail=lambda r: marker.update(hit=True))
        req.fail("x")
        assert marker.get("hit") is True


class _SimQueue:
    """Cola fake que simula un worker: tras enqueue, agenda complete/fail/nada."""

    def __init__(self, mode):
        self.mode = mode
        self.requests = []

    def enqueue(self, **kw):
        req = Request(
            priority=kw.get("priority", Priority.LOW),
            user_id=kw.get("user_id", ""),
            text=kw.get("text", ""),
            on_complete=kw.get("on_complete"),
            on_cancel=kw.get("on_cancel"),
            on_fail=kw.get("on_fail"),
        )
        self.requests.append(req)
        loop = asyncio.get_event_loop()
        if self.mode == "fail":
            loop.call_soon(lambda: req.fail("reasoner down"))
        elif self.mode == "complete":
            loop.call_soon(lambda: req.complete("respuesta ok"))
        # mode == "hang": no se agenda nada → fuerza timeout
        return req

    def get_position(self, request_id):
        return 1


def _dispatcher(queue):
    chroma = MagicMock()
    chroma.asearch_command = AsyncMock(return_value=None)
    ha = MagicMock()
    routine = MagicMock()
    routine.handle = AsyncMock(return_value={"handled": False, "response": "", "success": False})
    d = RequestDispatcher(
        chroma_sync=chroma,
        ha_client=ha,
        routine_manager=routine,
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=queue,
        vector_threshold=0.65,
    )
    return d


class TestSlowPathFailureContract:
    """Integración: _slow_path responde error accionable inmediato, no timeout falso."""

    @pytest.mark.asyncio
    async def test_reasoner_failure_returns_actionable_error_fast(self):
        d = _dispatcher(_SimQueue("fail"))
        t0 = asyncio.get_event_loop().time()
        result = await d._slow_path(
            text="explicá la relatividad",
            user_id="u1", user_name=None, zone_id=None,
            priority=Priority.LOW, on_response=None, timeout=5.0,
        )
        elapsed = asyncio.get_event_loop().time() - t0
        assert result.success is False
        assert result.intent == "error"          # NO "timeout"
        assert result.error == "reasoner down"
        assert "tarde" not in result.response.lower() and "demasiado" not in result.response.lower()
        assert elapsed < 1.0, "debe responder ya, no esperar el timeout de 5s"

    @pytest.mark.asyncio
    async def test_completed_request_still_returns_response(self):
        d = _dispatcher(_SimQueue("complete"))
        result = await d._slow_path(
            text="hola", user_id="u1", user_name=None, zone_id=None,
            priority=Priority.LOW, on_response=None, timeout=5.0,
        )
        assert result.success is True
        assert result.response == "respuesta ok"
        assert result.intent == "conversation"

    @pytest.mark.asyncio
    async def test_genuine_timeout_still_reported_as_timeout(self):
        # Si NADA responde (worker colgado de verdad), sigue siendo timeout.
        d = _dispatcher(_SimQueue("hang"))
        result = await d._slow_path(
            text="hola", user_id="u1", user_name=None, zone_id=None,
            priority=Priority.LOW, on_response=None, timeout=0.2,
        )
        assert result.success is False
        assert result.intent == "timeout"
