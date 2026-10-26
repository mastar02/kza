"""Deadline authority across the real router, worker, generation and output."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline.command_event import CommandEvent
from src.pipeline.request_router import RequestRouter
from tests.mocks.llm_http import event
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


def routed(monkeypatch, parts, *, timeout=.3, **voice_options):
    handler, engine, backend, endpoint, _ = voice_graph(monkeypatch, parts=parts, **voice_options)
    orchestrator = MultiUserOrchestrator(None, None, None, llm=handler.llm,
        response_handler=handler, response_timeout_s=timeout)
    processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
        text='Nexa explicame el cielo', user=None, emotion=None, timings={}, stt_confidence=None)))
    audio_manager = MagicMock()
    audio_manager.detect_source_zone.return_value = 'zone_origin'
    router = RequestRouter(processor, handler, audio_manager, orchestrator=orchestrator,
        orchestrator_enabled=True, llm_reasoner=handler.llm)
    orchestrator._running = True
    worker = asyncio.create_task(orchestrator._process_queue())
    return router, orchestrator, worker, handler, engine, backend, endpoint


async def test_outer_deadline_cannot_repeat_an_error_whose_device_tail_is_draining(monkeypatch):
    router, orchestrator, worker, handler, engine, backend, _ = routed(monkeypatch,
        [event('incomplete'), event(finish='length')])
    backend.release_tail.clear()
    request = asyncio.create_task(router.process_command(CommandEvent(
        np.zeros(16000, dtype=np.float32), room_id='origin')))
    try:
        await until(lambda: backend.outputs and backend.outputs[0].draining.is_set())
        await asyncio.sleep(.35)
        backend.release_tail.set()
        result = await asyncio.wait_for(request, 2)
        assert not result['success']
        assert engine.calls == ['No pude procesar eso ahora mismo, probá de nuevo.']
        assert orchestrator.context_manager.get('unknown').conversation_history == []
    finally:
        backend.release_tail.set()
        orchestrator._running = False
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await handler.aclose()
