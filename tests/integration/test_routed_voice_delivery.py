"""Actual routers, orchestrator, response owner, SDK, zones and PCM output."""

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


@pytest.mark.parametrize('orchestrated', [True, False])
@pytest.mark.parametrize('failure', [False, True])
async def test_real_router_delivers_each_reply_or_generation_error_once(monkeypatch, orchestrated, failure):
    text = 'Una respuesta completa de prueba.'
    parts = [event(text), event(finish='length' if failure else 'stop')]
    handler, engine, backend, endpoint, amplifier = voice_graph(monkeypatch, parts=parts)
    orchestrator = MultiUserOrchestrator(None, None, None, llm=handler.llm, response_handler=handler) if orchestrated else None
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=SimpleNamespace(
        text='Nexa explicame el cielo', user=None, emotion=None, timings={}, stt_confidence=None,
    ))
    audio_manager = MagicMock()
    audio_manager.detect_source_zone.return_value = 'zone_origin'
    memory = MagicMock()
    memory.build_context.return_value = ''
    chroma = MagicMock()
    chroma.search_command.return_value = None
    router = RequestRouter(processor, handler, audio_manager, orchestrator=orchestrator,
        orchestrator_enabled=orchestrated, llm_reasoner=handler.llm, memory_manager=memory,
        chroma_sync=chroma,
        routine_manager=SimpleNamespace(handle=AsyncMock(return_value={'handled': False})))
    worker = None
    if orchestrator:
        orchestrator._running = True
        worker = asyncio.create_task(orchestrator._process_queue())
    try:
        result = await router.process_command(CommandEvent(np.zeros(16000, dtype=np.float32), room_id='origin'))
        assert result['success'] is not failure
        expected = 'No pude procesar eso ahora mismo, probá de nuevo.' if failure else text
        assert engine.calls == [expected]
        np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio(expected))
        assert endpoint.body.closed and endpoint.client.is_closed()
        amplifier.select_zone.assert_called_once_with(1)
        if orchestrator:
            history = orchestrator.context_manager.get('unknown').conversation_history
            assert len(history) == (0 if failure else 2)
        elif failure:
            memory.record_interaction.assert_not_called()
    finally:
        if worker:
            orchestrator._running = False
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
