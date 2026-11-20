"""Routine requests cannot fall back to installing or deleting HA automations."""
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.routines.voice_routines import VoiceRoutineService
from tests.mocks.responses import response_handler_mock
@pytest.mark.asyncio
@pytest.mark.parametrize("orchestrated", [True, False])
@pytest.mark.parametrize(("utterance", "response_part"), [
    ("pensaba en una nueva rutina para la noche", "no creo rutinas por voz"),
    ("borrá la rutina riego", "desde la web"),
])
async def test_retired_routine_requests_are_refused_before_a_producer(orchestrated, utterance, response_part):
    text = f"nexa {utterance}"
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=ProcessedCommand(text=text, success=True, timings={}))
    chroma, ha = MagicMock(), AsyncMock()
    voice_routines = VoiceRoutineService()
    orchestrator = MultiUserOrchestrator(chroma, ha, voice_routines=voice_routines)
    orchestrator.dispatcher._slow_path = AsyncMock(side_effect=AssertionError("routine reached cloud"))
    audio_manager = MagicMock()
    audio_manager.detect_source_zone.return_value = None
    router = RequestRouter(
        processor, response_handler_mock(), audio_manager, orchestrator=orchestrator, orchestrator_enabled=orchestrated,
        chroma_sync=chroma, ha_client=ha, voice_routines=voice_routines, wake_acoustically_confirmed=True,
    )
    try:
        result = await router.process_command(np.zeros(16000, dtype=np.float32))
        assert result["intent"] == "routine" and response_part in result["response"]
        assert result["outcome"]["status"] == "unsupported"
        assert result["outcome"]["persistence"] == "not_written"
        assert result["outcome"]["dispatch"] == "not_sent"
        assert result["task_completed"] is False
        ha.create_automation.assert_not_called()
        ha.delete_automation.assert_not_called()
        ha.get_domotics_entities.assert_not_called()
        chroma.search_command.assert_not_called()
        chroma.asearch_command.assert_not_called()
        orchestrator.dispatcher._slow_path.assert_not_called()
    finally:
        await router.aclose()
        await orchestrator.stop()
