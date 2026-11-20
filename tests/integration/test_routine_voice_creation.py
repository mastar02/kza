"""Routine requests cannot fall back to installing or deleting HA automations."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.orchestrator.dispatcher import DispatchResult, MultiUserOrchestrator, PathType, Priority
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.routines.voice_routines import VoiceRoutineService, is_unclaimed_routine_creation
from tests.mocks.responses import response_handler_mock


@asynccontextmanager
async def voice_router(orchestrated, utterance, *, scheduler=None):
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=ProcessedCommand(
        text=f"Nexa, {utterance}", success=True, timings={},
    ))
    chroma, ha, reasoner = MagicMock(), AsyncMock(), MagicMock()
    voice_routines = VoiceRoutineService(scheduler)
    orchestrator = MultiUserOrchestrator(chroma, ha, llm=reasoner, voice_routines=voice_routines)
    orchestrator.dispatcher._slow_path = AsyncMock(side_effect=AssertionError("routine reached cloud"))
    audio_manager = MagicMock()
    audio_manager.detect_source_zone.return_value = None
    router = RequestRouter(
        processor, response_handler_mock(), audio_manager, orchestrator=orchestrator, orchestrator_enabled=orchestrated,
        chroma_sync=chroma, ha_client=ha, llm_reasoner=reasoner, voice_routines=voice_routines,
        wake_acoustically_confirmed=True,
    )
    try:
        yield router, orchestrator.dispatcher
    finally:
        await router.aclose()
        await orchestrator.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("orchestrated", [True, False])
@pytest.mark.parametrize(("utterance", "response_part"), [
    ("pensaba en una nueva rutina para la noche", "no creo rutinas por voz"),
    ("borrá la rutina riego", "desde la web"),
    ("elimina la automatización riego", "desde la web"),
    ("elimina las automatizaciones de riego", "desde la web"),
    ("para la rutina de riego configura que funcione de noche", "no creo rutinas por voz"),
    ("para el riego rutinario configura que funcione de noche", "no creo rutinas por voz"),
    ("configura el horario de la automatización de riego", "no creo rutinas por voz"),
    ("configurá a qué hora se activa la automatización de riego", "no creo rutinas por voz"),
    ("configura el horario de la automatización de riego para las ocho, ¿sí?", "no creo rutinas por voz"),
    ("para la automatización de riego configura el horario de noche", "no creo rutinas por voz"),
    ("para la automatización de riego configurá el horario a las ocho", "no creo rutinas por voz"),
    ("después del riego programá el horario de la automatización de jardín", "no creo rutinas por voz"),
    ("con tu permiso programá el horario de la automatización de riego", "no creo rutinas por voz"),
    ("programa que arranque el riego al anochecer", "no creo rutinas por voz"),
    ("quiero que cuando anochezca empiece el riego", "no creo rutinas por voz"),
    ("quiero que cuando anochezca empiece el riego, ¿podés?", "no creo rutinas por voz"),
    ("haz que cuando anochezca empiece el riego", "no creo rutinas por voz"),
    ("configura que el riego empiece de noche", "no creo rutinas por voz"),
    ("cada vez que anochezca que empiece el riego", "no creo rutinas por voz"),
    ("necesito que cada vez que anochezca empiece el riego", "no creo rutinas por voz"),
])
async def test_retired_routine_requests_are_refused_before_a_producer(orchestrated, utterance, response_part):
    async with voice_router(orchestrated, utterance) as (router, dispatcher):
        result = await router.process_command(np.zeros(16000, dtype=np.float32))
        assert result["intent"] == "routine" and response_part in result["response"]
        if orchestrated:
            assert result["path"] == "fast_routine"
        assert result["outcome"]["status"] == "unsupported"
        assert result["outcome"]["persistence"] == "not_written"
        assert result["outcome"]["dispatch"] == "not_sent"
        assert result["task_completed"] is False
        assert not router.ha.mock_calls and not router.chroma.mock_calls and not router.llm.mock_calls
        dispatcher._slow_path.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("orchestrated", [True, False])
async def test_catalog_question_with_automation_vocabulary_still_reads(orchestrated):
    scheduler = SimpleNamespace(get_all_routines=AsyncMock(return_value=[]))
    text = "qué rutinas tengo y cuál automatiza el riego"
    async with voice_router(orchestrated, text, scheduler=scheduler) as (router, dispatcher):
        event = CommandEvent(np.zeros(16000, dtype=np.float32), room_id="", capture_trigger="wake", wake_score=0.9)
        result = await router.process_command(event)
        assert result["intent"] == "routine" and result["outcome"]["status"] == "completed"
        assert result["response"] == "No hay rutinas guardadas. Podés crearlas desde la web."
        scheduler.get_all_routines.assert_awaited_once()
        assert not router.ha.mock_calls and not router.chroma.mock_calls and not router.llm.mock_calls
        dispatcher._slow_path.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("orchestrated", [True, False])
@pytest.mark.parametrize("utterance", [
    "cuándo juega Peñarol",
    "qué pasa cada vez que hay un empate",
    "cuál es el nombre del programa que ganó el premio",
    "cada vez que hay un empate cuándo se juega el desempate",
    "cada vez que hay un empate cuando se juega el desempate",
    "quiero escuchar algún programa de rutinas de ejercicio",
    "quiero escuchar un buen programa de rutinas de ejercicio",
    "¿se configura el horario de la automatización de riego desde la web?",
    "podés decirme quién configura el horario de la automatización de riego",
])
async def test_utterance_without_automation_intent_stays_conversational(orchestrated, utterance):
    async with voice_router(orchestrated, utterance) as (router, dispatcher):
        router.chroma.search_command.return_value = None
        dispatcher._slow_path = AsyncMock(return_value=DispatchResult(
            PathType.SLOW_LLM, Priority.LOW, True, "Respuesta sintética.", intent="conversation",
        ))
        result = await router.process_command(np.zeros(16000, dtype=np.float32))
        assert result["intent"] == "conversation"
        if orchestrated:
            assert result["path"] == "slow_llm"
            dispatcher._slow_path.assert_awaited_once()
        else:
            router.chroma.search_command.assert_called()


@pytest.mark.parametrize("utterance", [
    "quiero escuchar el programa que ganó el premio",
    "quiero escuchar mi programa de rutinas de ejercicio",
    "cada vez que hay un empate se juega el desempate?",
    "cual es el nombre del programa que gano el premio",
    "quién configura el horario de la automatización de riego",
    "podés decirme quién normalmente configura el horario de la automatización de riego",
    "se configura que empiece de noche la automatización de riego?",
    "me podés decir si el horario de la automatización de riego se configura desde la web?",
])
def test_indirect_routine_cues_preserve_nominal_programs_and_questions(utterance):
    assert not is_unclaimed_routine_creation(f"Nexa, {utterance}")


def test_accented_program_verb_survives_nominal_filtering():
    assert is_unclaimed_routine_creation("Nexa, después del último, programá el horario de la automatización de riego")
