"""Routine requests cannot fall back to installing or deleting HA automations."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.home_assistant.control_policy import HomeControlPolicy
from src.home_assistant.home_actions import HomeActionService
from src.nlu.llm_router import CommandClassification
from src.orchestrator.dispatcher import DispatchResult, MultiUserOrchestrator, PathType, Priority
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.routines.voice_routines import VoiceRoutineService, is_unclaimed_routine_creation
from tests.integration.test_home_control_policy import ObservedHA, write_selection
from tests.mocks.responses import response_handler_mock


@asynccontextmanager
async def voice_router(orchestrated, utterance, *, scheduler=None, home_actions=None):
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=ProcessedCommand(
        text=f"Nexa, {utterance}", success=True, timings={},
    ))
    chroma, reasoner = MagicMock(), MagicMock()
    ha = home_actions.ha if home_actions is not None else AsyncMock()
    voice_routines = VoiceRoutineService(scheduler)
    orchestrator = MultiUserOrchestrator(chroma, ha, llm=reasoner, voice_routines=voice_routines, home_actions=home_actions)
    orchestrator.dispatcher._slow_path = AsyncMock(side_effect=AssertionError("routine reached cloud"))
    audio_manager = MagicMock()
    audio_manager.detect_source_zone.return_value = None
    router = RequestRouter(
        processor, response_handler_mock(), audio_manager, orchestrator=orchestrator, orchestrator_enabled=orchestrated,
        chroma_sync=chroma, ha_client=ha, llm_reasoner=reasoner, voice_routines=voice_routines,
        wake_acoustically_confirmed=True, home_actions=home_actions,
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
    ("del riego programa la rutina para las ocho", "no creo rutinas por voz"),
    ("del riego programá la rutina para las ocho", "no creo rutinas por voz"),
    ("hacé que se programe el horario de la automatización de riego a las ocho", "no creo rutinas por voz"),
    ("necesito que se configure el horario de la automatización de riego", "no creo rutinas por voz"),
    ("quiero que programe el horario de la automatización de riego", "no creo rutinas por voz"),
    ("para el riego, programa dos rutinas para las ocho", "no creo rutinas por voz"),
    ("para el riego programa dos rutinas para las ocho", "no creo rutinas por voz"),
    ("para el riego, programa que arranque a las ocho", "no creo rutinas por voz"),
    ("para el riego programa que arranque a las ocho", "no creo rutinas por voz"),
    ("después del riego programa, la rutina de jardín para las ocho", "no creo rutinas por voz"),
    ("después del riego programa la rutina de jardín para las ocho", "no creo rutinas por voz"),
    ("después del riego, programa que vuelva a arrancar a las ocho", "no creo rutinas por voz"),
    ("después del riego programa, que vuelva a arrancar a las ocho", "no creo rutinas por voz"),
    ("después del riego programa que vuelva a arrancar a las ocho", "no creo rutinas por voz"),
    ("después del último programa que se encienda el riego a las ocho", "no creo rutinas por voz"),
    ("para el próximo programa que se encienda el riego a las ocho", "no creo rutinas por voz"),
    ("para el próximo programa que las luces prendan al salir el sol", "no creo rutinas por voz"),
    ("después del último programa que a las ocho se encienda el riego", "no creo rutinas por voz"),
    ("para el próximo programa que cada noche se encienda el riego", "no creo rutinas por voz"),
    ("después del anochecer programa que veamos que se encienda el riego a las ocho", "no creo rutinas por voz"),
    ("para el jardín programa que veamos el momento en que se enciendan las luces al salir el sol", "no creo rutinas por voz"),
    ("para el próximo programa que vamos a ver primero el momento en que se prenden las luces al salir el sol", "no creo rutinas por voz"),
    ("para el próximo programa que vamos a ver cuándo se prenden las luces al salir el sol", "no creo rutinas por voz"),
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
@pytest.mark.parametrize("utterance", [
    "después del último programa que se encienda el riego a las ocho",
    "para el próximo programa que las luces prendan al salir el sol",
    "después del último programa que a las ocho se encienda el riego",
    "para el próximo programa que cada noche se encienda el riego",
    "después del anochecer programa que veamos que se encienda el riego a las ocho",
    "para el jardín programa que veamos el momento en que se enciendan las luces al salir el sol",
    "para el próximo programa que vamos a ver primero el momento en que se prenden las luces al salir el sol",
    "para el próximo programa que vamos a ver cuándo se prenden las luces al salir el sol",
    "para el programa que vamos a ver cómo se prenden las luces al salir el sol",
    "para el próximo programa que vamos a ver quiero que se enciendan las luces al salir el sol",
    "para el próximo programa que vamos a ver quiero que me enciendan las luces al salir el sol",
    "para el próximo programa que vamos a ver quería que nos enciendan las luces al salir el sol",
    "para el próximo programa que vamos a ver queremos que te enciendan las luces al salir el sol",
    "para el próximo programa que vamos a ver quisiera que le enciendan las luces al salir el sol",
    "para el último programa que vimos quiero que me enciendan las luces al salir el sol",
    "para el próximo programa que vamos a ver, cuándo se prenden las luces al salir el sol",
    "para el próximo programa que escuche cuándo se prenden las luces al salir el sol",
])
async def test_programming_request_is_refused_even_if_7b_suggests_turn_on(orchestrated, utterance):
    async with voice_router(orchestrated, utterance) as (router, dispatcher):
        classify = AsyncMock(return_value=CommandClassification(is_command=True, confidence=0.99, intent="turn_on"))
        router.llm_command_router = SimpleNamespace(classify=classify)
        event = CommandEvent(
            np.zeros(16000, dtype=np.float32), room_id="", capture_trigger="wake", wake_score=0.9,
            textual_wake=True,
        )

        result = await router.process_command(event)

        assert result["intent"] == "routine"
        if orchestrated:
            classify.assert_awaited_once()
            assert result["path"] == "fast_routine"
        else:
            classify.assert_not_called()
        outcome = result["outcome"]
        assert (outcome["status"], outcome["persistence"], outcome["dispatch"]) == (
            "unsupported", "not_written", "not_sent",
        )
        assert not router.ha.mock_calls and not router.chroma.mock_calls and not router.llm.mock_calls
        dispatcher._slow_path.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("orchestrated", [True, False])
@pytest.mark.parametrize(("utterance", "orchestrated_stage"), [
    ("para el último programa que vimos, necesito que me enciendan las luces al salir el sol", "dispatcher"),
    ("para el último programa que vimos, haz que se enciendan las luces al salir el sol", "dispatcher"),
    ("para el último programa que vimos, hacé que se enciendan las luces al ponerse el sol", "dispatcher"),
    ("para el último programa que vimos, poné que se enciendan las luces al amanecer", "dispatcher"),
    ("para el último programa que vimos, quiero que me enciendan las luces al atardecer", "dispatcher"),
    ("para el último programa que vimos, prendé la luz a las 8", "gate"),
    ("necesito que prendas las luces a las ocho", "gate"),
    ("quiero que se riegue cada vez que anochezca", "verb_evidence"),
    ("quiero que pongas riego a las ocho", "verb_evidence"),
    ("necesito que me digas qué partido empieza a las ocho y que prendas las luces al salir el sol", "gate"),
    ("quiero que me cuentes el resultado y pongas la luz al 50 por ciento al amanecer", "verb_evidence"),
    ("quiero que me cuentes el resultado y dejes la luz al 50 por ciento al amanecer", "verb_evidence"),
    ("después del último programa que vimos, ¿podés prender la luz a las ocho?", "gate"),
    ("después del último programa que vimos, podes prender la luz a las ocho?", "gate"),
    ("¿podés prender la luz a las ocho?", "gate"),
    ("podes prender la luz a las ocho", "gate"),
])
async def test_scheduled_programming_cannot_turn_on_an_authorized_light(
    orchestrated, utterance, orchestrated_stage, tmp_path,
):
    async with authorized_light_router(orchestrated, utterance, tmp_path) as (router, dispatcher, ha, classify):
        event = CommandEvent(
            np.zeros(16000, dtype=np.float32), room_id="", capture_trigger="wake", wake_score=0.9,
            textual_wake=True,
        )

        result = await router.process_command(event)

        assert ha.sends == ha.reads == []
        assert result["intent"] == "routine"
        if orchestrated and orchestrated_stage != "gate":
            classify.assert_awaited_once()
        else:
            classify.assert_not_called()
        if orchestrated and orchestrated_stage == "dispatcher":
            assert result["path"] == "fast_routine"
        else:
            assert result.get("path") is None
        outcome = result["outcome"]
        assert (outcome["status"], outcome["persistence"], outcome["dispatch"]) == (
            "unsupported", "not_written", "not_sent",
        )
        assert not router.chroma.mock_calls and not router.llm.mock_calls
        dispatcher._slow_path.assert_not_called()


@asynccontextmanager
async def authorized_light_router(orchestrated, utterance, tmp_path):
    selection = tmp_path / "selection.json"
    write_selection(selection)
    ha = ObservedHA()
    actions = HomeActionService(ha, policy=HomeControlPolicy(selection_path=str(selection)))
    async with voice_router(orchestrated, utterance, home_actions=actions) as (router, dispatcher):
        command = {
            "domain": "light", "service": "turn_on", "entity_id": "light.test_lamp",
            "description": "luz de prueba", "similarity": 0.95,
        }
        router.chroma.search_command.return_value = command
        router.chroma.asearch_command = AsyncMock(return_value=command)
        classify = AsyncMock(return_value=CommandClassification(is_command=True, confidence=0.99, intent="turn_on"))
        router.llm_command_router = SimpleNamespace(classify=classify)
        yield router, dispatcher, ha, classify


@pytest.mark.asyncio
@pytest.mark.parametrize("orchestrated", [True, False])
@pytest.mark.parametrize("utterance", [
    "necesito que me digas a qué hora prenden las luces del estadio",
    "necesito que me digas a qué hora prenden las luces del estadio y apagan el marcador",
    "necesito que me digas a qué hora prenden las luces del estadio y ponen la música",
    "necesito que me digas qué partido empieza a las ocho",
    "quiero que me cuentes qué pasa cada vez que hay un empate",
    "después del último programa que vimos, ¿a las ocho entregaron el premio?",
    "después del último programa que vimos, a las ocho entregaron el premio?",
])
async def test_factual_questions_cannot_turn_on_an_authorized_light(orchestrated, utterance, tmp_path):
    async with authorized_light_router(orchestrated, utterance, tmp_path) as (router, dispatcher, ha, classify):
        dispatcher._slow_path = AsyncMock(return_value=DispatchResult(
            PathType.SLOW_LLM, Priority.LOW, True, "Respuesta sintética.", intent="conversation",
        ))
        event = CommandEvent(
            np.zeros(16000, dtype=np.float32), room_id="", capture_trigger="wake", wake_score=0.9,
            textual_wake=True,
        )

        result = await router.process_command(event)

        assert ha.sends == ha.reads == []
        assert (result["intent"], result["outcome"]["status"]) == ("conversation", "completed")
        assert not router.chroma.mock_calls
        if orchestrated:
            classify.assert_awaited_once()
            dispatcher._slow_path.assert_awaited_once()
        else:
            classify.assert_not_called()


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
    "quiero escuchar un excelente programa de rutinas de ejercicio",
    "quiero ver el primer programa que se emitió anoche",
    "quiero ver el polémico programa que ganó el premio",
    "después del último programa que vimos quiero hablar del premio",
    "para el próximo programa que grabemos buscá información sobre el premio",
    "después del último programa que escuché quiero hablar del premio",
    "para el próximo programa que vamos a grabar buscá información sobre el premio",
    "después del último programa que se emitió anoche, hablemos del premio",
    "para el programa que vamos a grabar mañana, me podés buscar información sobre el premio",
    "después del último programa que vimos, ¿quién ganó el premio?",
    "después del último programa que vimos, ¿a las ocho entregaron el premio?",
    "despues del ultimo programa que vimos, ¿a las ocho entregaron el premio?",
    "necesito que me digas qué partido empieza a las ocho",
    "necesito que me digas que partido empieza a las ocho",
    "quiero que me cuentes qué pasa cada vez que hay un empate",
    "quiero que me cuentes que pasa cada vez que hay un empate",
    "quiero que me expliques por que revisan las jugadas cada vez que hay un empate",
    "necesito que me confirmes si entregan el premio a las ocho",
    "quiero que lo revises y necesito que me digas qué partido empieza a las ocho",
    "quiero que verifiques a qué hora entregan el premio a las ocho",
    "cada vez que hay un empate quiero que me cuentes que pasa con el desempate",
    "después del último programa que vimos, necesito que me digas qué partido empieza a las ocho",
    "después del último programa que vimos quién ganó el premio",
    "para el próximo programa que vamos a grabar, ¿quién ganó el premio?",
    "para el próximo programa que vamos a grabar, quién ganó el premio",
    "para el próximo programa que vamos a grabar ¿qué premio ganó el invitado?",
    "para el próximo programa que vamos a grabar, cuándo pasa el programa a la radio",
    "después del último programa que escuché ¿quién ganó el premio?",
    "¿se configura el horario de la automatización de riego desde la web?",
    "podés decirme quién configura el horario de la automatización de riego",
    "¿Te parece que programa bien el horario la automatización de riego?",
])
async def test_utterance_without_automation_intent_stays_conversational(orchestrated, utterance):
    async with voice_router(orchestrated, utterance) as (router, dispatcher):
        router.chroma.search_command.return_value = None
        dispatcher._slow_path = AsyncMock(return_value=DispatchResult(
            PathType.SLOW_LLM, Priority.LOW, True, "Respuesta sintética.", intent="conversation",
        ))
        result = await router.process_command(np.zeros(16000, dtype=np.float32))
        assert result["intent"] == "conversation"
        assert result["outcome"]["status"] == "completed"
        assert not router.ha.mock_calls
        if orchestrated:
            assert result["path"] == "slow_llm"
            dispatcher._slow_path.assert_awaited_once()


@pytest.mark.parametrize("utterance", [
    "quiero escuchar el programa que ganó el premio",
    "quiero escuchar mi programa de rutinas de ejercicio",
    "cada vez que hay un empate se juega el desempate?",
    "cual es el nombre del programa que gano el premio",
    "quién configura el horario de la automatización de riego",
    "podés decirme quién normalmente configura el horario de la automatización de riego",
    "se configura que empiece de noche la automatización de riego?",
    "me podés decir si el horario de la automatización de riego se configura desde la web?",
    "¿sabés si ya se programa sola la automatización de riego?",
    "¿quién programa el horario de la automatización de riego?",
    "para el programa de rutinas de ejercicio",
    "para el excelente programa de rutinas de ejercicio",
    "quiero ver un insólito programa de rutinas de ejercicio",
    "quiero ver el primer programa que el público votó",
    "quiero ver el programa a las ocho sobre rutinas de ejercicio",
    "antes del polémico programa que ganó el premio, hablemos de la gala",
    "después del último programa que se emitió anoche, hablemos del premio",
    "luego del último programa que se transmitió anoche quiero hablar del premio",
    "para el nuevo programa que se estrenó anoche, buscá información sobre el elenco",
    "después del último programa que grabé quiero hablar del premio",
    "después del último programa que miré quiero hablar del premio",
    "después del último programa que vimos ayer",
    "después del último programa que escuche quiero hablar del premio",
])
def test_indirect_routine_cues_preserve_nominal_programs_and_questions(utterance):
    assert not is_unclaimed_routine_creation(f"Nexa, {utterance}")


def test_accented_program_verb_survives_nominal_filtering():
    assert is_unclaimed_routine_creation("Nexa, después del último, programá el horario de la automatización de riego")


@pytest.mark.parametrize("utterance", [
    "después del riego, programa ahora la rutina de jardín",
    "para el riego programa ahora el horario de la automatización",
    "después del riego programa dos rutinas para las ocho",
    "después del riego programa 2 rutinas para las ocho",
    "después del riego programa que arranque a las ocho",
    "después del último, programa ahora la rutina de jardín",
    "después del último programa, ahora la rutina de jardín",
    "después del riego programa de nuevo el horario de la automatización",
    "luego del riego programa de nuevo el horario de la automatización",
    "para el riego programa de nuevo el horario de la automatización",
    "antes del último programa que se encienda el riego",
    "luego de el último programa que se encienda el riego",
    "para el próximo programa a las ocho la rutina de jardín",
    "después del último programa a las ocho la rutina de jardín",
    "para el programa a las ocho la rutina de jardín",
    "después del último programa que cuando anochezca se encienda el riego",
    "para el jardín programa que grabemos al amanecer",
    "después del último programa que veamos que se encienda el riego",
    "para el próximo programa que veamos el momento en que se enciendan las luces",
    "para el próximo programa que vamos a ver que se encienda el riego",
    "para el próximo programa que vamos a ver si se enciende el riego al anochecer",
    "para el próximo programa que vamos a ver dónde se enciende el riego al anochecer",
    "para el próximo programa que vamos a ver hoy primero el momento en que se prenden las luces",
    "después del último programa que vimos anoche, quisiera que las luces prendan a las ocho",
    "para el próximo programa que vamos a ver nos enciendan las luces al salir el sol",
    "para el próximo programa que vamos a grabar mañana",
    "después del último programa que vimos quiero que me enciendan las luces",
    "después del último programa que vimos, me programás el riego",
    "para el último programa que vimos, necesito que funcione bien",
    "para el último programa que vimos, activar al caer el sol",
    "para el último programa que vimos, cerrar el acceso cada semana",
    "para el último programa que vimos, encendé las luces",
    "para el último programa que vimos, modificá el riego",
    "necesitamos que funcione al anochecer",
    "poné que empiece todos los días",
])
def test_program_directive_survives_nominal_filtering(utterance):
    assert is_unclaimed_routine_creation(f"Nexa, {utterance}")
