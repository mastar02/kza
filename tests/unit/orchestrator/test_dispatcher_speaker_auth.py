"""Tests: voice-auth opcional en el fast path (require_known_speaker_for_actions).

Defensa adicional contra disparos fantasma (todos User=unknown). Config-gated,
OFF por default: con el flag apagado el comportamiento no cambia.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.orchestrator.dispatcher import RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.orchestrator.context_manager import ContextManager


def _light_command():
    return {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.escritorio",
        "description": "Luz del escritorio encendida",
        "similarity": 0.9,
        "data": {},
        "capability": "onoff",
        "value_label": "prender",
    }


def _dispatcher(require_known: bool):
    chroma = MagicMock()
    chroma.asearch_command = AsyncMock(return_value=_light_command())
    routine = MagicMock()
    routine.handle = AsyncMock(return_value={"handled": False, "response": "", "success": False})
    d = RequestDispatcher(
        chroma_sync=chroma,
        ha_client=MagicMock(),
        routine_manager=routine,
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
        vector_threshold=0.65,
        require_known_speaker_for_actions=require_known,
    )
    d._fire_and_reconcile_ha = AsyncMock(return_value=None)
    return d


class TestSpeakerAuthGate:
    @pytest.mark.asyncio
    async def test_flag_off_unknown_speaker_still_fires(self):
        # Default OFF: sin cambio de comportamiento.
        d = _dispatcher(require_known=False)
        await d.dispatch(user_id="unknown", text="prendé la luz del escritorio", zone_id="zone_escritorio")
        assert d._fire_and_reconcile_ha.called

    @pytest.mark.asyncio
    async def test_flag_on_unknown_speaker_blocked(self):
        d = _dispatcher(require_known=True)
        result = await d.dispatch(user_id="unknown", text="prendé la luz del escritorio", zone_id="zone_escritorio")
        assert not d._fire_and_reconcile_ha.called, "speaker desconocido NO debe disparar la acción"
        assert result.intent == "speaker_auth"
        assert result.success is False

    @pytest.mark.asyncio
    async def test_flag_on_known_speaker_fires(self):
        d = _dispatcher(require_known=True)
        await d.dispatch(user_id="juan", text="prendé la luz del escritorio", zone_id="zone_escritorio")
        assert d._fire_and_reconcile_ha.called, "speaker enrolado SÍ debe disparar"
