"""Tests: un fallo de HA en el fire-and-forget debe ser audible (earcon).

Root cause (2026-07-25): el primer comando después de un hueco idle falló y el
usuario no recibió NINGÚN feedback. El log de producción:

    16:01:11.680 [HA-CALL] light.turn_on@light.grupo_escritorio success=False took=8ms
    16:01:11.680 HA fire-and-forget falló en light.turn_on@... sin response_handler
                 — usuario no fue notificado

Causa: `MultiUserOrchestrator.__init__` no aceptaba `response_handler` ni lo
pasaba al `RequestDispatcher` que construye. `RequestRouter` y `VoicePipeline`
sí lo reciben; el dispatcher del orquestador nunca. Por eso
`self.response_handler is None` SIEMPRE en producción y el fallo moría en un
WARNING.

Decisión (sesión 2026-07-25): earcon, no frase. La regla "domótica silenciosa"
aplica a los ÉXITOS que el usuario valida visualmente; un fallo silencioso es
justo lo que hace el bug invisible.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.orchestrator.dispatcher import RequestDispatcher, MultiUserOrchestrator


@pytest.fixture
def response_handler():
    rh = MagicMock()
    rh.play_earcon = MagicMock()
    rh.speak = MagicMock()
    return rh


def _dispatcher(ha_client, response_handler=None):
    return RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=ha_client,
        routine_manager=MagicMock(),
        response_handler=response_handler,
    )


class TestEarconOnFailure:
    """El fallo de HA dispara earcon en la zona del comando."""

    @pytest.mark.asyncio
    async def test_failed_call_plays_earcon(self, response_handler):
        ha = MagicMock()
        ha.call_service_ws = AsyncMock(return_value=False)
        d = _dispatcher(ha, response_handler)

        await d._fire_and_reconcile_ha({
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.grupo_escritorio",
            "description": "la luz del escritorio",
            "zone_id": "escritorio",
        })

        response_handler.play_earcon.assert_called_once()

    @pytest.mark.asyncio
    async def test_earcon_targets_the_command_zone(self, response_handler):
        """El earcon suena donde habló el usuario, no en todas las zonas."""
        ha = MagicMock()
        ha.call_service_ws = AsyncMock(return_value=False)
        d = _dispatcher(ha, response_handler)

        await d._fire_and_reconcile_ha({
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.grupo_cocina",
            "zone_id": "cocina",
        })

        kwargs = response_handler.play_earcon.call_args.kwargs
        args = response_handler.play_earcon.call_args.args
        assert "cocina" in (list(args) + list(kwargs.values())), (
            f"earcon no ruteado a la zona del comando: args={args} kwargs={kwargs}"
        )

    @pytest.mark.asyncio
    async def test_failed_call_does_not_speak(self, response_handler):
        """Decisión: earcon SIN frase. No debe hablar el error."""
        ha = MagicMock()
        ha.call_service_ws = AsyncMock(return_value=False)
        d = _dispatcher(ha, response_handler)

        await d._fire_and_reconcile_ha({
            "domain": "light", "service": "turn_on",
            "entity_id": "light.x", "description": "la luz",
        })

        response_handler.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_also_plays_earcon(self, response_handler):
        """Una excepción de HA es tan invisible como un success=False."""
        ha = MagicMock()
        ha.call_service_ws = AsyncMock(side_effect=RuntimeError("WS muerto"))
        d = _dispatcher(ha, response_handler)

        await d._fire_and_reconcile_ha({
            "domain": "light", "service": "turn_on", "entity_id": "light.x",
        })

        response_handler.play_earcon.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_call_stays_silent(self, response_handler):
        """Regla vigente: éxito de domótica = silencio (validación visual)."""
        ha = MagicMock()
        ha.call_service_ws = AsyncMock(return_value=True)
        d = _dispatcher(ha, response_handler)

        await d._fire_and_reconcile_ha({
            "domain": "light", "service": "turn_on", "entity_id": "light.x",
        })

        response_handler.play_earcon.assert_not_called()
        response_handler.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_earcon_failure_does_not_propagate(self, response_handler):
        """Si el earcon falla, no debe tumbar la task de reconciliación."""
        ha = MagicMock()
        ha.call_service_ws = AsyncMock(return_value=False)
        response_handler.play_earcon.side_effect = RuntimeError("sin audio")
        d = _dispatcher(ha, response_handler)

        await d._fire_and_reconcile_ha({
            "domain": "light", "service": "turn_on", "entity_id": "light.x",
        })  # no debe levantar


class TestOrchestratorWiresResponseHandler:
    """El seam que hacía el fallo inaudible en producción."""

    def test_orchestrator_forwards_response_handler_to_dispatcher(
        self, response_handler
    ):
        """Sin esto, dispatcher.response_handler es None SIEMPRE en prod."""
        orch = MultiUserOrchestrator(
            chroma_sync=MagicMock(),
            ha_client=MagicMock(),
            routine_manager=MagicMock(),
            response_handler=response_handler,
        )

        assert orch.dispatcher.response_handler is response_handler

    def test_orchestrator_without_response_handler_still_builds(self):
        """Backward compat: el parámetro es opcional."""
        orch = MultiUserOrchestrator(
            chroma_sync=MagicMock(),
            ha_client=MagicMock(),
            routine_manager=MagicMock(),
        )

        assert orch.dispatcher.response_handler is None
