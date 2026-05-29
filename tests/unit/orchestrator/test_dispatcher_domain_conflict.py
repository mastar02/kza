"""Tests: guarda de conflicto de dominio en el fast path (bug fantasma 2026-05-29).

Síntoma: `light.escritorio` prendía/apagaba sin comando. Causa raíz #2: un
intent no-luz (volumen/temperatura) caía al vector search con
prefer_area=Escritorio (zona del mic), que devolvía light.escritorio como
fallback y disparaba turn_on. La guarda rechaza el match de luz cuando el
texto pide explícitamente otro dominio y deja caer al router/slow path.

Ver project_escritorio_light_phantom_toggles_2026-05-29.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.orchestrator.dispatcher import (
    RequestDispatcher,
    _conflicting_domain,
)
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.orchestrator.context_manager import ContextManager


class TestConflictingDomainHelper:
    """Unit del helper puro `_conflicting_domain`."""

    @pytest.mark.parametrize("text,expected", [
        ("subí el volumen", "media_player"),
        ("Nexa subí el volumen", "media_player"),
        ("bajá el volume", "media_player"),
        ("bajá la temperatura del aire", "climate"),
        ("subí la temperatura", "climate"),
        ("prendé el aire", "climate"),
        ("poné el clima en veinte grados", "climate"),
        ("subí la calefacción", "climate"),
    ])
    def test_non_light_noun_conflicts_with_light_match(self, text, expected):
        assert _conflicting_domain(text, "light") == expected

    @pytest.mark.parametrize("text", [
        "poné la luz al cincuenta por ciento",
        "prendé la luz del escritorio",
        "apagá las luces",
        "subí la luz",            # 'subí' + 'luz' → es luz, no volumen
        "bajá la luz",
        "encendé el foco",
    ])
    def test_light_noun_present_means_no_conflict(self, text):
        # Si el usuario dijo 'luz'/'foco', confiamos en el match aunque haya
        # verbos compartidos con otros dominios (subí/bajá).
        assert _conflicting_domain(text, "light") is None

    def test_only_guards_light_matches(self):
        # Si el match NO es light, nunca hay conflicto (la guarda es asimétrica).
        assert _conflicting_domain("subí el volumen", "media_player") is None
        assert _conflicting_domain("subí el volumen", "climate") is None

    def test_empty_or_neutral_text_no_conflict(self):
        assert _conflicting_domain("", "light") is None
        assert _conflicting_domain("prendé el escritorio", "light") is None

    def test_word_boundary_avoids_substring_false_positive(self):
        # 'aire' no debe pegar dentro de otra palabra (ej: 'airear' inexistente
        # en dominio, pero validamos el boundary).
        assert _conflicting_domain("desairar el ambiente", "light") is None


def _light_command(entity_id="light.escritorio"):
    return {
        "domain": "light",
        "service": "turn_on",
        "entity_id": entity_id,
        "description": "Luz del escritorio encendida",
        "similarity": 0.82,
        "data": {},
        "capability": "onoff",
        "value_label": "prender",
    }


@pytest.fixture
def chroma_returns_light():
    cs = MagicMock()
    cs.search_command = MagicMock(return_value=_light_command())
    cs.asearch_command = AsyncMock(return_value=_light_command())
    return cs


@pytest.fixture
def ha_mock():
    ha = MagicMock()
    ha.call_service = MagicMock(return_value=True)
    ha.call_service_ws = AsyncMock(return_value=True)
    return ha


@pytest.fixture
def routine_mock():
    r = MagicMock()
    r.handle = AsyncMock(return_value={"handled": False, "response": "", "success": False})
    return r


@pytest.fixture
def dispatcher(chroma_returns_light, ha_mock, routine_mock):
    d = RequestDispatcher(
        chroma_sync=chroma_returns_light,
        ha_client=ha_mock,
        routine_manager=routine_mock,
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
        vector_threshold=0.65,
    )
    # Espía el disparo a HA para verificar que NO se prende la luz fantasma.
    d._fire_and_reconcile_ha = AsyncMock(return_value=None)
    return d


class TestDomainConflictBlocksPhantomLight:
    """Integración: 'subí el volumen' no debe disparar light.escritorio."""

    @pytest.mark.asyncio
    async def test_volume_command_does_not_fire_the_light(
        self, dispatcher, chroma_returns_light,
    ):
        result = await dispatcher.dispatch(
            user_id="unknown",
            text="Nexa subí el volumen",
            zone_id="zone_escritorio",
        )
        # El vector search corrió (devolvió light.escritorio)...
        assert chroma_returns_light.asearch_command.called
        # ...pero la guarda lo rechazó: NO se disparó la luz.
        assert not dispatcher._fire_and_reconcile_ha.called, (
            "El comando de volumen NO debe prender light.escritorio"
        )
        # ...y devuelve un resultado INMEDIATO y honesto (no cuelga en slow path
        # ni miente con timeout): success=False, intent dedicado, sin fingir éxito.
        assert result.intent == "domain_conflict"
        assert result.success is False
        assert result.response  # mensaje accionable, no vacío
        assert "volumen" in result.response.lower()

    @pytest.mark.asyncio
    async def test_temperature_command_does_not_fire_the_light(
        self, dispatcher,
    ):
        result = await dispatcher.dispatch(
            user_id="unknown",
            text="Nexa bajá la temperatura del aire",
            zone_id="zone_escritorio",
        )
        assert not dispatcher._fire_and_reconcile_ha.called
        assert result.intent == "domain_conflict"
        assert result.success is False
        assert "temperatura" in result.response.lower()

    @pytest.mark.asyncio
    async def test_conflict_does_not_enqueue_slow_path(self, dispatcher):
        # Regresión del hallazgo del silent-failure-hunter: el rechazo NO debe
        # caer al slow path (que colgaría 5s y devolvería un timeout falso).
        result = await dispatcher.dispatch(
            user_id="unknown",
            text="Nexa subí el volumen",
            zone_id="zone_escritorio",
        )
        assert len(dispatcher.queue) == 0, (
            "El conflicto de dominio NO debe encolar una petición de slow path"
        )
        assert result.intent != "timeout"

    @pytest.mark.asyncio
    async def test_legitimate_light_command_still_fires(
        self, dispatcher,
    ):
        # Control: un comando de luz real SÍ debe disparar (sin regresión).
        await dispatcher.dispatch(
            user_id="unknown",
            text="Nexa prendé la luz del escritorio",
            zone_id="zone_escritorio",
        )
        assert dispatcher._fire_and_reconcile_ha.called, (
            "Un comando de luz legítimo debe seguir disparando la acción HA"
        )
