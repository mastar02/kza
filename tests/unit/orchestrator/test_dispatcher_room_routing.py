"""Tests for dispatcher passing room/intent context to chroma.search_command.

Reproduce the routing bug from logs 2026-05-03 01:25 / 08:08:
- Wake text: 'Nexa bajá la luz al cincuenta por ciento' (no room mention)
- room_context resolved as Escritorio (mic-based)
- Dispatcher called chroma.search_command(text, threshold) WITHOUT
  passing zone_id/room nor service_filter — so light.cuarto won
  on raw embedding similarity and slot 50% was lost.

Post-fix: dispatcher must propagate (a) prefer_area derived from zone_id
when text has no literal room alias, and (b) service_filter+slots from
the LLMRouter classification.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.orchestrator.dispatcher import (
    RequestDispatcher,
)
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.orchestrator.context_manager import ContextManager


# Areas configuradas en el sistema (matchea metadata.area en Chroma).
KNOWN_AREAS = ("Cuarto", "Escritorio", "Living", "Cocina", "Baño", "Hall")


@pytest.fixture
def chroma_with_search():
    _result = {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.escritorio",
        "description": "Luz del escritorio encendida",
        "similarity": 0.88,
        "data": {},
        "capability": "onoff",
        "value_label": "prender",
    }
    cs = MagicMock()
    cs.search_command = MagicMock(return_value=_result)
    cs.asearch_command = AsyncMock(return_value=_result)
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
def dispatcher(chroma_with_search, ha_mock, routine_mock):
    return RequestDispatcher(
        chroma_sync=chroma_with_search,
        ha_client=ha_mock,
        routine_manager=routine_mock,
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
        vector_threshold=0.65,
    )


class TestDispatcherPropagatesRoomToSearch:
    """Decisión 1-B + 3-C: dispatcher debe pasar prefer_area al search_command."""

    @pytest.mark.asyncio
    async def test_zone_id_propagates_as_prefer_area_when_text_has_no_room(
        self, dispatcher, chroma_with_search,
    ):
        """Texto sin room → zone_id mic → prefer_area = area del room.

        Caso real: 'Nexa bajá la luz al cincuenta por ciento' desde
        zone_escritorio. El dispatcher debe pasar prefer_area='Escritorio'
        al search_command para que el re-scoring favorezca light.escritorio
        sobre light.cuarto.
        """
        await dispatcher.dispatch(
            user_id="unknown",
            text="Nexa bajá la luz al cincuenta por ciento",
            zone_id="zone_escritorio",
        )
        assert chroma_with_search.asearch_command.called, (
            "asearch_command should have been called for FAST_DOMOTICS path"
        )
        kwargs = chroma_with_search.asearch_command.call_args.kwargs
        assert kwargs.get("prefer_area") == "Escritorio", (
            f"Expected prefer_area='Escritorio' (derived from zone_escritorio), "
            f"got {kwargs.get('prefer_area')!r}"
        )

    @pytest.mark.asyncio
    async def test_text_with_literal_room_alias_overrides_mic(
        self, dispatcher, chroma_with_search,
    ):
        """Si el texto menciona 'living' explícito → prefer_area='Living', no Escritorio.

        Implementación de la decisión 1-B: el room hablado (literal,
        no embedding) gana sobre el mic.
        """
        await dispatcher.dispatch(
            user_id="unknown",
            text="prendé la luz del living",
            zone_id="zone_escritorio",
        )
        kwargs = chroma_with_search.asearch_command.call_args.kwargs
        assert kwargs.get("prefer_area") == "Living", (
            f"Literal 'living' in text must win over mic zone. "
            f"Got prefer_area={kwargs.get('prefer_area')!r}"
        )

    @pytest.mark.asyncio
    async def test_no_zone_no_alias_means_no_prefer_area(
        self, dispatcher, chroma_with_search,
    ):
        """Sin zone_id ni alias literal → prefer_area=None (sin restricción)."""
        await dispatcher.dispatch(
            user_id="unknown",
            text="prendé la luz",
            zone_id=None,
        )
        kwargs = chroma_with_search.asearch_command.call_args.kwargs
        assert kwargs.get("prefer_area") is None

    @pytest.mark.asyncio
    async def test_unknown_zone_does_not_pass_prefer_area(
        self, dispatcher, chroma_with_search,
    ):
        """zone_id que no mapea a ningún area conocida → prefer_area=None.

        Evita pasar strings basura a Chroma. El zone-to-area mapping debe
        ser explícito y conocido.
        """
        await dispatcher.dispatch(
            user_id="unknown",
            text="prendé la luz",
            zone_id="zone_inexistente",
        )
        kwargs = chroma_with_search.asearch_command.call_args.kwargs
        assert kwargs.get("prefer_area") is None


class TestDispatcherPassesIntentAsServiceFilter:
    """Decisión 4: portar service_filter del LLMRouter al dispatcher orchestrated.

    Hoy el path orchestrated descarta intent del LLMRouter; el legacy lo
    usa como service_filter. Después del fix, el orchestrated también.
    """

    @pytest.mark.asyncio
    async def test_dispatch_accepts_service_filter_kwarg(
        self, dispatcher, chroma_with_search,
    ):
        """RequestDispatcher.dispatch debe aceptar service_filter."""
        # The new kwarg must be accepted; if not, this raises TypeError.
        await dispatcher.dispatch(
            user_id="unknown",
            text="bajá la luz",
            zone_id="zone_escritorio",
            service_filter="turn_off",
        )
        kwargs = chroma_with_search.asearch_command.call_args.kwargs
        assert kwargs.get("service_filter") == "turn_off"

    @pytest.mark.asyncio
    async def test_dispatch_accepts_query_slots_kwarg(
        self, dispatcher, chroma_with_search,
    ):
        """RequestDispatcher.dispatch debe aceptar query_slots."""
        await dispatcher.dispatch(
            user_id="unknown",
            text="bajá la luz al cincuenta por ciento",
            zone_id="zone_escritorio",
            service_filter="set_brightness",
            query_slots={"brightness_pct": 50},
        )
        kwargs = chroma_with_search.asearch_command.call_args.kwargs
        assert kwargs.get("query_slots") == {"brightness_pct": 50}


class TestLivingMangledBySTT:
    """El STT rompe "living" far-field; el alias no puede cortarse acá.

    `_resolve_prefer_area` busca los aliases en el TEXTO, así que si la grammar
    resuelve room=living pero el dispatcher no reconoce "libby", el comando
    pierde el prefer_area y el vector search cae a la zona del micrófono:
    pedís el living y prende el escritorio (observado 2026-07-30, ~43% de los
    pedidos al living).
    """

    def test_libby_resuelve_a_living(self):
        from src.orchestrator.dispatcher import _resolve_prefer_area
        assert _resolve_prefer_area("nexa prende la luz del libby", None) == "Living"

    def test_el_mic_no_le_gana_al_destrozo(self):
        """Aun hablando desde el escritorio, "libby" manda al living."""
        from src.orchestrator.dispatcher import _resolve_prefer_area
        assert _resolve_prefer_area(
            "prende la luz del libby", "zone_escritorio"
        ) == "Living"

    def test_las_demas_rooms_siguen_intactas(self):
        from src.orchestrator.dispatcher import _resolve_prefer_area
        assert _resolve_prefer_area("prende la luz de la cocina", None) == "Cocina"
        assert _resolve_prefer_area("prende la luz del escritorio", None) == "Escritorio"


class TestGlobalScopeMergesQuerySlots:
    """El bloque de scope global ("toda la casa") bypassa el vector search,
    que es donde se mergean los query_slots (review 2026-08-09): "luces de
    toda la casa al 50%" prendía todo al brillo anterior y tiraba el 50% en
    silencio. Alcanzable desde que set/set_brightness rutean FAST_DOMOTICS."""

    @pytest.mark.asyncio
    async def test_turn_on_global_lleva_los_slots(self, dispatcher):
        result = await dispatcher.dispatch(
            user_id="unknown",
            text="nexa luces de toda la casa al 50 por ciento",
            zone_id="zone_escritorio",
            service_filter="turn_on",
            query_slots={"brightness_pct": 50},
        )
        assert result.action["entity_id"] == "light.hogar"
        assert result.action["service"] == "turn_on"
        assert result.action["data"] == {"brightness_pct": 50}

    @pytest.mark.asyncio
    async def test_turn_off_global_no_arrastra_slots(self, dispatcher):
        # brightness no es service_data de turn_off — un slot residual del
        # NLU no debe colarse en el apagado.
        result = await dispatcher.dispatch(
            user_id="unknown",
            text="nexa apagá las luces de toda la casa",
            zone_id="zone_escritorio",
            service_filter="turn_off",
            query_slots={"brightness_pct": 50},
        )
        assert result.action["entity_id"] == "light.hogar"
        assert result.action["service"] == "turn_off"
        assert result.action["data"] == {}

    @pytest.mark.asyncio
    async def test_slots_desconocidos_no_llegan_a_ha(self, dispatcher):
        # merge_service_data filtra contra el whitelist: una clave libre del
        # LLM haría que HA rechace el service_data completo.
        result = await dispatcher.dispatch(
            user_id="unknown",
            text="nexa luces de toda la casa al 50 por ciento",
            zone_id="zone_escritorio",
            service_filter="turn_on",
            query_slots={"brightness_pct": 50, "modo_fiesta": True},
        )
        assert result.action["data"] == {"brightness_pct": 50}
