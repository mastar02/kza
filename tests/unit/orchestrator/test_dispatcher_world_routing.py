"""Weather routing, and the thermostat collision it must not cause."""

from unittest.mock import MagicMock

import pytest

from src.orchestrator.context_manager import ContextManager
from src.orchestrator.dispatcher import PathType, RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue


@pytest.fixture
def dispatcher():
    d = RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=MagicMock(),
        routine_manager=MagicMock(),
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
    )
    # Truthy stand-in so the music classification branch runs (gated on
    # `self.music`) — same pattern as test_dispatcher_music_routing.py.
    # Needed for the FAST_MUSIC cases in test_existing_paths_do_not_regress.
    d.music = MagicMock()
    return d


@pytest.mark.parametrize("text", [
    "qué tiempo hace",
    "cómo está el clima",
    "qué temperatura hace",
    "qué temperatura hace afuera",
    "cuántos grados hay afuera",
    "llueve mañana",
    "va a llover mañana",
    "cómo está el clima mañana",
    "el pronóstico para mañana",
])
def test_weather_questions_route_to_fast_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_WEATHER


@pytest.mark.parametrize("text", [
    # The collision this task exists for: these are the THERMOSTAT.
    "poné la temperatura en 22",
    "subí la temperatura",
    "bajá la temperatura del living",
    "prendé la calefacción",
    "apagá el aire",
])
def test_thermostat_commands_do_not_become_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_DOMOTICS


@pytest.mark.parametrize("text,expected", [
    ("poné música de Spinetta", PathType.FAST_MUSIC),
    ("subí el volumen", PathType.FAST_MUSIC),
    ("prendé la luz del living", PathType.FAST_DOMOTICS),
    ("encender la luz", PathType.FAST_DOMOTICS),
    ("agrega leche a la lista de compras", PathType.FAST_LIST),
    ("recuérdame sacar la basura", PathType.FAST_REMINDER),
    ("por qué el cielo es azul", PathType.SLOW_LLM),
])
def test_existing_paths_do_not_regress(dispatcher, text, expected):
    path, _ = dispatcher._classify_request(text)
    assert path == expected


def test_service_filter_still_wins(dispatcher):
    # An upstream high-confidence domotics classification always wins.
    path, _ = dispatcher._classify_request(
        "subí la temperatura", service_filter="turn_on"
    )
    assert path == PathType.FAST_DOMOTICS
