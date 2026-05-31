"""
Tests for media control command routing to music path.

Media-control verbs (pausá, volumen, siguiente, etc.) that pass the command
gate must route to FAST_MUSIC or SLOW_MUSIC, not to FAST_DOMOTICS.
"""

import pytest
from unittest.mock import MagicMock

from src.orchestrator.dispatcher import RequestDispatcher, PathType
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.orchestrator.context_manager import ContextManager


@pytest.fixture
def dispatcher_with_music():
    """RequestDispatcher with a truthy music dispatcher so music paths are enabled."""
    d = RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=MagicMock(),
        routine_manager=MagicMock(),
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
    )
    d.music = MagicMock()  # truthy stand-in — enables music classification branch
    return d


@pytest.mark.parametrize("text", [
    "pausá la música",
    "subí el volumen",
    "poné música",
    "siguiente canción",
])
def test_media_routes_to_music(dispatcher_with_music, text):
    """Media control commands must route to a music path, not domotics."""
    path, _ = dispatcher_with_music._classify_request(text.lower())
    assert path in (PathType.FAST_MUSIC, PathType.SLOW_MUSIC), (
        f"Expected FAST_MUSIC or SLOW_MUSIC for {text!r}, got {path}"
    )


def test_subir_volumen_routes_via_volumen_keyword(dispatcher_with_music):
    """'subí el volumen' must match via 'volumen', NOT via generic 'subí'.

    This ensures we don't accidentally mark all 'subí' commands as music
    (e.g. 'subí la persiana' must still be domotics).
    """
    # Music path is enabled
    path, _ = dispatcher_with_music._classify_request("subí el volumen")
    assert path in (PathType.FAST_MUSIC, PathType.SLOW_MUSIC)

    # Domotics with 'subí' must NOT be stolen by the music path
    dispatcher_with_music.music = None  # disable music
    path_no_music, _ = dispatcher_with_music._classify_request("subí la persiana")
    assert path_no_music == PathType.FAST_DOMOTICS


def test_subir_persiana_stays_domotics(dispatcher_with_music):
    """'subí la persiana' must remain domotics even when music is enabled."""
    path, _ = dispatcher_with_music._classify_request("subí la persiana")
    assert path == PathType.FAST_DOMOTICS, (
        f"'subí la persiana' should be FAST_DOMOTICS, got {path}"
    )


def test_pone_luz_stays_domotics(dispatcher_with_music):
    """'poné la luz' must stay domotics even when music is enabled."""
    path, _ = dispatcher_with_music._classify_request("poné la luz")
    assert path == PathType.FAST_DOMOTICS, (
        f"'poné la luz' should be FAST_DOMOTICS, got {path}"
    )
