"""Tests for list and reminder dispatch integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.orchestrator.dispatcher import RequestDispatcher, PathType


@pytest.fixture
def dispatcher():
    d = RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=AsyncMock(),
        routine_manager=MagicMock(),
    )
    return d


class TestClassifyRequest:
    def test_add_to_list(self, dispatcher):
        path, _ = dispatcher._classify_request("agrega leche a la lista de compras")
        assert path == PathType.FAST_LIST

    def test_remove_from_list(self, dispatcher):
        path, _ = dispatcher._classify_request("quita el pan de la lista")
        assert path == PathType.FAST_LIST

    def test_what_on_list(self, dispatcher):
        path, _ = dispatcher._classify_request("qué hay en la lista de compras")
        assert path == PathType.FAST_LIST

    def test_reminder_set(self, dispatcher):
        path, _ = dispatcher._classify_request("recuérdame a las 5 sacar la basura")
        assert path == PathType.FAST_REMINDER

    def test_reminder_cancel(self, dispatcher):
        path, _ = dispatcher._classify_request("cancela el recordatorio de la basura")
        assert path == PathType.FAST_REMINDER

    def test_what_reminders(self, dispatcher):
        path, _ = dispatcher._classify_request("qué recordatorios tengo")
        assert path == PathType.FAST_REMINDER

    def test_what_pending_today(self, dispatcher):
        path, _ = dispatcher._classify_request("qué tengo pendiente hoy")
        assert path == PathType.FAST_REMINDER

    def test_every_monday(self, dispatcher):
        path, _ = dispatcher._classify_request("todos los lunes recuérdame poner la ropa")
        assert path == PathType.FAST_REMINDER

    def test_regular_domotics_still_works(self, dispatcher):
        path, _ = dispatcher._classify_request("prende la luz del living")
        assert path == PathType.FAST_DOMOTICS
