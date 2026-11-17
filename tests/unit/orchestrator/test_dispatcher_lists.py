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


# --------------------------------------------------------------------------
# P3-F2 owns R1 LIST CANCEL and this file (user answer Q30, "Como hoy"):
# whole-list delete and empty by voice keep today's handler (delete_list
# needs a name, clear_list empties it), now with voseo and accents; every C02
# counter is asserted and no other family claims these phrases.
# --------------------------------------------------------------------------

from src.nlu.household_intents import HouseholdFamily, classify_household_intent  # noqa: E402
from tests.mocks.c02_matrix import C02Entry, effects  # noqa: E402
from tests.mocks.household_dispatch import CountedLeaves  # noqa: E402


def _list(id_, text, status, persistence, mutations, spoken, *, effect_class="unauthorized_effect", note=""):
    return C02Entry(id_, text, "dispatcher", "list", effect_class, status, "fast_list", status, persistence, spoken,
                    effects(list_mutations=mutations), {"state": "shared_list_compras"}, note=note)


C02_LISTS = (
    _list("C02-LST-0101", "borrá la lista de compras", "completed", "committed", 2, "readback",
          note="delete_list removes the list and its item"),
    _list("C02-LST-0102", "eliminá la lista de compras", "completed", "committed", 2, "readback"),
    _list("C02-LST-0103", "borra la lista de compras", "completed", "committed", 2, "readback",
          note="today's tú form, unchanged"),
    _list("C02-LST-0104", "Nexa, eliminá la lista de compras.", "completed", "committed", 2, "readback"),
    _list("C02-LST-0105", "vaciá la lista de compras", "completed", "committed", 1, "readback"),
    _list("C02-LST-0106", "vacía la lista de compras", "completed", "committed", 1, "readback"),
    _list("C02-LST-0107", "limpiá la lista de compras", "completed", "committed", 1, "readback"),
    _list("C02-LST-0108", "borrá la lista de inexistente", "needs_input", "not_written", 0, "clarification"),
    _list("C02-LST-0109", "vaciá la lista de inexistente", "needs_input", "not_written", 0, "clarification"),
    _list("C02-LST-0110", "sacá la lista de compras", "needs_input", "not_written", 0, "clarification",
          note="ambiguous verb on the whole list: asks to delete or empty, zero effects"),
    _list("C02-LST-0111", "pará la lista de compras", "needs_input", "not_written", 0, "clarification"),
    _list("C02-LST-0112", "agregá leche a la lista de compras", "completed", "committed", 1, "readback",
          note="voseo add now reaches the add branch (before P3-F2 it listed the lists)"),
    _list("C02-LST-0113", "agregá pan y leche a la lista de compras", "completed", "committed", 1, "readback",
          note="an item enumeration is one list add, not a compound (C02-CMP-0008)"),
    _list("C02-LST-0114", "quitá el pan de la lista de compras", "completed", "committed", 1, "readback"),
    _list("C02-LST-0115", "sacá el pan de la lista de compras", "completed", "committed", 1, "readback"),
    _list("C02-LST-0116", "qué hay en la lista de compras", "completed", "not_required", 0, "answer"),
    _list("C02-LST-0117", "cancelá la lista de compras", "needs_input", "not_written", 0, "clarification"),
)


@pytest.fixture
async def counted(tmp_path):
    leaves = CountedLeaves(tmp_path)
    await leaves.start()
    yield leaves
    await leaves.close()


@pytest.mark.parametrize("entry", C02_LISTS, ids=lambda e: e.id)
async def test_c02_whole_list_and_item_entries(counted, entry):
    dispatcher = counted.dispatcher()
    await counted.apply_state(dispatcher, entry.conditions["state"])
    before = counted.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=entry.text)
    assert counted.effects(before) == entry.effects
    assert (result.path.value, result.outcome.status.value, result.outcome.persistence.value) == (
        entry.path, entry.task_status, entry.persistence)
    assert result.response.strip()
    assert classify_household_intent(entry.text).family is HouseholdFamily.LIST


async def test_whole_list_items_and_names_reach_the_store(counted):
    dispatcher = counted.dispatcher()
    await counted.apply_state(dispatcher, "shared_list_compras")
    added = await dispatcher.dispatch(user_id="unknown", text="agregá dulce de leche a la lista de compras")
    assert added.response == "Agregué dulce de leche"
    assert [i.text for i in await counted.list_manager.get_items(None, "compras", shared_only=True)] == [
        "pan", "dulce de leche"]
    emptied = await dispatcher.dispatch(user_id="unknown", text="vaciá la lista de compras")
    assert emptied.response == "Listo, vacié la lista"
    assert await counted.list_manager.get_items(None, "compras", shared_only=True) == []
    deleted = await dispatcher.dispatch(user_id="unknown", text="eliminá la lista de compras")
    assert deleted.response == "Borré la lista compras"
    assert await counted.list_manager.get_all_lists(None, shared_only=True) == []
