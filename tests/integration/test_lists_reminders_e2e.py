"""Integration tests for lists and reminders end-to-end flow."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, closing
import sqlite3
import time
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.lists.list_store import ListStore
from src.lists.list_manager import ListManager
from src.lists.ha_sync import HASyncManager
from src.reminders.reminder_store import ReminderStore, ReminderScope
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler
from src.users.user_manager import PermissionLevel, UserManager
from tests.mocks.home_control import selected_actions
from tests.mocks.responses import voice_output


@pytest_asyncio.fixture
async def list_system(tmp_path):
    """Full list system with store, manager, and HA sync."""
    store = ListStore(str(tmp_path / "lists.db"))
    await store.initialize()
    ha_client = AsyncMock()
    ha_sync = HASyncManager(ha_client=ha_client, entity_prefix="todo.kza")
    manager = ListManager(store=store, ha_client=ha_client, config={
        "default_list_name": "compras",
        "ha_sync_enabled": True,
    })
    yield {"store": store, "manager": manager, "ha_sync": ha_sync, "ha_client": ha_client}
    await store.close()


@pytest_asyncio.fixture
async def reminder_system(tmp_path):
    """Full reminder system with store, manager, and scheduler."""
    wake = asyncio.Event()
    store = ReminderStore(str(tmp_path / "reminders.db"), on_created=wake.set)
    await store.initialize()
    tts = voice_output()
    presence = MagicMock()
    presence.get_user_zone = MagicMock(return_value="living")
    ha_client = AsyncMock()
    actions = selected_actions(ha_client, ("light.jardin",))
    manager = ReminderManager(store=store, config={"tts_prefix": "Oye, recuerda"}, home_actions=actions)
    scheduler = ReminderScheduler(
        store=store, tts=tts, presence_detector=presence, home_actions=actions,
        ha_client=ha_client, config={"notice_max_attempts": 2, "notice_retry_min_gap_s": 1, "tts_prefix": "Oye, recuerda", "output_zone": "living"},
        wake_event=wake,
    )
    yield {
        "store": store,
        "manager": manager,
        "scheduler": scheduler,
        "tts": tts,
        "ha_client": ha_client,
        "presence": presence,
    }
    if scheduler._running:
        await scheduler.stop()
    await store.close()


class TestListFlow:
    """Test complete list lifecycle."""

    @pytest.mark.asyncio
    async def test_full_list_lifecycle(self, list_system):
        mgr = list_system["manager"]

        # Create list, add items
        await mgr.add_item(user_id="u1", item_text="leche")
        await mgr.add_item(user_id="u1", item_text="pan integral")
        await mgr.add_item(user_id="u1", item_text="huevos")

        # Query items
        items = await mgr.get_items(user_id="u1")
        assert len(items) == 3

        # Remove by fuzzy match
        removed = await mgr.remove_item(user_id="u1", item_text="pan")
        assert removed is True
        items = await mgr.get_items(user_id="u1")
        assert len(items) == 2

        # Clear list
        await mgr.clear_list(user_id="u1")
        items = await mgr.get_items(user_id="u1")
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_shared_list_between_users(self, list_system):
        mgr = list_system["manager"]

        # User 1 creates shared list
        await mgr.create_list(user_id="u1", list_name="casa", shared=True)
        await mgr.add_item(user_id="u1", item_text="jabon", list_name="casa")

        # User 2 can see and add to it
        items = await mgr.get_items(user_id="u2", list_name="casa")
        assert len(items) == 1
        await mgr.add_item(user_id="u2", item_text="detergente", list_name="casa")
        items = await mgr.get_items(user_id="u1", list_name="casa")
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_multiple_lists_per_user(self, list_system):
        mgr = list_system["manager"]

        await mgr.create_list(user_id="u1", list_name="oficina")
        await mgr.add_item(user_id="u1", item_text="leche")  # default "compras"
        await mgr.add_item(user_id="u1", item_text="lapiz", list_name="oficina")

        compras = await mgr.get_items(user_id="u1")
        oficina = await mgr.get_items(user_id="u1", list_name="oficina")
        assert len(compras) == 1
        assert len(oficina) == 1


class TestReminderFlow:
    """Test complete reminder lifecycle."""

    @pytest.mark.asyncio
    async def test_one_shot_reminder_fires(self, reminder_system):
        mgr = reminder_system["manager"]
        scheduler = reminder_system["scheduler"]
        tts = reminder_system["tts"]

        # Create reminder in the past (should fire immediately)
        r = await mgr.create(user_id="u1", text="sacar basura", trigger_at=time.time() - 10, scope=ReminderScope.SHARED)

        # Fire it through scheduler
        reminder = await reminder_system["store"].get_by_id(r.id)
        await scheduler._fire_reminder(reminder)

        # Verify TTS was called and reminder marked fired
        tts.aspeak.assert_awaited_once()
        assert "sacar basura" in tts.aspeak.call_args[0][0]
        updated = await reminder_system["store"].get_by_id(r.id)
        assert updated.state == "fired"

    @pytest.mark.asyncio
    async def test_recurring_reminder_reschedules(self, reminder_system):
        mgr = reminder_system["manager"]
        scheduler = reminder_system["scheduler"]

        r = await mgr.create(user_id="u1", text="pastilla", trigger_at=time.time() - 10, recurrence="daily", scope=ReminderScope.SHARED)
        reminder = await reminder_system["store"].get_by_id(r.id)
        await scheduler._fire_reminder(reminder)

        updated = await reminder_system["store"].get_by_id(r.id)
        assert updated.state == "active"
        assert updated.trigger_at > time.time()  # Rescheduled to tomorrow

    @pytest.mark.asyncio
    async def test_reminder_with_ha_actions(self, reminder_system):
        mgr = reminder_system["manager"]
        scheduler = reminder_system["scheduler"]
        ha = reminder_system["ha_client"]

        actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.jardin"}]
        r = await mgr.create(user_id="u1", text="regar plantas", trigger_at=time.time() - 10, ha_actions=actions, scope=ReminderScope.SHARED)

        reminder = await reminder_system["store"].get_by_id(r.id)
        await scheduler._fire_reminder(reminder)

        ha.call_service_once.assert_called_once_with("light", "turn_on", "light.jardin", None)

    @pytest.mark.asyncio
    async def test_cancel_reminder_by_fuzzy_text(self, reminder_system):
        mgr = reminder_system["manager"]

        await mgr.create(user_id="u1", text="sacar la basura", trigger_at=time.time() + 3600)
        await mgr.create(user_id="u1", text="ir al dentista", trigger_at=time.time() + 7200)

        result = await mgr.cancel_by_text(user_id="u1", search_text="basura")
        assert result is True

        active = await mgr.get_active(user_id="u1")
        assert len(active) == 1
        assert active[0].text == "ir al dentista"


@asynccontextmanager
async def personal_tasks_dashboard(tmp_path):
    async with AsyncExitStack() as stack:
        users = UserManager(str(tmp_path / "users.json"), retirement_proof=lambda ids: authority.users_retired(ids))
        authority = DashboardAuthority(tmp_path / "auth.db", account_provider=users.active_accounts)
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        lists = ListStore(str(tmp_path / "lists.db"))
        reminders = ReminderStore(str(tmp_path / "reminders.db"))
        for store in (lists, reminders):
            stack.push_async_callback(store.close)
            await store.initialize()
        ha, tts = AsyncMock(), voice_output()
        actions = selected_actions(ha, ("light.jardin",))
        scheduler = ReminderScheduler(reminders, tts, None, ha, config={"output_zone": "living"},
                                      home_actions=actions)
        api = DashboardAPI(authority=authority, user_manager=users, list_manager=ListManager(lists),
                           reminder_manager=ReminderManager(reminders, home_actions=actions),
                           reminder_scheduler=scheduler)
        await stack.enter_async_context(api.app.router.lifespan_context(api.app))
        client = await stack.enter_async_context(AsyncClient(
            transport=ASGITransport(app=api.app), base_url="http://testserver"))
        yield SimpleNamespace(users=users, authority=authority, client=client, reminders=reminders,
                              scheduler=scheduler, ha=ha, tts=tts)


def stored_ids(path, table):
    with closing(sqlite3.connect(path)) as db:
        return {row[0] for row in db.execute(f"SELECT id FROM {table}")}


@pytest.mark.asyncio
async def test_personal_task_retirement_survives_restoring_compatible_stores(tmp_path, dashboard_auth):
    lists_path, reminders_path = tmp_path / "lists.db", tmp_path / "reminders.db"
    lists_copy, reminders_copy = tmp_path / "lists.copy", tmp_path / "reminders.copy"
    def copy_database(source, destination):
        with closing(sqlite3.connect(source)) as original, closing(sqlite3.connect(destination)) as copy:
            original.backup(copy)

    async with personal_tasks_dashboard(tmp_path) as graph:
        users, authority, client = graph.users, graph.authority, graph.client
        retired, _ = await asyncio.to_thread(users.add_user, "Synthetic person", PermissionLevel.ADULT)
        survivor, _ = await asyncio.to_thread(users.add_user, "Synthetic survivor", PermissionLevel.ADULT)
        admin = authority.resolve(dashboard_auth["Authorization"])
        person_headers = {}
        for person in (retired, survivor):
            credential = await authority.issue_credential(
                admin, subject=person.user_id, role=DashboardRole.HOME, user_id=person.user_id,
                expires_at=time.time() + 3600)
            person_headers[person.user_id] = {"Authorization": f"Bearer {credential.token}"}
        owner, other = person_headers[retired.user_id], person_headers[survivor.user_id]

        async def post(url, headers, **body):
            response = await client.post(url, headers=headers, json=body)
            assert response.status_code == 200, response.text
            return response.json()

        personal = await post("/api/lists", owner, name="personal list", shared=False)
        item = await post(f"/api/lists/{personal['id']}/items", owner, text="personal item")
        shared = await post("/api/lists", dashboard_auth, name="shared list", shared=True)
        shared_item = await post(f"/api/lists/{shared['id']}/items", dashboard_auth, text="shared item")
        pending = await post("/api/reminders", owner, text="personal reminder", trigger_at=1.)
        other_job = await post("/api/reminders", other, text="other reminder", trigger_at=2.)
        shared_job = await post("/api/reminders", dashboard_auth, text="shared reminder",
                                trigger_at=3., scope="shared")
        for resource, identity in (("lists", personal["id"]), ("reminders", pending["id"])):
            response = await client.get(f"/api/{resource}", headers=owner, params={"scope": "personal"})
            assert response.status_code == 200 and [row["id"] for row in response.json()] == [identity]
            assert (await client.get(f"/api/{resource}", headers=dashboard_auth,
                                     params={"scope": "personal"})).status_code == 403
            response = await client.get(f"/api/{resource}", headers=other)
            assert response.status_code == 200 and identity not in {row["id"] for row in response.json()}
        assert (await client.get(f"/api/lists/{personal['id']}/items", headers=other)).status_code == 404

        await asyncio.to_thread(copy_database, lists_path, lists_copy)
        await asyncio.to_thread(copy_database, reminders_path, reminders_copy)
        response = await client.delete(f"/api/admin/users/{retired.user_id}", headers=dashboard_auth)
        assert response.status_code == 200 and response.json()["ok"], response.text
        assert authority.users_retired(frozenset({retired.user_id}))
        assert users.get_user(retired.user_id) is None
        assert (await client.get("/api/reminders", headers=owner)).status_code == 401

    assert stored_ids(lists_path, "lists") == {shared["id"]}
    assert stored_ids(lists_path, "list_items") == {shared_item["id"]}
    assert stored_ids(reminders_path, "reminders") == {other_job["id"], shared_job["id"]}

    # Recover only the compatible task stores. The authority DB and its
    # retirement mark remain current, as required by a compatible restore.
    await asyncio.to_thread(copy_database, lists_copy, lists_path)
    await asyncio.to_thread(copy_database, reminders_copy, reminders_path)
    assert personal["id"] in stored_ids(lists_path, "lists")
    assert item["id"] in stored_ids(lists_path, "list_items")
    assert pending["id"] in stored_ids(reminders_path, "reminders")

    async with personal_tasks_dashboard(tmp_path) as graph:
        authority, client = graph.authority, graph.client
        assert authority.users_retired(frozenset({retired.user_id}))
        assert (await client.get("/api/lists", headers=owner)).status_code == 401
        assert (await client.get(f"/api/lists/{personal['id']}/items", headers=other)).status_code == 404
        assert stored_ids(lists_path, "lists") == {shared["id"]}
        assert stored_ids(lists_path, "list_items") == {shared_item["id"]}
        assert stored_ids(reminders_path, "reminders") == {other_job["id"], shared_job["id"]}


@pytest.mark.asyncio
async def test_account_retirement_quarantines_tasks_after_second_store_failure(tmp_path, dashboard_auth):
    async with personal_tasks_dashboard(tmp_path) as graph:
        users, authority, client = graph.users, graph.authority, graph.client
        person, _ = await asyncio.to_thread(users.add_user, "Synthetic person", PermissionLevel.ADULT)
        other, _ = await asyncio.to_thread(users.add_user, "Synthetic other", PermissionLevel.ADULT)
        admin = authority.resolve(dashboard_auth["Authorization"])
        tokens = []
        for account in (person, other):
            credential = await authority.issue_credential(
                admin, subject=account.user_id, role=DashboardRole.HOME, user_id=account.user_id,
                expires_at=time.time() + 3600)
            tokens.append({"Authorization": f"Bearer {credential.token}"})
        owner, other_session = tokens

        created = await client.post("/api/lists", headers=owner, json={"name": "owned list", "shared": False})
        assert created.status_code == 200
        item = await client.post(f"/api/lists/{created.json()['id']}/items", headers=owner,
                                 json={"text": "owned item"})
        assert item.status_code == 200
        text = "private task marker"
        created = await client.post("/api/reminders", headers=owner, json={
            "text": text, "trigger_at": 1.,
            "ha_actions": [{"domain": "light", "service": "turn_on", "entity_id": "light.jardin"}],
        })
        assert created.status_code == 200
        reminder_id = created.json()["id"]
        reminder = await graph.reminders.get_by_id(reminder_id, owner=person.user_id)
        assert reminder and reminder.ha_actions

        # Fail inside the real second transaction, with the first store free
        # to commit. The temporary trigger only covers this connection.
        await graph.reminders._work.write(lambda db: db.execute("""
            CREATE TEMP TRIGGER fail_reminder_retirement BEFORE DELETE ON main.reminders
            BEGIN SELECT RAISE(ABORT, 'synthetic reminder purge failure'); END
        """).rowcount)
        url = f"/api/admin/users/{person.user_id}"
        refused = await client.delete(url, headers=dashboard_auth)
        assert refused.status_code == 503
        assert refused.json()["detail"] == {"code": "user_catalog_unavailable", "credentials_retired": True}
        assert authority.users_retired(frozenset({person.user_id}))
        assert users.get_user(person.user_id).is_active
        assert stored_ids(tmp_path / "lists.db", "lists") == set()
        assert stored_ids(tmp_path / "lists.db", "list_items") == set()
        assert stored_ids(tmp_path / "reminders.db", "reminders") == {reminder_id}

        assert (await client.get("/api/reminders", headers=owner)).status_code == 401
        for headers in (dashboard_auth, other_session):
            visible = await client.get("/api/reminders", headers=headers)
            assert visible.status_code == 200 and visible.json() == []
            notices = await client.get("/api/reminders/notices", headers=headers)
            assert notices.status_code == 200 and notices.json()["notices"] == []
        assert await graph.reminders.get_by_id(reminder_id, owner=person.user_id) is None
        await graph.scheduler._deliver_due(time.time())
        assert await graph.scheduler._fire_reminder(reminder) is False
        graph.tts.aspeak.assert_not_awaited()
        graph.ha.call_service_once.assert_not_called()
        assert await graph.reminders._work.read(lambda db: db.execute(
            "SELECT count(*) FROM reminder_home_actions").fetchone()[0]) == 0

        await graph.reminders._work.write(lambda db: db.execute("DROP TRIGGER fail_reminder_retirement").rowcount)
        retried = await client.delete(url, headers=dashboard_auth)
        assert retried.status_code == 200 and retried.json()["ok"]
        assert stored_ids(tmp_path / "reminders.db", "reminders") == set()
        assert users.get_user(person.user_id) is None
        assert authority.users_retired(frozenset({person.user_id}))
