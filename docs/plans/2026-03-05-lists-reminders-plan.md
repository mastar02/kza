# Lists & Reminders Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add voice-controlled shopping lists, generic lists (personal + shared), and recurring reminders with HA action chaining to KZA.

**Architecture:** Two separate modules (`src/lists/` and `src/reminders/`) sharing a SQLite database (`./data/lists.db`). Lists handle static CRUD data, reminders handle time-triggered events with recurrence. Both integrate into the existing dispatcher fast path, REST API, and HA todo platform.

**Tech Stack:** Python 3.13, asyncio, aiosqlite, pytest, FastAPI (existing dashboard), Home Assistant todo platform.

**Design doc:** `docs/plans/2026-03-05-lists-reminders-design.md`

---

## Task 1: SQLite Store Foundation — `list_store.py`

**Files:**
- Create: `src/lists/__init__.py`
- Create: `src/lists/list_store.py`
- Create: `tests/unit/lists/__init__.py`
- Create: `tests/unit/lists/test_list_store.py`

**Step 1: Write failing tests**

```python
# tests/unit/lists/test_list_store.py
"""Tests for ListStore — SQLite persistence for lists and items."""
import pytest
import pytest_asyncio
import aiosqlite
from pathlib import Path

from src.lists.list_store import ListStore, UserList, ListItem


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ListStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_list(store):
    lst = await store.create_list("compras", "user", "user_1")
    assert lst.name == "compras"
    assert lst.owner_type == "user"
    assert lst.owner_id == "user_1"
    assert lst.id  # non-empty


@pytest.mark.asyncio
async def test_create_shared_list(store):
    lst = await store.create_list("hogar", "shared", None)
    assert lst.owner_type == "shared"
    assert lst.owner_id is None


@pytest.mark.asyncio
async def test_get_lists_by_user(store):
    await store.create_list("mia", "user", "u1")
    await store.create_list("compartida", "shared", None)
    await store.create_list("otra", "user", "u2")
    lists = await store.get_lists_for_user("u1")
    names = [l.name for l in lists]
    assert "mia" in names
    assert "compartida" in names
    assert "otra" not in names


@pytest.mark.asyncio
async def test_add_and_get_items(store):
    lst = await store.create_list("compras", "user", "u1")
    item = await store.add_item(lst.id, "leche", added_by="u1")
    assert item.text == "leche"
    items = await store.get_items(lst.id)
    assert len(items) == 1
    assert items[0].text == "leche"


@pytest.mark.asyncio
async def test_remove_item(store):
    lst = await store.create_list("compras", "user", "u1")
    item = await store.add_item(lst.id, "pan")
    await store.remove_item(item.id)
    items = await store.get_items(lst.id)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_complete_item(store):
    lst = await store.create_list("tareas", "user", "u1")
    item = await store.add_item(lst.id, "lavar")
    updated = await store.complete_item(item.id)
    assert updated.completed is True
    assert updated.completed_at is not None


@pytest.mark.asyncio
async def test_clear_list(store):
    lst = await store.create_list("compras", "user", "u1")
    await store.add_item(lst.id, "a")
    await store.add_item(lst.id, "b")
    await store.clear_list(lst.id)
    items = await store.get_items(lst.id)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_delete_list(store):
    lst = await store.create_list("temp", "user", "u1")
    await store.add_item(lst.id, "x")
    await store.delete_list(lst.id)
    lists = await store.get_lists_for_user("u1")
    assert len(lists) == 0


@pytest.mark.asyncio
async def test_find_list_by_name(store):
    await store.create_list("compras", "user", "u1")
    found = await store.find_list_by_name("compras", "u1")
    assert found is not None
    assert found.name == "compras"


@pytest.mark.asyncio
async def test_find_list_by_name_shared(store):
    await store.create_list("casa", "shared", None)
    found = await store.find_list_by_name("casa", "u1")
    assert found is not None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lists/test_list_store.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.lists'`

**Step 3: Write minimal implementation**

```python
# src/lists/__init__.py
from src.lists.list_store import ListStore, UserList, ListItem

__all__ = ["ListStore", "UserList", "ListItem"]
```

```python
# src/lists/list_store.py
"""SQLite persistence for lists and items."""

import time
import uuid
import logging
import aiosqlite
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ListItem:
    """A single item in a list."""
    id: str
    list_id: str
    text: str
    completed: bool = False
    added_by: str | None = None
    created_at: float = 0.0
    completed_at: float | None = None


@dataclass
class UserList:
    """A named list with ownership."""
    id: str
    name: str
    owner_type: str  # "user" | "shared"
    owner_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    items: list[ListItem] = field(default_factory=list)


class ListStore:
    """SQLite store for lists and their items."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS lists (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS list_items (
                id TEXT PRIMARY KEY,
                list_id TEXT NOT NULL REFERENCES lists(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                added_by TEXT,
                created_at REAL NOT NULL,
                completed_at REAL
            );
        """)
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def create_list(self, name: str, owner_type: str, owner_id: str | None) -> UserList:
        now = time.time()
        list_id = uuid.uuid4().hex[:12]
        await self._db.execute(
            "INSERT INTO lists (id, name, owner_type, owner_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (list_id, name, owner_type, owner_id, now, now),
        )
        await self._db.commit()
        return UserList(id=list_id, name=name, owner_type=owner_type, owner_id=owner_id, created_at=now, updated_at=now)

    async def get_lists_for_user(self, user_id: str) -> list[UserList]:
        cursor = await self._db.execute(
            "SELECT * FROM lists WHERE owner_id = ? OR owner_type = 'shared' ORDER BY name",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [UserList(id=r["id"], name=r["name"], owner_type=r["owner_type"],
                         owner_id=r["owner_id"], created_at=r["created_at"], updated_at=r["updated_at"])
                for r in rows]

    async def find_list_by_name(self, name: str, user_id: str) -> UserList | None:
        cursor = await self._db.execute(
            "SELECT * FROM lists WHERE LOWER(name) = LOWER(?) AND (owner_id = ? OR owner_type = 'shared') LIMIT 1",
            (name, user_id),
        )
        r = await cursor.fetchone()
        if not r:
            return None
        return UserList(id=r["id"], name=r["name"], owner_type=r["owner_type"],
                        owner_id=r["owner_id"], created_at=r["created_at"], updated_at=r["updated_at"])

    async def add_item(self, list_id: str, text: str, added_by: str | None = None) -> ListItem:
        now = time.time()
        item_id = uuid.uuid4().hex[:12]
        await self._db.execute(
            "INSERT INTO list_items (id, list_id, text, added_by, created_at) VALUES (?, ?, ?, ?, ?)",
            (item_id, list_id, text, added_by, now),
        )
        await self._db.execute("UPDATE lists SET updated_at = ? WHERE id = ?", (now, list_id))
        await self._db.commit()
        return ListItem(id=item_id, list_id=list_id, text=text, added_by=added_by, created_at=now)

    async def get_items(self, list_id: str) -> list[ListItem]:
        cursor = await self._db.execute(
            "SELECT * FROM list_items WHERE list_id = ? ORDER BY created_at", (list_id,),
        )
        rows = await cursor.fetchall()
        return [ListItem(id=r["id"], list_id=r["list_id"], text=r["text"],
                         completed=bool(r["completed"]), added_by=r["added_by"],
                         created_at=r["created_at"], completed_at=r["completed_at"])
                for r in rows]

    async def remove_item(self, item_id: str) -> None:
        await self._db.execute("DELETE FROM list_items WHERE id = ?", (item_id,))
        await self._db.commit()

    async def complete_item(self, item_id: str) -> ListItem:
        now = time.time()
        await self._db.execute(
            "UPDATE list_items SET completed = 1, completed_at = ? WHERE id = ?", (now, item_id),
        )
        await self._db.commit()
        cursor = await self._db.execute("SELECT * FROM list_items WHERE id = ?", (item_id,))
        r = await cursor.fetchone()
        return ListItem(id=r["id"], list_id=r["list_id"], text=r["text"],
                        completed=bool(r["completed"]), added_by=r["added_by"],
                        created_at=r["created_at"], completed_at=r["completed_at"])

    async def clear_list(self, list_id: str) -> None:
        await self._db.execute("DELETE FROM list_items WHERE list_id = ?", (list_id,))
        await self._db.commit()

    async def delete_list(self, list_id: str) -> None:
        await self._db.execute("DELETE FROM list_items WHERE list_id = ?", (list_id,))
        await self._db.execute("DELETE FROM lists WHERE id = ?", (list_id,))
        await self._db.commit()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/lists/test_list_store.py -v
```
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/lists/ tests/unit/lists/
git commit -m "feat(lists): add ListStore with SQLite persistence and tests"
```

---

## Task 2: List Manager — `list_manager.py`

**Files:**
- Create: `src/lists/list_manager.py`
- Modify: `src/lists/__init__.py`
- Create: `tests/unit/lists/test_list_manager.py`

**Step 1: Write failing tests**

```python
# tests/unit/lists/test_list_manager.py
"""Tests for ListManager — business logic for list operations."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore, UserList, ListItem


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ListStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def manager(store):
    config = {"default_list_name": "compras", "ha_sync_enabled": False}
    return ListManager(store=store, ha_client=None, config=config)


@pytest.mark.asyncio
async def test_add_item_creates_default_list(manager):
    result = await manager.add_item(user_id="u1", item_text="leche")
    assert result.text == "leche"


@pytest.mark.asyncio
async def test_add_item_to_named_list(manager):
    await manager.create_list(user_id="u1", list_name="oficina")
    result = await manager.add_item(user_id="u1", item_text="lapiz", list_name="oficina")
    assert result.text == "lapiz"


@pytest.mark.asyncio
async def test_remove_item_fuzzy_match(manager):
    await manager.add_item(user_id="u1", item_text="leche descremada")
    removed = await manager.remove_item(user_id="u1", item_text="leche")
    assert removed is True


@pytest.mark.asyncio
async def test_get_items_returns_list(manager):
    await manager.add_item(user_id="u1", item_text="pan")
    await manager.add_item(user_id="u1", item_text="huevos")
    items = await manager.get_items(user_id="u1")
    assert len(items) == 2


@pytest.mark.asyncio
async def test_clear_list(manager):
    await manager.add_item(user_id="u1", item_text="a")
    await manager.add_item(user_id="u1", item_text="b")
    await manager.clear_list(user_id="u1")
    items = await manager.get_items(user_id="u1")
    assert len(items) == 0


@pytest.mark.asyncio
async def test_create_shared_list(manager):
    lst = await manager.create_list(user_id="u1", list_name="casa", shared=True)
    assert lst.owner_type == "shared"


@pytest.mark.asyncio
async def test_shared_list_visible_to_other_user(manager):
    await manager.create_list(user_id="u1", list_name="casa", shared=True)
    await manager.add_item(user_id="u1", item_text="jabon", list_name="casa")
    items = await manager.get_items(user_id="u2", list_name="casa")
    assert len(items) == 1


@pytest.mark.asyncio
async def test_delete_list(manager):
    await manager.create_list(user_id="u1", list_name="temp")
    result = await manager.delete_list(user_id="u1", list_name="temp")
    assert result is True


@pytest.mark.asyncio
async def test_get_all_lists(manager):
    await manager.create_list(user_id="u1", list_name="a")
    await manager.create_list(user_id="u1", list_name="b")
    lists = await manager.get_all_lists(user_id="u1")
    assert len(lists) >= 2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lists/test_list_manager.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.lists.list_manager'`

**Step 3: Write minimal implementation**

```python
# src/lists/list_manager.py
"""Business logic for list operations."""

import logging
from difflib import SequenceMatcher

from src.lists.list_store import ListStore, UserList, ListItem

logger = logging.getLogger(__name__)


class ListManager:
    """Manages list CRUD with fuzzy matching and auto-creation."""

    def __init__(self, store: ListStore, ha_client=None, config: dict | None = None):
        self._store = store
        self._ha = ha_client
        self._config = config or {}
        self._default_list = self._config.get("default_list_name", "compras")

    async def create_list(self, user_id: str, list_name: str, shared: bool = False) -> UserList:
        owner_type = "shared" if shared else "user"
        owner_id = None if shared else user_id
        lst = await self._store.create_list(list_name, owner_type, owner_id)
        logger.info(f"List created: '{list_name}' ({owner_type}) by {user_id}")
        return lst

    async def delete_list(self, user_id: str, list_name: str) -> bool:
        lst = await self._store.find_list_by_name(list_name, user_id)
        if not lst:
            return False
        await self._store.delete_list(lst.id)
        logger.info(f"List deleted: '{list_name}' by {user_id}")
        return True

    async def add_item(self, user_id: str, item_text: str, list_name: str | None = None) -> ListItem:
        lst = await self._resolve_list(user_id, list_name)
        item = await self._store.add_item(lst.id, item_text, added_by=user_id)
        logger.info(f"Item added: '{item_text}' to '{lst.name}' by {user_id}")
        return item

    async def remove_item(self, user_id: str, item_text: str, list_name: str | None = None) -> bool:
        lst = await self._resolve_list(user_id, list_name)
        items = await self._store.get_items(lst.id)
        match = self._fuzzy_find(item_text, items)
        if not match:
            return False
        await self._store.remove_item(match.id)
        logger.info(f"Item removed: '{match.text}' from '{lst.name}' by {user_id}")
        return True

    async def get_items(self, user_id: str, list_name: str | None = None) -> list[ListItem]:
        lst = await self._resolve_list(user_id, list_name)
        return await self._store.get_items(lst.id)

    async def clear_list(self, user_id: str, list_name: str | None = None) -> None:
        lst = await self._resolve_list(user_id, list_name)
        await self._store.clear_list(lst.id)
        logger.info(f"List cleared: '{lst.name}' by {user_id}")

    async def get_all_lists(self, user_id: str) -> list[UserList]:
        return await self._store.get_lists_for_user(user_id)

    async def _resolve_list(self, user_id: str, list_name: str | None) -> UserList:
        """Find or auto-create the target list."""
        name = list_name or self._default_list
        lst = await self._store.find_list_by_name(name, user_id)
        if not lst:
            lst = await self._store.create_list(name, "user", user_id)
            logger.info(f"Auto-created list: '{name}' for {user_id}")
        return lst

    def _fuzzy_find(self, query: str, items: list[ListItem], threshold: float = 0.5) -> ListItem | None:
        """Find best matching item by text similarity."""
        best_match = None
        best_score = threshold
        query_lower = query.lower()
        for item in items:
            # Exact substring match first
            if query_lower in item.text.lower():
                return item
            score = SequenceMatcher(None, query_lower, item.text.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = item
        return best_match
```

Update `src/lists/__init__.py`:
```python
from src.lists.list_store import ListStore, UserList, ListItem
from src.lists.list_manager import ListManager

__all__ = ["ListStore", "UserList", "ListItem", "ListManager"]
```

**Step 4: Run tests**

```bash
pytest tests/unit/lists/ -v
```
Expected: All tests PASS (store + manager)

**Step 5: Commit**

```bash
git add src/lists/ tests/unit/lists/
git commit -m "feat(lists): add ListManager with fuzzy matching and auto-creation"
```

---

## Task 3: Reminder Store — `reminder_store.py`

**Files:**
- Create: `src/reminders/__init__.py`
- Create: `src/reminders/reminder_store.py`
- Create: `tests/unit/reminders/__init__.py`
- Create: `tests/unit/reminders/test_reminder_store.py`

**Step 1: Write failing tests**

```python
# tests/unit/reminders/test_reminder_store.py
"""Tests for ReminderStore — SQLite persistence for reminders."""
import time
import pytest
import pytest_asyncio

from src.reminders.reminder_store import ReminderStore, Reminder


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_reminder(store):
    r = await store.create(user_id="u1", text="sacar basura", trigger_at=time.time() + 3600)
    assert r.text == "sacar basura"
    assert r.state == "active"
    assert r.user_id == "u1"


@pytest.mark.asyncio
async def test_create_recurring_reminder(store):
    r = await store.create(user_id="u1", text="pastilla", trigger_at=time.time() + 3600, recurrence="daily")
    assert r.recurrence == "daily"


@pytest.mark.asyncio
async def test_create_reminder_with_ha_actions(store):
    actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.patio"}]
    r = await store.create(user_id="u1", text="regar", trigger_at=time.time() + 3600, ha_actions=actions)
    assert r.ha_actions == actions


@pytest.mark.asyncio
async def test_get_active_reminders(store):
    await store.create(user_id="u1", text="a", trigger_at=time.time() + 100)
    await store.create(user_id="u1", text="b", trigger_at=time.time() + 200)
    await store.create(user_id="u2", text="c", trigger_at=time.time() + 300)
    reminders = await store.get_active_for_user("u1")
    assert len(reminders) == 2


@pytest.mark.asyncio
async def test_get_next_pending(store):
    now = time.time()
    await store.create(user_id="u1", text="later", trigger_at=now + 1000)
    await store.create(user_id="u1", text="soon", trigger_at=now + 10)
    nxt = await store.get_next_pending()
    assert nxt.text == "soon"


@pytest.mark.asyncio
async def test_mark_fired(store):
    r = await store.create(user_id="u1", text="x", trigger_at=time.time() - 10)
    await store.mark_fired(r.id)
    updated = await store.get_by_id(r.id)
    assert updated.state == "fired"
    assert updated.fire_count == 1


@pytest.mark.asyncio
async def test_cancel_reminder(store):
    r = await store.create(user_id="u1", text="x", trigger_at=time.time() + 100)
    await store.cancel(r.id)
    updated = await store.get_by_id(r.id)
    assert updated.state == "cancelled"


@pytest.mark.asyncio
async def test_update_trigger(store):
    r = await store.create(user_id="u1", text="x", trigger_at=1000.0)
    await store.update_trigger(r.id, 2000.0)
    updated = await store.get_by_id(r.id)
    assert updated.trigger_at == 2000.0


@pytest.mark.asyncio
async def test_get_due_reminders(store):
    now = time.time()
    await store.create(user_id="u1", text="past", trigger_at=now - 60)
    await store.create(user_id="u1", text="future", trigger_at=now + 3600)
    due = await store.get_due(now)
    assert len(due) == 1
    assert due[0].text == "past"


@pytest.mark.asyncio
async def test_cleanup_old_fired(store):
    old_time = time.time() - (31 * 86400)  # 31 days ago
    r = await store.create(user_id="u1", text="old", trigger_at=old_time)
    await store.mark_fired(r.id)
    cleaned = await store.cleanup_old(days=30)
    assert cleaned >= 1
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/reminders/test_reminder_store.py -v
```

**Step 3: Implement**

```python
# src/reminders/__init__.py
from src.reminders.reminder_store import ReminderStore, Reminder

__all__ = ["ReminderStore", "Reminder"]
```

```python
# src/reminders/reminder_store.py
"""SQLite persistence for reminders."""

import json
import time
import uuid
import logging
import aiosqlite
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Reminder:
    """A time-triggered reminder with optional recurrence and HA actions."""
    id: str
    user_id: str
    text: str
    trigger_at: float
    created_at: float = 0.0
    recurrence: str | None = None
    recurrence_end: float | None = None
    ha_actions: list[dict] | None = None
    state: str = "active"  # active | fired | cancelled
    last_fired_at: float | None = None
    fire_count: int = 0


class ReminderStore:
    """SQLite store for reminders."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                trigger_at REAL NOT NULL,
                created_at REAL NOT NULL,
                recurrence TEXT,
                recurrence_end REAL,
                ha_actions TEXT,
                state TEXT DEFAULT 'active',
                last_fired_at REAL,
                fire_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_reminders_trigger ON reminders(trigger_at)
                WHERE state = 'active';
            CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders(user_id, state);
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    def _row_to_reminder(self, r) -> Reminder:
        ha_actions = json.loads(r["ha_actions"]) if r["ha_actions"] else None
        return Reminder(
            id=r["id"], user_id=r["user_id"], text=r["text"],
            trigger_at=r["trigger_at"], created_at=r["created_at"],
            recurrence=r["recurrence"], recurrence_end=r["recurrence_end"],
            ha_actions=ha_actions, state=r["state"],
            last_fired_at=r["last_fired_at"], fire_count=r["fire_count"],
        )

    async def create(self, user_id: str, text: str, trigger_at: float,
                     recurrence: str | None = None, recurrence_end: float | None = None,
                     ha_actions: list[dict] | None = None) -> Reminder:
        now = time.time()
        reminder_id = uuid.uuid4().hex[:12]
        ha_json = json.dumps(ha_actions) if ha_actions else None
        await self._db.execute(
            """INSERT INTO reminders
               (id, user_id, text, trigger_at, created_at, recurrence, recurrence_end, ha_actions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (reminder_id, user_id, text, trigger_at, now, recurrence, recurrence_end, ha_json),
        )
        await self._db.commit()
        return Reminder(id=reminder_id, user_id=user_id, text=text, trigger_at=trigger_at,
                        created_at=now, recurrence=recurrence, recurrence_end=recurrence_end,
                        ha_actions=ha_actions)

    async def get_by_id(self, reminder_id: str) -> Reminder | None:
        cursor = await self._db.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        r = await cursor.fetchone()
        return self._row_to_reminder(r) if r else None

    async def get_active_for_user(self, user_id: str) -> list[Reminder]:
        cursor = await self._db.execute(
            "SELECT * FROM reminders WHERE user_id = ? AND state = 'active' ORDER BY trigger_at",
            (user_id,),
        )
        return [self._row_to_reminder(r) for r in await cursor.fetchall()]

    async def get_next_pending(self) -> Reminder | None:
        cursor = await self._db.execute(
            "SELECT * FROM reminders WHERE state = 'active' ORDER BY trigger_at ASC LIMIT 1",
        )
        r = await cursor.fetchone()
        return self._row_to_reminder(r) if r else None

    async def get_due(self, now: float) -> list[Reminder]:
        cursor = await self._db.execute(
            "SELECT * FROM reminders WHERE state = 'active' AND trigger_at <= ? ORDER BY trigger_at",
            (now,),
        )
        return [self._row_to_reminder(r) for r in await cursor.fetchall()]

    async def mark_fired(self, reminder_id: str) -> None:
        now = time.time()
        await self._db.execute(
            "UPDATE reminders SET state = 'fired', last_fired_at = ?, fire_count = fire_count + 1 WHERE id = ?",
            (now, reminder_id),
        )
        await self._db.commit()

    async def cancel(self, reminder_id: str) -> None:
        await self._db.execute("UPDATE reminders SET state = 'cancelled' WHERE id = ?", (reminder_id,))
        await self._db.commit()

    async def update_trigger(self, reminder_id: str, new_trigger: float) -> None:
        await self._db.execute(
            "UPDATE reminders SET trigger_at = ?, state = 'active' WHERE id = ?",
            (new_trigger, reminder_id),
        )
        await self._db.commit()

    async def cleanup_old(self, days: int = 30) -> int:
        cutoff = time.time() - (days * 86400)
        cursor = await self._db.execute(
            "DELETE FROM reminders WHERE state IN ('fired', 'cancelled') AND recurrence IS NULL AND created_at < ?",
            (cutoff,),
        )
        await self._db.commit()
        return cursor.rowcount
```

**Step 4: Run tests**

```bash
pytest tests/unit/reminders/test_reminder_store.py -v
```
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add src/reminders/ tests/unit/reminders/
git commit -m "feat(reminders): add ReminderStore with SQLite persistence and tests"
```

---

## Task 4: Recurrence Engine — `recurrence.py`

**Files:**
- Create: `src/reminders/recurrence.py`
- Create: `tests/unit/reminders/test_recurrence.py`

**Step 1: Write failing tests**

```python
# tests/unit/reminders/test_recurrence.py
"""Tests for recurrence engine — calculating next trigger dates."""
import pytest
from datetime import datetime

from src.reminders.recurrence import next_trigger, parse_recurrence, RecurrenceType


class TestParseRecurrence:
    def test_daily(self):
        r = parse_recurrence("daily")
        assert r == (RecurrenceType.DAILY, None)

    def test_weekdays(self):
        r = parse_recurrence("weekdays")
        assert r == (RecurrenceType.WEEKDAYS, None)

    def test_weekly_monday(self):
        r = parse_recurrence("weekly:1")
        assert r == (RecurrenceType.WEEKLY, 1)

    def test_weekly_friday(self):
        r = parse_recurrence("weekly:5")
        assert r == (RecurrenceType.WEEKLY, 5)

    def test_monthly(self):
        r = parse_recurrence("monthly:15")
        assert r == (RecurrenceType.MONTHLY, 15)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_recurrence("every_3_days")


class TestNextTrigger:
    def test_daily_next_day(self):
        # Monday 9:00 -> Tuesday 9:00
        base = datetime(2026, 3, 2, 9, 0).timestamp()  # Monday
        nxt = next_trigger(base, "daily")
        result = datetime.fromtimestamp(nxt)
        assert result.day == 3
        assert result.hour == 9

    def test_weekdays_friday_to_monday(self):
        # Friday 9:00 -> Monday 9:00
        base = datetime(2026, 3, 6, 9, 0).timestamp()  # Friday
        nxt = next_trigger(base, "weekdays")
        result = datetime.fromtimestamp(nxt)
        assert result.weekday() == 0  # Monday
        assert result.day == 9

    def test_weekdays_monday_to_tuesday(self):
        base = datetime(2026, 3, 2, 9, 0).timestamp()  # Monday
        nxt = next_trigger(base, "weekdays")
        result = datetime.fromtimestamp(nxt)
        assert result.weekday() == 1  # Tuesday

    def test_weekly_next_week(self):
        # Monday 9:00, weekly:1 (Monday) -> next Monday
        base = datetime(2026, 3, 2, 9, 0).timestamp()
        nxt = next_trigger(base, "weekly:1")
        result = datetime.fromtimestamp(nxt)
        assert result.day == 9
        assert result.weekday() == 0

    def test_monthly(self):
        # March 15 -> April 15
        base = datetime(2026, 3, 15, 9, 0).timestamp()
        nxt = next_trigger(base, "monthly:15")
        result = datetime.fromtimestamp(nxt)
        assert result.month == 4
        assert result.day == 15

    def test_monthly_end_of_year(self):
        # December 15 -> January 15
        base = datetime(2026, 12, 15, 9, 0).timestamp()
        nxt = next_trigger(base, "monthly:15")
        result = datetime.fromtimestamp(nxt)
        assert result.month == 1
        assert result.year == 2027
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/reminders/test_recurrence.py -v
```

**Step 3: Implement**

```python
# src/reminders/recurrence.py
"""Recurrence engine — calculates next trigger dates."""

from datetime import datetime, timedelta
from enum import StrEnum


class RecurrenceType(StrEnum):
    DAILY = "daily"
    WEEKDAYS = "weekdays"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


def parse_recurrence(recurrence: str) -> tuple[RecurrenceType, int | None]:
    """Parse a recurrence string into type and optional parameter.

    Args:
        recurrence: "daily", "weekdays", "weekly:1", "monthly:15"

    Returns:
        (RecurrenceType, parameter) where parameter is weekday (0-6) or day of month.

    Raises:
        ValueError: If recurrence format is invalid.
    """
    if recurrence == "daily":
        return RecurrenceType.DAILY, None
    if recurrence == "weekdays":
        return RecurrenceType.WEEKDAYS, None
    if recurrence.startswith("weekly:"):
        day = int(recurrence.split(":")[1])
        return RecurrenceType.WEEKLY, day
    if recurrence.startswith("monthly:"):
        day = int(recurrence.split(":")[1])
        return RecurrenceType.MONTHLY, day
    raise ValueError(f"Invalid recurrence format: {recurrence}")


def next_trigger(last_trigger: float, recurrence: str) -> float:
    """Calculate the next trigger time given the last trigger and recurrence rule.

    Args:
        last_trigger: Epoch timestamp of last trigger.
        recurrence: Recurrence string.

    Returns:
        Epoch timestamp of next trigger.
    """
    rec_type, param = parse_recurrence(recurrence)
    dt = datetime.fromtimestamp(last_trigger)

    if rec_type == RecurrenceType.DAILY:
        nxt = dt + timedelta(days=1)

    elif rec_type == RecurrenceType.WEEKDAYS:
        nxt = dt + timedelta(days=1)
        while nxt.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
            nxt += timedelta(days=1)

    elif rec_type == RecurrenceType.WEEKLY:
        nxt = dt + timedelta(days=7)

    elif rec_type == RecurrenceType.MONTHLY:
        month = dt.month + 1
        year = dt.year
        if month > 12:
            month = 1
            year += 1
        day = min(param, 28)  # Safe for all months
        nxt = dt.replace(year=year, month=month, day=day)

    else:
        raise ValueError(f"Unknown recurrence type: {rec_type}")

    return nxt.timestamp()
```

**Step 4: Run tests**

```bash
pytest tests/unit/reminders/test_recurrence.py -v
```
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add src/reminders/recurrence.py tests/unit/reminders/test_recurrence.py
git commit -m "feat(reminders): add recurrence engine with daily/weekdays/weekly/monthly"
```

---

## Task 5: Reminder Manager — `reminder_manager.py`

**Files:**
- Create: `src/reminders/reminder_manager.py`
- Modify: `src/reminders/__init__.py`
- Create: `tests/unit/reminders/test_reminder_manager.py`

**Step 1: Write failing tests**

```python
# tests/unit/reminders/test_reminder_manager.py
"""Tests for ReminderManager — CRUD and cancellation logic."""
import time
import pytest
import pytest_asyncio

from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_store import ReminderStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def manager(store):
    config = {"default_time": "09:00", "tts_prefix": "Oye, recuerda"}
    return ReminderManager(store=store, config=config)


@pytest.mark.asyncio
async def test_create_one_shot(manager):
    r = await manager.create(user_id="u1", text="dentista", trigger_at=time.time() + 3600)
    assert r.text == "dentista"
    assert r.recurrence is None


@pytest.mark.asyncio
async def test_create_recurring(manager):
    r = await manager.create(user_id="u1", text="pastilla", trigger_at=time.time() + 3600, recurrence="daily")
    assert r.recurrence == "daily"


@pytest.mark.asyncio
async def test_create_with_ha_actions(manager):
    actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.patio"}]
    r = await manager.create(user_id="u1", text="regar", trigger_at=time.time() + 3600, ha_actions=actions)
    assert len(r.ha_actions) == 1


@pytest.mark.asyncio
async def test_cancel_by_text(manager):
    await manager.create(user_id="u1", text="sacar la basura", trigger_at=time.time() + 3600)
    result = await manager.cancel_by_text(user_id="u1", search_text="basura")
    assert result is True


@pytest.mark.asyncio
async def test_cancel_nonexistent(manager):
    result = await manager.cancel_by_text(user_id="u1", search_text="inexistente")
    assert result is False


@pytest.mark.asyncio
async def test_get_active(manager):
    await manager.create(user_id="u1", text="a", trigger_at=time.time() + 100)
    await manager.create(user_id="u1", text="b", trigger_at=time.time() + 200)
    active = await manager.get_active(user_id="u1")
    assert len(active) == 2


@pytest.mark.asyncio
async def test_get_today(manager):
    now = time.time()
    await manager.create(user_id="u1", text="hoy", trigger_at=now + 100)
    await manager.create(user_id="u1", text="manana", trigger_at=now + 90000)
    today = await manager.get_today(user_id="u1")
    assert len(today) == 1
    assert today[0].text == "hoy"


@pytest.mark.asyncio
async def test_format_for_voice(manager):
    r = await manager.create(user_id="u1", text="tomar agua", trigger_at=time.time() + 3600)
    voice = manager.format_for_voice(r)
    assert "tomar agua" in voice
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/reminders/test_reminder_manager.py -v
```

**Step 3: Implement**

```python
# src/reminders/reminder_manager.py
"""Business logic for reminder operations."""

import logging
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from src.reminders.reminder_store import ReminderStore, Reminder

logger = logging.getLogger(__name__)


class ReminderManager:
    """Manages reminder CRUD with fuzzy cancellation."""

    def __init__(self, store: ReminderStore, config: dict | None = None):
        self._store = store
        self._config = config or {}
        self._tts_prefix = self._config.get("tts_prefix", "Oye, recuerda")

    async def create(self, user_id: str, text: str, trigger_at: float,
                     recurrence: str | None = None, ha_actions: list[dict] | None = None) -> Reminder:
        r = await self._store.create(
            user_id=user_id, text=text, trigger_at=trigger_at,
            recurrence=recurrence, ha_actions=ha_actions,
        )
        logger.info(f"Reminder created: '{text}' for {user_id} at {datetime.fromtimestamp(trigger_at)}")
        return r

    async def cancel_by_text(self, user_id: str, search_text: str) -> bool:
        active = await self._store.get_active_for_user(user_id)
        match = self._fuzzy_find(search_text, active)
        if not match:
            return False
        await self._store.cancel(match.id)
        logger.info(f"Reminder cancelled: '{match.text}' for {user_id}")
        return True

    async def get_active(self, user_id: str) -> list[Reminder]:
        return await self._store.get_active_for_user(user_id)

    async def get_today(self, user_id: str) -> list[Reminder]:
        now = datetime.now()
        end_of_day = now.replace(hour=23, minute=59, second=59).timestamp()
        active = await self._store.get_active_for_user(user_id)
        return [r for r in active if r.trigger_at <= end_of_day]

    def format_for_voice(self, reminder: Reminder) -> str:
        dt = datetime.fromtimestamp(reminder.trigger_at)
        time_str = dt.strftime("%H:%M")
        return f"{self._tts_prefix} {reminder.text} a las {time_str}"

    def _fuzzy_find(self, query: str, reminders: list[Reminder], threshold: float = 0.4) -> Reminder | None:
        best_match = None
        best_score = threshold
        query_lower = query.lower()
        for r in reminders:
            if query_lower in r.text.lower():
                return r
            score = SequenceMatcher(None, query_lower, r.text.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = r
        return best_match
```

Update `src/reminders/__init__.py`:
```python
from src.reminders.reminder_store import ReminderStore, Reminder
from src.reminders.reminder_manager import ReminderManager
from src.reminders.recurrence import next_trigger, parse_recurrence, RecurrenceType

__all__ = [
    "ReminderStore", "Reminder", "ReminderManager",
    "next_trigger", "parse_recurrence", "RecurrenceType",
]
```

**Step 4: Run all reminder tests**

```bash
pytest tests/unit/reminders/ -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reminders/ tests/unit/reminders/
git commit -m "feat(reminders): add ReminderManager with fuzzy cancel and voice formatting"
```

---

## Task 6: Reminder Scheduler — `reminder_scheduler.py`

**Files:**
- Create: `src/reminders/reminder_scheduler.py`
- Create: `tests/unit/reminders/test_reminder_scheduler.py`

**Step 1: Write failing tests**

```python
# tests/unit/reminders/test_reminder_scheduler.py
"""Tests for ReminderScheduler — asyncio scheduling and delivery."""
import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderStore, Reminder


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def mock_deps():
    return {
        "tts": AsyncMock(),
        "presence_detector": MagicMock(),
        "ha_client": AsyncMock(),
    }


@pytest_asyncio.fixture
async def scheduler(store, mock_deps):
    config = {
        "max_retries": 3,
        "retry_interval_seconds": 1,
        "missed_reminder_on_arrival": True,
        "tts_prefix": "Oye, recuerda",
    }
    s = ReminderScheduler(
        store=store,
        tts=mock_deps["tts"],
        presence_detector=mock_deps["presence_detector"],
        ha_client=mock_deps["ha_client"],
        config=config,
    )
    yield s
    if s._running:
        await s.stop()


@pytest.mark.asyncio
async def test_fire_reminder_with_tts(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    r = await store.create(user_id="u1", text="basura", trigger_at=time.time() - 10)
    await scheduler._fire_reminder(r)
    mock_deps["tts"].speak.assert_called_once()
    updated = await store.get_by_id(r.id)
    assert updated.state == "fired"


@pytest.mark.asyncio
async def test_fire_reminder_with_ha_actions(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.patio"}]
    r = await store.create(user_id="u1", text="regar", trigger_at=time.time() - 10, ha_actions=actions)
    await scheduler._fire_reminder(r)
    mock_deps["ha_client"].call_service.assert_called_once_with("light", "turn_on", "light.patio", {})


@pytest.mark.asyncio
async def test_fire_recurring_reschedules(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    r = await store.create(user_id="u1", text="pastilla", trigger_at=time.time() - 10, recurrence="daily")
    await scheduler._fire_reminder(r)
    updated = await store.get_by_id(r.id)
    assert updated.state == "active"
    assert updated.trigger_at > time.time()


@pytest.mark.asyncio
async def test_fire_user_not_present_queues_retry(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value=None)
    r = await store.create(user_id="u1", text="x", trigger_at=time.time() - 10)
    await scheduler._fire_reminder(r)
    # Should NOT be fired yet (user not present)
    updated = await store.get_by_id(r.id)
    assert updated.state == "active"


@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler, store):
    r = await store.create(user_id="u1", text="test", trigger_at=time.time() + 10000)
    task = asyncio.create_task(scheduler.start())
    await asyncio.sleep(0.1)
    assert scheduler._running is True
    await scheduler.stop()
    assert scheduler._running is False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/reminders/test_reminder_scheduler.py -v
```

**Step 3: Implement**

```python
# src/reminders/reminder_scheduler.py
"""Asyncio scheduler for firing reminders at their trigger time."""

import asyncio
import logging
import time

from src.reminders.reminder_store import ReminderStore, Reminder
from src.reminders.recurrence import next_trigger

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """Schedules and fires reminders using asyncio sleep-until-next pattern."""

    def __init__(self, store: ReminderStore, tts, presence_detector,
                 ha_client, config: dict | None = None):
        self._store = store
        self._tts = tts
        self._presence = presence_detector
        self._ha = ha_client
        self._config = config or {}
        self._running = False
        self._wake_event = asyncio.Event()
        self._max_retries = self._config.get("max_retries", 3)
        self._retry_interval = self._config.get("retry_interval_seconds", 300)
        self._tts_prefix = self._config.get("tts_prefix", "Oye, recuerda")
        self._retry_counts: dict[str, int] = {}

    async def start(self) -> None:
        """Main scheduler loop — sleeps until next reminder, fires, repeat."""
        self._running = True
        logger.info("ReminderScheduler started")
        while self._running:
            try:
                nxt = await self._store.get_next_pending()
                if not nxt:
                    # No reminders, wait until woken up
                    self._wake_event.clear()
                    await self._wake_event.wait()
                    continue

                wait_seconds = nxt.trigger_at - time.time()
                if wait_seconds > 0:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=wait_seconds)
                        continue  # Woken early, re-check
                    except asyncio.TimeoutError:
                        pass  # Time to fire

                # Fire all due reminders
                due = await self._store.get_due(time.time())
                for reminder in due:
                    await self._fire_reminder(reminder)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)

        logger.info("ReminderScheduler stopped")

    async def stop(self) -> None:
        self._running = False
        self._wake_event.set()

    def notify_new_reminder(self) -> None:
        """Wake the scheduler to re-evaluate next trigger."""
        self._wake_event.set()

    async def _fire_reminder(self, reminder: Reminder) -> None:
        """Attempt to deliver a reminder via TTS and HA actions."""
        zone = self._presence.get_user_zone(reminder.user_id) if self._presence else None

        if not zone:
            # User not present — retry later
            retry_count = self._retry_counts.get(reminder.id, 0)
            if retry_count < self._max_retries:
                self._retry_counts[reminder.id] = retry_count + 1
                new_trigger = time.time() + self._retry_interval
                await self._store.update_trigger(reminder.id, new_trigger)
                logger.info(f"Reminder '{reminder.text}' retry {retry_count + 1}/{self._max_retries}")
                return
            else:
                logger.warning(f"Reminder '{reminder.text}' missed after {self._max_retries} retries")
                self._retry_counts.pop(reminder.id, None)
                # Fall through to mark as fired

        # Deliver TTS
        if zone and self._tts:
            message = f"{self._tts_prefix} {reminder.text}"
            await self._tts.speak(message, zone_id=zone)
            logger.info(f"Reminder delivered: '{reminder.text}' to zone {zone}")

        # Execute HA actions
        if reminder.ha_actions and self._ha:
            for action in reminder.ha_actions:
                try:
                    await self._ha.call_service(
                        action["domain"], action["service"],
                        action.get("entity_id"), action.get("data", {}),
                    )
                except Exception as e:
                    logger.error(f"HA action failed for reminder '{reminder.text}': {e}")

        # Handle recurrence
        self._retry_counts.pop(reminder.id, None)
        if reminder.recurrence:
            new_time = next_trigger(reminder.trigger_at, reminder.recurrence)
            if reminder.recurrence_end and new_time > reminder.recurrence_end:
                await self._store.mark_fired(reminder.id)
            else:
                await self._store.update_trigger(reminder.id, new_time)
                logger.info(f"Recurring reminder rescheduled: '{reminder.text}'")
        else:
            await self._store.mark_fired(reminder.id)
```

**Step 4: Run tests**

```bash
pytest tests/unit/reminders/ -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reminders/ tests/unit/reminders/
git commit -m "feat(reminders): add ReminderScheduler with async delivery and retry logic"
```

---

## Task 7: Dispatcher Integration

**Files:**
- Modify: `src/orchestrator/dispatcher.py` — add PathType, keywords, handler methods
- Create: `tests/unit/orchestrator/test_dispatcher_lists.py`

**Step 1: Write failing tests**

```python
# tests/unit/orchestrator/test_dispatcher_lists.py
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
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/orchestrator/test_dispatcher_lists.py -v
```
Expected: FAIL — `PathType.FAST_LIST` does not exist

**Step 3: Modify `src/orchestrator/dispatcher.py`**

Add to `PathType` enum (line ~62):
```python
    FAST_LIST = "fast_list"           # List CRUD
    FAST_REMINDER = "fast_reminder"   # Reminder CRUD
```

Add keyword lists after existing keywords (~line 146):
```python
    LIST_KEYWORDS = [
        "lista de", "agrega", "agregale", "quita", "quitale",
        "qué hay en la lista", "vacía la lista", "vaciala",
        "crea una lista", "borra la lista", "lista compartida",
    ]

    REMINDER_KEYWORDS = [
        "recuérdame", "recuerdame", "recordatorio",
        "avísame", "avisame", "qué tengo pendiente",
        "que tengo pendiente", "qué recordatorios",
        "que recordatorios", "todos los lunes",
        "todos los días", "todos los dias",
        "cada día", "cada dia", "cada lunes",
        "cada martes", "de lunes a viernes",
        "cancela el recordatorio",
    ]
```

Modify `_classify_request` method to check lists/reminders BEFORE domotics (important for "agrega" which overlaps with domotics keywords):

In `_classify_request`, add BEFORE the domotics check (before line ~318):
```python
        # Detect lists
        for keyword in self.LIST_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_LIST, Priority.HIGH

        # Detect reminders
        for keyword in self.REMINDER_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_REMINDER, Priority.HIGH
```

Update `_stats` init in `__init__` — `by_path` already uses `{p: 0 for p in PathType}` so new PathTypes auto-included.

Add to `dispatch` method, in the path routing section (after the FAST_MUSIC/SLOW_MUSIC/FAST_DOMOTICS blocks, before the `else` for slow path):

```python
        elif path == PathType.FAST_LIST:
            result = await self._fast_list_path(text, user_id, zone_id)
            self._stats["fast_path"] += 1

        elif path == PathType.FAST_REMINDER:
            result = await self._fast_reminder_path(text, user_id, zone_id)
            self._stats["fast_path"] += 1
```

Add stub handler methods:
```python
    async def _fast_list_path(self, text: str, user_id: str, zone_id: str = None) -> DispatchResult:
        """Handle list commands via ListManager."""
        # Will be connected in Task 9 (main.py integration)
        return DispatchResult(
            path=PathType.FAST_LIST, priority=Priority.HIGH,
            success=False, response="Listas no configuradas",
        )

    async def _fast_reminder_path(self, text: str, user_id: str, zone_id: str = None) -> DispatchResult:
        """Handle reminder commands via ReminderManager."""
        # Will be connected in Task 9 (main.py integration)
        return DispatchResult(
            path=PathType.FAST_REMINDER, priority=Priority.HIGH,
            success=False, response="Recordatorios no configurados",
        )
```

**Step 4: Run tests**

```bash
pytest tests/unit/orchestrator/test_dispatcher_lists.py -v
```
Expected: All 9 tests PASS

Also run existing dispatcher tests to make sure nothing broke:
```bash
pytest tests/unit/orchestrator/ -v
```

**Step 5: Commit**

```bash
git add src/orchestrator/dispatcher.py tests/unit/orchestrator/test_dispatcher_lists.py
git commit -m "feat(dispatcher): add FAST_LIST and FAST_REMINDER path types with keyword detection"
```

---

## Task 8: Config + settings.yaml

**Files:**
- Modify: `config/settings.yaml` — add lists and reminders sections

**Step 1: Add config sections**

Add after the `memory` section (~line 280) in `config/settings.yaml`:

```yaml
# =============================================================================
# Lists & Reminders
# =============================================================================
lists:
  db_path: "./data/lists.db"
  default_list_name: "compras"
  ha_sync_enabled: true
  ha_entity_prefix: "todo.kza"

reminders:
  max_retries: 3
  retry_interval_seconds: 300
  missed_reminder_on_arrival: true
  default_time: "09:00"
  tts_prefix: "Oye, recuerda"
```

**Step 2: Verify config loads**

```bash
python -c "import yaml; c = yaml.safe_load(open('config/settings.yaml')); print(c['lists']); print(c['reminders'])"
```
Expected: Both sections print without error

**Step 3: Commit**

```bash
git add config/settings.yaml
git commit -m "feat(config): add lists and reminders sections to settings.yaml"
```

---

## Task 9: Main.py Integration + DI Wiring

**Files:**
- Modify: `src/main.py` — add imports, instantiation, DI wiring

**Step 1: Add imports to `src/main.py`** (after line ~54):

```python
from src.lists.list_store import ListStore
from src.lists.list_manager import ListManager
from src.reminders.reminder_store import ReminderStore
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler
```

**Step 2: Add instantiation** in `async def main()`, after the analytics section (~line 243) and before the multi-room section:

```python
    # Lists & Reminders
    lists_config = config.get("lists", {})
    reminders_config = config.get("reminders", {})
    list_store = ListStore(lists_config.get("db_path", "./data/lists.db"))
    await list_store.initialize()
    list_manager = ListManager(store=list_store, ha_client=ha_client, config=lists_config)

    reminder_store = ReminderStore(lists_config.get("db_path", "./data/lists.db"))
    await reminder_store.initialize()
    reminder_manager = ReminderManager(store=reminder_store, config=reminders_config)
    reminder_scheduler = None  # Created after presence_detector is ready
    logger.info("Lists & reminders initialized")
```

**Step 3: After presence_detector is created** (~line 310), add scheduler creation:

```python
    # Reminder scheduler (needs presence_detector)
    reminder_scheduler = ReminderScheduler(
        store=reminder_store, tts=tts,
        presence_detector=presence_detector,
        ha_client=ha_client, config=reminders_config,
    )
```

**Step 4: Wire into dispatcher** — find where RequestDispatcher is instantiated and add `list_manager` and `reminder_manager`. The dispatcher needs new init params. Add to the `RequestDispatcher.__init__` signature:

In `src/orchestrator/dispatcher.py`, add params:
```python
    def __init__(self, ..., list_manager=None, reminder_manager=None):
        ...
        self.list_manager = list_manager
        self.reminder_manager = reminder_manager
```

In `src/main.py`, pass them when creating the dispatcher.

**Step 5: Start scheduler task** in the main run section (where `alert_scheduler` starts):

```python
    if reminder_scheduler:
        asyncio.create_task(reminder_scheduler.start())
        logger.info("Reminder scheduler started")
```

**Step 6: Verify it imports**

```bash
python -c "from src.lists import ListStore, ListManager; from src.reminders import ReminderStore, ReminderManager, ReminderScheduler; print('OK')"
```

**Step 7: Commit**

```bash
git add src/main.py src/orchestrator/dispatcher.py
git commit -m "feat: wire lists and reminders into main.py DI chain"
```

---

## Task 10: REST API Endpoints

**Files:**
- Modify: `src/dashboard/api.py` — add list and reminder endpoints
- Create: `tests/unit/dashboard/test_api_lists.py`

**Step 1: Write failing tests**

```python
# tests/unit/dashboard/test_api_lists.py
"""Tests for list and reminder REST API endpoints."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.dashboard.api import DashboardAPI
from src.lists.list_store import ListStore
from src.lists.list_manager import ListManager
from src.reminders.reminder_store import ReminderStore
from src.reminders.reminder_manager import ReminderManager


@pytest_asyncio.fixture
async def api(tmp_path):
    list_store = ListStore(str(tmp_path / "test.db"))
    await list_store.initialize()
    list_manager = ListManager(store=list_store, ha_client=None, config={"default_list_name": "compras", "ha_sync_enabled": False})

    reminder_store = ReminderStore(str(tmp_path / "test.db"))
    await reminder_store.initialize()
    reminder_manager = ReminderManager(store=reminder_store, config={})

    dashboard = DashboardAPI(list_manager=list_manager, reminder_manager=reminder_manager)
    yield dashboard

    await list_store.close()
    await reminder_store.close()


@pytest_asyncio.fixture
async def client(api):
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_create_list(client):
    resp = await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "compras"


@pytest.mark.asyncio
async def test_get_lists(client):
    await client.post("/api/lists", json={"name": "a", "user_id": "u1"})
    resp = await client.get("/api/lists", params={"user_id": "u1"})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_add_item(client):
    lst = (await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})).json()
    resp = await client.post(f"/api/lists/{lst['id']}/items", json={"text": "leche", "user_id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["text"] == "leche"


@pytest.mark.asyncio
async def test_get_items(client):
    lst = (await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})).json()
    await client.post(f"/api/lists/{lst['id']}/items", json={"text": "pan", "user_id": "u1"})
    resp = await client.get(f"/api/lists/{lst['id']}/items")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_delete_item(client):
    lst = (await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})).json()
    item = (await client.post(f"/api/lists/{lst['id']}/items", json={"text": "pan", "user_id": "u1"})).json()
    resp = await client.delete(f"/api/lists/{lst['id']}/items/{item['id']}")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_create_reminder(client):
    import time
    resp = await client.post("/api/reminders", json={
        "user_id": "u1", "text": "dentista", "trigger_at": time.time() + 3600,
    })
    assert resp.status_code == 200
    assert resp.json()["text"] == "dentista"


@pytest.mark.asyncio
async def test_get_reminders(client):
    import time
    await client.post("/api/reminders", json={"user_id": "u1", "text": "a", "trigger_at": time.time() + 100})
    resp = await client.get("/api/reminders", params={"user_id": "u1"})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_delete_reminder(client):
    import time
    r = (await client.post("/api/reminders", json={"user_id": "u1", "text": "x", "trigger_at": time.time() + 100})).json()
    resp = await client.delete(f"/api/reminders/{r['id']}")
    assert resp.status_code == 200
```

**Step 2: Run tests — expect FAIL**

**Step 3: Add endpoints to `DashboardAPI`**

Add `list_manager` and `reminder_manager` to `DashboardAPI.__init__` params and register new routes in `_register_routes`. Implementation follows the same pattern as existing routine endpoints.

**Step 4: Run tests**

```bash
pytest tests/unit/dashboard/test_api_lists.py -v
```

**Step 5: Commit**

```bash
git add src/dashboard/api.py tests/unit/dashboard/
git commit -m "feat(api): add REST endpoints for lists and reminders"
```

---

## Task 11: HA Sync — `ha_sync.py`

**Files:**
- Create: `src/lists/ha_sync.py`
- Create: `tests/unit/lists/test_ha_sync.py`

**Step 1: Write tests for HA sync**

Test that `HASyncManager` calls HA WebSocket to create/update todo entities when lists change. Mock `ha_client`.

**Step 2: Implement `ha_sync.py`**

Uses `ha_client.call_service("todo", "add_item", ...)` for syncing items. Subscribes to HA WebSocket events for bidirectional sync.

**Step 3: Run tests, commit**

```bash
git commit -m "feat(lists): add HA todo platform sync"
```

---

## Task 12: Full Integration Test

**Files:**
- Create: `tests/integration/test_lists_reminders_e2e.py`

**Step 1: Write end-to-end test**

Test the full flow: create list via ListManager, add items, verify in store. Create reminder, verify scheduler picks it up, fires with mock TTS. Test recurring reminder reschedules.

**Step 2: Run all tests**

```bash
pytest tests/ -v --tb=short
```
Expected: 617+ existing tests + ~80-100 new tests all PASS

**Step 3: Final commit**

```bash
git commit -m "test: add integration tests for lists and reminders"
```

---

## Summary

| Task | Component | Estimated Tests |
|------|-----------|----------------|
| 1 | ListStore (SQLite) | 10 |
| 2 | ListManager (business logic) | 9 |
| 3 | ReminderStore (SQLite) | 11 |
| 4 | Recurrence engine | 11 |
| 5 | ReminderManager (business logic) | 8 |
| 6 | ReminderScheduler (async delivery) | 5 |
| 7 | Dispatcher integration | 9 |
| 8 | Config (settings.yaml) | 0 |
| 9 | Main.py DI wiring | 0 |
| 10 | REST API endpoints | 8 |
| 11 | HA sync | ~5 |
| 12 | Integration tests | ~5 |
| **Total** | | **~81** |

**Dependencies:**
- Tasks 1-6 can be parallelized in pairs: (1,3), (2,5), (4,6)
- Task 7 depends on Task 1-6 (needs PathTypes and managers)
- Task 8 is independent
- Task 9 depends on 1-8
- Task 10 depends on 1-5, 9
- Task 11 depends on 1-2
- Task 12 depends on all
