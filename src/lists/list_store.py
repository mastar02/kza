"""SQLite persistence for user lists and list items."""
import asyncio
from contextvars import Context
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

import aiosqlite

from src.core.owned_work import ClosePhase, WorkCloseTimeout
from src.core.sqlite_work import AuthorityGuard, SQLiteWork
from src.memory.output_registration import (
    DestinationRegistration, before_store_open, store_open_returned, store_close_returned,
)

logger = logging.getLogger(__name__)


class OwnerType(StrEnum):
    USER = "user"
    SHARED = "shared"


@dataclass
class ListItem:
    """A single item within a list."""

    id: str
    list_id: str
    text: str
    completed: bool = False
    added_by: str | None = None
    created_at: float = 0.0
    completed_at: float | None = None


@dataclass
class UserList:
    """A named list owned by a user or shared."""

    id: str
    name: str
    owner_type: OwnerType
    owner_id: str | None
    created_at: float = 0.0
    updated_at: float = 0.0
    items: list[ListItem] = field(default_factory=list)


def _generate_id() -> str:
    """Generate a short unique identifier."""
    return uuid.uuid4().hex[:12]


class ListStore:
    """SQLite-backed storage for lists and their items."""

    def __init__(self, db_path: str, *, registration: "DestinationRegistration | None" = None) -> None:
        self._db_path = db_path
        self._destination_registration = registration
        self._db: aiosqlite.Connection | None = None
        self._work: SQLiteWork | None = None
        self._closing: asyncio.Task | None = None
        self._lifecycle = asyncio.Lock()

    async def initialize(self) -> None:
        """Prepare schema before admitting consumers to the shared native owner."""
        if self._closing is not None and not self._closing.done():
            raise RuntimeError("list_store_closing")
        async with self._lifecycle:
            if self._db is not None:
                raise RuntimeError("list_store_already_initialized")
            if self._closing is not None and not self._closing.done():
                raise RuntimeError("list_store_closing")
            self._closing = None
            await before_store_open(self, "lists", self._db_path, self._destination_registration)
            self._db = await aiosqlite.connect(self._db_path)
            if self._closing is not None:
                raise RuntimeError("list_store_closing")
            self._db.row_factory = aiosqlite.Row
            self._work = SQLiteWork(self._db)
            await self._work.read(lambda db: db.execute("PRAGMA foreign_keys = ON").fetchall())

            def schema(db):
                db.execute("""
                    CREATE TABLE IF NOT EXISTS lists (
                        id TEXT PRIMARY KEY, name TEXT NOT NULL, owner_type TEXT NOT NULL,
                        owner_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
                    )
                """)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS list_items (
                        id TEXT PRIMARY KEY, list_id TEXT NOT NULL, text TEXT NOT NULL,
                        completed INTEGER NOT NULL DEFAULT 0, added_by TEXT,
                        created_at REAL NOT NULL, completed_at REAL,
                        FOREIGN KEY (list_id) REFERENCES lists(id)
                    )
                """)
            await self._work.write(schema)
            await store_open_returned(self)
            logger.info("ListStore initialized at %s", self._db_path)

    async def close(self) -> None:
        """Retain actual drain and connection close after an observer cancellation."""
        if self._closing is None:
            if self._work is not None:
                self._work.begin_close()

            async def finish():
                async with self._lifecycle:
                    work, database = self._work, self._db
                    if work is not None:
                        await work.begin_close()
                    if database is not None:
                        await database.close()
                    if self._db is database:
                        self._db = None
                    if self._work is work:
                        self._work = None
                    if database is not None:
                        await store_close_returned(self)

            self._closing = asyncio.create_task(finish(), context=Context())
            def observed(done):
                if not done.cancelled():
                    done.exception()
            self._closing.add_done_callback(observed, context=Context())
        closing = self._closing
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(closing)
        except TimeoutError as error:
            raise WorkCloseTimeout("list_store_close_pending", phase=ClosePhase.DRAIN,
                diagnostics=self._work.diagnostics() if self._work is not None else {}) from error

    async def create_list(
        self, name: str, owner_type: OwnerType | str, owner_id: str | None, *,
        guard: AuthorityGuard | None = None, shared_only: bool = False,
    ) -> UserList:
        """Commit a new list under its original native authority and scope."""
        owner_type = OwnerType(owner_type)
        if shared_only:
            if owner_type is not OwnerType.SHARED:
                raise ValueError("shared_list_required")
            owner_id = None
        now, list_id = time.time(), _generate_id()
        def operation(db):
            db.execute(
                "INSERT INTO lists (id, name, owner_type, owner_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (list_id, name, owner_type.value, owner_id, now, now))
            return UserList(id=list_id, name=name, owner_type=owner_type, owner_id=owner_id,
                            created_at=now, updated_at=now)
        return await self._work.write(operation, guard=guard)

    async def get_lists_for_user(self, user_id: str | None, *,
                                 guard: AuthorityGuard | None = None,
                                 shared_only: bool = False) -> list[UserList]:
        """Project only eligible lists inside one native queue operation."""
        def operation(db):
            rows = db.execute("SELECT * FROM lists WHERE owner_type = 'shared' "
                              "OR (? = 0 AND owner_id = ?)", (shared_only, user_id)).fetchall()
            return [self._row_to_list(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def find_list_by_name(
        self, name: str, user_id: str | None, *, guard: AuthorityGuard | None = None,
        shared_only: bool = False,
    ) -> UserList | None:
        """Apply ownership before selecting a matching list or projecting its name."""
        def operation(db):
            row = db.execute("SELECT * FROM lists WHERE LOWER(name) = LOWER(?) "
                "AND (owner_type = 'shared' OR (? = 0 AND owner_id = ?)) LIMIT 1",
                (name, shared_only, user_id)).fetchone()
            return self._row_to_list(row) if row is not None else None
        return await self._work.read(operation, guard=guard)

    async def add_item(
        self, list_id: str, text: str, added_by: str | None = None, *,
        guard: AuthorityGuard | None = None, shared_only: bool = False,
    ) -> ListItem | None:
        """Check the current parent scope and commit its item in one transaction."""
        now, item_id = time.time(), _generate_id()
        if shared_only:
            added_by = None
        def operation(db):
            parent = db.execute("SELECT 1 FROM lists WHERE id = ? "
                "AND (? = 0 OR owner_type = 'shared')", (list_id, shared_only)).fetchone()
            if parent is None:
                return None
            db.execute("INSERT INTO list_items (id, list_id, text, completed, added_by, created_at) "
                "VALUES (?, ?, ?, 0, ?, ?)", (item_id, list_id, text, added_by, now))
            return ListItem(id=item_id, list_id=list_id, text=text, completed=False,
                            added_by=added_by, created_at=now)
        return await self._work.write(operation, guard=guard)

    async def get_items(self, list_id: str, *, guard: AuthorityGuard | None = None,
                        shared_only: bool = False) -> list[ListItem]:
        """Filter the parent scope before projecting any requested item."""
        def operation(db):
            rows = db.execute("SELECT * FROM list_items WHERE list_id = ? "
                "AND (? = 0 OR EXISTS (SELECT 1 FROM lists WHERE lists.id = list_items.list_id "
                "AND owner_type = 'shared')) ORDER BY created_at", (list_id, shared_only)).fetchall()
            return [self._row_to_item(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def remove_item(self, item_id: str, *, list_id: str | None = None,
                          guard: AuthorityGuard | None = None, shared_only: bool = False) -> bool:
        """Delete only from the selected list while its parent scope is eligible."""
        def operation(db):
            return db.execute("DELETE FROM list_items WHERE id = ? AND (? IS NULL OR list_id = ?) "
                "AND (? = 0 OR EXISTS (SELECT 1 FROM lists WHERE lists.id = list_items.list_id "
                "AND owner_type = 'shared'))", (item_id, list_id, list_id, shared_only)).rowcount > 0
        return await self._work.write(operation, guard=guard)

    async def complete_item(self, item_id: str, *, list_id: str | None = None,
                            guard: AuthorityGuard | None = None,
                            shared_only: bool = False) -> ListItem | None:
        """Commit completion only within the selected current parent scope."""
        now = time.time()
        def operation(db):
            changed = db.execute("UPDATE list_items SET completed = 1, completed_at = ? "
                "WHERE id = ? AND (? IS NULL OR list_id = ?) "
                "AND (? = 0 OR EXISTS (SELECT 1 FROM lists WHERE lists.id = list_items.list_id "
                "AND owner_type = 'shared'))", (now, item_id, list_id, list_id, shared_only)).rowcount
            if not changed:
                return None
            return self._row_to_item(db.execute("SELECT * FROM list_items WHERE id = ?", (item_id,)).fetchone())
        return await self._work.write(operation, guard=guard)

    async def clear_list(self, list_id: str, *, guard: AuthorityGuard | None = None,
                         shared_only: bool = False) -> bool:
        """Verify the parent scope before deleting any item in this transaction."""
        def operation(db):
            if db.execute("SELECT 1 FROM lists WHERE id = ? AND (? = 0 OR owner_type = 'shared')",
                          (list_id, shared_only)).fetchone() is None:
                return False
            db.execute("DELETE FROM list_items WHERE list_id = ?", (list_id,))
            return not db.execute("SELECT 1 FROM list_items WHERE list_id = ? LIMIT 1", (list_id,)).fetchone()
        return await self._work.write(operation, guard=guard)

    async def delete_list(self, list_id: str, *, guard: AuthorityGuard | None = None,
                          shared_only: bool = False) -> bool:
        """Delete an eligible parent and its children atomically under authority."""
        def operation(db):
            if db.execute("SELECT 1 FROM lists WHERE id = ? AND (? = 0 OR owner_type = 'shared')",
                          (list_id, shared_only)).fetchone() is None:
                return False
            db.execute("DELETE FROM list_items WHERE list_id = ?", (list_id,))
            return db.execute("DELETE FROM lists WHERE id = ?", (list_id,)).rowcount > 0
        return await self._work.write(operation, guard=guard)

    @staticmethod
    def _row_to_list(row: aiosqlite.Row) -> UserList:
        """Convert a database row to a UserList dataclass."""
        return UserList(
            id=row["id"],
            name=row["name"],
            owner_type=OwnerType(row["owner_type"]),
            owner_id=row["owner_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_item(row: aiosqlite.Row) -> ListItem:
        """Convert a database row to a ListItem dataclass."""
        return ListItem(
            id=row["id"],
            list_id=row["list_id"],
            text=row["text"],
            completed=bool(row["completed"]),
            added_by=row["added_by"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )
