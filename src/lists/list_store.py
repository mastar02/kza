"""SQLite persistence for user lists and list items."""
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

import aiosqlite

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

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open database and create tables if needed."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS lists (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS list_items (
                id TEXT PRIMARY KEY,
                list_id TEXT NOT NULL,
                text TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                added_by TEXT,
                created_at REAL NOT NULL,
                completed_at REAL,
                FOREIGN KEY (list_id) REFERENCES lists(id)
            )
            """
        )
        await self._db.commit()
        logger.info("ListStore initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def create_list(
        self, name: str, owner_type: OwnerType | str, owner_id: str | None
    ) -> UserList:
        """Create a new list and return it."""
        now = time.time()
        list_id = _generate_id()
        await self._db.execute(
            "INSERT INTO lists (id, name, owner_type, owner_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (list_id, name, owner_type, owner_id, now, now),
        )
        await self._db.commit()
        logger.info("Created list '%s' (id=%s, owner_type=%s)", name, list_id, owner_type)
        return UserList(
            id=list_id,
            name=name,
            owner_type=OwnerType(owner_type),
            owner_id=owner_id,
            created_at=now,
            updated_at=now,
        )

    async def get_lists_for_user(self, user_id: str) -> list[UserList]:
        """Return all lists owned by user_id plus all shared lists."""
        cursor = await self._db.execute(
            "SELECT * FROM lists WHERE owner_id = ? OR owner_type = 'shared'",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_list(row) for row in rows]

    async def find_list_by_name(
        self, name: str, user_id: str
    ) -> UserList | None:
        """Find a list by name (case-insensitive) for user or shared."""
        cursor = await self._db.execute(
            "SELECT * FROM lists WHERE LOWER(name) = LOWER(?) "
            "AND (owner_id = ? OR owner_type = 'shared') LIMIT 1",
            (name, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_list(row)

    async def add_item(
        self, list_id: str, text: str, added_by: str | None = None
    ) -> ListItem:
        """Add an item to a list."""
        now = time.time()
        item_id = _generate_id()
        await self._db.execute(
            "INSERT INTO list_items (id, list_id, text, completed, added_by, created_at) "
            "VALUES (?, ?, ?, 0, ?, ?)",
            (item_id, list_id, text, added_by, now),
        )
        await self._db.commit()
        return ListItem(
            id=item_id,
            list_id=list_id,
            text=text,
            completed=False,
            added_by=added_by,
            created_at=now,
        )

    async def get_items(self, list_id: str) -> list[ListItem]:
        """Get all items for a given list."""
        cursor = await self._db.execute(
            "SELECT * FROM list_items WHERE list_id = ? ORDER BY created_at",
            (list_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_item(row) for row in rows]

    async def remove_item(self, item_id: str) -> None:
        """Delete an item by its ID."""
        await self._db.execute("DELETE FROM list_items WHERE id = ?", (item_id,))
        await self._db.commit()

    async def complete_item(self, item_id: str) -> ListItem | None:
        """Mark an item as completed. Returns None if item not found."""
        now = time.time()
        await self._db.execute(
            "UPDATE list_items SET completed = 1, completed_at = ? WHERE id = ?",
            (now, item_id),
        )
        await self._db.commit()
        cursor = await self._db.execute(
            "SELECT * FROM list_items WHERE id = ?", (item_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    async def clear_list(self, list_id: str) -> None:
        """Remove all items from a list."""
        await self._db.execute(
            "DELETE FROM list_items WHERE list_id = ?", (list_id,)
        )
        await self._db.commit()

    async def delete_list(self, list_id: str) -> None:
        """Delete a list and all its items."""
        await self._db.execute(
            "DELETE FROM list_items WHERE list_id = ?", (list_id,)
        )
        await self._db.execute("DELETE FROM lists WHERE id = ?", (list_id,))
        await self._db.commit()
        logger.info("Deleted list id=%s", list_id)

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
