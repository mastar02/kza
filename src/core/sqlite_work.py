"""Own complete SQLite operations on an existing aiosqlite worker."""

from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass
import asyncio
import sqlite3

import aiosqlite

from src.core.owned_work import OwnedBlockingWork, observe_native

AuthorityGuard = Callable[[], bool]


@dataclass
class _Health:
    poisoned: bool = False


def _require_current(guard: AuthorityGuard | None) -> None:
    """Check an explicit native capability; None selects trusted internal work."""
    if guard is not None:
        try:
            allowed = guard()
        except Exception:
            raise PermissionError('sqlite_authority_unavailable') from None
        if allowed is not True:
            raise PermissionError('sqlite_authority_denied')


def _native[T](connection: sqlite3.Connection, operation: Callable[[sqlite3.Connection], T],
               guard: AuthorityGuard | None, write: bool, health: _Health) -> T:
    """Run a detached read or an entire guarded transaction without queue gaps."""
    result = None
    admitted = False
    try:
        if health.poisoned:
            raise RuntimeError('sqlite_connection_unavailable')
        if connection.in_transaction:
            # It predates this callback; rollback would damage another owner.
            raise RuntimeError('sqlite_foreign_transaction')
        admitted = True
        _require_current(guard)
        if write:
            connection.execute('BEGIN IMMEDIATE')
            _require_current(guard)
        result = operation(connection)
        if isinstance(result, sqlite3.Cursor):
            result.close()
            raise TypeError('sqlite_detached_result_required')
        if not write and connection.in_transaction:
            raise RuntimeError('sqlite_read_started_transaction')
        _require_current(guard)
        if write:
            connection.commit()
        return result
    except BaseException:
        # The connection was idle at admission; any transaction here belongs to
        # this callback. A failed rollback prevents every subsequent projection.
        if admitted:
            try:
                if connection.in_transaction:
                    connection.rollback()
            except BaseException:
                health.poisoned = True
                raise
        raise
    finally:
        result = operation = guard = None


class SQLiteWork:
    """Bound and retain native operations; the injected connection stays caller-owned.

    Callbacks must return detached DTOs or values and must not commit or roll
    back themselves. After initialization all connection consumers must use this
    owner, so reads cannot interleave with provisional writes.
    """

    def __init__(self, connection: aiosqlite.Connection, *, capacity: int = 32,
                 close_timeout_s: float = 1):
        self._connection = connection
        self._health = _Health()
        self._work = OwnedBlockingWork(worker=self._invoke, capacity=capacity,
            close_timeout_s=close_timeout_s, isolate_context=True)

    async def _invoke[T](self, operation: Callable[[sqlite3.Connection], T], *,
                         guard: AuthorityGuard | None, write: bool,
                         publish: Callable[[T], None] | None = None) -> T:
        context = copy_context()
        result = None

        def invoke():
            return context.run(observe_native, _native, self._connection._conn,
                               operation, guard, write, self._health)

        try:
            # A no-argument callback keeps private arguments out of aiosqlite's
            # DEBUG partial repr. Only the isolated owner's timing context crosses.
            result = await self._connection._execute(invoke)
            if publish is not None:
                publish(result)
            return result
        finally:
            operation = guard = context = invoke = publish = result = None

    async def read[T](self, operation: Callable[[sqlite3.Connection], T], *,
                      guard: AuthorityGuard | None = None) -> T:
        """Project all rows in one retained native callback."""
        return await self._work.run(operation, guard=guard, write=False)

    async def write[T](self, operation: Callable[[sqlite3.Connection], T], *,
                       guard: AuthorityGuard | None = None,
                       publish: Callable[[T], None] | None = None) -> T:
        """Own native commit and its synchronous publication; never recheck after commit."""
        return await self._work.run(operation, guard=guard, write=True, publish=publish)

    async def aclose(self) -> None:
        """Stop admission and drain actual work before the caller closes SQLite."""
        await self._work.aclose()

    def begin_close(self) -> asyncio.Task:
        """Return actual drain ownership for a retained connection-close task."""
        return self._work.begin_close()

    async def drain_admitted(self) -> None:
        """Drain the current operations, including publication, keeping admission open."""
        await self._work.drain_admitted()

    def diagnostics(self) -> dict:
        """Report bounded numeric work evidence without retained SQL values."""
        return self._work.diagnostics()
