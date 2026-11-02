"""Queued SQLite work preserves authority, transaction ownership and actual drain."""

import asyncio
from contextvars import ContextVar
import logging
import sqlite3
import threading
from types import SimpleNamespace

import aiosqlite
import pytest

from src.core.owned_work import WorkCloseTimeout
from src.core.sqlite_work import SQLiteWork

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def native_store(tmp_path):
    state = SimpleNamespace(hold_commit=False, fail_rollback=False,
        commit_entered=threading.Event(), commit_release=threading.Event(), releases=[])

    class ObservedConnection(sqlite3.Connection):
        def commit(self):
            super().commit()
            if state.hold_commit:
                state.commit_entered.set()
                assert state.commit_release.wait(5)

        def rollback(self):
            if state.fail_rollback:
                raise sqlite3.OperationalError('synthetic_rollback_failure')
            super().rollback()

    path = tmp_path / 'native.db'
    database = await aiosqlite.connect(path, factory=ObservedConnection, timeout=1)
    await database.execute('CREATE TABLE entries (value TEXT PRIMARY KEY)')
    await database.commit()
    work = SQLiteWork(database, close_timeout_s=.05)
    try:
        yield SimpleNamespace(database=database, work=work, state=state, path=path)
    finally:
        state.commit_release.set()
        for release in state.releases:
            release.set()
        await work.aclose()
        await database.close()


def insert(value):
    def operation(database):
        database.execute('INSERT INTO entries VALUES (?)', (value,))
        return value
    return operation


def values(database):
    return [row[0] for row in database.execute('SELECT value FROM entries ORDER BY value').fetchall()]


async def test_revoked_authority_is_checked_inside_queued_callback(native_store):
    graph = native_store
    entered, release, current = threading.Event(), threading.Event(), threading.Event()
    graph.state.releases.append(release)
    current.set()

    def block(database):
        entered.set()
        assert release.wait(5)

    first = asyncio.create_task(graph.work.read(block))
    assert await asyncio.to_thread(entered.wait, 3)
    pending = asyncio.create_task(graph.work.write(insert('withdrawn'), guard=current.is_set))
    await asyncio.sleep(0)
    current.clear()
    release.set()
    await first
    with pytest.raises(PermissionError):
        await pending
    assert await graph.work.write(insert('authorized')) == 'authorized'
    assert await graph.work.read(values) == ['authorized']


async def test_authority_is_rechecked_after_real_sqlite_writer_lock(native_store):
    graph = native_store
    entered, current = threading.Event(), threading.Event()
    current.set()
    await graph.database.set_trace_callback(lambda sql: entered.set() if sql == 'BEGIN IMMEDIATE' else None)
    with sqlite3.connect(graph.path) as competing:
        competing.execute('BEGIN IMMEDIATE')
        pending = asyncio.create_task(graph.work.write(insert('withdrawn'), guard=current.is_set))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            current.clear()
        finally:
            competing.commit()
        with pytest.raises(PermissionError):
            await pending
    assert await graph.work.read(values) == []


@pytest.mark.parametrize('cancel_observer', [False, True])
async def test_provisional_write_rolls_back_before_other_writer_or_reader(native_store, cancel_observer):
    graph = native_store
    entered, release, current = threading.Event(), threading.Event(), threading.Event()
    graph.state.releases.append(release)
    current.set()

    def provisional(database):
        insert('withdrawn')(database)
        entered.set()
        assert release.wait(5)

    pending = asyncio.create_task(graph.work.write(provisional, guard=current.is_set))
    assert await asyncio.to_thread(entered.wait, 3)
    independent = asyncio.create_task(graph.work.write(insert('authorized')))
    reader = asyncio.create_task(graph.work.read(values))
    await asyncio.sleep(0)
    if cancel_observer:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    current.clear()
    release.set()
    if not cancel_observer:
        with pytest.raises(PermissionError):
            await pending
    assert await independent == 'authorized'
    assert await reader == ['authorized']
    assert await graph.work.read(values) == ['authorized']


async def test_cancelled_commit_observer_retains_actual_worker_and_close_drain(native_store):
    graph = native_store
    graph.state.hold_commit = True
    pending = asyncio.create_task(graph.work.write(insert('committed')))
    assert await asyncio.to_thread(graph.state.commit_entered.wait, 3)
    with sqlite3.connect(graph.path) as independent:
        assert values(independent) == ['committed']
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    with pytest.raises(WorkCloseTimeout):
        await graph.work.aclose()
    with pytest.raises(RuntimeError, match='blocking_work_closed'):
        await graph.work.read(values)
    graph.state.commit_release.set()
    await graph.work.aclose()
    await graph.work.aclose()
    with sqlite3.connect(graph.path) as independent:
        assert values(independent) == ['committed']


async def test_foreign_transaction_is_rejected_without_rollback_or_commit(native_store):
    graph = native_store
    await graph.database.execute("INSERT INTO entries VALUES ('foreign')")
    with pytest.raises(RuntimeError, match='sqlite_foreign_transaction'):
        await graph.work.write(insert('unexpected'))
    await graph.database.commit()
    assert await graph.work.read(values) == ['foreign']


async def test_failed_rollback_poisons_connection_before_any_later_projection(native_store):
    graph = native_store
    current = threading.Event()
    current.set()

    def provisional(database):
        insert('uncommitted')(database)
        current.clear()
        graph.state.fail_rollback = True

    with pytest.raises(sqlite3.OperationalError, match='synthetic_rollback_failure'):
        await graph.work.write(provisional, guard=current.is_set)
    with pytest.raises(RuntimeError, match='sqlite_connection_unavailable'):
        await graph.work.read(values)
    with pytest.raises(RuntimeError, match='sqlite_connection_unavailable'):
        await graph.work.write(insert('unexpected'))
    with sqlite3.connect(graph.path) as independent:
        assert values(independent) == []


@pytest.mark.parametrize('allowed', [False, None, 1, 'true'])
async def test_native_authority_requires_exact_true(native_store, allowed):
    with pytest.raises(PermissionError):
        await native_store.work.write(insert('unexpected'), guard=lambda: allowed)
    assert await native_store.work.read(values) == []


async def test_native_guard_failure_is_denied_and_a_later_writer_progresses(native_store):
    def unavailable():
        raise RuntimeError('synthetic authority unavailable')
    with pytest.raises(PermissionError, match='sqlite_authority_unavailable'):
        await native_store.work.write(insert('unexpected'), guard=unavailable)
    assert await native_store.work.write(insert('authorized')) == 'authorized'


async def test_private_caller_context_and_values_do_not_reach_worker_context_or_debug_repr(native_store, caplog):
    private = 'Synthetic private invocation and SQL value.'
    marker = ContextVar('synthetic_private_caller', default=None)
    token = marker.set(private)
    caplog.set_level(logging.DEBUG, logger='aiosqlite')
    try:
        def operation(database):
            assert marker.get() is None
            return insert(private)(database)
        assert await native_store.work.write(operation) == private
        assert await native_store.work.read(values) == [private]
    finally:
        marker.reset(token)
    assert private not in caplog.text
    assert all(row['native_ms'] is not None for row in native_store.work._work.diagnostics()['recent'])


async def test_live_cursor_cannot_escape_a_native_transaction(native_store):
    with pytest.raises(TypeError, match='sqlite_detached_result_required'):
        await native_store.work.write(lambda database: database.execute("INSERT INTO entries VALUES ('provisional')"))
    assert await native_store.work.read(values) == []


async def test_accidental_write_from_read_callback_rolls_back_its_own_transaction(native_store):
    def invalid_read(database):
        insert('provisional')(database)
    with pytest.raises(RuntimeError, match='sqlite_read_started_transaction'):
        await native_store.work.read(invalid_read)
    assert await native_store.work.read(values) == []
