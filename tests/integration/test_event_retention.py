"""Feature-independent physical event TTL with real SQLite and source authority."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import sqlite3
import threading
import time

import pytest

from src.analytics.erasure import build_event_erasure
from src.analytics.event_logger import EventLogger
from src.analytics.event_storage import event_descriptor, sweep_events
from src.analytics.retention import EventRetention
from src.analytics.services import build_event_logger
from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(tmp_path, *, days=90, background=False, clock=None):
    clock = clock or Clock()
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(
            str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=0.01
        )
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        retention = EventRetention(privacy, retention_days=days)
        privacy.add_borrower(retention.begin_close)

        async def close_retention():
            await retention.begin_close()

        stack.push_async_callback(close_retention)
        events = EventLogger(
            str(tmp_path / "events.db"),
            source_policy=privacy.ledger,
            clock=clock,
            retention_notify=retention.notify,
        )
        stack.push_async_callback(events.aclose)
        privacy.add_borrower(events.begin_close)
        await events.acall(events.initialize)
        if background:
            retention.start()
        yield privacy, retention, events, clock


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def drain(retention):
    for _ in range(10):
        await retention.reconcile()
        if not retention._active:
            return
        await asyncio.gather(*(asyncio.shield(t) for t in retention._active.values()))
    raise AssertionError("synthetic retention did not quiesce")


def rows(path):
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT trigger_phrase,user_name,context_json,expires_at,text_expires_at FROM events_policy_v1 ORDER BY id"
        ).fetchall()


async def test_raw_deadline_scrubs_all_text_but_keeps_numeric_metadata_without_erase_request(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, retention, events, clock):
        source = memory_source(now=clock.now, ttl=60)
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            user_name="synthetic name",
            extra_context={"private": "synthetic private copy", "rms": 0.2},
            sources=(source,),
        )
        clock.now += 60
        await drain(retention)
        result = rows(events.db_path)[0]
        assert result[:2] == (None, None) and json.loads(result[2]) == {"rms": 0.2}
        assert result[-1] == source.expires_at
        assert not privacy.ledger.conversation_revoked(source.conversation_id)
        clock.now += 90 * 86400
        await drain(retention)
        assert rows(events.db_path) == []


async def test_metadata_expiry_and_raw_dates_survive_restart_and_reads_do_not_renew(
    tmp_path,
):
    async with graph(tmp_path, days=3) as (_, retention, events, clock):
        source = memory_source(now=clock.now)
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(source,),
        )
        await drain(retention)
        original = rows(events.db_path)[0]
        assert original[3] == clock.now + 3 * 86400
    async with graph(tmp_path, days=90, clock=clock) as (_, retention, events, clock):
        await events.acall(events.get_events)
        assert rows(events.db_path)[0][3:] == original[3:]
        clock.now += 3 * 86400
        await drain(retention)
        assert rows(events.db_path) == []


async def test_existing_earlier_raw_deadline_is_never_extended_by_source_deadline(
    tmp_path,
):
    async with graph(tmp_path) as (_, retention, events, clock):
        source = memory_source(now=clock.now, ttl=600)
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(source,),
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "UPDATE events_policy_v1 SET text_expires_at=?", (clock.now + 30,)
            )
        await drain(retention)
        assert rows(events.db_path)[0][-1] == clock.now + 30
        clock.now += 30
        await drain(retention)
        assert rows(events.db_path)[0][0] is None


async def test_disabled_factory_cleans_historical_bases_and_leaves_legacy_unchanged(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, _, events, _):
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(memory_source(now=clock.now),),
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute("CREATE TABLE events (raw TEXT)")
            db.execute("INSERT INTO events VALUES ('synthetic unsourced legacy')")
    clock.now += 48 * 3600
    config = {"analytics": {"enabled": False, "events_db": str(tmp_path / "new.db")}}
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path / "privacy.db"), clock=clock)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        driver = await build_event_erasure(config, privacy=privacy, owner=owner)
        assert (
            await build_event_logger(
                config, shutdown=stack, privacy=privacy, retention=driver.retention
            )
            is None
        )
        await drain(driver.retention)
        assert rows(tmp_path / "events.db")[0][0] is None
        assert not (tmp_path / "new.db").exists()
        with sqlite3.connect(tmp_path / "events.db") as db:
            assert db.execute("SELECT * FROM events").fetchall() == [
                ("synthetic unsourced legacy",)
            ]


async def test_new_earlier_write_wakes_sleeping_retention_at_actual_deadline(tmp_path):
    async with graph(tmp_path, background=True, clock=time.time) as (
        _,
        retention,
        events,
        _,
    ):
        await until(lambda: retention._status and not retention._active)
        assert min(retention._due.values()) > time.time() + 20
        source = memory_source(now=time.time(), ttl=0.3)
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic short-lived text",
            sources=(source,),
        )
        await until(lambda: rows(events.db_path) and rows(events.db_path)[0][0] is None)
        assert rows(events.db_path)[0][-1] == source.expires_at


async def test_corrupt_and_unsupported_targets_remain_visible_while_valid_target_finishes(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, retention, events, clock):
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_text("synthetic invalid database")
        for version, path in ((1, corrupt), (2, tmp_path / "unknown.db")):
            await privacy.acall(
                privacy.ledger.erasures.register_target,
                ErasureParticipant.EVENTS,
                version,
                event_descriptor(path),
            )
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(memory_source(now=clock.now, ttl=30),),
        )
        clock.now += 30
        await drain(retention)
        assert rows(events.db_path)[0][0] is None
        result = retention.status()
        assert {row["error"] for row in result["targets"]} == {
            None,
            "event_retention_failed",
            "unsupported_event_retention_target",
        }
        assert (
            str(tmp_path) not in json.dumps(result)
            and not (tmp_path / "unknown.db").exists()
        )


@pytest.mark.parametrize(
    "field,value",
    [("sources_json", "[{}]"), ("expires_at", "unknown"), ("text_expires_at", None)],
)
async def test_unknown_metadata_rolls_back_batch_and_stays_pending(
    tmp_path, field, value
):
    async with graph(tmp_path) as (_, retention, events, clock):
        for _ in range(2):
            await events.alog(
                "light.synthetic",
                "turn_on",
                trigger_phrase="synthetic private text",
                sources=(memory_source(now=clock.now, ttl=30),),
            )
        with sqlite3.connect(events.db_path) as db:
            db.execute(f"UPDATE events_policy_v1 SET {field}=? WHERE id=2", (value,))
        clock.now += 30
        await drain(retention)
        assert all(row[0] == "synthetic private text" for row in rows(events.db_path))
        assert retention.status()["targets"][0]["error"] == "event_retention_failed"


async def test_native_batches_limit_128_rows_and_keep_cursor_until_tail_is_scrubbed(
    tmp_path, monkeypatch
):
    import src.analytics.retention as module

    async with graph(tmp_path) as (_, retention, events, clock):
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(memory_source(now=clock.now, ttl=30),),
        )
        with sqlite3.connect(events.db_path) as db:
            names = ",".join(
                r[1]
                for r in db.execute("PRAGMA table_info(events_policy_v1)")
                if r[1] != "id"
            )
            for _ in range(256):
                db.execute(
                    f"INSERT INTO events_policy_v1 ({names}) SELECT {names} FROM events_policy_v1 WHERE id=1"
                )
        clock.now += 30
        original = module.sweep_events
        receipts = []

        def observe(*args):
            result = original(*args)
            receipts.append(result)
            return result

        monkeypatch.setattr(module, "sweep_events", observe)
        await drain(retention)
        assert [r.scrubbed for r in receipts] == [128, 128, 1]
        assert [r.cursor for r in receipts] == [128, 256, None]
        assert all(row[0] is None for row in rows(events.db_path))


async def test_late_harvest_uses_computation_time_instead_of_renewing_wait(tmp_path):
    async with graph(tmp_path) as (_, retention, events, clock):
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(memory_source(now=clock.now, ttl=30),),
        )
        await retention.reconcile()
        await asyncio.gather(*retention._active.values())
        clock.now += 30
        await drain(retention)
        assert rows(events.db_path)[0][0] is None


async def test_pending_failed_fact_commit_denies_reads_but_does_not_destroy_valid_events(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, retention, events, clock):
        from dataclasses import replace
        from src.memory.fact_ledger import FactDependency

        source = memory_source(now=clock.now)
        fact = await privacy.acall(
            privacy.ledger.store, "synthetic valid fact", "fact", 0.8, (source,)
        )
        derived = replace(
            memory_source(now=clock.now),
            fact_dependencies=(
                FactDependency(fact.id, fact.revision, fact.expires_at),
            ),
        )
        await events.alog("light.synthetic", "turn_on", sources=(derived,))
        entered, release = threading.Event(), threading.Event()
        db = privacy.ledger._db

        class FailingCommit:
            def __getattr__(self, name):
                return getattr(db, name)

            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError("synthetic failed commit")

        privacy.ledger._db = FailingCommit()
        deleting = asyncio.create_task(
            asyncio.to_thread(
                privacy.ledger.delete_fact, fact.id, MemoryReviewer("synthetic-admin")
            )
        )
        try:
            await until(entered.is_set)
            assert not privacy.sources_available((derived,))
            result = await retention._work.run(
                sweep_events, event_descriptor(events.db_path), 0, privacy.ledger, 90
            )
            assert result.deleted == 0 and len(rows(events.db_path)) == 1
            assert await events.acall(events.cleanup_old_events) == 0
        finally:
            release.set()
            with pytest.raises(OSError, match="synthetic failed commit"):
                await deleting
            privacy.ledger._db = db
        assert privacy.sources_available((derived,))
        assert len(rows(events.db_path)) == 1


async def test_cancelled_observer_keeps_native_retention_work_and_privacy_lease(
    tmp_path, monkeypatch
):
    import src.analytics.retention as module

    async with graph(tmp_path) as (privacy, retention, events, clock):
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(memory_source(now=clock.now, ttl=30),),
        )
        clock.now += 30
        entered, release = threading.Event(), threading.Event()
        original = module.sweep_events

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(module, "sweep_events", blocked)
        await retention.reconcile()
        await until(entered.is_set)
        try:
            # Cancelling a separate observer cannot cancel the retained task.
            observing = asyncio.Event()

            async def watch():
                observing.set()
                await asyncio.shield(next(iter(retention._active.values())))

            observer = asyncio.create_task(watch())
            await observing.wait()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert (
                privacy.resource_status()["writer_lease"]
                and retention._work.status()["operations"] == 1
            )
        finally:
            release.set()
            await privacy.begin_close()
        assert (
            not privacy.resource_status()["writer_lease"]
            and rows(events.db_path)[0][0] is None
        )


async def test_saturated_default_executor_does_not_starve_event_retention(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        release.wait(3)

    waiting = asyncio.create_task(asyncio.to_thread(blocked))
    try:
        await until(entered.is_set)
        async with graph(tmp_path) as (_, retention, events, clock):
            await events.alog(
                "light.synthetic",
                "turn_on",
                trigger_phrase="synthetic private text",
                sources=(memory_source(now=clock.now, ttl=30),),
            )
            clock.now += 30
            await drain(retention)
            assert rows(events.db_path)[0][0] is None
            release.set()
    finally:
        release.set()
        await waiting


async def test_blocked_historical_sqlite_does_not_hold_authority_or_other_native_sweep(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, retention, events, clock):
        other = EventLogger(
            str(tmp_path / "locked.db"), source_policy=privacy.ledger, clock=clock
        )
        await other.acall(other.initialize)
        await other.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic locked row",
            sources=(memory_source(now=clock.now, ttl=30),),
        )
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic available row",
            sources=(memory_source(now=clock.now, ttl=30),),
        )
        clock.now += 30
        lock = sqlite3.connect(other.db_path)
        lock.execute("BEGIN IMMEDIATE")
        try:
            await retention.reconcile()
            await until(lambda: rows(events.db_path)[0][0] is None)
            assert any(not task.done() for task in retention._active.values())
            # The blocked event DB never holds the authority's transaction lock.
            await privacy.acall(privacy.ledger.status)
        finally:
            lock.rollback()
            lock.close()
            await drain(retention)
            await other.aclose()
        assert rows(other.db_path)[0][0] is None


async def test_failed_native_sweep_retains_expired_data_for_retry_after_restart(
    tmp_path, monkeypatch
):
    import src.analytics.retention as module

    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, retention, events, _):
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic retry row",
            sources=(memory_source(now=clock.now, ttl=30),),
        )
        clock.now += 30
        with monkeypatch.context() as patch:

            def fail(*args):
                raise OSError("synthetic private path must not appear in status")

            patch.setattr(module, "sweep_events", fail)
            await drain(retention)
        assert rows(events.db_path)[0][0] == "synthetic retry row"
        assert "synthetic private path" not in json.dumps(retention.status())
    async with graph(tmp_path, clock=clock) as (_, retention, events, _):
        await drain(retention)
        assert rows(events.db_path)[0][0] is None


async def test_fact_dependency_expiry_removes_event_metadata_without_erase_operation(
    tmp_path,
):
    from dataclasses import replace
    from src.memory.fact_ledger import FactDependency

    async with graph(tmp_path) as (privacy, retention, events, clock):
        source = memory_source(now=clock.now)
        fact = await privacy.acall(
            privacy.ledger.store, "synthetic automatic fact", "fact", 0.8, (source,)
        )
        derived = replace(
            memory_source(now=clock.now),
            fact_dependencies=(
                FactDependency(fact.id, fact.revision, fact.expires_at),
            ),
        )
        await events.alog("light.synthetic", "turn_on", sources=(derived,))
        clock.now += 30 * 86400
        await drain(retention)
        assert rows(events.db_path) == []
        with sqlite3.connect(privacy.ledger.path) as db:
            assert (
                db.execute("SELECT COUNT(*) FROM conversation_erasures").fetchone()[0]
                == 0
            )


async def test_disabled_native_retention_never_imports_models(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import asyncio,importlib.abc,sqlite3,sys
        from contextlib import AsyncExitStack
        from pathlib import Path
        from uuid import uuid4
        import json
        class RejectModels(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,*args):
                if fullname.split('.')[0] in {'torch','chromadb','sentence_transformers','faster_whisper','onnxruntime'}:
                    raise AssertionError('disabled retention imported a model')
        sys.meta_path.insert(0,RejectModels())
        from src.memory.privacy_services import build_privacy_services
        from src.memory.erasure_coordinator import ErasureCoordinator
        from src.memory.fact_ledger import MemorySource,MemoryOrigin
        from src.analytics.erasure import build_event_erasure
        async def main():
            root=Path(sys.argv[1]); now=2000000000.
            source=MemorySource(str(uuid4()),str(uuid4()),now-100,now-1,MemoryOrigin.VOICE)
            path=root/'events.db'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE events_policy_v1 (id INTEGER PRIMARY KEY,timestamp REAL,expires_at REAL,text_expires_at REAL,sources_json TEXT,trigger_phrase TEXT,user_name TEXT,context_json TEXT,metrics_json TEXT)')
                db.execute('INSERT INTO events_policy_v1 VALUES (1,?,?,?,?,?,?,?,?)',(now-100,now+86400,now-1,json.dumps([source.to_dict()]),'synthetic private text',None,'{}','{}'))
            config={'memory':{'enabled':False,'privacy_db_path':str(root/'privacy.db')},'analytics':{'enabled':False,'events_db':str(path)}}
            async with AsyncExitStack() as stack:
                privacy=await build_privacy_services(config,shutdown=stack,clock=lambda:now)
                owner=ErasureCoordinator(privacy); stack.push_async_callback(owner.aclose)
                driver=await build_event_erasure(config,privacy=privacy,owner=owner)
                for _ in range(4):
                    await driver.retention.reconcile()
                    await asyncio.gather(*driver.retention._active.values())
                with sqlite3.connect(path) as db:
                    assert db.execute('SELECT trigger_phrase FROM events_policy_v1').fetchone()[0] is None
        asyncio.run(main())
    """)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


async def test_large_histories_yield_between_batches_to_other_registered_databases(
    tmp_path, monkeypatch
):
    async with graph(tmp_path) as (privacy, retention, events, clock):
        writers = [events]
        try:
            for n in range(2):
                writer = EventLogger(
                    str(tmp_path / f"history-{n}.db"),
                    source_policy=privacy.ledger,
                    clock=clock,
                )
                await writer.acall(writer.initialize)
                writers.append(writer)
            for writer in writers:
                await writer.alog(
                    "light.synthetic",
                    "turn_on",
                    trigger_phrase="synthetic private text",
                    sources=(memory_source(now=clock.now, ttl=30),),
                )
                with sqlite3.connect(writer.db_path) as db:
                    names = ",".join(
                        r[1]
                        for r in db.execute("PRAGMA table_info(events_policy_v1)")
                        if r[1] != "id"
                    )
                    for _ in range(256):
                        db.execute(
                            f"INSERT INTO events_policy_v1 ({names}) SELECT {names} FROM events_policy_v1 WHERE id=1"
                        )
            clock.now += 30
            starts = []
            run = retention._run

            async def observe(target, cursor):
                starts.append((target["id"], cursor))
                return await run(target, cursor)

            monkeypatch.setattr(retention, "_run", observe)
            await drain(retention)
            assert len({key for key, _ in starts[:3]}) == 3
            assert len(starts) == 9
            assert all(
                row[0] is None for writer in writers for row in rows(writer.db_path)
            )
        finally:
            for writer in writers[1:]:
                await writer.aclose()


async def test_runtime_factory_uses_bounded_retention_instead_of_a_second_logger_sweeper(
    tmp_path,
):
    config = {"analytics": {"events_db": str(tmp_path / "events.db")}}
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path / "privacy.db"))
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        driver = await build_event_erasure(config, privacy=privacy, owner=owner)
        events = await build_event_logger(
            config, shutdown=stack, privacy=privacy, retention=driver.retention
        )
        assert events._maintenance is None
        source = memory_source(now=time.time(), ttl=0.3)
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(source,),
        )
        await until(lambda: rows(events.db_path)[0][0] is None)
        assert rows(events.db_path)[0][-1] == source.expires_at


@pytest.mark.parametrize("encoded", ["{}", "[]", "[{}]"])
async def test_unverified_source_shapes_cannot_enter_any_live_event_projection(
    tmp_path, encoded
):
    async with graph(tmp_path) as (_, _, events, clock):
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(memory_source(now=clock.now),),
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute("UPDATE events_policy_v1 SET sources_json=?", (encoded,))
        assert await events.acall(events.get_events) == []
        assert (await events.acall(events.get_stats))["total_events"] == 0
        assert (
            sum(
                (
                    await events.acall(
                        events.get_hourly_distribution, "light.synthetic"
                    )
                ).values()
            )
            == 0
        )
        assert (
            sum(
                (
                    await events.acall(
                        events.get_weekday_distribution, "light.synthetic"
                    )
                ).values()
            )
            == 0
        )
        assert await events.acall(events.get_sequences, min_occurrences=1) == []
        assert rows(events.db_path)[0][0] == "synthetic private text"


async def test_source_expiry_hides_raw_text_even_when_persisted_deadline_is_later(
    tmp_path,
):
    async with graph(tmp_path) as (_, _, events, clock):
        source = memory_source(now=clock.now, ttl=30)
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            user_name="synthetic name",
            extra_context={"rms": 0.2, "private": "synthetic private context"},
            sources=(source,),
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "UPDATE events_policy_v1 SET text_expires_at=?", (clock.now + 600,)
            )
        clock.now += 30
        result = (await events.acall(events.get_events))[0]
        assert (
            result.trigger_phrase is None
            and result.user_name is None
            and result.context == {"rms": 0.2}
        )
        assert rows(events.db_path)[0][0] == "synthetic private text"


async def test_unrecognized_metadata_deadline_is_excluded_before_any_purge(tmp_path):
    async with graph(tmp_path) as (_, _, events, clock):
        await events.alog(
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic private text",
            sources=(memory_source(now=clock.now),),
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute("UPDATE events_policy_v1 SET expires_at='unknown'")
        assert await events.acall(events.get_events) == []
        assert (await events.acall(events.get_stats))["total_events"] == 0
        assert rows(events.db_path)[0][0] == "synthetic private text"
