"""Real policy-event cleanup, historical targets and late native writer fencing."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4

import pytest

from src.analytics.erasure import build_event_erasure, event_descriptor
from src.analytics.event_logger import EventLogger
from src.analytics.services import build_event_logger
from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant as Participant
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer("synthetic-admin")


@asynccontextmanager
async def graph(tmp_path, *, enabled=True, timeout=1):
    config = {
        "analytics": {"enabled": enabled, "events_db": str(tmp_path / "events.db")}
    }
    async with AsyncExitStack() as stack:
        clock = Clock()
        privacy = PrivacyServices(
            str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=timeout
        )
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy, close_timeout_s=timeout)
        stack.push_async_callback(owner.aclose)
        driver = await build_event_erasure(config, privacy=privacy, owner=owner)
        # This file observes conversation cleanup independently of TTL sweeps.
        driver.retention._maintenance.cancel()
        await asyncio.gather(driver.retention._maintenance, return_exceptions=True)
        events = await build_event_logger(config, shutdown=stack, privacy=privacy)
        if events is not None:
            events._maintenance.cancel()
            await asyncio.gather(events._maintenance, return_exceptions=True)
        yield privacy, owner, driver, events, clock, config


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def flush(owner):
    await owner.reconcile()
    await asyncio.gather(*(a.task for a in owner._active.values()))
    await owner.reconcile()


async def seed_event(events, *args, **kwargs):
    """Await the actual seed write; erasure setup has no caller-wait assertion."""
    return await events.acall(events.log, *args, **kwargs)


async def test_erasure_seed_waits_for_actual_writer_beyond_caller_observation_deadline(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        entered, release = threading.Event(), threading.Event()

        def context():
            entered.set()
            assert release.wait(3)
            return {}

        events.context_provider = context
        assert events.observation_timeout_s == .1
        observing = asyncio.create_task(events.alog('light.synthetic', 'turn_on', sources=(source,)))
        try:
            await until(entered.is_set)
            with pytest.raises(TimeoutError):
                await observing
            assert events.resource_status()['operations'] == 1
            assert stored_rows(events.db_path) == []
        finally:
            release.set()
            await until(lambda: events.resource_status()['operations'] == 0)
        assert len(stored_rows(events.db_path)) == 1

        entered.clear()
        release.clear()
        seeding = asyncio.create_task(seed_event(events, 'light.synthetic', 'turn_on', sources=(source,)))
        try:
            await until(entered.is_set)
            await asyncio.sleep(events.observation_timeout_s + .02)
            assert not seeding.done() and events.resource_status()['operations'] == 1
        finally:
            release.set()
            await seeding
        assert len(stored_rows(events.db_path)) == 2
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert stored_rows(events.db_path) == []
        assert (await status(privacy, receipt))[0]['state'] == 'complete'
        assert events.observation_timeout_s == .1


async def status(privacy, receipt):
    value = await privacy.acall(privacy.ledger.erasures.snapshot, receipt["id"])
    return [row for row in value["steps"] if row["participant"] == "events"]


def stored_rows(path):
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT id,trigger_phrase FROM events_policy_v1 ORDER BY id"
        ).fetchall()


async def test_real_event_erasure_removes_mixed_sources_and_keeps_unrelated_and_numeric_rows(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        erased, survivor = memory_source(now=clock.now), memory_source(now=clock.now)
        await seed_event(events,
            "light.first",
            "turn_on",
            trigger_phrase="synthetic erased",
            sources=(erased,),
        )
        await seed_event(events,
            "light.mixed",
            "turn_on",
            trigger_phrase="synthetic mixed",
            sources=(erased, survivor),
        )
        await seed_event(events,
            "light.survives",
            "turn_on",
            trigger_phrase="synthetic survives",
            sources=(survivor,),
        )
        await seed_event(events,
            "asr_quality:synthetic", "accepted:ok", extra_context={"rms": 0.2}
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute("CREATE TABLE events (private_text TEXT)")
            db.execute(
                "INSERT INTO events VALUES ('synthetic legacy without provenance')"
            )
        receipt = await owner.request(erased.conversation_id, ACTOR)
        assert len(stored_rows(events.db_path)) == 4
        assert len(await events.acall(events.get_events)) == 2
        await flush(owner)
        assert stored_rows(events.db_path) == [(3, "synthetic survives"), (4, None)]
        assert (await status(privacy, receipt))[0]["state"] == "complete"
        assert (await privacy.acall(privacy.ledger.erasures.snapshot, receipt["id"]))[
            "state"
        ] == "blocked"
        with sqlite3.connect(events.db_path) as db:
            assert db.execute("SELECT * FROM events").fetchall() == [
                ("synthetic legacy without provenance",)
            ]


async def test_indirect_event_fact_dependency_is_removed_after_source_conversation_erasure(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        fact = await privacy.acall(
            privacy.ledger.store, "synthetic dependency", "fact", 0.8, (source,)
        )
        from src.memory.fact_ledger import FactDependency

        derived = replace(
            memory_source(now=clock.now),
            fact_dependencies=(
                FactDependency(fact.id, fact.revision, fact.expires_at),
            ),
        )
        await seed_event(events,
            "light.derived",
            "turn_on",
            trigger_phrase="synthetic fact-derived response",
            sources=(derived,),
        )
        receipt = await owner.request(source.conversation_id, ACTOR)
        assert not privacy.ledger.conversation_revoked(derived.conversation_id)
        await flush(owner)
        assert stored_rows(events.db_path) == []
        assert (await status(privacy, receipt))[0]["state"] == "complete"


async def test_disabled_analytics_cleans_registered_historical_database_after_restart(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        await seed_event(events,
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic old database",
            sources=(source,),
        )
        receipt = await owner.request(source.conversation_id, ACTOR)
    async with graph(tmp_path, enabled=False) as (privacy, owner, _, events, _, config):
        assert events is None
        await flush(owner)
        assert stored_rows(config["analytics"]["events_db"]) == []
        assert (await status(privacy, receipt))[0]["state"] == "complete"


async def test_new_location_adds_target_to_prior_operation_and_preserves_unrelated_rows(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        await seed_event(events, "light.synthetic", "turn_on", sources=(source,))
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        other = EventLogger(str(tmp_path / "new.db"), source_policy=privacy.ledger)
        try:
            await other.acall(other.initialize)
            await other.alog(
                "light.live", "turn_on", sources=(memory_source(now=clock.now),)
            )
            await flush(owner)
            steps = await status(privacy, receipt)
            assert len(steps) == 2 and all(row["state"] == "complete" for row in steps)
            assert len(stored_rows(other.db_path)) == 1
        finally:
            await other.aclose()


async def test_writer_registers_actual_canonical_location_before_open_even_if_symlink_changes(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(a, target_is_directory=True)
        other = EventLogger(str(alias / "events.db"), source_policy=privacy.ledger)
        try:
            await other.acall(other.initialize)
            alias.unlink()
            alias.symlink_to(b, target_is_directory=True)
            source = memory_source(now=clock.now)
            await other.alog("light.synthetic", "turn_on", sources=(source,))
            assert other.db_path == a / "events.db" and not (b / "events.db").exists()
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            assert stored_rows(a / "events.db") == []
            assert all(
                row["state"] == "complete" for row in await status(privacy, receipt)
            )
        finally:
            await other.aclose()


async def test_absent_database_is_not_created_or_initialized_by_disabled_cleanup(
    tmp_path,
):
    async with graph(tmp_path, enabled=False) as (privacy, owner, _, _, _, config):
        path = Path(config["analytics"]["events_db"])
        assert not path.exists()
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert (
            not path.exists()
            and (await status(privacy, receipt))[0]["state"] == "complete"
        )


async def test_legacy_only_database_is_not_migrated_or_assigned_sources(tmp_path):
    path = tmp_path / "events.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE events (raw TEXT)")
        db.execute("INSERT INTO events VALUES ('synthetic unsourced legacy')")
    async with graph(tmp_path, enabled=False) as (privacy, owner, _, _, _, _):
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert (await status(privacy, receipt))[0]["state"] == "complete"
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT * FROM events").fetchall() == [
                ("synthetic unsourced legacy",)
            ]
            assert db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall() == [("events",)]


@pytest.mark.parametrize("encoded", ["not json", "{}", "[{}]", "[]"])
async def test_unknown_or_unattributed_event_sources_remain_pending_without_private_errors(
    tmp_path, encoded
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        await seed_event(events,
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic unknown raw text",
            sources=(source,),
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute("UPDATE events_policy_v1 SET sources_json=?", (encoded,))
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        step = (await status(privacy, receipt))[0]
        assert step["state"] == "pending" and step["last_error"] == "cleanup_failed"
        assert len(stored_rows(events.db_path)) == 1
        assert str(events.db_path) not in json.dumps(
            step
        ) and "synthetic unknown" not in json.dumps(step)


async def test_partial_event_batch_rolls_back_if_an_unknown_source_is_encountered(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        for _ in range(2):
            await seed_event(events, "light.synthetic", "turn_on", sources=(source,))
        with sqlite3.connect(events.db_path) as db:
            db.execute("UPDATE events_policy_v1 SET sources_json='[{}]' WHERE id=2")
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert len(stored_rows(events.db_path)) == 2
        assert (await status(privacy, receipt))[0]["state"] == "pending"


async def test_batches_scan_at_most_128_metadata_rows_and_cannot_skip_later_match(
    tmp_path, monkeypatch
):
    async with graph(tmp_path) as (privacy, owner, driver, events, clock, _):
        source, live = memory_source(now=clock.now), memory_source(now=clock.now)
        await seed_event(events, "light.live", "turn_on", sources=(live,))
        await seed_event(events, "light.erased", "turn_on", sources=(source,))
        with sqlite3.connect(events.db_path) as db:
            columns = [
                row[1]
                for row in db.execute("PRAGMA table_info(events_policy_v1)")
                if row[1] != "id"
            ]
            names = ",".join(columns)
            for _ in range(129):
                db.execute(
                    f"INSERT INTO events_policy_v1 ({names}) SELECT {names} FROM events_policy_v1 WHERE id=1"
                )
            db.execute(
                f"INSERT INTO events_policy_v1 ({names}) SELECT {names} FROM events_policy_v1 WHERE id=2"
            )
        calls = []
        clean = driver._clean

        def observe(step):
            result = clean(step)
            calls.append(result)
            return result

        monkeypatch.setattr(driver, "_clean", observe)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert calls[0].cursor == "128" and not calls[0].done
        assert len(stored_rows(events.db_path)) == 131
        clock.now += 0.02
        await flush(owner)
        assert len(calls) == 2 and calls[-1].done
        assert len(stored_rows(events.db_path)) == 130
        assert (await status(privacy, receipt))[0]["state"] == "complete"


async def test_leased_restart_reopens_completed_receipt_for_restored_event_row(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        await seed_event(events,
            "light.synthetic",
            "turn_on",
            trigger_phrase="synthetic restored row",
            sources=(source,),
        )
        with sqlite3.connect(events.db_path) as db:
            row = db.execute("SELECT * FROM events_policy_v1").fetchone()
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert stored_rows(events.db_path) == []
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "INSERT INTO events_policy_v1 VALUES ("
                + ",".join("?" for _ in row)
                + ")",
                row,
            )
    async with graph(tmp_path, enabled=False) as (privacy, owner, _, _, _, config):
        assert (await status(privacy, receipt))[0]["state"] == "pending"
        await flush(owner)
        assert stored_rows(config["analytics"]["events_db"]) == []
        assert (await status(privacy, receipt))[0]["state"] == "complete"


async def test_late_timed_out_writer_cannot_repopulate_after_event_cleanup(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        entered, release = threading.Event(), threading.Event()

        def context():
            entered.set()
            assert release.wait(3)
            return {"private": "synthetic late event"}

        events.context_provider = context
        events.observation_timeout_s = 0.01
        writing = asyncio.create_task(
            events.alog("light.synthetic", "turn_on", sources=(source,))
        )
        try:
            await until(entered.is_set)
            with pytest.raises(TimeoutError):
                await writing
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            assert (await status(privacy, receipt))[0]["state"] == "complete"
            assert events.resource_status()["operations"] == 1
        finally:
            release.set()
            await until(lambda: events.resource_status()["operations"] == 0)
        assert stored_rows(events.db_path) == []
        with pytest.raises(PermissionError):
            await events.acall(
                events.log, "light.synthetic", "turn_on", sources=(source,)
            )


async def test_writer_native_drain_keeps_privacy_lease_after_close_observation_timeout(
    tmp_path,
):
    async with graph(tmp_path, timeout=0.01) as (privacy, _, _, events, clock, _):
        entered, release = threading.Event(), threading.Event()

        def context():
            entered.set()
            assert release.wait(3)
            return {}

        events.context_provider = context
        writing = asyncio.create_task(
            events.acall(
                events.log,
                "light.synthetic",
                "turn_on",
                sources=(memory_source(now=clock.now),),
            )
        )
        try:
            await until(entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert (
                privacy.resource_status()["writer_lease"]
                and events.resource_status()["operations"] == 1
            )
            assert events.resource_status()["closed"]
        finally:
            release.set()
            await asyncio.gather(writing, return_exceptions=True)
            await privacy.begin_close()
        assert (
            not privacy.resource_status()["writer_lease"] and privacy.ledger._db is None
        )


async def test_cancelled_cleanup_observer_retains_actual_sqlite_work_and_close_lease(
    tmp_path, monkeypatch
):
    async with graph(tmp_path, timeout=0.01) as (
        privacy,
        owner,
        driver,
        events,
        clock,
        _,
    ):
        source = memory_source(now=clock.now)
        await seed_event(events, "light.synthetic", "turn_on", sources=(source,))
        entered, release = threading.Event(), threading.Event()
        clean = driver._clean

        def blocked(step):
            entered.set()
            assert release.wait(3)
            return clean(step)

        monkeypatch.setattr(driver, "_clean", blocked)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await owner.reconcile()
        await until(entered.is_set)
        try:
            assert (await status(privacy, receipt))[0]["state"] == "running"
            with pytest.raises(WorkCloseTimeout):
                await owner.aclose()
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
        finally:
            release.set()
            await privacy.begin_close()
        assert stored_rows(events.db_path) == []


async def test_failed_historical_database_does_not_prevent_independent_event_cleanup(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        bad = tmp_path / "bad.db"
        bad.write_text("synthetic corrupt database")
        await privacy.acall(
            privacy.ledger.erasures.register_target,
            Participant.EVENTS,
            1,
            event_descriptor(bad),
        )
        source = memory_source(now=clock.now)
        await seed_event(events, "light.synthetic", "turn_on", sources=(source,))
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert stored_rows(events.db_path) == []
        steps = await status(privacy, receipt)
        assert sorted(row["state"] for row in steps) == ["complete", "pending"]


async def test_unsupported_registered_event_version_stays_blocked(tmp_path):
    async with graph(tmp_path, enabled=False) as (privacy, owner, _, _, _, config):
        await privacy.acall(
            privacy.ledger.erasures.register_target,
            Participant.EVENTS,
            2,
            event_descriptor(config["analytics"]["events_db"]),
        )
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        steps = await status(privacy, receipt)
        assert sorted(row["state"] for row in steps) == ["complete", "pending"]
        assert (
            next(row for row in steps if row["state"] == "pending")["last_error"]
            == "participant_unavailable"
        )
        assert (await privacy.acall(privacy.ledger.erasures.snapshot, receipt["id"]))[
            "state"
        ] == "blocked"


async def test_cancelled_driver_observer_keeps_native_batch_owned_until_real_close(
    tmp_path, monkeypatch
):
    async with graph(tmp_path, timeout=0.01) as (
        privacy,
        owner,
        driver,
        events,
        clock,
        _,
    ):
        source = memory_source(now=clock.now)
        await seed_event(events, "light.synthetic", "turn_on", sources=(source,))
        await owner.request(source.conversation_id, ACTOR)
        row = next(
            row
            for row in await privacy.acall(privacy.ledger.erasures.due)
            if row["participant"] == "events"
        )
        claim = await privacy.acall(privacy.ledger.erasures.claim, row)
        entered, release = threading.Event(), threading.Event()
        clean = driver._clean

        def blocked(step):
            entered.set()
            assert release.wait(3)
            return clean(step)

        monkeypatch.setattr(driver, "_clean", blocked)
        observer = asyncio.create_task(driver.clean(claim))
        try:
            await until(entered.is_set)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert driver._work.status()["operations"] == 1
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
        finally:
            release.set()
            await privacy.begin_close()
        assert stored_rows(events.db_path) == []


async def test_event_commit_inside_admission_barrier_precedes_revocation_and_physical_receipt(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        entered, release = threading.Event(), threading.Event()
        connect = events._connect

        @contextmanager
        def blocked():
            entered.set()
            assert release.wait(3)
            with connect() as db:
                yield db

        monkeypatch.setattr(events, "_connect", blocked)
        writing = asyncio.create_task(
            events.acall(events.log, "light.synthetic", "turn_on", sources=(source,))
        )
        deleting = None
        try:
            await until(entered.is_set)
            deleting = asyncio.create_task(owner.request(source.conversation_id, ACTOR))
            await asyncio.sleep(0.02)
            assert not deleting.done()
        finally:
            release.set()
            await writing
        receipt = await deleting
        assert len(stored_rows(events.db_path)) == 1
        await flush(owner)
        assert (
            stored_rows(events.db_path) == []
            and (await status(privacy, receipt))[0]["state"] == "complete"
        )


async def test_saturated_default_executor_does_not_starve_event_writer_or_cleanup(
    tmp_path,
):
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    release = threading.Event()
    blocked = asyncio.create_task(asyncio.to_thread(release.wait, 3))
    try:
        async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
            source = memory_source(now=clock.now)
            await events.alog("light.synthetic", "turn_on", sources=(source,))
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            assert (
                stored_rows(events.db_path) == []
                and (await status(privacy, receipt))[0]["state"] == "complete"
            )
            # Executor shutdown itself uses the general pool only after native
            # event work is drained; release it before closing the fixture.
            assert not blocked.done()
            release.set()
    finally:
        release.set()
        await blocked


async def test_repeated_ledger_initialization_does_not_reopen_live_event_claim(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, _, _, _):
        receipt = await owner.request(str(uuid4()), ACTOR)
        row = next(
            row
            for row in await privacy.acall(privacy.ledger.erasures.due)
            if row["participant"] == "events"
        )
        claim = await privacy.acall(privacy.ledger.erasures.claim, row)
        await privacy.acall(privacy.initialize)
        assert (await status(privacy, receipt))[0]["state"] == "running"
        from src.memory.erasure_journal import ErasureResult

        assert await privacy.acall(
            privacy.ledger.erasures.finish, claim, ErasureResult(True, pending_count=0)
        )


@pytest.mark.parametrize("row_id", [0, -1])
async def test_unrecognized_nonpositive_row_id_cannot_be_skipped_by_cursor(
    tmp_path, row_id
):
    async with graph(tmp_path) as (privacy, owner, _, events, clock, _):
        source = memory_source(now=clock.now)
        await seed_event(events, "light.synthetic", "turn_on", sources=(source,))
        with sqlite3.connect(events.db_path) as db:
            db.execute("UPDATE events_policy_v1 SET id=?", (row_id,))
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert (await status(privacy, receipt))[0]["state"] == "pending"
        assert len(stored_rows(events.db_path)) == 1
