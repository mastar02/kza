"""Physical proposal expiry without a consumer, optional features or CVO request."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sqlite3
import threading
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.analytics.event_evidence import event_target_id
from src.analytics.event_storage import event_descriptor, sweep_events
from src.analytics.suggestion_sweep import REFERENCE_BUDGET, sweep_suggestions
from src.analytics.suggestion_engine import SuggestionEngine
from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import FactDependency
from src.pipeline.request_router import RequestRouter
from tests.integration.test_pattern_evidence import ACTOR, DAY, START, Clock, seed_days
from tests.integration.test_suggestion_erasure import graph, stored, until
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


async def seed(events, engine, clock, count=5):
    await seed_days(events, clock, count=count)
    proposals = await events.acall(engine.generate_suggestions)
    assert proposals
    return proposals


async def batch(privacy, engine, events, cursor=None, *, days=90):
    descriptor = event_descriptor(events.db_path)
    return await privacy.acall(
        sweep_suggestions,
        privacy.ledger,
        engine.store.descriptor,
        cursor,
        {event_target_id(descriptor): descriptor},
        days,
    )


async def seed_large(events, engine, clock):
    for day in range(5):
        for item in range(60):
            clock.now = START + day * DAY + item
            await events.acall(
                events.log,
                "light.synthetic",
                "turn_on",
                sources=(memory_source(now=clock.now),),
            )
    proposals = await events.acall(engine.generate_suggestions)
    assert proposals
    return proposals


async def drain(retention, *, force=True):
    await retention.refresh()
    if force:
        retention._due = dict.fromkeys(retention._due, 0)
    for _ in range(20):
        await retention.reconcile()
        tasks = [*retention._active.values()]
        tasks.extend(
            t
            for t in (retention._runtime_task, retention._registry_task)
            if t is not None
        )
        await asyncio.gather(*tasks)
        await retention.reconcile()
        if not retention._active:
            return
    raise AssertionError("retention did not finish bounded batches")


@pytest.mark.parametrize(
    "status", ["pending", "accepted", "rejected", "snoozed", "implemented"]
)
async def test_expiry_physically_removes_all_statuses_without_reads_or_erasure(
    tmp_path, status
):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        with sqlite3.connect(engine.store.path) as db:
            db.execute("UPDATE suggestions_policy_v1 SET status=?", (status,))
        legacy = tmp_path / "suggestions.json"
        legacy.write_bytes(b"unverified synthetic legacy")
        clock.now = max(p.expires_at for p in proposals)
        await drain(driver.retention)
        assert stored(engine.store.path) == []
        assert legacy.read_bytes() == b"unverified synthetic legacy"
        assert await privacy.acall(privacy.ledger.erasures.due) == []


async def test_disabled_restart_cleans_historical_roots_with_original_deadlines(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, _, _, events, engine, _, _):
        proposals = await seed(events, engine, clock)
        sidecar = engine.store.path
    clock.now = max(p.expires_at for p in proposals)
    async with graph(
        tmp_path, enabled=False, clock=clock, legacy=tmp_path / "moved.json"
    ) as (_, _, driver, events, engine, _, _):
        assert events is engine is None
        driver.retention._maintenance = None
        driver.retention.start()
        driver.retention.notify()
        await until(lambda: not stored(sidecar))
        assert not (tmp_path / "moved.json.policy-v1.sqlite3").exists()


async def test_raw_expiry_preserves_metadata_proposals_and_does_not_extend_deadlines(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        clock.now += 3 * DAY
        await privacy.acall(
            sweep_events, event_descriptor(events.db_path), 0, privacy.ledger, 90
        )
        await drain(driver.retention)
        assert len(stored(engine.store.path)) == len(proposals)
        with sqlite3.connect(engine.store.path) as db:
            assert {
                r[0] for r in db.execute("SELECT expires_at FROM suggestions_policy_v1")
            } == {p.expires_at for p in proposals}


@pytest.mark.parametrize("change", ["deleted", "changed", "shortened"])
async def test_retired_actual_event_removes_derived_proposal_before_own_expiry(
    tmp_path, change
):
    async with graph(tmp_path) as (privacy, _, _, events, engine, clock, _):
        await seed(events, engine, clock)
        with sqlite3.connect(events.db_path) as db:
            if change == "deleted":
                db.execute("DELETE FROM events_policy_v1")
            elif change == "changed":
                db.execute("UPDATE events_policy_v1 SET action='turn_off'")
            else:
                db.execute("UPDATE events_policy_v1 SET expires_at=expires_at-86400")
        result = await batch(privacy, engine, events)
        assert result.deleted and not result.pending
        assert stored(engine.store.path) == []


@pytest.mark.parametrize(
    "unavailable", ["missing_file", "missing_table", "malformed_sources"]
)
async def test_unknown_event_source_is_preserved_until_own_expiry(
    tmp_path, unavailable
):
    async with graph(tmp_path) as (privacy, _, _, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        await events.aclose()
        if unavailable == "missing_file":
            events.db_path.unlink()
        else:
            with sqlite3.connect(events.db_path) as db:
                if unavailable == "missing_table":
                    db.execute("DROP TABLE events_policy_v1")
                else:
                    db.execute("UPDATE events_policy_v1 SET sources_json='unknown'")
        result = await batch(privacy, engine, events)
        assert result.pending == len(proposals) and not result.deleted
        assert len(stored(engine.store.path)) == len(proposals)
        clock.now = max(p.expires_at for p in proposals)
        result = await batch(privacy, engine, events)
        assert result.deleted == len(proposals) and not result.pending


async def test_large_lineage_is_checked_incrementally_and_late_retirement_is_seen(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, _, events, engine, clock, _):
        proposals = await seed_large(events, engine, clock)
        # Use actual generated evidence, not invented dependency rows.
        with sqlite3.connect(engine.store.path) as db:
            row = db.execute(
                "SELECT row_id,evidence_json FROM suggestions_policy_v1 ORDER BY row_id LIMIT 1"
            ).fetchone()
            evidence = json.loads(row[1])
            assert len(evidence) > 2 * REFERENCE_BUDGET
            db.execute("DELETE FROM suggestions_policy_v1 WHERE row_id!=?", (row[0],))
        first = await batch(privacy, engine, events)
        assert (
            first.references == REFERENCE_BUDGET
            and first.cursor.offset == REFERENCE_BUDGET
        )
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "DELETE FROM events_policy_v1 WHERE id=?", (evidence[-1]["event_id"],)
            )
        second = await batch(privacy, engine, events, first.cursor)
        third = await batch(privacy, engine, events, second.cursor)
        assert all(r.references <= REFERENCE_BUDGET for r in (first, second, third))
        assert third.deleted == 1 and not stored(engine.store.path)
        assert proposals


async def test_unknown_receipt_does_not_hide_later_committed_retirement(tmp_path):
    async with graph(tmp_path) as (privacy, _, _, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "UPDATE events_policy_v1 SET sources_json='unknown' WHERE id=(SELECT MIN(id) FROM events_policy_v1)"
            )
            db.execute(
                "DELETE FROM events_policy_v1 WHERE id=(SELECT MAX(id) FROM events_policy_v1)"
            )
        result = await batch(privacy, engine, events)
        assert result.deleted == len(proposals) and not stored(engine.store.path)


async def test_budget_exhausted_on_deleted_row_resumes_next_row_at_offset_zero(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, _, events, engine, clock, _):
        await seed_large(events, engine, clock)
        with sqlite3.connect(engine.store.path) as db:
            rows = db.execute(
                "SELECT row_id,evidence_json FROM suggestions_policy_v1 ORDER BY row_id"
            ).fetchall()
            assert len(rows) == 2
            evidence = json.loads(rows[0][1])[:REFERENCE_BUDGET]
            db.execute(
                "UPDATE suggestions_policy_v1 SET evidence_json=? WHERE row_id=?",
                (json.dumps(evidence), rows[0][0]),
            )
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "DELETE FROM events_policy_v1 WHERE id=?", (evidence[-1]["event_id"],)
            )
        first = await batch(privacy, engine, events)
        assert first.references == REFERENCE_BUDGET and first.deleted == 1
        assert first.cursor.after == rows[1][0] and first.cursor.offset == 0
        assert first.cursor.identity is not None
        second = await batch(privacy, engine, events, first.cursor)
        assert second.deleted == 1 and not stored(engine.store.path)


async def test_changed_row_between_batches_restarts_receipts_instead_of_skipping(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, _, events, engine, clock, _):
        await seed_large(events, engine, clock)
        with sqlite3.connect(engine.store.path) as db:
            row = db.execute(
                "SELECT row_id,evidence_json FROM suggestions_policy_v1 ORDER BY row_id LIMIT 1"
            ).fetchone()
            db.execute("DELETE FROM suggestions_policy_v1 WHERE row_id!=?", (row[0],))
        first = await batch(privacy, engine, events)
        assert first.cursor.offset == REFERENCE_BUDGET
        evidence = json.loads(row[1])
        with sqlite3.connect(engine.store.path) as db:
            db.execute(
                "UPDATE suggestions_policy_v1 SET evidence_json=?",
                (json.dumps(list(reversed(evidence))),),
            )
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "DELETE FROM events_policy_v1 WHERE id=?", (evidence[-1]["event_id"],)
            )
        result = await batch(privacy, engine, events, first.cursor)
        assert result.references == 1 and result.deleted == 1


async def test_expiry_rejects_late_native_generation_and_reports_actual_work(tmp_path):
    async with graph(tmp_path) as (_, _, driver, events, engine, clock, _):
        await seed_days(events, clock)
        entered, release = threading.Event(), threading.Event()

        class Model:
            def generate(self, *args, **kwargs):
                entered.set()
                assert release.wait(3)
                return "Synthetic late proposal"

        engine.llm = Model()
        generating = asyncio.create_task(events.acall(engine.generate_suggestions))
        try:
            await until(entered.is_set)
            clock.now += 30 * DAY
            status = await driver._sweep_runtime()
            assert status["pending_operations"] > 0 and not generating.done()
            assert not stored(engine.store.path)
            release.set()
            assert await generating == []
            assert not stored(engine.store.path)
        finally:
            release.set()
            await asyncio.gather(generating, return_exceptions=True)


async def test_expiry_stops_scoped_voice_but_retains_noncooperative_tts_until_exit(
    tmp_path, monkeypatch
):
    response, tts, backend, _, _ = voice_graph(monkeypatch)
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        router = RequestRouter(
            None, response, None, suggestion_engine=engine, privacy=privacy
        )
        driver.attach_router(router)
        tts.release.clear()
        observer = asyncio.create_task(router._maybe_present_suggestion())
        try:
            await until(tts.entered.is_set)
            clock.now = max(p.expires_at for p in proposals)
            await asyncio.wait_for(observer, 2)
            status = await driver._sweep_runtime()
            assert status["pending_operations"] > 0
            assert response.retained_delivery_scopes() and backend.outputs == []
            tts.release.set()
            await until(lambda: not response.retained_delivery_scopes())
            assert backend.outputs == []
            assert (await driver._sweep_runtime())["pending_operations"] == 0
        finally:
            tts.release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await router.aclose()
            await response.aclose()


async def test_shorter_policy_is_persisted_even_when_it_precedes_generation(tmp_path):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        await seed_days(events, clock)
        clock.now += 10 * DAY
        proposals = await events.acall(engine.generate_suggestions)
        assert proposals
        privacy.ledger.automatic_days = 1
        restarted = await events.acall(
            SuggestionEngine,
            events,
            suggestions_path=str(tmp_path / "suggestions.json"),
        )
        with sqlite3.connect(engine.store.path) as db:
            assert all(
                expiry == START + 5 * DAY and expiry < created
                for created, expiry in db.execute(
                    "SELECT created_at,expires_at FROM suggestions_policy_v1"
                )
            )
        privacy.ledger.automatic_days = 30
        llm = MagicMock()
        restarted.llm = llm
        assert await events.acall(restarted.get_pending_suggestions) == []
        await drain(driver.retention)
        assert not stored(engine.store.path)


async def test_shorter_policy_rejects_expired_candidates_before_model_call(tmp_path):
    async with graph(tmp_path) as (privacy, _, _, events, engine, clock, _):
        await seed_days(events, clock)
        clock.now += 2 * DAY
        privacy.ledger.automatic_days = 1
        llm = MagicMock()
        engine.llm = llm
        assert await events.acall(engine.generate_suggestions) == []
        assert llm.mock_calls == []


async def test_unknown_rows_do_not_starve_later_expired_known_rows(tmp_path):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        with sqlite3.connect(engine.store.path) as db:
            db.execute(
                "UPDATE suggestions_policy_v1 SET row_id=100 WHERE row_id=(SELECT MAX(row_id) FROM suggestions_policy_v1)"
            )
            db.execute(
                "UPDATE suggestions_policy_v1 SET evidence_json='unknown' WHERE row_id=(SELECT MIN(row_id) FROM suggestions_policy_v1)"
            )
            for row_id in range(2, 37):
                db.execute(
                    """INSERT INTO suggestions_policy_v1
                    SELECT ?,?,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json,message
                    FROM suggestions_policy_v1 WHERE row_id=1""",
                    (row_id, "sug_" + uuid4().hex),
                )
        clock.now = max(p.expires_at for p in proposals)
        await drain(driver.retention)
        assert len(stored(engine.store.path)) == 36
        assert any(t.get("pending") == 36 for t in driver.retention.status()["targets"])


async def test_runtime_sweep_prunes_idle_router_cache_at_original_expiry(tmp_path):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        router = RequestRouter(
            None, None, None, suggestion_engine=engine, privacy=privacy
        )
        driver.attach_router(router)
        router._pending_suggestion = proposals[0]
        clock.now = proposals[0].expires_at
        await driver._sweep_runtime()
        assert router._pending_proposal is None
        await router.aclose()


async def test_default_executor_saturation_does_not_block_cached_root_expiry(tmp_path):
    async with graph(tmp_path) as (_, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        await driver.retention.refresh()
        await drain(driver.retention)
        loop = asyncio.get_running_loop()
        previous = loop._default_executor
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        entered, release = threading.Event(), threading.Event()

        def block():
            entered.set()
            release.wait(5)

        blocker = loop.run_in_executor(None, block)
        try:
            await until(entered.is_set)
            clock.now = max(p.expires_at for p in proposals)
            await drain(driver.retention)
            assert not blocker.done() and not stored(engine.store.path)
        finally:
            release.set()
            await blocker
            loop._default_executor = previous
            executor.shutdown(wait=True)


async def test_registered_historical_event_target_is_used_without_logger(tmp_path):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        await seed(events, engine, clock)
        await events.aclose()
        with sqlite3.connect(events.db_path) as db:
            db.execute("DELETE FROM events_policy_v1")
        await drain(driver.retention)
        assert not stored(engine.store.path)
        targets = await privacy.acall(
            privacy.ledger.erasures.targets, ErasureParticipant.EVENTS
        )
        assert (
            event_target_id(event_descriptor(events.db_path))
            in driver.retention._event_targets
        )
        assert targets


async def test_committed_fact_retirement_cleans_without_a_conversation_erasure(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        fact = await privacy.acall(
            privacy.ledger.store,
            "synthetic fact",
            "fact",
            0.8,
            (memory_source(now=clock.now),),
        )
        for day in range(5):
            clock.now = START + day * DAY
            source = replace(
                memory_source(now=clock.now),
                fact_dependencies=(
                    FactDependency(fact.id, fact.revision, fact.expires_at),
                ),
            )
            await events.acall(
                events.log, "light.synthetic", "turn_on", sources=(source,)
            )
        assert await events.acall(engine.generate_suggestions)
        assert await privacy.acall(privacy.ledger.delete_fact, fact.id, ACTOR)
        await drain(driver.retention)
        assert not stored(engine.store.path)
        assert await privacy.acall(privacy.ledger.erasures.due) == []


async def test_failed_pending_marker_does_not_authorize_irreversible_ttl_cleanup(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        sources = await seed_days(events, clock)
        proposals = await events.acall(engine.generate_suggestions)
        original = privacy.ledger._db
        entered, release = threading.Event(), threading.Event()

        class FailedCommit:
            def __getattr__(self, key):
                return getattr(original, key)

            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError("synthetic marker failed")

        privacy.ledger._db = FailedCommit()
        marking = asyncio.create_task(
            privacy.acall(
                privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
            )
        )
        try:
            await until(entered.is_set)
            # A real pending fence hides reads, but its rollback must retain data.
            assert not engine.available_now(proposals[0])
            descriptor = event_descriptor(events.db_path)
            result = await driver.retention._work.run(
                sweep_suggestions,
                privacy.ledger,
                engine.store.descriptor,
                None,
                {event_target_id(descriptor): descriptor},
                90,
            )
            assert not result.deleted and len(stored(engine.store.path)) == len(
                proposals
            )
        finally:
            release.set()
            with pytest.raises(OSError, match="marker failed"):
                await marking
            privacy.ledger._db = original
        assert await events.acall(engine.get_pending_suggestions)


async def test_blocked_registry_and_one_root_do_not_starve_another_known_root(tmp_path):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        other = await events.acall(
            SuggestionEngine, events, suggestions_path=str(tmp_path / "other.json")
        )
        assert await events.acall(other.generate_suggestions)
        await drain(driver.retention)
        entered, release = threading.Event(), threading.Event()

        def block():
            with privacy.ledger._transaction():
                with sqlite3.connect(engine.store.path) as db:
                    db.execute("BEGIN IMMEDIATE")
                    entered.set()
                    assert release.wait(4)

        blocked = asyncio.create_task(privacy.acall(block))
        try:
            await until(entered.is_set)
            clock.now = max(p.expires_at for p in proposals)
            driver.retention.notify()
            await asyncio.sleep(0)
            async with asyncio.timeout(2):
                while stored(other.store.path):
                    await driver.retention.reconcile()
                    await asyncio.sleep(0.005)
            assert not blocked.done()
            assert (
                driver.retention._registry_task is not None
                and not driver.retention._registry_task.done()
            )
        finally:
            release.set()
            await blocked
        await drain(driver.retention)
        assert not stored(engine.store.path)


async def test_failed_batch_exposes_fixed_error_and_retries_original_deadline(
    tmp_path, monkeypatch
):
    import src.analytics.suggestion_retention as module

    async with graph(tmp_path) as (_, _, driver, events, engine, clock, _):
        proposals = await seed(events, engine, clock)
        clock.now = max(p.expires_at for p in proposals)

        def fail(*args):
            raise OSError("synthetic private path and message")

        with monkeypatch.context() as patch:
            patch.setattr(module, "sweep_suggestions", fail)
            await drain(driver.retention)
            status = driver.retention.status()
            assert "synthetic private" not in json.dumps(status)
            assert any(
                t.get("error") == "suggestion_retention_failed"
                for t in status["targets"]
            )
            assert stored(engine.store.path)
        clock.now += 2
        await drain(driver.retention, force=False)
        assert not stored(engine.store.path)


async def test_cancelled_batch_observer_and_close_keep_actual_native_work_and_lease(
    tmp_path, monkeypatch
):
    import src.analytics.suggestion_retention as module

    entered, release = threading.Event(), threading.Event()
    original = module.sweep_suggestions

    def block(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    async with graph(tmp_path, close_timeout=0.03) as (
        privacy,
        owner,
        driver,
        events,
        engine,
        clock,
        _,
    ):
        proposals = await seed(events, engine, clock)
        clock.now = max(p.expires_at for p in proposals)
        monkeypatch.setattr(module, "sweep_suggestions", block)
        await driver.retention.refresh()
        driver.retention._due = dict.fromkeys(driver.retention._due, 0)
        await driver.retention.reconcile()
        await until(entered.is_set)

        async def observe():
            await asyncio.shield(next(iter(driver.retention._active.values())))

        observer = asyncio.create_task(observe())
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        try:
            with pytest.raises(WorkCloseTimeout):
                await owner.aclose()
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
            assert driver.retention._work.status()["operations"]
            assert not driver.retention._closing.done()
        finally:
            release.set()
            await owner.begin_close()
            await privacy.begin_close()
        assert not privacy.resource_status()["writer_lease"]
        assert stored(engine.store.path) == []
        assert observer.cancelled()
