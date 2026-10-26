"""Connected proposal erasure, real native lifetimes and historical targets."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import shutil
import sqlite3
import threading
from types import SimpleNamespace
from uuid import uuid4
from weakref import ref

import pytest

from src.analytics.erasure import build_event_erasure
from src.analytics.services import build_event_logger
from src.analytics.suggestion_engine import SuggestionEngine
from src.analytics.suggestion_erasure import build_suggestion_erasure
from src.analytics.suggestion_storage import cleanup_suggestions
from src.core.owned_work import WorkCloseTimeout
from src.core.voice_delivery import ResponseBudgets
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant as Participant
from src.memory.fact_ledger import FactDependency
from src.memory.privacy_services import PrivacyServices
from src.pipeline.request_router import RequestRouter
from src.pipeline.response_handler import ResponseHandler
from tests.integration.test_pattern_evidence import ACTOR, DAY, START, Clock
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(tmp_path, *, enabled=True, clock=None, legacy=None, close_timeout=1):
    clock = clock or Clock()
    config = {
        "analytics": {
            "enabled": enabled,
            "events_db": str(tmp_path / "events.db"),
            "suggestions_path": str(legacy or tmp_path / "suggestions.json"),
        }
    }
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(
            str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=close_timeout
        )
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy, close_timeout_s=close_timeout)
        stack.push_async_callback(owner.aclose)
        events_driver = await build_event_erasure(config, privacy=privacy, owner=owner)
        events_driver.retention._maintenance.cancel()
        await asyncio.gather(
            events_driver.retention._maintenance, return_exceptions=True
        )
        driver = await build_suggestion_erasure(config, privacy=privacy, owner=owner)
        # Observe CVO independently of periodic TTL. Keep the actual pools open
        # so shutdown cannot wait behind a deliberately saturated general pool.
        driver.retention._maintenance.cancel()
        await asyncio.gather(driver.retention._maintenance, return_exceptions=True)
        if driver.retention._cycle is not None:
            await asyncio.gather(driver.retention._cycle, return_exceptions=True)
        await asyncio.gather(*driver.retention._active.values(), return_exceptions=True)
        events = await build_event_logger(
            config, shutdown=stack, privacy=privacy, retention=events_driver.retention
        )
        engine = None
        if events is not None:
            engine = await events.acall(
                SuggestionEngine,
                events,
                suggestions_path=config["analytics"]["suggestions_path"],
            )
            driver.attach_engine(engine)
        yield privacy, owner, driver, events, engine, clock, config


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def flush(owner):
    await owner.reconcile()
    await asyncio.gather(*(a.task for a in owner._active.values()))
    await owner.reconcile()


async def steps(privacy, receipt):
    snapshot = await privacy.acall(privacy.ledger.erasures.snapshot, receipt["id"])
    return [s for s in snapshot["steps"] if s["participant"] == "suggestions"]


def stored(path):
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT id,sources_json FROM suggestions_policy_v1 ORDER BY row_id"
        ).fetchall()


async def seed(events, engine, clock, *, derived=None):
    erased, survivor = str(uuid4()), str(uuid4())
    for day in range(5):
        clock.now = START + day * DAY
        source = memory_source(now=clock.now, conversation_id=erased)
        if derived is not None:
            source = replace(source, fact_dependencies=(derived,))
        other = memory_source(now=clock.now, conversation_id=survivor)
        await events.acall(
            events.log,
            "light.synthetic",
            "turn_on",
            sources=(source, other) if day == 0 else (source,),
        )
        clock.now += 10
        await events.acall(events.log, "light.survives", "turn_on", sources=(other,))
    proposals = await events.acall(engine.generate_suggestions)
    assert proposals
    return erased, survivor, proposals


async def test_physical_erasure_covers_all_statuses_and_mixed_sources_without_reading_messages(
    tmp_path, monkeypatch
):
    import src.analytics.suggestion_storage as module

    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, proposals = await seed(events, engine, clock)
        affected = [
            p for p in proposals if any(s.conversation_id == erased for s in p.sources)
        ]
        survivors = {p.id for p in proposals if p not in affected}
        assert len(affected) >= 2 and survivors
        await events.acall(engine.respond_to_suggestion, affected[0].id, True)
        await events.acall(engine.respond_to_suggestion, affected[1].id, False, 24)
        legacy = tmp_path / "suggestions.json"
        legacy.write_text("synthetic unverified legacy data")
        original = module.existing_suggestion_database
        statements = []
        from contextlib import contextmanager

        @contextmanager
        def observed(descriptor):
            with original(descriptor) as db:
                if db is not None:
                    db.set_trace_callback(statements.append)
                yield db

        monkeypatch.setattr(module, "existing_suggestion_database", observed)
        receipt = await owner.request(erased, ACTOR)
        assert len(stored(engine.store.path)) == len(proposals)
        await flush(owner)
        assert {row[0] for row in stored(engine.store.path)} == survivors
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))
        assert (await privacy.acall(privacy.ledger.erasures.snapshot, receipt["id"]))[
            "state"
        ] == "blocked"
        assert legacy.read_text() == "synthetic unverified legacy data"
        assert not any(
            "SELECT message" in sql or "pattern_json" in sql for sql in statements
        )


async def test_indirect_fact_revocation_removes_proposals_with_different_conversation_ids(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        source = memory_source(now=clock.now)
        fact = await privacy.acall(
            privacy.ledger.store, "synthetic dependent fact", "fact", 0.8, (source,)
        )
        _, _, proposals = await seed(
            events,
            engine,
            clock,
            derived=FactDependency(fact.id, fact.revision, fact.expires_at),
        )
        affected = {
            p.id for p in proposals if any(s.fact_dependencies for s in p.sources)
        }
        assert affected and all(
            s.conversation_id != source.conversation_id
            for p in proposals
            for s in p.sources
        )
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert not affected & {row[0] for row in stored(engine.store.path)}
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))


async def test_disabled_analytics_cleans_historical_target_and_restored_completed_rows(
    tmp_path,
):
    clock = Clock()
    backup = tmp_path / "saved-policy.db"
    async with graph(tmp_path, clock=clock) as (
        privacy,
        owner,
        _,
        events,
        engine,
        _,
        config,
    ):
        erased, _, proposals = await seed(events, engine, clock)
        affected = {
            p.id
            for p in proposals
            if any(s.conversation_id == erased for s in p.sources)
        }
        old_path = engine.store.path
        with sqlite3.connect(old_path) as db, sqlite3.connect(backup) as copy:
            db.backup(copy)
        receipt = await owner.request(erased, ACTOR)
        await flush(owner)
        assert not affected & {row[0] for row in stored(old_path)}
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))
    shutil.copyfile(backup, old_path)
    async with graph(
        tmp_path, enabled=False, clock=clock, legacy=tmp_path / "unused" / "new.json"
    ) as (privacy, owner, _, events, engine, _, _):
        assert events is None and engine is None
        assert any(row["state"] == "pending" for row in await steps(privacy, receipt))
        await flush(owner)
        assert not affected & {row[0] for row in stored(old_path)}
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))
        assert not (tmp_path / "unused").exists()


async def test_unknown_first_row_does_not_starve_later_known_batches_or_claim_completion(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, proposals = await seed(events, engine, clock)
        chosen = next(
            p for p in proposals if any(s.conversation_id == erased for s in p.sources)
        )
        with sqlite3.connect(engine.store.path) as db:
            for _ in range(70):
                db.execute(
                    """INSERT INTO suggestions_policy_v1
                    (id,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json,message)
                    SELECT ?,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json,message
                    FROM suggestions_policy_v1 WHERE id=?""",
                    ("sug_" + uuid4().hex, chosen.id),
                )
            unknown_id = db.execute(
                "SELECT id FROM suggestions_policy_v1 ORDER BY row_id LIMIT 1"
            ).fetchone()[0]
            db.execute(
                "UPDATE suggestions_policy_v1 SET sources_json='unknown source protocol' WHERE id=?",
                (unknown_id,),
            )
        assert unknown_id not in {
            p.id for p in await events.acall(engine.get_pending_suggestions)
        }
        receipt = await owner.request(erased, ACTOR)
        for _ in range(5):
            await flush(owner)
            clock.now += 61
        remaining = stored(engine.store.path)
        assert any(row[0] == unknown_id for row in remaining)
        assert all(
            row[0] == unknown_id
            or all(s["conversation_id"] != erased for s in json.loads(row[1]))
            for row in remaining
        )
        status = await steps(privacy, receipt)
        assert any(
            row["state"] == "pending" and row["pending_count"] == 1 for row in status
        )


async def test_nonpositive_row_id_remains_quarantined_without_starving_known_rows(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, proposals = await seed(events, engine, clock)
        affected = {
            p.id
            for p in proposals
            if any(s.conversation_id == erased for s in p.sources)
        }
        unknown_id = "sug_" + uuid4().hex
        with sqlite3.connect(engine.store.path) as db:
            db.execute(
                """INSERT INTO suggestions_policy_v1
                (row_id,id,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json,message)
                SELECT -1,?,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json,message
                FROM suggestions_policy_v1 LIMIT 1""",
                (unknown_id,),
            )
        assert unknown_id not in {
            p.id for p in await events.acall(engine.get_pending_suggestions)
        }
        receipt = await owner.request(erased, ACTOR)
        await flush(owner)
        ids = {row[0] for row in stored(engine.store.path)}
        assert unknown_id in ids and not affected & ids
        assert any(
            row["pending_count"] == 1 and row["state"] == "pending"
            for row in await steps(privacy, receipt)
        )


async def test_late_native_generation_keeps_runtime_pending_after_observer_and_file_cleanup(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        entered, release = threading.Event(), threading.Event()

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return "synthetic late message"

        # Remove current proposals so the actual engine enters its model path.
        with sqlite3.connect(engine.store.path) as db:
            db.execute("DELETE FROM suggestions_policy_v1")
        engine.llm = SimpleNamespace(generate=blocked)
        observer = asyncio.create_task(events.acall(engine.generate_suggestions))
        try:
            await until(entered.is_set)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            receipt = await owner.request(erased, ACTOR)
            await flush(owner)
            assert any(
                row["state"] == "pending" and row["pending_count"]
                for row in await steps(privacy, receipt)
            )
            assert events.resource_status()["operations"] == 1
            release.set()
            await until(lambda: events.resource_status()["operations"] == 0)
            clock.now += 61
            await flush(owner)
            assert all(
                row["state"] == "complete" for row in await steps(privacy, receipt)
            )
            assert all(
                all(s["conversation_id"] != erased for s in json.loads(row[1]))
                for row in stored(engine.store.path)
            )
        finally:
            release.set()


async def test_native_jobs_admitted_after_barrier_do_not_delay_its_receipt(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        one_entered, one_release = threading.Event(), threading.Event()
        two_entered, two_release = threading.Event(), threading.Event()

        def blocked(entered, release):
            entered.set()
            assert release.wait(3)

        one = asyncio.create_task(events.acall(blocked, one_entered, one_release))
        two = None
        try:
            await until(one_entered.is_set)
            receipt = await owner.request(erased, ACTOR)
            await flush(owner)
            two = asyncio.create_task(events.acall(blocked, two_entered, two_release))
            await until(two_entered.is_set)
            one_release.set()
            await one
            clock.now += 61
            await flush(owner)
            assert all(
                row["state"] == "complete" for row in await steps(privacy, receipt)
            )
            assert not two.done() and not two_release.is_set()
        finally:
            one_release.set()
            two_release.set()
            await asyncio.gather(*(task for task in (one, two) if task is not None))


async def test_erasure_waits_for_real_noncooperative_tts_and_cancels_only_scoped_voice(
    tmp_path, monkeypatch
):
    response, tts, backend, _, _ = voice_graph(monkeypatch)
    async with graph(tmp_path) as (privacy, owner, driver, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        router = RequestRouter(
            None, response, None, suggestion_engine=engine, privacy=privacy
        )
        driver.attach_router(router)
        tts.release.clear()
        observer = asyncio.create_task(router._maybe_present_suggestion())
        try:
            await until(tts.entered.is_set)
            receipt = await owner.request(erased, ACTOR)
            await flush(owner)
            await observer
            assert not observer.cancelled()
            assert router._pending_suggestion is None
            assert response.retained_delivery_scopes()
            assert any(
                row["state"] == "pending" and row["pending_count"]
                for row in await steps(privacy, receipt)
            )
            assert backend.outputs == []
            tts.release.set()
            await until(lambda: not response.retained_delivery_scopes())
            clock.now += 61
            await flush(owner)
            assert all(
                row["state"] == "complete" for row in await steps(privacy, receipt)
            )
            assert backend.outputs == []
        finally:
            tts.release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await router.aclose()
            await response.aclose()


async def test_pending_cache_is_pruned_without_a_new_voice_request(tmp_path):
    async with graph(tmp_path) as (privacy, owner, driver, events, engine, clock, _):
        erased, _, proposals = await seed(events, engine, clock)
        router = RequestRouter(
            None, None, None, suggestion_engine=engine, privacy=privacy
        )
        driver.attach_router(router)
        router._pending_suggestion = next(
            p for p in proposals if any(s.conversation_id == erased for s in p.sources)
        )
        receipt = await owner.request(erased, ACTOR)
        await flush(owner)
        assert router._pending_proposal is None
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))
        await router.aclose()


async def test_shutdown_keeps_actual_tts_and_writer_lease_after_flow_cancel(
    tmp_path, monkeypatch
):
    response, tts, _, _, _ = voice_graph(monkeypatch)
    async with graph(tmp_path, close_timeout=0.01) as (
        privacy,
        _,
        driver,
        events,
        engine,
        clock,
        _,
    ):
        await seed(events, engine, clock)
        router = RequestRouter(
            None, response, None, suggestion_engine=engine, privacy=privacy
        )
        driver.attach_router(router)
        tts.release.clear()
        observer = asyncio.create_task(router._maybe_present_suggestion())
        contender = PrivacyServices(privacy.ledger.path, clock=clock)
        try:
            await until(tts.entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
            assert response.retained_delivery_scopes()
            with pytest.raises(RuntimeError, match="privacy_writer_already_active"):
                await contender.acall(contender.initialize)
            tts.release.set()
            await until(lambda: privacy._closing.done())
            await privacy.aclose()
            assert not privacy.resource_status()["writer_lease"]
        finally:
            tts.release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await router.aclose()
            await response.aclose()
            await contender.aclose()


async def test_abort_worker_remains_visible_after_controller_and_synthesis_exit(
    tmp_path, monkeypatch
):
    response = ResponseHandler(None, response_budgets=ResponseBudgets(cleanup_s=0.02))
    rendered, stopping, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def stop():
        stopping.set()
        assert release.wait(3)

    def render(session, _):
        session.player = SimpleNamespace(stop=stop)
        rendered.set()
        assert session.cancelled.wait(3)

    monkeypatch.setattr(response, "_render", render)
    async with graph(tmp_path) as (privacy, owner, driver, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        router = RequestRouter(
            None, response, None, suggestion_engine=engine, privacy=privacy
        )
        driver.attach_router(router)
        observer = asyncio.create_task(router._maybe_present_suggestion())
        try:
            await until(rendered.is_set)
            receipt = await owner.request(erased, ACTOR)
            await flush(owner)
            await until(stopping.is_set)
            await observer
            status = response.get_delivery_status()
            assert (
                status["workers"] == status["controllers"] == 0
                and status["aborts"] == 1
            )
            assert response.retained_delivery_scopes()
            clock.now += 61
            await flush(owner)
            assert any(
                row["state"] == "pending" and row["pending_count"]
                for row in await steps(privacy, receipt)
            )
            release.set()
            await until(lambda: not response.retained_delivery_scopes())
            clock.now += 61
            await flush(owner)
            assert all(
                row["state"] == "complete" for row in await steps(privacy, receipt)
            )
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await router.aclose()
            await response.aclose()


async def test_new_voice_session_does_not_reuse_the_erased_delivery_scope(
    tmp_path, monkeypatch
):
    response, tts, backend, _, _ = voice_graph(monkeypatch)
    async with graph(tmp_path) as (privacy, owner, driver, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        router = RequestRouter(
            None, response, None, suggestion_engine=engine, privacy=privacy
        )
        driver.attach_router(router)
        tts.release.clear()
        old = asyncio.create_task(router._maybe_present_suggestion())
        new = None
        try:
            await until(tts.entered.is_set)
            old_scopes = response.retained_delivery_scopes()
            receipt = await owner.request(erased, ACTOR)
            await flush(owner)
            await old
            tts.release.set()
            await until(lambda: not response.retained_delivery_scopes())
            tts.release.clear()
            tts.entered.clear()
            new = asyncio.create_task(response.aspeak("synthetic unrelated new voice"))
            await until(tts.entered.is_set)
            assert not old_scopes & response.retained_delivery_scopes()
            clock.now += 61
            await flush(owner)
            assert all(
                row["state"] == "complete" for row in await steps(privacy, receipt)
            )
            assert not new.done() and backend.outputs == []
        finally:
            tts.release.set()
            await asyncio.gather(
                *(task for task in (old, new) if task is not None),
                return_exceptions=True,
            )
            await router.aclose()
            await response.aclose()


async def test_cancelled_cleanup_observer_keeps_sqlite_work_and_lease_owned(
    tmp_path, monkeypatch
):
    import src.analytics.suggestion_erasure as module

    async with graph(tmp_path, close_timeout=0.01) as (
        privacy,
        owner,
        driver,
        events,
        engine,
        clock,
        _,
    ):
        erased, _, _ = await seed(events, engine, clock)
        entered, release = threading.Event(), threading.Event()
        original = module.cleanup_suggestions

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(module, "cleanup_suggestions", blocked)
        receipt = await owner.request(erased, ACTOR)
        await owner.reconcile()
        try:
            await until(entered.is_set)
            active = next(
                a
                for a in owner._active.values()
                if a.step.participant is Participant.SUGGESTIONS
                and "path" in a.step.descriptor
            )
            active.task.cancel()
            await asyncio.gather(active.task, return_exceptions=True)
            assert driver._disk.status()["operations"] == 1
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
            release.set()
            await until(lambda: privacy._closing.done())
            await privacy.aclose()
            assert not privacy.resource_status()["writer_lease"]
        finally:
            release.set()
    async with graph(tmp_path, enabled=False, clock=clock) as (
        privacy,
        owner,
        _,
        _,
        _,
        _,
        _,
    ):
        await flush(owner)
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))


async def test_failed_acknowledgement_retries_only_receipt_not_physical_cleanup(
    tmp_path, monkeypatch
):
    import src.analytics.suggestion_erasure as module

    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        calls = []
        original = module.cleanup_suggestions

        def tracked(*args):
            calls.append(1)
            return original(*args)

        monkeypatch.setattr(module, "cleanup_suggestions", tracked)
        receipt = await owner.request(erased, ACTOR)
        finish = owner.journal.finish
        failed = False

        def fail_once(step, *args):
            nonlocal failed
            if (
                not failed
                and step.participant is Participant.SUGGESTIONS
                and "path" in step.descriptor
            ):
                failed = True
                raise OSError("synthetic private acknowledgement")
            return finish(step, *args)

        monkeypatch.setattr(owner.journal, "finish", fail_once)
        with pytest.raises(OSError):
            await flush(owner)
        assert len(calls) == 1
        await flush(owner)
        assert len(calls) == 1
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))


async def test_failure_is_sanitized_and_survives_restart_until_actual_retry(
    tmp_path, monkeypatch
):
    import src.analytics.suggestion_erasure as module

    clock = Clock()
    async with graph(tmp_path, clock=clock) as (
        privacy,
        owner,
        _,
        events,
        engine,
        _,
        _,
    ):
        erased, _, _ = await seed(events, engine, clock)
        affected = {
            p.id
            for p in await events.acall(engine.get_pending_suggestions)
            if any(s.conversation_id == erased for s in p.sources)
        }
        path = engine.store.path

        def fail(*args):
            raise OSError("synthetic private path and transcript")

        with monkeypatch.context() as patch:
            patch.setattr(module, "cleanup_suggestions", fail)
            receipt = await owner.request(erased, ACTOR)
            await flush(owner)
            status = await privacy.acall(
                privacy.ledger.erasures.snapshot, receipt["id"]
            )
            assert "synthetic private" not in json.dumps(status)
            assert any(
                row["last_error"] == "cleanup_failed"
                for row in status["steps"]
                if row["participant"] == "suggestions"
            )
            assert affected <= {row[0] for row in stored(path)}
    async with graph(tmp_path, enabled=False, clock=clock) as (
        privacy,
        owner,
        _,
        _,
        _,
        _,
        _,
    ):
        await flush(owner)
        assert not affected & {row[0] for row in stored(path)}
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))


async def test_redirected_sidecar_is_not_followed_or_falsely_acknowledged(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        original = stored(engine.store.path)
        moved = tmp_path / "unregistered.db"
        engine.store.path.rename(moved)
        engine.store.path.symlink_to(moved)
        receipt = await owner.request(erased, ACTOR)
        await flush(owner)
        assert stored(moved) == original
        assert any(row["state"] == "pending" for row in await steps(privacy, receipt))
        engine.store.path.unlink()
        moved.rename(engine.store.path)
        clock.now += 61
        await flush(owner)
        assert all(row["state"] == "complete" for row in await steps(privacy, receipt))


@pytest.mark.parametrize(
    "cursor",
    [
        "0",
        "s1:0:0",
        "s1:01:0",
        "s1:1:-1",
        "s1:1:01",
        "s1:9223372036854775808:0",
        "private/path",
    ],
)
async def test_unknown_cleanup_cursor_is_rejected_before_any_data_access(
    tmp_path, cursor
):
    async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        await owner.request(erased, ACTOR)
        before = stored(engine.store.path)
        with pytest.raises(ValueError, match="invalid_suggestion_cursor"):
            await privacy.acall(
                cleanup_suggestions,
                privacy.ledger,
                engine.store.descriptor,
                erased,
                cursor,
            )
        assert stored(engine.store.path) == before


async def test_unsupported_registered_version_remains_pending_without_opening_private_path(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, _, _, _, _):
        absent = tmp_path / "must-not-open.db"
        await privacy.acall(
            privacy.ledger.erasures.register_target,
            Participant.SUGGESTIONS,
            2,
            {"path": str(absent), "table": "suggestions_policy_v2"},
        )
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert not absent.exists()
        assert any(
            row["last_error"] == "participant_unavailable"
            for row in await steps(privacy, receipt)
        )


async def test_owner_registration_cannot_silently_adopt_already_running_untracked_work(
    tmp_path,
):
    async with graph(tmp_path) as (_, _, driver, events, engine, _, _):
        other = await events.acall(
            SuggestionEngine, events, suggestions_path=str(tmp_path / "other.json")
        )
        entered, release = threading.Event(), threading.Event()

        def blocked():
            entered.set()
            assert release.wait(3)

        work = asyncio.create_task(events.acall(blocked))
        try:
            await until(entered.is_set)
            with pytest.raises(ValueError, match="registration_before_work"):
                driver.attach_engine(other)
        finally:
            release.set()
            await work
        driver.attach_engine(other)


async def test_pending_failed_marker_cannot_authorize_physical_proposal_removal(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, driver, events, engine, clock, _):
        erased, _, proposals = await seed(events, engine, clock)
        entered, release = threading.Event(), threading.Event()
        original = privacy.ledger._db

        class FailedCommit:
            def __getattr__(self, key):
                return getattr(original, key)

            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError("synthetic failed marker")

        privacy.ledger._db = FailedCommit()
        marker = asyncio.create_task(
            privacy.acall(privacy.ledger.delete_conversation, erased, ACTOR)
        )
        try:
            await until(entered.is_set)
            assert privacy.ledger.conversation_revoked(erased)
            assert not privacy.ledger.conversation_committed_revoked(erased)
            with pytest.raises(
                RuntimeError, match="suggestion_erasure_authority_unavailable"
            ):
                await driver._disk.run(
                    cleanup_suggestions,
                    privacy.ledger,
                    engine.store.descriptor,
                    erased,
                    None,
                )
            assert len(stored(engine.store.path)) == len(proposals)
        finally:
            release.set()
            with pytest.raises(OSError, match="synthetic failed marker"):
                await marker
            privacy.ledger._db = original
        assert {p.id for p in await events.acall(engine.get_pending_suggestions)} == {
            p.id for p in proposals
        }


async def test_saturated_default_executor_cannot_starve_proposal_cleanup(tmp_path):
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(3)

    try:
        async with graph(tmp_path) as (privacy, owner, _, events, engine, clock, _):
            erased, _, _ = await seed(events, engine, clock)
            blocker = asyncio.create_task(asyncio.to_thread(block))
            try:
                await until(entered.is_set)
                receipt = await owner.request(erased, ACTOR)
                await flush(owner)
                assert all(
                    row["state"] == "complete" for row in await steps(privacy, receipt)
                )
                assert not blocker.done() and not release.is_set()
            finally:
                release.set()
                await blocker
    finally:
        release.set()
        pool.shutdown(wait=False)


async def test_completed_native_result_is_not_retained_by_a_pending_retry_barrier(
    tmp_path,
):
    @dataclass
    class PrivateResult:
        text: str

    async with graph(tmp_path) as (privacy, owner, driver, events, engine, clock, _):
        erased, _, _ = await seed(events, engine, clock)
        entered, release = threading.Event(), threading.Event()
        weak = []

        def produce():
            result = PrivateResult("synthetic private result")
            weak.append(ref(result))
            entered.set()
            assert release.wait(3)
            return result

        observer = asyncio.create_task(events.acall(produce))
        try:
            await until(entered.is_set)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            receipt = await owner.request(erased, ACTOR)
            await flush(owner)
            assert driver._barriers[receipt["id"]].pending()
            release.set()
            await until(lambda: weak[0]() is None)
            assert receipt["id"] in driver._barriers
            assert not driver._barriers[receipt["id"]].tasks
        finally:
            release.set()


async def test_disabled_suggestion_cleanup_imports_no_models_or_audio_backend(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import asyncio,importlib.abc,sys
        from contextlib import AsyncExitStack
        from pathlib import Path
        class DenyModels(importlib.abc.MetaPathFinder):
            def find_spec(self,name,*args):
                if name.split('.')[0] in {'torch','numpy','chromadb','sounddevice','sentence_transformers','faster_whisper','onnxruntime'} or name.startswith('src.pipeline'):
                    raise AssertionError('disabled proposal cleanup imported optional runtime')
        sys.meta_path.insert(0,DenyModels())
        from src.memory.privacy_services import build_privacy_services
        from src.memory.erasure_coordinator import ErasureCoordinator
        from src.memory.fact_ledger import MemoryReviewer
        from src.analytics.suggestion_erasure import build_suggestion_erasure
        from uuid import uuid4
        async def main():
            root=Path(sys.argv[1])
            config={'memory':{'enabled':False,'privacy_db_path':str(root/'privacy.db')},'analytics':{'enabled':False,'suggestions_path':str(root/'absent'/'proposals.json')}}
            async with AsyncExitStack() as stack:
                privacy=await build_privacy_services(config,shutdown=stack)
                owner=ErasureCoordinator(privacy)
                stack.push_async_callback(owner.aclose)
                await build_suggestion_erasure(config,privacy=privacy,owner=owner)
                receipt=await owner.request(str(uuid4()),MemoryReviewer('synthetic-admin'))
                for _ in range(3):
                    await owner.reconcile()
                    await asyncio.gather(*(a.task for a in owner._active.values()))
                value=await privacy.acall(privacy.ledger.erasures.snapshot,receipt['id'])
                assert all(s['state']=='complete' for s in value['steps'] if s['participant']=='suggestions')
                assert not (root/'absent').exists()
        asyncio.run(main())
    """)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
