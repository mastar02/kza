"""Real snapshot and idle-buffer expiry without context producers or read access."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import threading
import time

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import MemoryReviewer
from src.memory.memory_manager import ShortTermMemory
from src.memory.privacy_services import PrivacyServices
from src.orchestrator.context_erasure import build_context_erasure
from src.orchestrator.context_manager import ContextManager, UserContext
from src.orchestrator.context_persister import ContextPersister
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(tmp_path, *, clock=None, base=None, background=False, compactor=None):
    clock = clock or Clock()
    config = {
        "orchestrator": {
            "enabled": False,
            "context": {
                "persistence": {
                    "enabled": False,
                    "base_path": str(base or tmp_path / "contexts"),
                }
            },
        }
    }
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(
            str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=0.01
        )
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy, close_timeout_s=0.01)
        stack.push_async_callback(owner.aclose)
        driver = await build_context_erasure(config, privacy=privacy, owner=owner)
        retention = driver.retention
        if not background:
            retention._maintenance.cancel()
            await asyncio.gather(retention._maintenance, return_exceptions=True)
        persister = ContextPersister(
            config["orchestrator"]["context"]["persistence"]["base_path"],
            source_policy=privacy.ledger,
            clock=clock,
        )
        manager = ContextManager(
            source_policy=privacy.ledger,
            clock=clock,
            persister=persister,
            compactor=compactor,
            compaction_threshold=100,
        )
        stack.push_async_callback(manager.aclose)
        driver.attach_manager(manager)
        short = ShortTermMemory(source_policy=privacy.ledger, clock=clock)
        driver.attach_short_term(short)
        yield privacy, driver, manager, persister, short, clock


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def drain(retention):
    for _ in range(16):
        await retention.reconcile()
        jobs = [*retention._active.values()]
        if retention._runtime_task is not None:
            jobs.append(retention._runtime_task)
        if retention._registry_task is not None:
            jobs.append(retention._registry_task)
        if not jobs:
            return
        await asyncio.gather(*(asyncio.shield(t) for t in jobs))
    raise AssertionError("synthetic retention did not quiesce")


async def snapshot(manager, persister, user, source):
    ctx = await manager.aget_or_create(user)
    manager.add_turn(user, "user", "synthetic private turn", sources=(source,))
    ctx.compacted_summary = "synthetic private summary"
    ctx.summary_sources = (source,)
    ctx.preserved_ids = ["light.synthetic"]
    await manager._offload(persister.save, ctx)
    return ctx


async def test_expiry_removes_idle_summary_turns_short_term_and_file_without_reads_or_erase(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, driver, manager, persister, short, clock):
        source = memory_source(now=clock.now, ttl=30)
        ctx = await snapshot(manager, persister, "erased", source)
        short.add_turn(
            "synthetic private turn", "synthetic private answer", sources=(source,)
        )
        clock.now += 30
        await drain(driver.retention)
        assert (
            ctx.compacted_summary is None
            and ctx.summary_sources == ()
            and ctx.preserved_ids == []
            and ctx.conversation_history == []
        )
        assert not short._turns and not (persister.base_path / "erased.json").exists()
        assert not privacy.ledger.conversation_revoked(source.conversation_id)
        assert not any(row["error"] for row in driver.retention.status()["targets"])


async def test_live_snapshot_and_confirmed_fact_authority_do_not_extend_raw_context_expiry(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, driver, manager, persister, _, clock):
        from dataclasses import replace
        from src.memory.fact_ledger import FactDependency

        source = memory_source(now=clock.now, ttl=30)
        proposed = await privacy.acall(
            privacy.ledger.store,
            "synthetic permanent fact",
            "fact",
            0.8,
            (source,),
            proposed=True,
        )
        fact = await privacy.acall(
            privacy.ledger.confirm, proposed.id, MemoryReviewer("synthetic-admin")
        )
        derived = replace(
            source,
            fact_dependencies=(
                FactDependency(fact.id, fact.revision, fact.expires_at),
            ),
        )
        ctx = await snapshot(manager, persister, "expired", derived)
        live = await snapshot(
            manager, persister, "live", memory_source(now=clock.now, ttl=600)
        )
        clock.now += 30
        await drain(driver.retention)
        assert ctx.compacted_summary is None and live.compacted_summary
        assert (
            not (persister.base_path / "expired.json").exists()
            and (persister.base_path / "live.json").exists()
        )
        assert await privacy.acall(privacy.ledger.get, fact.id) is not None


async def test_historical_policy_directories_expire_with_all_features_disabled_and_legacy_intact(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock, base=tmp_path / "old") as (
        _,
        _,
        manager,
        persister,
        _,
        _,
    ):
        source = memory_source(now=clock.now, ttl=30)
        ctx = await snapshot(manager, persister, "expired", source)
        payload = persister.build_payload(ctx)
        payload["version"] = 2
        persister.previous_policy_path.mkdir(parents=True, exist_ok=True)
        (persister.previous_policy_path / "expired.json").write_text(
            json.dumps(payload)
        )
        legacy = tmp_path / "old" / "expired.json"
        legacy.write_text("synthetic unsourced legacy")
    clock.now += 30
    async with graph(tmp_path, clock=clock, base=tmp_path / "new") as (
        _,
        driver,
        _,
        _,
        _,
        _,
    ):
        await drain(driver.retention)
        assert not (tmp_path / "old" / "policy_v2" / "expired.json").exists()
        assert not (tmp_path / "old" / "policy_v3" / "expired.json").exists()
        assert (
            legacy.read_text() == "synthetic unsourced legacy"
            and not (tmp_path / "new").exists()
        )


async def test_reading_restarting_and_later_sources_never_renew_original_snapshot_expiry(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, _, manager, persister, _, _):
        source = memory_source(now=clock.now, ttl=60)
        await snapshot(manager, persister, "expired", source)
        clock.now += 30
        original = (await manager._offload(persister.load, "expired"))[
            "summary_sources"
        ][0]["expires_at"]
        assert original == source.expires_at
    async with graph(tmp_path, clock=clock) as (_, driver, manager, persister, _, _):
        ctx = await manager.aget_or_create("expired")
        assert ctx.summary_sources[0].expires_at == original
        clock.now += 30
        await drain(driver.retention)
        assert (
            ctx.compacted_summary is None
            and not (persister.base_path / "expired.json").exists()
        )


async def test_new_earlier_deadline_wakes_sleeping_native_and_runtime_retention(
    tmp_path,
):
    async with graph(tmp_path, clock=time.time, background=True) as (
        _,
        driver,
        manager,
        persister,
        short,
        _,
    ):
        await until(lambda: driver.retention._status and not driver.retention._active)
        source = memory_source(now=time.time(), ttl=0.3)
        ctx = await snapshot(manager, persister, "expired", source)
        short.add_turn("synthetic private turn", "synthetic answer", sources=(source,))
        await until(
            lambda: (
                not (persister.base_path / "expired.json").exists()
                and not ctx.conversation_history
                and not short._turns
            )
        )
        assert ctx.compacted_summary is None


async def test_unknown_entry_does_not_starve_expired_files_in_later_batches(tmp_path):
    async with graph(tmp_path) as (_, driver, manager, persister, _, clock):
        source = memory_source(now=clock.now, ttl=30)
        await snapshot(manager, persister, "expired", source)
        for n in range(70):
            (persister.base_path / f"unknown_{n}.json").write_text(
                "{synthetic corrupt data"
            )
        clock.now += 30
        await drain(driver.retention)
        assert not (persister.base_path / "expired.json").exists()
        status = next(
            s for s in driver.retention.status()["targets"] if s.get("pending")
        )
        assert status["pending"] == 70 and status["error"] == "unknown_context_entries"
        assert str(tmp_path) not in json.dumps(driver.retention.status())
        assert len(list(persister.base_path.glob("unknown_*.json"))) == 70


async def test_missing_roots_and_unknown_symlinks_are_never_created_or_followed(
    tmp_path,
):
    async with graph(tmp_path) as (_, driver, _, persister, _, _):
        await drain(driver.retention)
        assert not persister.base_path.exists()
        persister.base_path.mkdir(parents=True)
        external = tmp_path / "external.json"
        external.write_text("synthetic foreign content")
        link = persister.base_path / "foreign.json"
        link.symlink_to(external)
        driver.retention.notify(str(persister.base_path))
        await asyncio.sleep(0)
        await drain(driver.retention)
        assert external.read_text() == "synthetic foreign content" and link.is_symlink()
        assert any(s.get("pending") == 1 for s in driver.retention.status()["targets"])


async def test_native_failure_survives_restart_and_retry_uses_original_source_deadline(
    tmp_path, monkeypatch
):
    import src.orchestrator.context_retention as module

    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, driver, manager, persister, _, _):
        await snapshot(
            manager, persister, "expired", memory_source(now=clock.now, ttl=30)
        )
        clock.now += 30
        with monkeypatch.context() as patch:

            def fail(*args):
                raise OSError("synthetic private path must not be retained")

            patch.setattr(module, "sweep_snapshots", fail)
            await drain(driver.retention)
        assert (persister.base_path / "expired.json").exists()
        assert "synthetic private path" not in json.dumps(driver.retention.status())
    async with graph(tmp_path, clock=clock) as (_, driver, _, persister, _, _):
        await drain(driver.retention)
        assert not (persister.base_path / "expired.json").exists()


async def test_inactive_orphan_draft_removed_without_deleting_unknown_quarantine(
    tmp_path,
):
    async with graph(tmp_path) as (_, driver, _, persister, _, _):
        persister.base_path.mkdir(parents=True)
        orphan = persister.base_path / ".snapshot-abcdefgh"
        orphan.write_text("synthetic abandoned draft")
        corrupt = persister.base_path / "expired.json.corrupt-123"
        corrupt.write_text("synthetic unknown quarantine")
        await drain(driver.retention)
        assert not orphan.exists() and corrupt.exists()
        assert any(s.get("pending") == 1 for s in driver.retention.status()["targets"])


async def test_late_harvest_never_extends_an_already_due_snapshot_deadline(tmp_path):
    async with graph(tmp_path) as (_, driver, manager, persister, _, clock):
        await snapshot(
            manager, persister, "expired", memory_source(now=clock.now, ttl=30)
        )
        await asyncio.sleep(0)
        await driver.retention.reconcile()
        await asyncio.gather(*driver.retention._active.values())
        clock.now += 30
        await drain(driver.retention)
        assert not (persister.base_path / "expired.json").exists()


async def test_expired_noncooperative_compaction_does_not_block_other_runtime_buffers(
    tmp_path,
):
    from tests.integration.test_context_erasure import SlowCompactor

    compactor = SlowCompactor()
    async with graph(tmp_path, compactor=compactor) as (
        _,
        driver,
        manager,
        persister,
        short,
        clock,
    ):
        source = memory_source(now=clock.now, ttl=30)
        await manager.aget_or_create("first")
        manager.add_turn("first", "user", "synthetic private turn", sources=(source,))
        working = asyncio.create_task(
            manager._compact_background("first", keep_recent=0)
        )
        try:
            await compactor.entered.wait()
            clock.now += 30
            await drain(driver.retention)
            await compactor.cancelled.wait()
            assert (
                not working.done()
                and driver.retention.status()["runtime"]["pending_operations"] >= 1
            )
            ctx = await snapshot(
                manager, persister, "second", memory_source(now=clock.now, ttl=30)
            )
            short.add_turn(
                "synthetic second turn",
                "synthetic answer",
                sources=(memory_source(now=clock.now, ttl=30),),
            )
            clock.now += 30
            driver.retention.notify()
            await asyncio.sleep(0)
            await drain(driver.retention)
            assert (
                ctx.compacted_summary is None
                and not short._turns
                and not working.done()
            )
        finally:
            compactor.release.set()
            await working
        assert manager.get("first").compacted_summary is None


async def test_cancelled_retention_observer_keeps_native_work_and_writer_lease(
    tmp_path, monkeypatch
):
    import src.orchestrator.context_retention as module

    async with graph(tmp_path) as (privacy, driver, manager, persister, _, clock):
        await snapshot(
            manager, persister, "expired", memory_source(now=clock.now, ttl=30)
        )
        clock.now += 30
        entered, release = threading.Event(), threading.Event()
        original = module.sweep_snapshots

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(module, "sweep_snapshots", blocked)
        await driver.retention.reconcile()
        await until(entered.is_set)
        observed = asyncio.Event()

        async def watch():
            observed.set()
            await asyncio.shield(next(iter(driver.retention._active.values())))

        observer = asyncio.create_task(watch())
        try:
            await observed.wait()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
        finally:
            release.set()
            await privacy.begin_close()
        assert (
            not privacy.resource_status()["writer_lease"]
            and not (persister.base_path / "expired.json").exists()
        )


async def test_no_snapshot_without_sources_is_written_or_hydrated(tmp_path):
    async with graph(tmp_path) as (_, _, manager, persister, _, _):
        empty = UserContext("unsourced", "synthetic private name")
        with pytest.raises(PermissionError):
            await manager._offload(persister.save, empty)
        persister.base_path.mkdir(parents=True, exist_ok=True)
        path = persister.base_path / "unsourced.json"
        path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "user_id": "unsourced",
                    "user_name": "synthetic private name",
                    "compacted_summary": None,
                    "preserved_ids": [],
                }
            )
        )
        assert await manager._offload(persister.load, "unsourced") is None
        assert path.exists()


async def test_snapshot_user_id_validation_matches_cleanup_protocol_for_newlines(
    tmp_path,
):
    async with graph(tmp_path) as (_, _, manager, persister, _, clock):
        ctx = UserContext(
            "unsafe\n",
            "synthetic private name",
            compacted_summary="synthetic summary",
            summary_sources=(memory_source(now=clock.now),),
        )
        with pytest.raises(ValueError):
            await manager._offload(persister.save, ctx)
        assert not persister.base_path.exists()


async def test_blocked_snapshot_writer_and_registry_do_not_starve_another_known_root(
    tmp_path, monkeypatch
):
    import os

    async with graph(tmp_path) as (privacy, driver, manager, persister, _, clock):
        source = memory_source(now=clock.now, ttl=30)
        ctx = await snapshot(manager, persister, "blocked", source)
        peer = ContextPersister(
            tmp_path / "other", source_policy=privacy.ledger, clock=clock
        )
        other = UserContext(
            "expired",
            "synthetic profile",
            compacted_summary="synthetic private summary",
            summary_sources=(source,),
            _source_policy=privacy.ledger,
            _clock=clock,
        )
        await manager._offload(peer.save, other)
        await driver.retention.refresh()
        await drain(driver.retention)
        entered, release = threading.Event(), threading.Event()
        original = os.replace

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(os, "replace", blocked)
        writing = asyncio.create_task(manager._offload(persister.save, ctx))
        try:
            await until(entered.is_set)
            clock.now += 30
            driver.retention.notify()
            await asyncio.sleep(0)
            # A metadata refresh now waits on the writer's ledger guard. Keep
            # reconciling cached roots instead of awaiting that refresh.
            async with asyncio.timeout(2):
                while (peer.base_path / "expired.json").exists():
                    await driver.retention.reconcile()
                    await asyncio.sleep(0.005)
            assert (
                driver.retention._registry_task is not None
                and not driver.retention._registry_task.done()
            )
            assert not writing.done()
            assert len(list(persister.base_path.glob(".snapshot-*"))) == 1
        finally:
            release.set()
            await writing
        await drain(driver.retention)
        assert not (persister.base_path / "blocked.json").exists()


async def test_saturated_default_executor_remains_occupied_while_context_retention_finishes(
    tmp_path,
):
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        release.wait(3)

    waiting = asyncio.create_task(asyncio.to_thread(blocked))
    try:
        await until(entered.is_set)
        async with graph(tmp_path) as (_, driver, manager, persister, _, clock):
            await snapshot(
                manager, persister, "expired", memory_source(now=clock.now, ttl=30)
            )
            clock.now += 30
            await drain(driver.retention)
            assert (
                not (persister.base_path / "expired.json").exists()
                and not waiting.done()
            )
            release.set()
    finally:
        release.set()
        await waiting


async def test_pending_failed_commit_never_authorizes_snapshot_or_idle_buffer_deletion(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, driver, manager, persister, short, clock):
        source = memory_source(now=clock.now, ttl=300)
        ctx = await snapshot(manager, persister, "valid", source)
        short.add_turn("synthetic valid turn", "synthetic answer", sources=(source,))
        entered, release = threading.Event(), threading.Event()
        db = privacy.ledger._db

        class FailedCommit:
            def __getattr__(self, name):
                return getattr(db, name)

            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError("synthetic failed commit")

        privacy.ledger._db = FailedCommit()
        deleting = asyncio.create_task(
            asyncio.to_thread(
                privacy.ledger.delete_conversation,
                source.conversation_id,
                MemoryReviewer("synthetic-admin"),
            )
        )
        try:
            await until(entered.is_set)
            assert not privacy.sources_available((source,))
            from src.orchestrator.context_storage import sweep_snapshots

            descriptor = next(
                t["descriptor"]
                for t in driver.retention._targets.values()
                if t["descriptor"]["format"] == 3
            )
            result = await driver.retention._work.run(
                sweep_snapshots, privacy.ledger, descriptor, None
            )
            assert result.deleted == 0
            await driver._sweep_runtime()
            assert ctx.compacted_summary and ctx.conversation_history and short._turns
        finally:
            release.set()
            with pytest.raises(OSError):
                await deleting
            privacy.ledger._db = db
        assert (
            persister.base_path / "valid.json"
        ).exists() and privacy.sources_available((source,))


async def test_individual_fact_rejection_cleans_derived_snapshot_without_conversation_request(
    tmp_path,
):
    from dataclasses import replace
    from src.memory.fact_ledger import FactDependency

    async with graph(tmp_path) as (privacy, driver, manager, persister, _, clock):
        source = memory_source(now=clock.now, ttl=300)
        fact = await privacy.acall(
            privacy.ledger.store, "synthetic automatic fact", "fact", 0.8, (source,)
        )
        derived = replace(
            source,
            fact_dependencies=(
                FactDependency(fact.id, fact.revision, fact.expires_at),
            ),
        )
        ctx = await snapshot(manager, persister, "expired", derived)
        await drain(driver.retention)
        await privacy.acall(
            privacy.ledger.delete_fact, fact.id, MemoryReviewer("synthetic-admin")
        )
        clock.now += 30
        await drain(driver.retention)
        assert (
            ctx.compacted_summary is None
            and not (persister.base_path / "expired.json").exists()
        )
        assert not privacy.ledger.conversation_revoked(source.conversation_id)


async def test_snapshot_batches_yield_and_remove_65_files_as_32_32_1(
    tmp_path, monkeypatch
):
    import src.orchestrator.context_retention as module

    async with graph(tmp_path) as (_, driver, manager, persister, _, clock):
        source = memory_source(now=clock.now, ttl=30)
        ctx = await snapshot(manager, persister, "seed", source)
        payload = persister.build_payload(ctx)
        for n in range(64):
            await manager._offload(
                persister.save_payload, {**payload, "user_id": f"user_{n}"}
            )
        clock.now += 30
        receipts = []
        original = module.sweep_snapshots

        def observe(ledger, descriptor, cursor):
            result = original(ledger, descriptor, cursor)
            if descriptor["format"] == 3:
                receipts.append(result)
            return result

        monkeypatch.setattr(module, "sweep_snapshots", observe)
        await drain(driver.retention)
        assert [r.deleted for r in receipts] == [32, 32, 1]
        assert list(persister.base_path.glob("*.json")) == []


async def test_unsupported_target_version_stays_visible_without_opening_its_path(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, driver, _, _, _, _):
        path = tmp_path / "unknown"
        await privacy.acall(
            privacy.ledger.erasures.register_target,
            ErasureParticipant.CONTEXTS,
            2,
            {"kind": "snapshots", "format": 3, "path": str(path)},
        )
        driver.retention.notify()
        await asyncio.sleep(0)
        await drain(driver.retention)
        assert not path.exists()
        assert any(
            s["error"] == "unsupported_context_retention_target"
            for s in driver.retention.status()["targets"]
        )


async def test_runtime_sweep_yields_between_user_batches(tmp_path):
    async with graph(tmp_path) as (_, driver, manager, _, _, clock):
        source = memory_source(now=clock.now, ttl=30)
        for n in range(70):
            await manager.aget_or_create(f"user_{n}")
            manager.add_turn(
                f"user_{n}", "user", "synthetic private turn", sources=(source,)
            )
        clock.now += 30
        observed = []
        running = True

        async def observe():
            while running:
                observed.append(
                    sum(
                        not ctx.conversation_history
                        for ctx in manager._contexts.values()
                    )
                )
                await asyncio.sleep(0)

        observer = asyncio.create_task(observe())
        try:
            await drain(driver.retention)
        finally:
            running = False
            await observer
        assert any(0 < count < 70 for count in observed)
        assert all(not ctx.conversation_history for ctx in manager._contexts.values())


async def test_disabled_context_expiry_never_imports_optional_models(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import asyncio,importlib.abc,json,sys
        from pathlib import Path
        from contextlib import AsyncExitStack
        from uuid import uuid4
        class RejectModels(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,*args):
                if fullname.split('.')[0] in {'torch','chromadb','sentence_transformers','faster_whisper','onnxruntime'}:
                    raise AssertionError('disabled retention imported a model')
        sys.meta_path.insert(0,RejectModels())
        from src.memory.fact_ledger import MemorySource,MemoryOrigin
        from src.memory.privacy_services import build_privacy_services
        from src.memory.erasure_coordinator import ErasureCoordinator
        from src.orchestrator.context_erasure import build_context_erasure
        async def main():
            root=Path(sys.argv[1]); now=2000000000.
            source=MemorySource(str(uuid4()),str(uuid4()),now-100,now-1,MemoryOrigin.VOICE)
            path=root/'contexts'/'policy_v3'/'expired.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'version':3,'user_id':'expired','compacted_summary':'synthetic private summary','summary_sources':[source.to_dict()]}))
            config={'memory':{'enabled':False,'privacy_db_path':str(root/'privacy.db')},'orchestrator':{'enabled':False,'context':{'persistence':{'enabled':False,'base_path':str(root/'contexts')}}}}
            async with AsyncExitStack() as stack:
                privacy=await build_privacy_services(config,shutdown=stack,clock=lambda:now)
                owner=ErasureCoordinator(privacy); stack.push_async_callback(owner.aclose)
                driver=await build_context_erasure(config,privacy=privacy,owner=owner)
                for _ in range(6):
                    await driver.retention.reconcile()
                    jobs=list(driver.retention._active.values())
                    if driver.retention._runtime_task is not None: jobs.append(driver.retention._runtime_task)
                    if driver.retention._registry_task is not None: jobs.append(driver.retention._registry_task)
                    await asyncio.gather(*jobs)
                assert not path.exists() and not privacy.ledger.conversation_revoked(source.conversation_id)
        asyncio.run(main())
    """)
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
