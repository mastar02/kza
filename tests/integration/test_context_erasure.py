"""Physical sourced snapshots, live buffers and retained late context operations."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
import json
import threading
from uuid import uuid4

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import FactDependency, MemoryReviewer
from src.memory.memory_manager import ShortTermMemory
from src.memory.privacy_services import PrivacyServices
from src.orchestrator.compactor import CompactionResult
from src.orchestrator.context_erasure import build_context_erasure
from src.orchestrator.context_manager import ContextManager, ConversationTurn
from src.orchestrator.context_persister import ContextPersister
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer("synthetic-admin")


@asynccontextmanager
async def graph(tmp_path, *, clock=None, base=None, compactor=None, timeout=1):
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
            str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=timeout
        )
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy, close_timeout_s=timeout)
        stack.push_async_callback(owner.aclose)
        driver = await build_context_erasure(config, privacy=privacy, owner=owner)
        persister = ContextPersister(
            config["orchestrator"]["context"]["persistence"]["base_path"],
            source_policy=privacy.ledger,
            clock=clock,
        )
        manager = ContextManager(
            persister=persister,
            source_policy=privacy.ledger,
            clock=clock,
            compactor=compactor,
            compaction_threshold=100,
        )
        driver.attach_manager(manager)
        stack.push_async_callback(manager.aclose)
        short = ShortTermMemory(source_policy=privacy.ledger, clock=clock)
        driver.attach_short_term(short)
        yield privacy, owner, driver, manager, persister, short, clock, config


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def flush(owner):
    """Drain all immediately scheduled batches without advancing retry time."""
    for _ in range(8):
        await owner.reconcile()
        if not owner._active:
            due = await owner.privacy.acall(owner.journal.due, 128)
            if not any(
                ErasureParticipant(row["participant"]) in owner._drivers
                and row["version"]
                in owner._drivers[ErasureParticipant(row["participant"])].versions
                for row in due
            ):
                return
        await asyncio.gather(*(a.task for a in owner._active.values()))
    raise AssertionError("synthetic callbacks did not quiesce")


async def steps(privacy, receipt):
    status = await privacy.acall(privacy.ledger.erasures.snapshot, receipt["id"])
    return [s for s in status["steps"] if s["participant"] == "contexts"]


async def snapshot(manager, persister, user, source):
    ctx = await manager.aget_or_create(user)
    manager.add_turn(user, "user", "synthetic private turn", sources=(source,))
    ctx.compacted_summary = "synthetic private summary"
    ctx.summary_sources = (source,)
    ctx.preserved_ids = ["light.synthetic"]
    await manager._offload(persister.save, ctx)
    return ctx


async def test_connected_erasure_clears_history_summary_short_term_and_file_but_preserves_others(
    tmp_path,
):
    async with graph(tmp_path) as (
        privacy,
        owner,
        _,
        manager,
        persister,
        short,
        clock,
        _,
    ):
        source, live = memory_source(now=clock.now), memory_source(now=clock.now)
        erased = await snapshot(manager, persister, "erased", source)
        survivor = await snapshot(manager, persister, "survivor", live)
        short.add_turn("synthetic erased", "synthetic answer", sources=(source,))
        short.add_turn("synthetic live", "synthetic answer", sources=(live,))
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert (
            erased.compacted_summary is None
            and erased.conversation_history == []
            and erased.preserved_ids == []
        )
        assert (
            survivor.compacted_summary == "synthetic private summary"
            and len(survivor.conversation_history) == 1
        )
        assert (
            not (persister.base_path / "erased.json").exists()
            and (persister.base_path / "survivor.json").exists()
        )
        assert [turn.user_input for turn in short._turns] == ["synthetic live"]
        result = await steps(privacy, receipt)
        assert all(s["state"] == "complete" for s in result), result
        assert (await privacy.acall(privacy.ledger.erasures.snapshot, receipt["id"]))[
            "state"
        ] == "blocked"
        manager.add_turn("erased", "user", "must not return", sources=(source,))
        short.add_turn("must not return", "answer", sources=(source,))
        assert erased.conversation_history == [] and len(short._turns) == 1


async def test_fact_dependency_erasure_reaches_contexts_with_other_conversation_id(
    tmp_path,
):
    async with graph(tmp_path) as (
        privacy,
        owner,
        _,
        manager,
        persister,
        short,
        clock,
        _,
    ):
        source = memory_source(now=clock.now)
        fact = await privacy.acall(
            privacy.ledger.store, "synthetic fact", "fact", 0.8, (source,)
        )
        derived = replace(
            memory_source(now=clock.now),
            fact_dependencies=(
                FactDependency(fact.id, fact.revision, fact.expires_at),
            ),
        )
        ctx = await snapshot(manager, persister, "derived", derived)
        short.add_turn("derived synthetic turn", "synthetic answer", sources=(derived,))
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert not privacy.ledger.conversation_revoked(derived.conversation_id)
        assert ctx.compacted_summary is None and not short._turns
        assert not (persister.base_path / "derived.json").exists()
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


async def test_disabled_features_clean_registered_historical_roots_after_restart(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock, base=tmp_path / "old") as (
        privacy,
        owner,
        _,
        manager,
        persister,
        _,
        _,
        _,
    ):
        source = memory_source(now=clock.now)
        await snapshot(manager, persister, "erased", source)
        path = persister.base_path / "erased.json"
        receipt = await owner.request(source.conversation_id, ACTOR)
    async with graph(tmp_path, clock=clock, base=tmp_path / "new") as (
        privacy,
        owner,
        _,
        _,
        _,
        _,
        _,
        _,
    ):
        await flush(owner)
        assert not path.exists() and not (tmp_path / "new").exists()
        assert len(await steps(privacy, receipt)) == 5
        result = await steps(privacy, receipt)
        assert all(s["state"] == "complete" for s in result), result


async def test_policy_v2_is_erased_by_recorded_sources_while_original_legacy_is_untouched(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, manager, persister, _, clock, _):
        source = memory_source(now=clock.now)
        ctx = await snapshot(manager, persister, "erased", source)
        payload = persister.build_payload(ctx)
        payload["version"] = 2
        persister.previous_policy_path.mkdir(parents=True, exist_ok=True)
        old = persister.previous_policy_path / "erased.json"
        old.write_text(json.dumps(payload))
        legacy = tmp_path / "contexts" / "erased.json"
        legacy.write_text("synthetic unsourced legacy")
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert not old.exists() and legacy.read_text() == "synthetic unsourced legacy"
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


async def test_missing_roots_are_not_created_and_no_optional_models_are_required(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, _, _, _, _, _):
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert not (tmp_path / "contexts").exists()
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


class SlowCompactor:
    def __init__(self):
        self.entered, self.release, self.cancelled = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

    async def compact(self, turns, **kwargs):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        return CompactionResult(
            "synthetic late summary", (), len(turns), "synthetic", 1
        )


async def test_erasure_waits_noncooperative_compaction_and_rejects_late_summary(
    tmp_path,
):
    compactor = SlowCompactor()
    async with graph(tmp_path, compactor=compactor) as (
        privacy,
        owner,
        _,
        manager,
        _,
        _,
        clock,
        _,
    ):
        source = memory_source(now=clock.now)
        ctx = await manager.aget_or_create("erased")
        manager.add_turn("erased", "user", "synthetic private turn", sources=(source,))
        working = asyncio.create_task(
            manager._compact_background("erased", keep_recent=0)
        )
        try:
            await compactor.entered.wait()
            receipt = await owner.request(source.conversation_id, ACTOR)
            await owner.reconcile()
            await compactor.cancelled.wait()
            assert not working.done()
            assert any(s["state"] == "running" for s in await steps(privacy, receipt))
        finally:
            compactor.release.set()
            await working
        await flush(owner)
        assert ctx.compacted_summary is None and ctx.conversation_history == []
        assert not manager._active_compactions
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


async def test_late_native_save_is_retained_then_rejected_after_erasure(
    tmp_path, monkeypatch
):
    async with graph(tmp_path) as (privacy, owner, _, manager, persister, _, clock, _):
        source = memory_source(now=clock.now)
        ctx = await snapshot(manager, persister, "erased", source)
        payload = persister.build_payload(ctx)
        entered, release = threading.Event(), threading.Event()
        save = persister.save_payload

        def blocked(data):
            entered.set()
            assert release.wait(3)
            save(data)

        monkeypatch.setattr(persister, "save_payload", blocked)
        writing = asyncio.create_task(manager._offload(persister.save_payload, payload))
        try:
            await until(entered.is_set)
            receipt = await owner.request(source.conversation_id, ACTOR)
            await owner.reconcile()
            await until(
                lambda: any(
                    s.participant == ErasureParticipant.CONTEXTS
                    for s in (a.step for a in owner._active.values())
                )
            )
            assert not writing.done()
        finally:
            release.set()
            with pytest.raises(PermissionError):
                await writing
        await flush(owner)
        assert not (persister.base_path / "erased.json").exists()
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


async def test_file_replace_finishes_before_marker_and_cannot_outlive_physical_receipt(
    tmp_path, monkeypatch
):
    import os

    async with graph(tmp_path) as (privacy, owner, _, manager, persister, _, clock, _):
        source = memory_source(now=clock.now)
        ctx = await snapshot(manager, persister, "erased", source)
        entered, release = threading.Event(), threading.Event()
        original = os.replace

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(os, "replace", blocked)
        writing = asyncio.create_task(manager._offload(persister.save, ctx))
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
        await flush(owner)
        assert not (persister.base_path / "erased.json").exists()
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


@pytest.mark.parametrize(
    "content",
    [
        "{bad json",
        json.dumps(
            {
                "version": 3,
                "user_id": "unknown",
                "compacted_summary": "synthetic unknown summary",
            }
        ),
    ],
)
async def test_unknown_snapshots_remain_pending_without_exposing_private_errors(
    tmp_path, content
):
    async with graph(tmp_path) as (privacy, owner, _, _, persister, _, _, _):
        persister.base_path.mkdir(parents=True)
        path = persister.base_path / "unknown.json"
        path.write_text(content)
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        result = await steps(privacy, receipt)
        assert any(
            s["state"] == "pending" and s["last_error"] == "cleanup_failed"
            for s in result
        )
        assert path.read_text() == content and str(tmp_path) not in json.dumps(result)


async def test_orphan_draft_is_removed_but_unknown_corrupt_quarantine_stays_pending(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, owner, _, _, persister, _, _, _):
        persister.base_path.mkdir(parents=True)
        orphan = persister.base_path / ".snapshot-abcdefgh"
        orphan.write_text("synthetic abandoned draft")
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert not orphan.exists()
        bad = persister.base_path / "erased.json.corrupt-123"
        bad.write_text("synthetic unknown quarantine")
        # A new operation must inspect unknown quarantines instead of claiming
        # that a previous completed receipt covered later files.
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert bad.exists() and any(
            s["state"] == "pending" for s in await steps(privacy, receipt)
        )


async def test_shared_persisters_do_not_delete_an_active_draft(tmp_path, monkeypatch):
    import os

    async with graph(tmp_path) as (privacy, _, _, manager, persister, _, clock, _):
        source = memory_source(now=clock.now)
        ctx = await snapshot(manager, persister, "erased", source)
        peer = ContextPersister(
            tmp_path / "contexts", source_policy=privacy.ledger, clock=clock
        )
        await manager._offload(peer.load, "erased")
        entered, release = threading.Event(), threading.Event()
        original = os.replace

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(os, "replace", blocked)
        writing = asyncio.create_task(manager._offload(persister.save, ctx))
        purging = None
        try:
            await until(entered.is_set)
            purging = asyncio.create_task(manager._offload(peer.purge_expired))
            await asyncio.sleep(0.02)
            assert not purging.done()
        finally:
            release.set()
            await writing
            await purging
        assert (persister.base_path / "erased.json").exists()


async def test_unknown_unsourced_live_buffers_prevent_runtime_completion(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, manager, _, short, _, _):
        ctx = await manager.aget_or_create("unknown")
        ctx.conversation_history.append(
            ConversationTurn("user", "synthetic unknown raw turn")
        )
        ctx.pending_confirmation = {"synthetic": "unknown provenance"}
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert any(
            s["state"] == "pending" and s["pending_count"] == 2
            for s in await steps(privacy, receipt)
        )
        assert ctx.conversation_history


async def test_context_native_save_keeps_writer_lease_after_close_observation_timeout(
    tmp_path, monkeypatch
):
    import os

    async with graph(tmp_path, timeout=0.01) as (
        privacy,
        _,
        _,
        manager,
        persister,
        _,
        clock,
        _,
    ):
        ctx = await snapshot(manager, persister, "erased", memory_source(now=clock.now))
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
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert (
                privacy.resource_status()["writer_lease"]
                and len(manager._io_tasks) == 1
            )
            assert manager._io_closed
        finally:
            release.set()
            await writing
            await privacy.begin_close()
        assert not privacy.resource_status()["writer_lease"] and not manager._io_tasks


async def test_cancelled_disk_cleanup_observer_does_not_abandon_claim_or_lease(
    tmp_path, monkeypatch
):
    import src.orchestrator.context_erasure as module

    async with graph(tmp_path, timeout=0.01) as (
        privacy,
        owner,
        driver,
        manager,
        persister,
        _,
        clock,
        _,
    ):
        source = memory_source(now=clock.now)
        await snapshot(manager, persister, "erased", source)
        await owner.request(source.conversation_id, ACTOR)
        targets = await privacy.acall(
            privacy.ledger.erasures.targets, ErasureParticipant.CONTEXTS
        )
        target = next(t for t in targets if t["descriptor"].get("format") == 3)
        due = next(
            row
            for row in await privacy.acall(privacy.ledger.erasures.due)
            if row["target_id"] == target["id"]
        )
        claim = await privacy.acall(privacy.ledger.erasures.claim, due)
        entered, release = threading.Event(), threading.Event()
        original = module.cleanup_snapshots

        def blocked(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(module, "cleanup_snapshots", blocked)
        observer = asyncio.create_task(driver.clean(claim))
        try:
            await until(entered.is_set)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert (
                privacy.resource_status()["writer_lease"]
                and driver._disk.status()["operations"] == 1
            )
        finally:
            release.set()
            await privacy.begin_close()
        assert not (persister.base_path / "erased.json").exists()


async def test_cancelled_hydration_is_retained_until_erasure_and_cannot_publish_old_summary(
    tmp_path, monkeypatch
):
    async with graph(tmp_path) as (privacy, owner, _, manager, persister, _, clock, _):
        source = memory_source(now=clock.now)
        await snapshot(manager, persister, "erased", source)
        manager.remove_context("erased")
        entered, release = threading.Event(), threading.Event()
        load = persister.load

        def blocked(user):
            payload = load(user)
            entered.set()
            assert release.wait(3)
            return payload

        monkeypatch.setattr(persister, "load", blocked)
        reading = asyncio.create_task(manager.aget_or_create("erased"))
        try:
            await until(entered.is_set)
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
            receipt = await owner.request(source.conversation_id, ACTOR)
            await owner.reconcile()
            assert len(manager._io_tasks) == 1
            assert any(s["state"] == "running" for s in await steps(privacy, receipt))
        finally:
            release.set()
        await flush(owner)
        assert manager.get("erased").compacted_summary is None
        assert not (persister.base_path / "erased.json").exists()
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


async def test_persister_uses_registered_physical_directory_after_config_alias_changes(
    tmp_path,
):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(old, target_is_directory=True)
    async with graph(tmp_path, base=alias) as (
        privacy,
        owner,
        _,
        manager,
        persister,
        _,
        clock,
        _,
    ):
        source = memory_source(now=clock.now)
        ctx = await snapshot(manager, persister, "erased", source)
        alias.unlink()
        alias.symlink_to(new, target_is_directory=True)
        await manager._offload(persister.save, ctx)
        assert (
            persister.base_path == old / "policy_v3"
            and not (new / "policy_v3").exists()
        )
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert not (old / "policy_v3" / "erased.json").exists()
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


async def test_completed_context_receipts_are_rechecked_after_restored_file_on_leased_restart(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (
        privacy,
        owner,
        _,
        manager,
        persister,
        _,
        _,
        _,
    ):
        source = memory_source(now=clock.now)
        await snapshot(manager, persister, "erased", source)
        path = persister.base_path / "erased.json"
        original = path.read_bytes()
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert not path.exists()
        path.write_bytes(original)
    async with graph(tmp_path, clock=clock) as (privacy, owner, _, _, _, _, _, _):
        assert any(s["state"] == "pending" for s in await steps(privacy, receipt))
        await flush(owner)
        assert not path.exists() and all(
            s["state"] == "complete" for s in await steps(privacy, receipt)
        )


async def test_snapshot_cleanup_batches_65_files_with_private_hash_cursors(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, manager, persister, _, clock, _):
        source = memory_source(now=clock.now)
        ctx = await snapshot(manager, persister, "seed", source)
        payload = persister.build_payload(ctx)
        for n in range(64):
            data = {**payload, "user_id": f"user_{n}"}
            await manager._offload(persister.save_payload, data)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert len(list(persister.base_path.glob("*.json"))) == 33
        clock.now += 0.02
        await flush(owner)
        assert len(list(persister.base_path.glob("*.json"))) == 1
        clock.now += 0.02
        await flush(owner)
        assert list(persister.base_path.glob("*.json")) == []
        assert all(s["state"] == "complete" for s in await steps(privacy, receipt))


async def test_pending_failed_commit_never_authorizes_irreversible_snapshot_purge(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, _, manager, persister, _, clock, _):
        source = memory_source(now=clock.now)
        await snapshot(manager, persister, "erased", source)
        db = privacy.ledger._db
        entered, release = threading.Event(), threading.Event()

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
                privacy.ledger.delete_conversation, source.conversation_id, ACTOR
            )
        )
        try:
            await until(entered.is_set)
            assert not privacy.sources_available((source,))
            assert await manager._offload(persister.purge_expired) == 0
            assert (persister.base_path / "erased.json").exists()
        finally:
            release.set()
            with pytest.raises(OSError):
                await deleting
            privacy.ledger._db = db
        assert privacy.sources_available((source,))


async def test_router_idle_query_cache_is_physically_pruned_and_cannot_be_refilled(
    tmp_path,
):
    from src.pipeline.request_router import RequestRouter

    async with graph(tmp_path) as (privacy, owner, driver, _, _, _, clock, _):
        router = RequestRouter(
            None, None, None, privacy=privacy, orchestrator_enabled=False
        )
        driver.attach_router(router)
        try:
            source, live = memory_source(now=clock.now), memory_source(now=clock.now)
            router._query_cache.put("erased", "synthetic private query", (source,))
            router._query_cache.put("live", "synthetic live query", (live,))
            receipt = await owner.request(source.conversation_id, ACTOR)
            assert "erased" in router._query_cache._entries
            await flush(owner)
            assert set(router._query_cache._entries) == {"live"}
            assert not router._query_cache.put(
                "erased", "synthetic late copy", (source,)
            )
            assert all(s["state"] == "complete" for s in await steps(privacy, receipt))
        finally:
            await router.aclose()


async def test_saturated_default_pool_does_not_starve_context_writes_or_physical_cleanup(
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
        async with graph(tmp_path) as (
            privacy,
            owner,
            _,
            manager,
            persister,
            _,
            clock,
            _,
        ):
            source = memory_source(now=clock.now)
            await snapshot(manager, persister, "erased", source)
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            assert not (persister.base_path / "erased.json").exists()
            assert all(s["state"] == "complete" for s in await steps(privacy, receipt))
            release.set()
    finally:
        release.set()
        await waiting


async def test_disabled_context_cleanup_does_not_import_optional_models(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import asyncio,importlib.abc,json,sys
        from contextlib import AsyncExitStack
        from pathlib import Path
        from uuid import uuid4
        class RejectModels(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,*args):
                if fullname.split('.')[0] in {'torch','chromadb','sentence_transformers','faster_whisper','onnxruntime'}:
                    raise AssertionError('cleanup imported an optional model')
        sys.meta_path.insert(0,RejectModels())
        from src.memory.privacy_services import build_privacy_services
        from src.memory.erasure_coordinator import ErasureCoordinator
        from src.memory.fact_ledger import MemorySource,MemoryOrigin,MemoryReviewer
        from src.orchestrator.context_erasure import build_context_erasure
        async def main():
            root=Path(sys.argv[1]); now=2000000000.
            source=MemorySource(str(uuid4()),str(uuid4()),now,now+3600,MemoryOrigin.VOICE)
            path=root/'contexts'/'policy_v3'/'erased.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'version':3,'user_id':'erased','compacted_summary':'synthetic private summary','summary_sources':[source.to_dict()]}))
            config={'memory':{'enabled':False,'privacy_db_path':str(root/'privacy.db')},'orchestrator':{'enabled':False,'context':{'persistence':{'enabled':False,'base_path':str(root/'contexts')}}}}
            async with AsyncExitStack() as stack:
                privacy=await build_privacy_services(config,shutdown=stack,clock=lambda:now)
                owner=ErasureCoordinator(privacy); stack.push_async_callback(owner.aclose)
                await build_context_erasure(config,privacy=privacy,owner=owner)
                await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
                for _ in range(4):
                    await owner.reconcile()
                    await asyncio.gather(*(a.task for a in owner._active.values()))
                assert not path.exists()
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
