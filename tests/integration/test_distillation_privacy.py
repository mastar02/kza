"""Real capture/catalog scopes through inference, cancellation and late storage."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
import json
import logging
import threading
from types import SimpleNamespace
from weakref import ref

import pytest

from src.ambient.distiller import Distiller
from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance
from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryOrigin, MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.prompt_context import SourcesInvalidated, run_private
from tests.integration.test_pattern_evidence import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer("synthetic-admin")
FACTS = '[{"fact":"Synthetic useful fact","category":"fact","confidence":0.8}]'


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


@asynccontextmanager
async def graph(tmp_path, *, chat=None, save=None, ttl=60, count=2):
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(
            str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=0.03
        )
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        store = AmbientStore(
            str(tmp_path / "ambient.db"), source_policy=privacy.ledger, clock=clock
        )
        await store.init()
        stack.push_async_callback(store.close)
        sources = []
        for item in range(count):
            source = replace(
                memory_source(now=clock.now, ttl=ttl), origin=MemoryOrigin.AMBIENT
            )
            sources.append(source)
            await store.add(
                AmbientUtterance(
                    "synthetic-room",
                    clock.now,
                    clock.now + 1,
                    text=f"Synthetic private transcript {item}",
                    source="live",
                    vad_prob=0.9,
                    capture_id=source.capture_id,
                    conversation_id=source.conversation_id,
                    expires_at=source.expires_at,
                )
            )

        async def extract(prompt):
            return FACTS

        def persist(fact, category, *, confidence, metadata, sources):
            return privacy.ledger.store(fact, category, confidence, sources)

        distiller = Distiller(
            store,
            chat or extract,
            save or persist,
            min_batch=1,
            privacy=privacy,
            close_timeout_s=0.03,
        )
        stack.push_async_callback(distiller.aclose)
        yield privacy, store, distiller, clock, sources


async def test_actual_catalog_fact_keeps_exact_batch_sources_and_original_30_day_deadline(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, store, distiller, clock, sources):
        assert await distiller.distill_once() == 1
        facts = await privacy.acall(privacy.ledger.list)
        assert len(facts) == 1 and set(facts[0].sources) == set(sources)
        assert facts[0].expires_at == clock.now + 30 * 86400
        clock.now += 60
        await store.purge_expired(sweep_orphans=False)
        assert await store.undistilled_live() == []
        assert (
            await privacy.acall(privacy.ledger.get, facts[0].id)
        ).expires_at == facts[0].expires_at
        assert not distiller.snapshot_operations()


@pytest.mark.parametrize("invalidation", ["expiry", "revocation"])
async def test_inflight_chat_is_cancelled_and_inputs_not_marked_on_invalidation(
    tmp_path, invalidation
):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def extract(prompt):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with graph(tmp_path, chat=extract) as (
        privacy,
        store,
        distiller,
        clock,
        sources,
    ):
        running = asyncio.create_task(distiller.distill_once())
        await entered.wait()
        if invalidation == "expiry":
            clock.now += 60
        else:
            await privacy.acall(
                privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
            )
        assert await asyncio.wait_for(running, 2) == 0
        assert cancelled.is_set() and not distiller.snapshot_operations()
        rows = await (
            await store._db.execute("SELECT distilled FROM utterances")
        ).fetchall()
        assert rows and all(not r[0] for r in rows)
        assert await privacy.acall(privacy.ledger.list) == []


async def test_revocation_after_read_before_prompt_prevents_model_admission(
    tmp_path, monkeypatch
):
    called = []

    async def extract(prompt):
        called.append(prompt)
        return FACTS

    async with graph(tmp_path, chat=extract) as (privacy, store, distiller, _, sources):
        read = store.undistilled_live

        async def interleave(**kwargs):
            rows = await read(**kwargs)
            await privacy.acall(
                privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
            )
            return rows

        monkeypatch.setattr(store, "undistilled_live", interleave)
        assert await distiller.distill_once() == 0 and called == []


async def test_mixed_batch_loses_entire_derivation_then_other_conversation_can_retry(
    tmp_path,
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def extract(prompt):
        entered.set()
        await release.wait()
        return FACTS

    async with graph(tmp_path, chat=extract) as (privacy, store, distiller, _, sources):
        running = asyncio.create_task(distiller.distill_once())
        await entered.wait()
        await privacy.acall(
            privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
        )
        assert await asyncio.wait_for(running, 2) == 0
        release.set()
        assert await distiller.distill_once() == 1
        fact = (await privacy.acall(privacy.ledger.list))[0]
        assert fact.sources == (sources[1],)
        assert len(await store.undistilled_live()) == 0


async def test_noncooperative_model_remains_owned_after_caller_cancel_and_close_timeout(
    tmp_path,
):
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def extract(prompt):
        entered.set()
        try:
            await asyncio.shield(release.wait())
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return FACTS

    async with graph(tmp_path, chat=extract) as (privacy, _, distiller, _, _):
        observer = asyncio.create_task(distiller.distill_once())
        await entered.wait()
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        await cancelled.wait()
        try:
            assert distiller.resource_status()["cycles"] == 1
            with pytest.raises(RuntimeError, match="capacity"):
                await distiller.distill_once()
            with pytest.raises(WorkCloseTimeout):
                await distiller.aclose()
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert (
                privacy.resource_status()["writer_lease"]
                and distiller.snapshot_operations()
            )
        finally:
            release.set()
            await distiller.begin_close()
            await privacy.begin_close()
        assert (
            not distiller.snapshot_operations()
            and not privacy.resource_status()["writer_lease"]
        )


@pytest.mark.parametrize("invalidation", ["expiry", "revocation"])
async def test_native_store_is_retained_and_late_guard_rejects_invalid_sources(
    tmp_path, invalidation
):
    entered, release = threading.Event(), threading.Event()
    async with graph(tmp_path) as (privacy, store, distiller, clock, sources):
        actual = distiller._store_fact

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return actual(*args, **kwargs)

        distiller._store_fact = blocked
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await until(entered.is_set)
            if invalidation == "expiry":
                clock.now += 60
            else:
                await privacy.acall(
                    privacy.ledger.delete_conversation,
                    sources[0].conversation_id,
                    ACTOR,
                )
            assert await asyncio.wait_for(observer, 2) == 0
            assert distiller.resource_status()["native_operations"] == 1
            assert distiller.resource_status()["store_callbacks"] == 1
            with pytest.raises(RuntimeError, match="capacity"):
                await distiller.distill_once()
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await until(lambda: not distiller.snapshot_operations())
        assert await privacy.acall(privacy.ledger.list) == []
        assert all(
            not r[0]
            for r in await (
                await store._db.execute("SELECT distilled FROM utterances")
            ).fetchall()
        )


async def test_cancelled_native_observer_keeps_writer_lease_until_actual_callback_exit(
    tmp_path,
):
    entered, release = threading.Event(), threading.Event()
    async with graph(tmp_path) as (privacy, _, distiller, _, _):
        actual = distiller._store_fact

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return actual(*args, **kwargs)

        distiller._store_fact = blocked
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await until(entered.is_set)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
            assert (
                not distiller._closing.done()
                and distiller.resource_status()["native_operations"] == 1
            )
        finally:
            release.set()
            await distiller.begin_close()
            await privacy.begin_close()
        assert not privacy.resource_status()["writer_lease"]


@pytest.mark.parametrize(
    "response",
    [
        "malformed",
        "{}",
        "Synthetic error: []",
        '{"error":"[]"}',
        '[{"fact":"Synthetic","category":"unknown"}]',
        '[{"fact":"Synthetic","category":"fact","confidence":"NaN"}]',
        " " * 65537,
    ],
)
async def test_malformed_or_unbounded_model_response_does_not_acknowledge_input(
    tmp_path, response
):
    async def extract(prompt):
        return response

    async with graph(tmp_path, chat=extract) as (_, store, distiller, _, _):
        assert await distiller.distill_once() == 0
        assert len(await store.undistilled_live()) == 2


@pytest.mark.parametrize("response", ["[]", "```json\n[]\n```"])
async def test_valid_empty_response_acknowledges_inputs_without_creating_facts(
    tmp_path,
    response,
):
    async def extract(prompt):
        return response

    async with graph(tmp_path, chat=extract) as (privacy, store, distiller, _, _):
        assert await distiller.distill_once() == 0
        assert await store.undistilled_live() == []
        assert await privacy.acall(privacy.ledger.list) == []


@pytest.mark.parametrize("stage", ["chat", "store"])
async def test_provider_failure_logs_fixed_type_and_can_retry_without_private_exception(
    tmp_path, caplog, stage
):
    async with graph(tmp_path) as (_, store, distiller, _, _):
        actual = distiller._chat if stage == "chat" else distiller._store_fact

        def fail(*args, **kwargs):
            raise OSError("Synthetic private transcript and path")

        if stage == "chat":

            async def failing(prompt):
                fail()

            distiller._chat = failing
        else:
            distiller._store_fact = fail
        with caplog.at_level(logging.WARNING):
            assert await distiller.distill_once() == 0
        own_messages = [
            r.getMessage() for r in caplog.records if r.name == "src.ambient.distiller"
        ]
        assert own_messages and all(
            "Synthetic private" not in message for message in own_messages
        )
        assert "Synthetic private" not in json.dumps(distiller.resource_status())
        assert len(await store.undistilled_live()) == 2
        if stage == "chat":
            distiller._chat = actual
        else:
            distiller._store_fact = actual
        assert await distiller.distill_once() == 1


async def test_invalidated_native_callback_waiting_for_default_executor_never_enters_store(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, _, distiller, clock, _):
        loop = asyncio.get_running_loop()
        previous = loop._default_executor
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def hold():
            entered.set()
            assert release.wait(3)

        def persist(*args, **kwargs):
            calls.append(args)

        distiller._store_fact = persist
        blocker = loop.run_in_executor(None, hold)
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await until(entered.is_set)
            await until(lambda: distiller.resource_status()["native_operations"] == 1)
            clock.now += 60
            assert await asyncio.wait_for(observer, 2) == 0
            assert (
                not blocker.done()
                and distiller.resource_status()["native_operations"] == 1
            )
        finally:
            release.set()
            await blocker
            await until(lambda: not distiller.snapshot_operations())
            loop._default_executor = previous
            pool.shutdown(wait=True)
        assert calls == [] and await privacy.acall(privacy.ledger.list) == []


async def test_idle_periodic_distiller_stop_wakes_without_waiting_six_hours(tmp_path):
    async with graph(tmp_path) as (_, _, distiller, _, _):
        task = asyncio.create_task(distiller.run_forever())
        await asyncio.sleep(0)
        distiller.stop()
        await asyncio.wait_for(task, 0.1)
        assert not distiller.snapshot_operations()


async def test_cancelled_private_monitor_does_not_keep_late_result_in_observer_traceback():
    @dataclass
    class Result:
        text: str

    entered, release = asyncio.Event(), asyncio.Event()
    references = []

    async def provider():
        entered.set()
        try:
            await asyncio.shield(release.wait())
        except asyncio.CancelledError:
            await release.wait()
        value = Result("Synthetic private late result")
        references.append(ref(value))
        return value

    observer = asyncio.create_task(run_private(provider(), lambda: True))
    await entered.wait()
    observer.cancel()
    release.set()
    result = await asyncio.gather(observer, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    await until(lambda: references and references[0]() is None)
    assert observer.cancelled() and result  # Keep the observer and its traceback alive.


async def test_closed_distiller_rejects_new_work_and_other_authority(tmp_path):
    async with graph(tmp_path) as (privacy, store, distiller, _, _):
        with pytest.raises(ValueError, match="authority_mismatch"):
            Distiller(
                type("OtherStore", (), {"source_policy": None})(),
                distiller._chat,
                distiller._store_fact,
                privacy=privacy,
            )
        await distiller.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await distiller.distill_once()
        with pytest.raises(RuntimeError, match="closed"):
            await distiller.run_forever()
        assert len(await store.undistilled_live()) == 2


async def test_failed_pending_marker_aborts_derivation_but_preserves_retryable_input(
    tmp_path,
):
    entered = asyncio.Event()

    async def extract(prompt):
        entered.set()
        await asyncio.Event().wait()

    async with graph(tmp_path, chat=extract) as (privacy, store, distiller, _, sources):
        observer = asyncio.create_task(distiller.distill_once())
        await entered.wait()
        writing, release = threading.Event(), threading.Event()
        original = privacy.ledger._db

        class FailedCommit:
            def __getattr__(self, key):
                return getattr(original, key)

            def commit(self):
                writing.set()
                assert release.wait(3)
                raise OSError("Synthetic failed marker")

        privacy.ledger._db = FailedCommit()
        marking = asyncio.create_task(
            privacy.acall(
                privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
            )
        )
        try:
            await until(writing.is_set)
            assert await asyncio.wait_for(observer, 2) == 0
            assert (
                await store.undistilled_live() != []
            )  # The unrelated capture is still available.
            assert all(
                not r[0]
                for r in await (
                    await store._db.execute("SELECT distilled FROM utterances")
                ).fetchall()
            )
        finally:
            release.set()
            with pytest.raises(OSError, match="failed marker"):
                await marking
            privacy.ledger._db = original

        async def retry(prompt):
            return FACTS

        distiller._chat = retry
        assert len(await store.undistilled_live()) == 2
        assert await distiller.distill_once() == 1


async def test_original_row_objects_are_released_before_waiting_for_model(
    tmp_path, monkeypatch
):
    class Text(str):
        pass

    references = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def extract(prompt):
        entered.set()
        await release.wait()
        return FACTS

    async with graph(tmp_path, chat=extract) as (_, store, distiller, _, _):
        read = store.undistilled_live

        async def tracked(**kwargs):
            rows = await read(**kwargs)
            for row in rows:
                row["text"] = Text(row["text"])
                references.append(ref(row["text"]))
            return rows

        monkeypatch.setattr(store, "undistilled_live", tracked)
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await entered.wait()
            assert references and all(r() is None for r in references)
        finally:
            release.set()
            await observer


async def test_real_async_memory_callback_remains_visible_while_its_native_delegate_runs(
    tmp_path, monkeypatch
):
    from src.memory.long_term import LongTermMemory
    from tests.mocks.memory_graph import MemoryClient

    entered, release = threading.Event(), threading.Event()
    async with graph(tmp_path) as (privacy, _, distiller, _, _):
        memory = LongTermMemory(
            MemoryClient(tmp_path / "chroma"), privacy=privacy, shutdown_timeout_s=0.03
        )
        await memory.acall(memory.initialize)
        persist = memory.store_fact

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return persist(*args, **kwargs)

        monkeypatch.setattr(memory, "store_fact", blocked)
        distiller._store_fact = memory.astore_fact
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await until(entered.is_set)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            await until(lambda: distiller.resource_status()["cycles"] == 0)
            assert distiller.resource_status()["store_callbacks"] == 1
            assert distiller.resource_status()["native_operations"] == 0
            assert memory._operations
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()["writer_lease"]
            assert not distiller._closing.done()
        finally:
            release.set()
            await distiller.begin_close()
            await memory.begin_close()
            await privacy.begin_close()
        assert not distiller.snapshot_operations() and not memory._operations


async def test_real_ambient_builder_attaches_distiller_to_shared_privacy_before_start(
    tmp_path,
):
    from src.ambient.transcriber import build_ambient_path

    async with graph(tmp_path) as (privacy, _, _, clock, _):

        def persist(fact, category, *, confidence, metadata, sources):
            return privacy.ledger.store(fact, category, confidence, sources)

        path = build_ambient_path(
            {
                "shadow_mode": False,
                "db_path": str(tmp_path / "other.db"),
                "distill": {"min_batch": 1, "destination": "local"},
            },
            {},
            [],
            persist,
            privacy=privacy,
        )
        try:
            assert (
                path.distiller._privacy is privacy
                and path.distiller._policy is privacy.ledger
            )
            await path.store.init()
            source = memory_source(now=clock.now, ttl=60)
            await path.store.add(
                AmbientUtterance(
                    "synthetic-room",
                    clock.now,
                    clock.now + 1,
                    text="Synthetic builder capture",
                    source="live",
                    vad_prob=0.9,
                    capture_id=source.capture_id,
                    conversation_id=source.conversation_id,
                    expires_at=source.expires_at,
                )
            )

            async def extract(prompt):
                return FACTS

            path.distiller._chat = extract
            assert await path.distiller.distill_once() == 1
        finally:
            await path.stop()
        assert path.distiller._closed and path.store._db is None


async def test_cancelled_ambient_stop_keeps_store_open_until_actual_distillation_exit(
    tmp_path,
):
    from src.ambient.transcriber import AmbientPath

    entered, release = threading.Event(), threading.Event()
    async with graph(tmp_path) as (_, store, distiller, _, _):
        persist = distiller._store_fact

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return persist(*args, **kwargs)

        distiller._store_fact = blocked

        async def stop_transcriber():
            pass

        path = AmbientPath(
            None, SimpleNamespace(stop=stop_transcriber), store, distiller
        )
        observer = asyncio.create_task(distiller.distill_once())
        closing = None
        try:
            await until(entered.is_set)
            closing = asyncio.create_task(path.stop())
            await until(lambda: distiller._closed)
            closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)
            assert store._db is not None and not path._closing.done()
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await path.begin_close()
        assert store._db is None
        with pytest.raises(RuntimeError, match="ambient_path_closed"):
            await path.start()


async def test_invalidated_monitor_releases_completed_private_result():
    available = [True]
    entered, release = asyncio.Event(), asyncio.Event()

    @dataclass
    class Value:
        text: str

    references = []

    async def provider():
        entered.set()
        try:
            await asyncio.shield(release.wait())
        except asyncio.CancelledError:
            await release.wait()
        value = Value("Synthetic private revoked result")
        references.append(ref(value))
        return value

    observer = asyncio.create_task(run_private(provider(), lambda: available[0]))
    await entered.wait()
    available[0] = False
    release.set()
    error = None
    try:
        await observer
    except SourcesInvalidated as caught:
        error = caught
    assert error is not None
    await until(lambda: references and references[0]() is None)
    assert error.__traceback__ is not None  # No GC manipulation or traceback clearing.
