"""Actual raw read/handoff ownership survives cancellation and native delays."""

import asyncio
from dataclasses import replace
import json
import threading
from uuid import uuid4
from weakref import ref

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import FactDependency
from src.memory.producer_scopes import ProducerScopes
from tests.integration.test_ambient_producer_privacy import active_source, until
from tests.integration.test_distillation_privacy import ACTOR, FACTS, graph as distillation_graph
from tests.mocks.ambient_graph import ambient_graph, chunk

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('native_sqlite', [False, True], ids=['row-handoff', 'sqlite-thread'])
async def test_cancelled_unresolved_read_keeps_scope_and_lease_until_actual_exit(tmp_path, monkeypatch, native_sqlite):
    async with distillation_graph(tmp_path) as (privacy, store, distiller, _, sources):
        entered, release = threading.Event(), threading.Event()
        async_release = asyncio.Event()
        read = store.undistilled_live
        if native_sqlite:
            def held(text):
                entered.set()
                assert release.wait(3)
                return text
            await store._db.create_function('held_text', 1, held)
            async def stalled(**kwargs):
                cursor = await store._db.execute('SELECT *,held_text(text) AS held FROM utterances')
                return [dict(row) for row in await cursor.fetchall()]
        else:
            async def stalled(**kwargs):
                rows = await read(**kwargs)
                entered.set()
                await async_release.wait()
                return rows
        monkeypatch.setattr(store, 'undistilled_live', stalled)
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await until(entered.is_set)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            assert distiller.resource_status()['cycles'] == 0
            assert distiller.resource_status()['reads'] == 1
            assert distiller.resource_status()['source_scopes']['unknown'] == 1
            await privacy.acall(privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR)
            assert distiller.conversation_busy(sources[0].conversation_id)
            assert distiller.conversation_busy(str(uuid4()))  # Read has not resolved its lineage yet.
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert privacy.resource_status()['writer_lease']
            assert not distiller._closing.done()
        finally:
            release.set()
            async_release.set()
            await distiller.begin_close()
            await privacy.begin_close()
        assert not distiller.snapshot_operations()
        assert distiller.resource_status()['source_scopes']['scopes'] == 0
        assert not privacy.resource_status()['writer_lease']


@pytest.mark.parametrize('selection', ['all', 'budget', 'language'])
async def test_known_prompt_affinity_excludes_unrelated_and_retired_input_locals(tmp_path, selection):
    entered, release = asyncio.Event(), asyncio.Event()
    async def chat(prompt):
        entered.set()
        await release.wait()
        return FACTS
    async with distillation_graph(tmp_path, chat=chat) as (privacy, store, distiller, _, sources):
        if selection == 'budget':
            rows = await store.undistilled_live()
            distiller.max_batch_chars = len(rows[0]['text'])+40
            rows = None
        if selection == 'language':
            distiller._lang_detect = lambda text: ('en' if text.endswith('1') else 'es', 1.)
            distiller.drop_language = 'en'
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert distiller.conversation_busy(sources[0].conversation_id)
            assert distiller.conversation_busy(sources[1].conversation_id) is (selection == 'all')
            assert not distiller.conversation_busy(str(uuid4()))
            status = json.dumps(distiller.resource_status())
            assert 'Synthetic private' not in status and sources[0].conversation_id not in status
        finally:
            release.set()
            await observer
        assert not distiller.conversation_busy(sources[0].conversation_id)
        assert distiller.resource_status()['source_scopes']['scopes'] == 0


@pytest.mark.parametrize('stop', ['cancel', 'revoke'])
async def test_native_save_keeps_exact_affinity_after_cycle_observer_exits(tmp_path, stop):
    async with distillation_graph(tmp_path) as (privacy, _, distiller, _, sources):
        entered, release = threading.Event(), threading.Event()
        original = distiller._store_fact
        def save(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return original(*args, **kwargs)
        distiller._store_fact = save
        observer = asyncio.create_task(distiller.distill_once())
        try:
            await until(entered.is_set)
            if stop == 'cancel':
                observer.cancel()
            else:
                await privacy.acall(privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR)
            await asyncio.gather(observer, return_exceptions=True)
            await until(lambda: distiller.resource_status()['cycles'] == 0)
            assert distiller.resource_status()['cycles'] == 0
            assert distiller.resource_status()['native_operations'] == 1
            assert distiller.conversation_busy(sources[0].conversation_id)
            assert not distiller.conversation_busy(str(uuid4()))
        finally:
            release.set()
            await until(lambda: not distiller.snapshot_operations())
        assert not distiller.conversation_busy(sources[0].conversation_id)


async def test_read_exception_does_not_escape_with_raw_locals_or_diagnostics(tmp_path, monkeypatch, caplog):
    async with distillation_graph(tmp_path) as (_, store, distiller, _, sources):
        references = []
        class Payload:
            text = 'synthetic private exception payload'
        async def failed(**kwargs):
            payload = Payload()
            references.append(ref(payload))
            raise ValueError(payload)
        monkeypatch.setattr(store, 'undistilled_live', failed)
        assert await distiller.distill_once() == 0
        assert references and references[0]() is None
        assert 'synthetic private exception payload' not in caplog.text
        assert not distiller.conversation_busy(sources[0].conversation_id)
        assert not distiller.snapshot_operations()


async def test_short_batch_leaves_no_raw_read_or_unknown_scope(tmp_path):
    async with distillation_graph(tmp_path) as (_, _, distiller, _, sources):
        distiller.min_batch = 3
        assert await distiller.distill_once() == 0
        assert not distiller.conversation_busy(sources[0].conversation_id)
        assert distiller.resource_status()['reads'] == 0
        assert distiller.resource_status()['source_scopes']['scopes'] == 0


@pytest.mark.parametrize('stage', ['stt', 'profiles', 'speaker', 'quality', 'doa'])
async def test_sourced_ambient_native_work_keeps_scope_after_observer_cancel(tmp_path, stage):
    async with ambient_graph(tmp_path, stage=stage) as g:
        observer = asyncio.create_task(g.transcriber._handle_segment('synthetic-a', g.segment()))
        try:
            await until(g.gates[stage].entered.is_set)
            source = active_source(g.transcriber)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            assert g.transcriber.conversation_busy(source.conversation_id)
            assert not g.transcriber.conversation_busy(str(uuid4()))
            assert g.privacy.resource_status()['writer_lease']
            status = json.dumps(g.transcriber.resource_status())
            assert source.conversation_id not in status and 'synthetic-transcript' not in status
        finally:
            g.gates[stage].release.set()
            await until(lambda: not g.transcriber.snapshot_operations())
        assert not g.transcriber.conversation_busy(source.conversation_id)
        assert g.transcriber.resource_status()['source_scopes']['scopes'] == 0


async def test_room_handoff_stays_finite_and_sourced_after_model_task_finishes(tmp_path, monkeypatch):
    async with ambient_graph(tmp_path) as g:
        entered, release = asyncio.Event(), asyncio.Event()
        captured = []
        original = g.transcriber._handle_segment
        async def handoff(room, segment):
            captured.append(ref(segment.audio))
            await original(room, segment)
            entered.set()
            await release.wait()
        monkeypatch.setattr(g.transcriber, '_handle_segment', handoff)
        await g.path.start()
        g.tap.push('synthetic-a', chunk(), ts=g.clock.now)
        g.tap.push('synthetic-a', chunk(0), ts=g.clock.now+.08)
        try:
            await asyncio.wait_for(entered.wait(), 3)
            rows = await g.store.undistilled_live(require_provenance=True)
            assert len(rows) == 1
            conversation_id = rows[0]['conversation_id']
            assert not g.transcriber._segments and g.transcriber._batches
            assert g.transcriber.conversation_busy(conversation_id)
            assert not g.transcriber.conversation_busy(str(uuid4()))
            # The long-lived idle controller has no audio locals across awaits.
            controller = next(task for task in g.transcriber._tasks if task.get_coro().__name__ == '_room_worker')
            assert not {'items', 'seg'} & set(controller.get_coro().cr_frame.f_locals.keys())
            assert captured[0]() is not None
        finally:
            release.set()
            await until(lambda: not g.transcriber._batches)
        assert captured[0]() is None
        assert not g.transcriber.conversation_busy(conversation_id)


async def test_cancelled_unknown_vad_batch_keeps_scope_until_actual_arrays_retire(tmp_path):
    async with ambient_graph(tmp_path, stage='vad') as g:
        await g.path.start()
        audio = chunk()
        reference = ref(audio)
        g.tap.push('synthetic-a', audio, ts=g.clock.now)
        del audio
        try:
            await until(g.gates['vad'].entered.is_set)
            closing = g.transcriber.begin_close()
            await until(lambda: not g.transcriber._batches)
            assert g.transcriber.conversation_busy(str(uuid4()))
            assert g.transcriber.resource_status()['source_scopes']['unknown'] >= 1
            assert reference() is not None and not closing.done()
        finally:
            g.gates['vad'].release.set()
            await g.transcriber.begin_close()
        assert reference() is None
        assert not g.transcriber.conversation_busy(str(uuid4()))


class SnapshotOwner:
    def __init__(self):
        self.tasks = set()

    def snapshot(self):
        return tuple(self.tasks)


async def test_scope_capacity_is_owned_by_real_children_after_parent_exit(tmp_path):
    async with distillation_graph(tmp_path) as (privacy, _, _, _, sources):
        owner = SnapshotOwner()
        registry = ProducerScopes(owner.snapshot, capacity=1)
        entered, release = asyncio.Event(), asyncio.Event()
        references = []
        class Payload:
            text = 'synthetic raw controller payload'
        async def child():
            payload = Payload()
            references.append(ref(payload))
            entered.set()
            try:
                await release.wait()
            finally:
                payload = None
        with registry.operation((sources[0],)):
            task = asyncio.create_task(child())
            owner.tasks.add(task)
            task.add_done_callback(owner.tasks.discard)
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert references[0]() is not None
            assert registry.busy(sources[0].conversation_id, privacy.ledger)
            with pytest.raises(RuntimeError, match='producer_scope_capacity'):
                with registry.operation((sources[1],)):
                    raise AssertionError('capacity was released before actual child exit')
        finally:
            release.set()
            await task
            await until(lambda: not owner.tasks)
        assert references[0]() is None
        assert registry.resource_status()['scopes'] == 0
        with registry.operation((sources[1],)):
            assert registry.busy(sources[1].conversation_id, privacy.ledger)


async def test_lost_scope_owner_cannot_be_reported_as_proof_of_native_exit(tmp_path):
    async with distillation_graph(tmp_path) as (privacy, _, _, _, sources):
        owner = SnapshotOwner()
        reference = ref(owner)
        registry = ProducerScopes(owner.snapshot)
        with registry.operation((sources[0],)):
            del owner
        assert reference() is None
        assert registry.busy(sources[0].conversation_id, privacy.ledger)
        assert registry.resource_status()['scopes'] == 1


async def test_derived_scope_stays_busy_when_its_fact_source_is_revoked(tmp_path):
    async with distillation_graph(tmp_path) as (privacy, _, _, _, sources):
        fact = await privacy.acall(privacy.ledger.store, 'synthetic derived fact', 'fact', .9,
                                  sources=(sources[0],))
        derived = replace(sources[1], fact_dependencies=(FactDependency(fact.id, fact.revision, fact.expires_at),))
        owner = SnapshotOwner()
        registry = ProducerScopes(owner.snapshot)
        with registry.operation((derived,)):
            assert not registry.busy(sources[0].conversation_id, privacy.ledger)
            await privacy.acall(privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR)
            assert registry.busy(sources[0].conversation_id, privacy.ledger)
        assert not registry.busy(sources[0].conversation_id, privacy.ledger)
