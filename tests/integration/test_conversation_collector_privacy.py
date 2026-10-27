"""Actual turn files, feedback replacement, derived exports and native lifetime."""

import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import producer_context
from src.pipeline.request_router import RequestRouter
from src.training.conversation_collector import build_conversation_collector, LoRATrainer
from src.training.habit_dataset_generator import HabitDatasetGenerator
from src.training.nightly_trainer import NightlyTrainer, NightlyConfig
from src.wakeword.clip_store import build_wake_clip_store
from tests.mocks.conversations import graph
from tests.mocks.wake_clips import until

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


async def marked(g, *, sources=None):
    assert await g.add(sources=sources)
    view, = await g.collector.get_turns()
    assert await g.collector.mark_response(view,'good',sources=(g.base.source(),))
    return (await g.collector.get_turns())[0]


def turn_files(g):
    return list((g.collector.data_dir/'policy_v1').glob('*'))


def block_encode(g, monkeypatch, *, exports_only=False):
    entered,release = threading.Event(),threading.Event()
    encode = g.collector._encode
    def blocked(job):
        if not exports_only or job.target.extension=='.jsonl':
            entered.set()
            assert release.wait(5)
        encode(job)
    monkeypatch.setattr(g.collector,'_encode',blocked)
    return entered,release


@pytest.mark.parametrize('which',['input','response'])
async def test_either_source_revocation_fences_record_and_export_before_cleanup(tmp_path,which):
    async with graph(tmp_path) as g:
        sources = g.sources()
        view = await marked(g,sources=sources)
        export = await g.collector.export_for_training(str(tmp_path/'exports'))
        path = Path(export.output_path)
        source = sources[0] if which=='input' else sources[1]
        await g.privacy.acall(g.privacy.ledger.delete_conversation,source.conversation_id,ACTOR)
        assert not view.available and view.user_input==view.assistant_response==''
        assert not export.available and export.output_path==''
        assert not await g.collector.get_turns()
        await g.store.reconcile()
        assert not path.exists() and not turn_files(g) and not await g.rows()


async def test_idle_ttl_clears_held_raw_fields_and_removes_all_file_kinds(tmp_path):
    async with graph(tmp_path,background=True) as g:
        sources = g.sources(input_ttl=2)
        view = await marked(g,sources=sources)
        export = await g.collector.export_for_training(str(tmp_path/'exports'))
        path = Path(export.output_path)
        g.clock.now = sources[0].expires_at
        await until(lambda: view._scope.retired.is_set() and export._scope.retired.is_set())
        assert view.__dict__['user_input']==view.__dict__['assistant_response']==''
        await until(lambda: not path.exists() and not turn_files(g))
        assert not await g.rows()
        g.clock.now = sources[0].captured_at
        assert not view.available and export.output_path==''


async def test_feedback_invalidates_previously_exported_version_even_with_live_captures(tmp_path):
    async with graph(tmp_path) as g:
        view = await marked(g)
        export = await g.collector.export_for_training(str(tmp_path/'exports'))
        old_path = Path(export.output_path)
        assert await g.collector.mark_response(view,'corrected','synthetic correction',sources=(g.base.source(),))
        assert g.privacy.sources_available(export._scope.sources)
        assert not export.available and export.output_path==''
        await g.store.reconcile()
        assert not old_path.exists()
        current, = await g.collector.get_turns()
        assert current.correction=='synthetic correction'
        fresh = await g.collector.export_for_training(str(tmp_path/'exports'))
        assert json.loads(Path(fresh.output_path).read_text())['output']=='synthetic correction'


async def test_expired_parent_during_export_encoding_cannot_publish(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        sources = g.sources(input_ttl=1)
        await marked(g,sources=sources)
        entered,release = block_encode(g,monkeypatch,exports_only=True)
        exporting = asyncio.create_task(g.collector.export_for_training(str(tmp_path/'exports')))
        try:
            await until(entered.is_set)
            g.clock.now = sources[0].expires_at
            await g.store.reconcile()
        finally:
            release.set()
            assert await exporting is None
        assert not list((tmp_path/'exports'/'policy_v1').glob('*'))


async def test_feedback_replacement_while_export_is_blocked_prevents_stale_copy(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        view = await marked(g)
        entered,release = block_encode(g,monkeypatch,exports_only=True)
        exporting = asyncio.create_task(g.collector.export_for_training(str(tmp_path/'exports')))
        try:
            await until(entered.is_set)
            assert await g.collector.mark_response(view,'bad',sources=(g.base.source(),))
            await g.store.reconcile()
        finally:
            release.set()
            assert await exporting is None
        assert not list((tmp_path/'exports'/'policy_v1').glob('*'))


async def test_cancelled_turn_observer_keeps_native_writer_and_privacy_lease(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        entered,release = block_encode(g,monkeypatch)
        adding = asyncio.create_task(g.add())
        try:
            await until(entered.is_set)
            adding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await adding
            for close in (g.collector.aclose,g.privacy.aclose):
                with pytest.raises(WorkCloseTimeout):
                    await close()
            assert g.privacy.resource_status()['writer_lease'] and g.collector.resource_status()['active_writes']
        finally:
            release.set()
            await g.privacy.begin_close()
        assert not turn_files(g)
        assert not g.privacy.resource_status()['writer_lease']


async def test_failed_unlink_keeps_file_locator_and_blocks_expired_reads(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        sources = g.sources(input_ttl=1)
        assert await g.add(sources=sources)
        g.clock.now = sources[0].expires_at
        unlink = g.store._unlink
        def fail(*args):
            raise OSError('synthetic private path')
        monkeypatch.setattr(g.store,'_unlink',fail)
        await g.store.reconcile()
        assert g.store.resource_status()['state']=='unavailable'
        assert turn_files(g) and len(await g.rows())==1
        assert not await g.collector.get_turns()
        monkeypatch.setattr(g.store,'_unlink',unlink)
        await g.store.reconcile()
        assert not turn_files(g) and not await g.rows()


async def test_raw_reader_is_owned_after_cancel_and_cannot_deliver_revoked_data(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        sources = g.sources()
        assert await g.add(sources=sources)
        entered,release = threading.Event(),threading.Event()
        load = g.collector._load
        def blocked(row):
            entered.set()
            assert release.wait(5)
            return load(row)
        monkeypatch.setattr(g.collector,'_load',blocked)
        reading = asyncio.create_task(g.collector.get_turns())
        try:
            await until(entered.is_set)
            await g.privacy.acall(g.privacy.ledger.delete_conversation,sources[0].conversation_id,ACTOR)
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            await g.privacy.begin_close()
        assert g.collector.resource_status()['read_views']==0


async def test_native_error_and_blocked_log_do_not_keep_job_payload(tmp_path,monkeypatch):
    import src.training.conversation_collector as module
    async with graph(tmp_path) as g:
        entered,release = threading.Event(),threading.Event()
        jobs = []
        def fail(job):
            jobs.append(job)
            raise OSError('synthetic private message')
        def log(message,error_type):
            assert 'synthetic private message' not in message and error_type=='OSError'
            entered.set()
            assert release.wait(5)
        monkeypatch.setattr(g.collector,'_encode',fail)
        monkeypatch.setattr(module,'logger',SimpleNamespace(warning=log))
        adding = asyncio.create_task(g.add())
        try:
            await until(entered.is_set)
            assert jobs and all(not job.data and not job.body for job in jobs)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
        finally:
            release.set()
            assert await adding is None
            await g.privacy.begin_close()


async def test_legacy_conversations_are_not_read_exported_or_rotated(tmp_path):
    async with graph(tmp_path) as g:
        legacy = g.collector.data_dir/'conv_legacy.json'
        legacy.parent.mkdir(parents=True)
        legacy.write_text(json.dumps({'id':'synthetic legacy','turns':[{'user_input':'synthetic private legacy'}]}))
        assert (await g.collector.get_stats())['total_turns']==0
        assert not await g.collector.get_turns()
        assert await g.collector.export_for_training(str(tmp_path/'exports'),only_marked=False) is None
        assert 'synthetic private legacy' in legacy.read_text()


@pytest.mark.parametrize('damage',['content','oversize','symlink'])
async def test_unrecognized_turn_file_cannot_become_an_input(tmp_path,damage):
    async with graph(tmp_path) as g:
        identifier = await g.add()
        path = g.collector.data_dir/'policy_v1'/(identifier+'.json')
        if damage=='content':
            path.write_text('{}')
        elif damage=='oversize':
            path.write_bytes(b'x'*65537)
        else:
            target = tmp_path/'synthetic-sentinel'
            target.write_text('synthetic untouched')
            path.unlink()
            path.symlink_to(target)
        with pytest.raises(RuntimeError,match='conversation_operation_failed'):
            await g.collector.get_turns()


async def test_feedback_never_extends_original_deadline_and_uses_feedback_source(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources(input_ttl=3)
        assert await g.add(sources=sources)
        view, = await g.collector.get_turns()
        feedback = g.base.source(ttl=20)
        g.clock.now += 1
        assert await g.collector.mark_response(view,'good',sources=(feedback,))
        current, = await g.collector.get_turns()
        assert current.expires_at==sources[0].expires_at and feedback in current.sources
        await g.privacy.acall(g.privacy.ledger.delete_conversation,feedback.conversation_id,ACTOR)
        assert current.user_input==''
        await g.store.reconcile()
        assert not await g.rows()


async def test_router_only_acknowledges_actual_scoped_feedback(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources()
        assert await g.add(sources=sources)
        router = RequestRouter(MagicMock(),MagicMock(),MagicMock(),conversation_collector=g.collector,
                               privacy=g.privacy,orchestrator_enabled=False)
        try:
            feedback = g.feedback(sources[0])
            token = router._memory_source.set(feedback)
            try:
                with producer_context('another-room',(feedback,)):
                    denied = await router._check_feedback('buena respuesta')
                assert denied['is_feedback'] and not denied['success']
                with producer_context('synthetic-room',(feedback,)):
                    saved = await router._check_feedback('buena respuesta')
                assert saved['success']
            finally:
                router._memory_source.reset(token)
        finally:
            await router.aclose()


async def test_closed_views_and_exports_are_scrubbed_while_valid_files_keep_original_ttl(tmp_path):
    async with graph(tmp_path) as g:
        view = await marked(g)
        export = await g.collector.export_for_training(str(tmp_path/'exports'))
        path = Path(export.output_path)
        await g.collector.aclose()
        assert view.user_input=='' and export.output_path==''
        assert path.exists()


async def test_restart_without_nightly_training_cleans_historical_exports(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources(input_ttl=1)
        await marked(g,sources=sources)
        export = await g.collector.export_for_training(str(tmp_path/'exports'))
        path = Path(export.output_path)
        clock = g.clock
    clock.now = sources[0].expires_at
    config = {'rooms':{'wake_word':{'capture_clips':{'enabled':False,'dir':str(tmp_path/'unused_wake')}}},
        'training':{'nightly':{'enabled':False},'conversations_dir':str(tmp_path/'unused_conversations'),
                    'stt_corrections':{'enabled':False,'data_dir':str(tmp_path/'unused_stt')}}}
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path/'privacy.db'),clock=clock)
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        store = await build_wake_clip_store(config,privacy=privacy,shutdown=stack)
        await build_conversation_collector(config,store=store)
        await until(lambda: not path.exists() and not list((tmp_path/'conversations'/'policy_v1').glob('*')))
        assert not (tmp_path/'unused_conversations').exists()


async def test_untracked_learning_consumers_reject_managed_conversations_before_copy(tmp_path):
    async with graph(tmp_path) as g:
        await marked(g)
        generator = HabitDatasetGenerator(data_dir=str(tmp_path/'habits'),conversation_collector=g.collector)
        with pytest.raises(RuntimeError,match='private_training_session_unavailable'):
            generator._generate_from_conversations()
        trainer = NightlyTrainer(NightlyConfig(data_dir=str(tmp_path/'nightly'),output_dir=str(tmp_path/'models')),
                                 habit_generator=generator)
        with pytest.raises(RuntimeError,match='private_training_session_unavailable'):
            trainer.collect_daily_data()
        export = await g.collector.export_for_training(str(tmp_path/'exports'))
        lora = LoRATrainer(output_dir=str(tmp_path/'adapters'))
        with pytest.raises(RuntimeError,match='private_training_session_unavailable'):
            lora.train(export.output_path)


async def test_candidate_profile_keeps_unverified_nightly_training_disabled():
    import yaml
    config = yaml.safe_load(Path('config/settings.yaml').read_text())
    assert config['training']['nightly']['enabled'] is False


async def test_export_batches_continue_without_silently_dropping_source_lineage(tmp_path):
    async with graph(tmp_path) as g:
        for i in range(23):
            assert await g.add(user_input=f'synthetic input {i}')
            view = (await g.collector.get_turns(limit=1))[0]
            assert await g.collector.mark_response(view,'good',sources=(g.base.source(),))
        first = await g.collector.export_for_training(str(tmp_path/'exports'))
        assert first.exported==21 and first.has_more and len(first._scope.sources)==63
        second = await g.collector.export_for_training(str(tmp_path/'exports'),after_sequence=first.next_cursor)
        assert second.exported==2 and not second.has_more
        entries = [json.loads(line) for export in (first,second) for line in Path(export.output_path).read_text().splitlines()]
        assert [row['instruction'] for row in entries]==[f'synthetic input {i}' for i in range(23)]
        assert len({row['_kza']['record_id'] for row in entries})==23


async def test_full_executor_queue_expires_without_waiting_for_physical_sweep(tmp_path,monkeypatch):
    async with graph(tmp_path,background=True) as g:
        entered,release = threading.Event(),threading.Event()
        calls = []
        jobs = []
        encode = g.collector._encode
        def blocked(job):
            calls.append(threading.get_ident())
            entered.set()
            assert release.wait(5)
            encode(job)
        monkeypatch.setattr(g.collector,'_encode',blocked)
        sources = g.sources(input_ttl=1)
        adding = [asyncio.create_task(g.add(sources=sources)) for _ in range(8)]
        try:
            await until(lambda: len(g.collector._jobs)==8 and len(calls)==2)
            await until(lambda: g.store._sweep is not None and not g.store._sweep.done()
                        and g.store._work.status()['operations']>0)
            assert await g.add() is None
            jobs = [job for job in g.collector._jobs.values() if job.id not in g.collector._active]
            assert len(jobs)==6
            g.clock.now = sources[0].expires_at
            await until(lambda: all(not job.data and not job.body for job in jobs))
            assert g.collector.resource_status()['active_writes']==2
            g.clock.now = sources[0].captured_at
        finally:
            release.set()
            await asyncio.gather(*adding)
        assert not turn_files(g) and not await g.rows()


async def test_export_partial_failure_keeps_durable_intent_for_later_cleanup(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        await marked(g)
        encode,discard = g.collector._encode,g.store.discard
        def interrupted(job):
            encode(job)
            raise OSError('synthetic interrupted export')
        def unavailable(*args):
            raise OSError('synthetic unlink unavailable')
        monkeypatch.setattr(g.collector,'_encode',interrupted)
        monkeypatch.setattr(g.store,'discard',unavailable)
        assert await g.collector.export_for_training(str(tmp_path/'exports')) is None
        parts = list((tmp_path/'exports'/'policy_v1').glob('*.part'))
        assert len(parts)==1
        def intents():
            with g.privacy.ledger._transaction() as db:
                return [dict(row) for row in db.execute("SELECT * FROM wake_clips_v1 WHERE bucket='conversation_export'")]
        pending, = await g.store._work.run(intents)
        assert not pending['published']
        monkeypatch.setattr(g.store,'discard',discard)
        await g.store.reconcile()
        assert not parts[0].exists() and not await g.store._work.run(intents)


async def test_replacement_race_has_one_winner_and_cannot_fall_back_to_an_older_turn(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        assert await g.add()
        original, = await g.collector.get_turns()
        entered,release = block_encode(g,monkeypatch)
        first = asyncio.create_task(g.collector.mark_response(original,'corrected','synthetic first',sources=(g.base.source(),)))
        try:
            await until(entered.is_set)
            # Restore only new encoder admission; the first native invocation
            # remains inside its already-entered gate.
            monkeypatch.setattr(g.collector,'_encode',lambda job: type(g.collector)._encode(g.collector,job))
            assert await g.collector.mark_response(original,'corrected','synthetic second',sources=(g.base.source(),))
        finally:
            release.set()
            assert not await first
        current, = await g.collector.get_turns()
        assert current.correction=='synthetic second'


async def test_source_metadata_and_digest_are_required_for_legacy_like_files(tmp_path):
    async with graph(tmp_path) as g:
        identifier = await g.add()
        path = g.collector.data_dir/'policy_v1'/(identifier+'.json')
        envelope = json.loads(path.read_text())
        envelope['sources'] = []
        raw = json.dumps(envelope).encode()
        path.write_bytes(raw)
        def align_digest_only():
            with g.privacy.ledger._transaction() as db:
                db.execute('UPDATE conversation_turns_v1 SET digest=? WHERE id=?',(hashlib.sha256(raw).hexdigest(),identifier))
        await g.store._work.run(align_digest_only)
        with pytest.raises(RuntimeError,match='conversation_operation_failed'):
            await g.collector.get_turns()


async def test_offline_cli_uses_authority_and_managed_output_paths(tmp_path,capsys):
    import importlib.util
    import yaml
    import time
    spec = importlib.util.spec_from_file_location('synthetic_train_cli',Path('scripts/train_assistant.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    async with graph(tmp_path) as g:
        g.clock.now = time.time()  # The independent CLI uses the real recording clock.
        await marked(g)
    config = {'memory':{'privacy_db_path':str(tmp_path/'privacy.db')},
              'rooms':{'wake_word':{'capture_clips':{'dir':str(tmp_path/'unused_wake')}}},
              'training':{'conversations_dir':str(tmp_path/'conversations'),
                          'stt_corrections':{'enabled':False,'data_dir':str(tmp_path/'unused_stt')}}}
    config_path = tmp_path/'settings.yaml'
    config_path.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(action='export',config=str(config_path),output=None,output_dir=str(tmp_path/'exports'),
                           format='alpaca',include_all=False,limit=32,after_sequence=0)
    assert await asyncio.to_thread(module.cmd_training_data,args)==0
    assert 'Archivo temporal:' in capsys.readouterr().out
    paths = list((tmp_path/'exports'/'policy_v1').glob('*.jsonl'))
    assert len(paths)==1 and json.loads(paths[0].read_text())['instruction']=='synthetic input'
    with sqlite3.connect(tmp_path/'privacy.db') as db:
        assert db.execute("SELECT COUNT(*) FROM wake_clips_v1 WHERE bucket='conversation_export' AND published=1").fetchone()[0]==1


async def test_deadline_expiring_inside_commit_callback_cannot_publish(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        entered,release = threading.Event(),threading.Event()
        commit = g.collector._commit
        def blocked(db,job):
            commit(db,job)
            entered.set()
            assert release.wait(5)
        monkeypatch.setattr(g.collector,'_commit',blocked)
        sources = g.sources(input_ttl=1)
        adding = asyncio.create_task(g.add(sources=sources))
        try:
            await until(entered.is_set)
            g.clock.now = sources[0].expires_at
        finally:
            release.set()
            identifier = await adding
        assert identifier is None
        assert not turn_files(g) and not await g.rows()


async def test_deadline_expiring_during_actual_sqlite_commit_cannot_publish(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        connection = g.privacy.ledger._db
        entered,release = threading.Event(),threading.Event()
        class Connection:
            armed = False
            def __getattr__(self,name):
                return getattr(connection,name)
            def commit(self):
                if self.armed:
                    entered.set()
                    assert release.wait(5)
                return connection.commit()
        proxy = Connection()
        g.privacy.ledger._db = proxy
        commit = g.collector._commit
        def arm(db,job):
            commit(db,job)
            proxy.armed = True
        monkeypatch.setattr(g.collector,'_commit',arm)
        sources = g.sources(input_ttl=1)
        adding = asyncio.create_task(g.add(sources=sources))
        try:
            await until(entered.is_set)
            g.clock.now = sources[0].expires_at
        finally:
            release.set()
            identifier = await adding
            g.privacy.ledger._db = connection
        assert identifier is None
        assert not turn_files(g) and not await g.rows()
