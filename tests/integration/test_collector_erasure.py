"""Physical collector receipts include actual native ownership and unknown data."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import json
from pathlib import Path
import threading
from uuid import uuid4

import numpy as np
import pytest

from src.memory.erasure_journal import ErasureParticipant
from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import FactDependency
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.privacy_services import PrivacyServices
from src.memory.legacy_quarantine import stage_legacy_quarantine
from src.wakeword.clip_store import build_wake_clip_store
from tests.mocks.legacy_memory import legacy_memory
from tests.mocks.collector_erasure import configuration
from tests.mocks.collector_erasure import ACTOR, graph
from tests.mocks.wake_clips import until

pytestmark=pytest.mark.asyncio


async def claim(g, source, target):
    await g.receipt(source)
    due=await g.privacy.acall(g.privacy.ledger.erasures.due,128)
    row=next(row for row in due if row['target_id']==target.id)
    return await g.privacy.acall(g.privacy.ledger.erasures.claim,row)


@pytest.mark.parametrize('kind',['wake','stt','conversations','exports'])
async def test_journal_removes_each_managed_dataset_and_keeps_unrelated_sources(tmp_path,kind):
    async with graph(tmp_path) as g:
        source,other=g.source(),g.source()
        result=await g.add(kind,source)
        assert result is not None
        if kind!='exports':
            assert await g.add(kind,other)
        await g.receipt(source)
        await g.settle(source)
        records=await g.records()
        assert not any(any(s['conversation_id']==source.conversation_id for s in json.loads(row['sources_json'])) for row in records)
        if kind!='exports':
            assert any(any(s['conversation_id']==other.conversation_id for s in json.loads(row['sources_json'])) for row in records)
        receipt=await g.receipt(source)
        assert receipt['state']!='complete'  # Other participants are still unavailable.
        assert not any('root' in row or 'descriptor' in row for row in receipt['steps'])


@pytest.mark.parametrize('kind',['wake','stt','conversations','exports'])
async def test_native_writer_retains_claim_and_lease_until_its_real_exit(tmp_path,monkeypatch,kind):
    async with graph(tmp_path) as g:
        source=g.source()
        entered,release=threading.Event(),threading.Event()
        writer={'wake':g.wake,'stt':g.stt,'conversations':g.conversations,'exports':g.conversations}[kind]
        original=writer._encode
        def held(job):
            if kind!='exports' or job.target.kind.value=='conversation_exports':
                entered.set()
                assert release.wait(2)
            original(job)
        monkeypatch.setattr(writer,'_encode',held)
        if kind=='wake':
            assert g.wake.submit('synthetic-room',.99,np.ones(1600,dtype=np.float32),sources=(source,))
            adding=None
        else:
            adding=asyncio.create_task(g.add(kind,source))
        cleaning=None
        try:
            await until(entered.is_set)
            step=await claim(g,source,g.targets[kind])
            cleaning=asyncio.create_task(g.driver.clean(step))
            await asyncio.sleep(.02)
            assert not cleaning.done() and g.privacy.resource_status()['writer_lease']
            assert g.store.conversation_busy(step.target_id,source.conversation_id)
            release.set()
            assert (await cleaning).done
            if adding is not None:
                await adding
            assert not g.store.conversation_busy(step.target_id,source.conversation_id)
        finally:
            release.set()
            if adding is not None:
                await adding
            if cleaning is not None:
                await cleaning


async def test_unrelated_blocked_encoder_does_not_block_another_conversation(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source,other=g.source(),g.source()
        fact=await g.privacy.acall(g.privacy.ledger.store,'unrelated live fact','fact',.9,sources=(g.source(),))
        other=replace(other,fact_dependencies=(FactDependency(fact.id,fact.revision,fact.expires_at),))
        entered,release=threading.Event(),threading.Event()
        original=g.wake._encode
        def held(job):
            original(job)
            entered.set()
            assert release.wait(2)
        monkeypatch.setattr(g.wake,'_encode',held)
        assert g.wake.submit('synthetic-room',.99,np.ones(1600,dtype=np.float32),sources=(other,))
        try:
            await until(entered.is_set)
            await g.receipt(source)
            await g.settle(source)
            assert g.wake._busy is not None
        finally:
            release.set()


@pytest.mark.parametrize('location',['legacy','orphan','symlink'])
async def test_unattributed_files_remain_pending_and_are_never_deleted(tmp_path,location):
    async with graph(tmp_path) as g:
        source=g.source()
        await g.add('conversations',source)
        root=Path(g.targets['conversations'].root)
        outside=tmp_path/'outside'
        outside.write_text('preserve')
        path=root/('legacy.json' if location=='legacy' else 'policy_v1/'+str(uuid4())+'.json')
        if location=='symlink':
            path.symlink_to(outside)
        else:
            path.write_text('unattributed private marker')
        step=await claim(g,source,g.targets['conversations'])
        result=await g.driver.clean(step)
        assert not result.done and path.exists() and outside.read_text()=='preserve'
        assert not await g.records()


async def test_configured_legacy_training_root_blocks_only_its_receipt(tmp_path):
    async with graph(tmp_path) as g:
        source=g.source()
        root=tmp_path/'habits'
        root.mkdir()
        path=root/'user_profiles.json'
        path.write_text('{"unverified":"private legacy marker"}')
        await g.add('wake',source)
        await g.receipt(source)
        for _ in range(30):
            await g.owner.reconcile()
            g.clock.now+=.02
            await asyncio.sleep(.001)
        steps=await g.steps(source)
        receipt=await g.receipt(source)
        def state():
            with g.privacy.ledger._transaction() as db:
                return db.execute('SELECT state FROM erasure_steps WHERE operation_id=? AND target_id=?',
                    (receipt['id'],g.targets['wake'].id)).fetchone()[0]
        assert await g.privacy.acall(state)=='complete'
        assert any(row['state']!='complete' for row in steps)
        assert path.read_text()=='{"unverified":"private legacy marker"}'


async def test_old_numeric_cursor_cannot_hide_raw_metadata_behind_it(tmp_path):
    async with graph(tmp_path) as g:
        source=g.source()
        await g.add('stt',source)
        step=await claim(g,source,g.targets['stt'])
        result=await g.driver.clean(replace(step,cursor='s:9223372036854775807:9223372036854775807'))
        assert result.done and not await g.records()
        def count():
            with g.privacy.ledger._transaction() as db:
                return db.execute('SELECT COUNT(*) FROM stt_corrections_v1').fetchone()[0]
        assert await g.privacy.acall(count)==0


@pytest.mark.parametrize('kind',['stt','conversations'])
async def test_native_reader_scope_survives_observer_cancellation_and_source_revocation(tmp_path,monkeypatch,kind):
    async with graph(tmp_path) as g:
        source=g.source()
        await g.add(kind,source)
        entered,release=threading.Event(),threading.Event()
        captured=[]
        if kind=='conversations':
            original=g.conversations._load
            def held(row):
                data=original(row)
                captured.append(data)
                entered.set()
                assert release.wait(2)
                return data
            monkeypatch.setattr(g.conversations,'_load',held)
            reading=asyncio.create_task(g.conversations.get_turns())
        else:
            import src.training.stt_correction_collector as module
            original=module.STTCorrection
            def held(*args,**kwargs):
                view=original(*args,**kwargs)
                captured.append(view)
                entered.set()
                assert release.wait(2)
                return view
            monkeypatch.setattr(module,'STTCorrection',held)
            reading=asyncio.create_task(g.stt.get_training_pairs())
        cleaning=None
        try:
            await until(entered.is_set)
            assert g.store.resource_status()['native_readers']==1
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
            step=await claim(g,source,g.targets[kind])
            cleaning=asyncio.create_task(g.driver.clean(step))
            await asyncio.sleep(.02)
            assert not cleaning.done() and g.store.conversation_busy(step.target_id,source.conversation_id)
            # A blocked raw reader cannot hold the sole authority transaction.
            async with asyncio.timeout(.2):
                fresh=await g.privacy.acall(g.privacy.source_for_capture,str(uuid4()),g.clock.now,'another-room')
            assert fresh.conversation_id!=source.conversation_id
            release.set()
            assert (await cleaning).done
            await until(lambda:g.store.resource_status()['native_readers']==0)
            if kind=='conversations':
                assert captured==[{}]
            else:
                assert captured[0].__dict__['original_text']==captured[0].__dict__['corrected_text']==''
        finally:
            release.set()
            if cleaning is not None:
                await cleaning


async def test_duplicate_claim_observers_share_actual_unlink_and_cancel_does_not_release_lease(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source=g.source()
        await g.add('conversations',source)
        step=await claim(g,source,g.targets['conversations'])
        entered,release=threading.Event(),threading.Event()
        original=g.store._unlink
        calls=0
        def held(*args):
            nonlocal calls
            calls+=1
            entered.set()
            assert release.wait(2)
            return original(*args)
        monkeypatch.setattr(g.store,'_unlink',held)
        first=asyncio.create_task(g.driver.clean(step))
        second=None
        try:
            await until(entered.is_set)
            second=asyncio.create_task(g.driver.clean(step))
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert g.driver.resource_status()['callbacks']==1
            assert g.privacy.resource_status()['writer_lease']
            release.set()
            assert (await second).done and calls==1
        finally:
            release.set()
            if second is not None:
                await second


async def test_missing_unlink_keeps_locator_and_retry_removes_file_before_ack(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source=g.source()
        identifier=await g.add('conversations',source)
        step=await claim(g,source,g.targets['conversations'])
        with monkeypatch.context() as patch:
            patch.setattr(g.store,'_unlink',lambda *_: (_ for _ in ()).throw(OSError('synthetic unlink failure')))
            with pytest.raises(OSError):
                await g.driver.clean(step)
        assert any(row['id']==identifier for row in await g.records())
        assert (Path(g.targets['conversations'].root)/'policy_v1'/(identifier+'.json')).exists()
        assert (await g.driver.clean(step)).done
        assert not await g.records()


@pytest.mark.parametrize('damage',['claim','descriptor','version','cursor'])
async def test_unowned_claim_or_unknown_protocol_cannot_delete_known_files(tmp_path,damage):
    async with graph(tmp_path) as g:
        source=g.source()
        identifier=await g.add('conversations',source)
        step=await claim(g,source,g.targets['conversations'])
        changed={'claim':{'claim_token':str(uuid4())},'descriptor':{'descriptor':{'kind':'conversations','root':str(tmp_path/'elsewhere')}},
            'version':{'version':9},'cursor':{'cursor':'s:-1:2'}}[damage]
        with pytest.raises(ValueError):
            await g.driver.clean(replace(step,**changed))
        assert any(row['id']==identifier for row in await g.records())


async def test_closing_observation_keeps_native_cleanup_and_shared_authority_lease(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source=g.source()
        await g.add('conversations',source)
        step=await claim(g,source,g.targets['conversations'])
        entered,release=threading.Event(),threading.Event()
        original=g.store._unlink
        def held(*args):
            entered.set()
            assert release.wait(2)
            return original(*args)
        monkeypatch.setattr(g.store,'_unlink',held)
        cleaning=asyncio.create_task(g.driver.clean(step))
        try:
            await until(entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
            release.set()
            await cleaning
            await g.privacy.aclose()
            assert not g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            await cleaning


@pytest.mark.parametrize('operation',['reader','unreserved_writer'])
async def test_removed_fact_dependency_keeps_derived_native_scope_owned(tmp_path,monkeypatch,operation):
    async with graph(tmp_path) as g:
        original_source=g.source()
        fact=await g.privacy.acall(g.privacy.ledger.store,'synthetic private fact','fact',.9,sources=(original_source,))
        derived=replace(g.source(),fact_dependencies=(FactDependency(fact.id,fact.revision,fact.expires_at),))
        entered,release=threading.Event(),threading.Event()
        if operation=='reader':
            await g.add('conversations',derived)
            load=g.conversations._load
            def held(row):
                data=load(row)
                entered.set()
                assert release.wait(2)
                return data
            monkeypatch.setattr(g.conversations,'_load',held)
            active=asyncio.create_task(g.conversations.get_turns())
        else:
            reserve=g.store.reserve
            def held(*args):
                entered.set()
                assert release.wait(2)
                return reserve(*args)
            monkeypatch.setattr(g.store,'reserve',held)
            active=asyncio.create_task(g.add('conversations',derived))
        cleaning=None
        try:
            await until(entered.is_set)
            step=await claim(g,original_source,g.targets['conversations'])
            cleaning=asyncio.create_task(g.driver.clean(step))
            await asyncio.sleep(.02)
            assert not cleaning.done(), 'derived raw data is still retained by native work'
            release.set()
            assert (await cleaning).done
            await active
            assert not await g.records()
        finally:
            release.set()
            await active
            if cleaning is not None:
                await cleaning


async def test_feedback_retains_original_conversation_before_unreserved_raw_read(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source,interaction=g.source(),g.source()
        await g.add('conversations',source)
        view,=await g.conversations.get_turns()
        entered,release=threading.Event(),threading.Event()
        original=g.conversations._load
        def held(row):
            data=original(row)
            entered.set()
            assert release.wait(2)
            return data
        monkeypatch.setattr(g.conversations,'_load',held)
        marking=asyncio.create_task(g.conversations.mark_response(view,'corrected','synthetic correction',sources=(interaction,)))
        cleaning=None
        try:
            await until(entered.is_set)
            step=await claim(g,source,g.targets['conversations'])
            cleaning=asyncio.create_task(g.driver.clean(step))
            await asyncio.sleep(.02)
            assert not cleaning.done()
            release.set()
            assert (await cleaning).done and not await marking
        finally:
            release.set()
            await marking
            if cleaning is not None:
                await cleaning


async def test_metadata_and_directory_pages_cross_128_without_losing_live_rows(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source,other=g.source(),g.source()
        for _ in range(257):
            await g.add('conversations',source)
        for _ in range(129):
            await g.add('conversations',other)
        step=await claim(g,source,g.targets['conversations'])
        cursors=[]
        page=g.driver._page
        def observed(current):
            cursors.append(current.cursor)
            return page(current)
        monkeypatch.setattr(g.driver,'_page',observed)
        assert (await g.driver.clean(step)).done
        records=await g.records()
        assert len(records)==129 and all(other.conversation_id in row['sources_json'] for row in records)
        assert any(cursor and cursor.startswith('s:') for cursor in cursors)
        assert 'a:scan' in cursors


async def test_replaced_policy_directory_cannot_complete_an_existing_audit(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source,other=g.source(),g.source()
        for _ in range(130):
            await g.add('conversations',other)
        step=await claim(g,source,g.targets['conversations'])
        page=g.driver._page
        replaced=False
        def replacing(current):
            nonlocal replaced
            result=page(current)
            if result.more and result.result.cursor=='a:scan' and not replaced:
                directory=Path(g.targets['conversations'].root)/'policy_v1'
                directory.rename(directory.with_name('retained-old-policy'))
                directory.mkdir()
                replaced=True
            return result
        monkeypatch.setattr(g.driver,'_page',replacing)
        with pytest.raises(ValueError,match='collector_policy_directory_replaced'):
            await g.driver.clean(step)
        assert replaced and not g.driver.resource_status()['audits']


async def test_runtime_factory_registers_collectors_when_all_learning_flags_are_off(tmp_path):
    config=configuration(tmp_path)
    async with AsyncExitStack() as stack:
        privacy=PrivacyServices(str(tmp_path/'privacy.db'))
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        owner=ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        await build_wake_clip_store(config,privacy=privacy,shutdown=stack,owner=owner)
        assert ErasureParticipant.COLLECTORS in owner._drivers
        targets=await privacy.acall(privacy.ledger.erasures.targets,ErasureParticipant.COLLECTORS)
        kinds={row['descriptor']['kind'] for row in targets}
        assert {'wake_clips','stt_corrections','conversations','conversation_exports','legacy_training'}<=kinds
        assert not (tmp_path/'conversations').exists() and not (tmp_path/'models').exists()


async def test_restored_files_behind_complete_receipt_are_rechecked_on_leased_restart(tmp_path):
    async with graph(tmp_path) as g:
        source=g.source()
        identifier=await g.add('conversations',source)
        record,=await g.records()
        file=Path(g.targets['conversations'].root)/'policy_v1'/(identifier+'.json')
        saved=file.read_bytes()
        await g.receipt(source)
        await g.settle(source)
        config=g.config
        clock=g.clock
        assert not file.exists()
    file.write_bytes(saved)
    import sqlite3
    with sqlite3.connect(tmp_path/'privacy.db') as db:
        db.execute('INSERT INTO wake_clips_v1 VALUES (?,?,?,?,?,?,?,?,?,?)',tuple(record.values()))
    config['training']['conversations_dir']=str(tmp_path/'next-conversations')
    async with AsyncExitStack() as stack:
        privacy=PrivacyServices(str(tmp_path/'privacy.db'),clock=clock)
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        owner=ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        store=await build_wake_clip_store(config,privacy=privacy,shutdown=stack,owner=owner)
        # Local retention cannot provide the journal's proof in this test.
        store._maintenance.cancel()
        await asyncio.gather(store._maintenance,return_exceptions=True)
        receipt=await owner.request(source.conversation_id,ACTOR)
        assert any(row['state']!='complete' for row in receipt['steps'] if row['participant']=='collectors')
        async with asyncio.timeout(3):
            while True:
                await owner.reconcile()
                receipt=await owner.request(source.conversation_id,ACTOR)
                if all(row['state']=='complete' for row in receipt['steps'] if row['participant']=='collectors'):
                    break
                clock.now+=.02
                await asyncio.sleep(.001)
        assert not file.exists()


async def test_legacy_quarantine_rows_and_snapshot_never_get_a_false_collector_receipt(tmp_path):
    fixture=legacy_memory(tmp_path)
    await stage_legacy_quarantine(str(fixture.preview),str(fixture.output))
    config=configuration(fixture.output)
    async with AsyncExitStack() as stack:
        privacy=PrivacyServices(str(fixture.output/'candidate.sqlite3'))
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        owner=ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        await build_wake_clip_store(config,privacy=privacy,shutdown=stack,owner=owner)
        targets=await privacy.acall(privacy.ledger.erasures.targets,ErasureParticipant.COLLECTORS)
        root=next(row for row in targets if row['descriptor']['kind']=='legacy_quarantine')
        coverage=next(row for row in targets if row['descriptor']['kind']=='quarantine_locator_coverage')
        source_id=str(uuid4())
        await owner.request(source_id,ACTOR)
        driver=owner._drivers[ErasureParticipant.COLLECTORS].callback.__self__
        async def clean(target):
            due=await privacy.acall(privacy.ledger.erasures.due,128)
            step=await privacy.acall(privacy.ledger.erasures.claim,next(row for row in due if row['target_id']==target['id']))
            return await driver.clean(step)
        assert (await clean(coverage)).done
        result=await clean(root)
        assert not result.done and result.pending_count==2
        assert (fixture.output/'legacy.sqlite3').is_file()
        def retire_rows():
            with privacy.ledger._transaction() as db:
                db.execute('DELETE FROM legacy_quarantine_records_v1')
        await privacy.acall(retire_rows)
        # Explicit retirement of rows is not proof that the raw backup retired.
        # The same retained claim remains valid until its receipt is finished.
        def root_claim():
            with privacy.ledger._transaction() as db:
                return dict(db.execute('SELECT * FROM erasure_steps WHERE target_id=?',(root['id'],)).fetchone())
        state=await privacy.acall(root_claim)
        from src.memory.erasure_journal import ErasureStep
        current=ErasureStep(state['operation_id'],source_id,ErasureParticipant.COLLECTORS,root['id'],1,
            root['descriptor'],state['attempts'],state['claim_token'],state['cursor'])
        assert not (await driver.clean(current)).done
        (fixture.output/'legacy.sqlite3').unlink()
        assert (await driver.clean(current)).done


async def test_missing_quarantine_location_remains_pending_until_explicit_copy_registration(tmp_path):
    fixture=legacy_memory(tmp_path)
    await stage_legacy_quarantine(str(fixture.preview),str(fixture.output))
    import sqlite3
    with sqlite3.connect(fixture.output/'candidate.sqlite3') as db:
        db.execute("DELETE FROM erasure_targets WHERE descriptor LIKE '%legacy_quarantine%'")
    config=configuration(fixture.output)
    async with AsyncExitStack() as stack:
        privacy=PrivacyServices(str(fixture.output/'candidate.sqlite3'))
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        owner=ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        await build_wake_clip_store(config,privacy=privacy,shutdown=stack,owner=owner)
        await owner.request(str(uuid4()),ACTOR)
        due=await privacy.acall(privacy.ledger.erasures.due,128)
        targets=await privacy.acall(privacy.ledger.erasures.targets,ErasureParticipant.COLLECTORS)
        coverage=next(row for row in targets if row['descriptor']['kind']=='quarantine_locator_coverage')
        step=await privacy.acall(privacy.ledger.erasures.claim,next(row for row in due if row['target_id']==coverage['id']))
        result=await owner._drivers[ErasureParticipant.COLLECTORS].callback(step)
        assert not result.done
        manifest=json.loads((fixture.output/'manifest.json').read_text())
        await privacy.acall(privacy.ledger.erasures.register_target,ErasureParticipant.COLLECTORS,1,
            {'kind':'legacy_quarantine','root':str(fixture.output.resolve()),'batch_id':manifest['batch_id']})
        assert (await owner._drivers[ErasureParticipant.COLLECTORS].callback(step)).done
        receipt=await owner.request(step.conversation_id,ACTOR)
        assert any(row['state']!='complete' for row in receipt['steps'] if row['participant']=='collectors')


async def test_blocked_physical_unlink_does_not_hold_the_authority_writer(tmp_path,monkeypatch):
    async with graph(tmp_path) as g:
        source,other=g.source(),g.source()
        await g.add('conversations',source)
        step=await claim(g,source,g.targets['conversations'])
        entered,release=threading.Event(),threading.Event()
        original=g.store._unlink
        def held(*args):
            entered.set()
            assert release.wait(2)
            return original(*args)
        monkeypatch.setattr(g.store,'_unlink',held)
        cleaning=asyncio.create_task(g.driver.clean(step))
        try:
            await until(entered.is_set)
            async with asyncio.timeout(.2):
                receipt=await g.owner.request(other.conversation_id,ACTOR)
            assert receipt['revoked'] and g.privacy.ledger.conversation_committed_revoked(other.conversation_id)
            assert not cleaning.done()
        finally:
            release.set()
            await cleaning


async def test_unrecognized_source_metadata_keeps_file_and_only_fixed_receipt_errors(tmp_path):
    async with graph(tmp_path) as g:
        source=g.source()
        identifier=await g.add('conversations',source)
        def corrupt():
            with g.privacy.ledger._transaction() as db:
                db.execute('UPDATE wake_clips_v1 SET sources_json=? WHERE id=?',('private malformed marker',identifier))
        await g.privacy.acall(corrupt)
        await g.receipt(source)
        async with asyncio.timeout(2):
            while True:
                await g.owner.reconcile()
                receipt=await g.receipt(source)
                if any(row['last_error']=='cleanup_failed' for row in receipt['steps'] if row['participant']=='collectors'):
                    break
                await asyncio.sleep(.001)
        assert 'private malformed marker' not in json.dumps(receipt)
        assert any(row['id']==identifier for row in await g.records())
        assert (Path(g.targets['conversations'].root)/'policy_v1'/(identifier+'.json')).is_file()


async def test_default_cli_export_uses_configured_dedicated_root(tmp_path):
    import os
    import sys
    import time
    import yaml
    async with graph(tmp_path) as g:
        g.clock.now=time.time()
        source=g.source()
        await g.add('conversations',source)
        view,=await g.conversations.get_turns()
        assert await g.conversations.mark_response(view,'good',sources=(source,))
        config=g.config
        config['memory']={'privacy_db_path':str(tmp_path/'privacy.db')}
    config_path=tmp_path/'settings.yaml'
    config_path.write_text(yaml.safe_dump(config))
    (tmp_path/'data').mkdir()
    (tmp_path/'data/unrelated.marker').write_text('preserve')
    script=Path(__file__).resolve().parents[2]/'scripts/train_assistant.py'
    process=await asyncio.create_subprocess_exec(sys.executable,str(script),'training','--action','export',
        '--config',str(config_path),cwd=tmp_path,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    stdout,stderr=await asyncio.wait_for(process.communicate(),10)
    assert process.returncode==0,stderr.decode()
    assert str(tmp_path/'exports/policy_v1').encode() in stdout
    assert not (tmp_path/'data/policy_v1').exists()
    assert (tmp_path/'data/unrelated.marker').read_text()=='preserve'


async def test_distinct_registered_collectors_can_share_a_policy_directory(tmp_path):
    from src.training.stt_correction_collector import STTCorrectionCollector
    from src.wakeword.clip_store import CollectorKind
    async with graph(tmp_path) as g:
        source,other=g.source(),g.source()
        shared=await g.store._work.run(g.store.register,g.targets['conversations'].root,CollectorKind.STT_CORRECTIONS)
        extra=STTCorrectionCollector(shared.root,store=g.store,target=shared)
        assert await extra.add_correction(audio=np.ones(1600,dtype=np.float32),original_text='unrelated raw',
            corrected_text='unrelated corrected',sources=(other,),correction_sources=(other,))
        await g.add('conversations',source)
        await g.receipt(source)
        await g.settle(source)
        assert len(await extra.get_training_pairs())==1


@pytest.mark.parametrize('failure_at',[1,2],ids=['root','policy'])
async def test_directory_iterator_failure_releases_descriptors_with_exception_retained(tmp_path,monkeypatch,failure_at):
    import os
    async with graph(tmp_path) as g:
        source,other=g.source(),g.source()
        await g.add('conversations',other)
        step=await claim(g,source,g.targets['conversations'])
        original=os.scandir
        calls=0
        def failing(path):
            nonlocal calls
            if type(path) is int:
                calls+=1
                if calls==failure_at:
                    raise OSError('synthetic iterator allocation failure')
            return original(path)
        monkeypatch.setattr(os,'scandir',failing)
        baseline=len(os.listdir('/proc/self/fd'))
        with pytest.raises(OSError) as retained:
            await g.driver.clean(step)
        assert retained.value is not None and calls==failure_at
        assert len(os.listdir('/proc/self/fd'))==baseline
        assert not g.driver._audits
        assert len(await g.records())==1
