"""Voice TTL, manual retention and local erasure use the actual authority."""

import asyncio
import json
import threading
from types import SimpleNamespace
from uuid import uuid4
from weakref import ref
import sqlite3

import httpx
import numpy as np
import pytest

from src.core.logging import LogContext
from src.core.owned_work import WorkCloseTimeout
from src.dashboard.api import DashboardAPI
from src.memory.fact_ledger import MemoryReviewer
from src.users.user_manager import PermissionLevel
from src.users.voice_enrollment import VoiceEnrollment
from src.ambient.speaker_tagger import SpeakerTagger
from src.dashboard.auth import DashboardPrincipal,require_dashboard_session
from src.memory.producer_context import producer_context
from src.pipeline.command_processor import CommandProcessor
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.stt.whisper_fast import STTResult
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_live_stream_receipts import flush
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import ACTOR,AUDIO,Encoder,VECTOR,graph,idle,seed

from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import web_graph

pytestmark = pytest.mark.asyncio


async def test_fingerprint_read_retry_and_restart_keep_the_original_capture_deadline(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        source = memory_source(now=clock.now,ttl=10)
        identity = await voices.acall(voices.ledger.put,'synthetic',VECTOR,1.,(source,))
        clock.now+=9
        assert await voices.acall(voices.ledger.put,'synthetic',VECTOR,1.,(source,))==identity
        [row] = (await voices.acall(voices.ledger.snapshot))['fingerprints']
        assert row['expires_at']==source.expires_at
    async with graph(tmp_path) as (_,_,voices,clock):
        clock.now = source.expires_at
        assert (await voices.acall(voices.ledger.snapshot))['fingerprints']==[]
        await voices.acall(voices.ledger.prune)
        def count():
            with voices.privacy.ledger._transaction() as db:
                return db.execute('SELECT COUNT(*) FROM voice_fingerprints').fetchone()[0]
        assert await voices.acall(count)==0


async def test_manual_profile_persists_after_capture_expiry_and_restart_without_granting_permissions(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        _,identities = await seed(voices,clock)
        request = str(uuid4())
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic reviewed voice',request,ACTOR)
        clock.now+=48*3600
        await voices.acall(voices.ledger.prune)
        snapshot = await voices.acall(voices.ledger.snapshot)
        assert not snapshot['fingerprints'] and snapshot['profiles']==[profile]
        assert profile['expires_at'] is None and not snapshot['recognition_grants_permissions']
        assert await voices.acall(voices.ledger.confirm,identities,'Synthetic reviewed voice',request,ACTOR)==profile
    async with graph(tmp_path) as (_,_,voices,clock):
        clock.now+=3*86400
        assert (await voices.acall(voices.ledger.snapshot))['profiles']==[profile]


async def test_review_requires_current_distinct_captures_and_compatible_encoders(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        sources,identities = await seed(voices,clock)
        with pytest.raises(ValueError,match='distinct'):
            await voices.acall(voices.ledger.confirm,(identities[0],)*3,'Synthetic',str(uuid4()),ACTOR)
        other = await voices.acall(voices.ledger.put,'other-model',VECTOR,1.,(sources[0],))
        with pytest.raises(ValueError,match='incompatible'):
            await voices.acall(voices.ledger.confirm,(identities[0],identities[1],other),'Synthetic',str(uuid4()),ACTOR)
        clock.now+=60
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.confirm,identities,'Synthetic',str(uuid4()),ACTOR)


async def test_deleted_profile_cannot_be_recreated_by_a_retried_review_request(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        _,identities = await seed(voices,clock)
        request = str(uuid4())
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic name',request,ACTOR)
        with pytest.raises(ValueError,match='request_conflict'):
            await voices.acall(voices.ledger.confirm,identities,'Different name',request,ACTOR)
        with pytest.raises(ValueError,match='revision_conflict'):
            await voices.acall(voices.ledger.delete_profile,profile['id'],2,ACTOR)
        assert await voices.acall(voices.ledger.delete_profile,profile['id'],1,ACTOR)
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.confirm,identities,'Synthetic name',request,ACTOR)
        assert not (await voices.acall(voices.ledger.snapshot))['profiles']
        def tombstone():
            with voices.privacy.ledger._transaction() as db:
                row = db.execute('SELECT name,embedding FROM voice_profiles WHERE id=?',(profile['id'],)).fetchone()
                digest = db.execute('SELECT digest FROM voice_review_requests WHERE profile_id=?',(profile['id'],)).fetchone()[0]
                return tuple(row),digest
        assert await voices.acall(tombstone)==((None,None),'deleted')


async def test_conversation_erasure_removes_its_vectors_profile_and_independent_suggestion_link(tmp_path):
    async with graph(tmp_path) as (privacy,owner,voices,clock):
        sources,identities = await seed(voices,clock)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic name',str(uuid4()),ACTOR)
        independent = memory_source(now=clock.now)
        await voices.acall(voices.ledger.put,'synthetic-model',VECTOR,1.,(independent,))
        assert any(row['suggested_profile']==profile['id'] for row in (await voices.acall(voices.ledger.snapshot))['fingerprints'])
        operation = await owner.request(sources[0].conversation_id,MemoryReviewer('synthetic-admin'))
        await flush(owner)
        snapshot = await voices.acall(voices.ledger.snapshot)
        assert not snapshot['profiles']
        assert all(row['suggested_profile'] is None and row['similarity'] is None for row in snapshot['fingerprints'])
        assert await voices.acall(voices.ledger.pending,sources[0].conversation_id)==0
        receipt = await privacy.acall(privacy.ledger.erasures.snapshot,operation['id'])
        assert receipt['state']!='complete'


async def test_expired_or_revoked_output_of_a_running_model_cannot_commit(tmp_path):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        entered,release = threading.Event(),threading.Event()
        class Held(Encoder):
            def get_embedding(self,audio):
                entered.set()
                assert release.wait(3)
                return super().get_embedding(audio)
        encoder = Held()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now,ttl=10)
        try:
            with LogContext(private='Synthetic caller payload'):
                assert voices.observe(encoder,AUDIO,(source,))
            await until(entered.is_set)
            clock.now+=10
            await privacy.acall(privacy.ledger.delete_conversation,source.conversation_id,MemoryReviewer('synthetic-admin'))
        finally:
            release.set()
            await idle(voices)
        assert (await voices.acall(voices.ledger.snapshot))['fingerprints']==[]


async def test_queued_model_is_not_entered_after_its_source_expires(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        encoder = Encoder()
        lock = voices.register_encoder(encoder)
        source = memory_source(now=clock.now,ttl=10)
        lock.acquire()
        try:
            assert voices.observe(encoder,AUDIO,(source,))
            await until(lambda:voices.resource_status()['operations']>0)
            clock.now+=10
        finally:
            lock.release()
            await idle(voices)
        assert encoder.calls==0


async def test_idle_maintenance_physically_prunes_without_a_new_voice_query(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        source = memory_source(now=clock.now,ttl=1)
        await voices.acall(voices.ledger.put,'synthetic',VECTOR,1.,(source,))
        clock.now+=1
        def empty():
            with voices.privacy.ledger._transaction() as db:
                return db.execute('SELECT COUNT(*) FROM voice_fingerprints').fetchone()[0]==0
        async with asyncio.timeout(3):
            while not await voices.privacy.acall(empty):
                await asyncio.sleep(.02)


async def test_native_model_and_reserved_post_keep_local_receipt_pending_until_actual_exit(tmp_path):
    async with graph(tmp_path) as (privacy,owner,voices,clock):
        entered,release = threading.Event(),threading.Event()
        class Held(Encoder):
            def get_embedding(self,audio):
                entered.set()
                assert release.wait(3)
                return super().get_embedding(audio)
        encoder = Held()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now)
        try:
            assert voices.observe(encoder,AUDIO,(source,))
            await until(entered.is_set)
            operation = await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
            await flush(owner)
            def local():
                with privacy.ledger._transaction() as db:
                    return dict(db.execute("""SELECT s.* FROM erasure_steps s JOIN erasure_targets t ON t.id=s.target_id
                        WHERE s.operation_id=? AND t.participant='voices' AND t.version=1""",(operation['id'],)).fetchone())
            assert (await privacy.acall(local))['state']=='pending'
            with pytest.raises(WorkCloseTimeout):
                await voices.aclose()
            assert voices.resource_status()['pending']==1 and not voices.begin_close().done()
        finally:
            release.set()
            await voices.begin_close()
        assert voices.resource_status()['pending']==0


async def test_legacy_profiles_are_excluded_and_account_writes_preserve_originals(tmp_path):
    async with graph(tmp_path) as (_,_,voices,_):
        original = tmp_path/'users.json'
        payload = {'users':[{'user_id':'legacy-admin','name':'Synthetic account','permission_level':4,
            'voice_embedding':[.1]*192}]}
        encoded = json.dumps(payload).encode()
        original.write_bytes(encoded)
        users = await voices.load_accounts(str(original))
        assert users.get_all_embeddings()=={} and users.get_user('legacy-admin').voice_embedding is None
        assert not users.set_voice_embedding('legacy-admin',VECTOR)
        await voices.acall(users.update_user,'legacy-admin')
        assert original.read_bytes()==encoded
        assert users.data_path!=original and users.data_path.stat().st_mode&0o777==0o600
        assert 'voice_embedding' in users.data_path.read_text() and json.loads(users.data_path.read_text())['users'][0]['voice_embedding'] is None
        reloaded = await voices.load_accounts(str(original))
        assert reloaded.get_user('legacy-admin').permission_level is PermissionLevel.ADMIN


async def test_voice_enrollment_cannot_create_accounts_or_permissions_in_managed_mode(tmp_path):
    async with graph(tmp_path) as (_,_,voices,_):
        users = await voices.load_accounts(str(tmp_path/'users.json'))
        workflow = VoiceEnrollment(users,Encoder(),review_required=True)
        result = workflow.handle('agregar persona',AUDIO)
        assert result['handled'] and not workflow.is_active and users.get_all_users()==[]
        created,_ = users.add_user('Synthetic',PermissionLevel.ADMIN,voice_embedding=VECTOR)
        assert created is None and not users.data_path.exists()


@pytest.mark.parametrize('payload',[
    {'schema':'unrecognized','users':[]},
    {'schema':'kza-users-accounts-v1','users':[
        {'user_id':'partial-admin','name':'Synthetic','permission_level':4},
        {'user_id':'invalid','permission_level':4}]},
])
async def test_unverified_account_catalog_is_read_only_and_drops_partial_accounts(tmp_path,payload):
    async with graph(tmp_path) as (_,_,voices,_):
        catalog = tmp_path/'users_accounts_v1.json'
        original = json.dumps(payload).encode()
        catalog.write_bytes(original)
        users = await voices.load_accounts(str(tmp_path/'users.json'))
        assert users.get_all_users()==[]
        with pytest.raises(RuntimeError,match='unverified_account_catalog'):
            await voices.acall(users.add_user,'Synthetic',PermissionLevel.ADMIN)
        assert users.get_all_users()==[] and catalog.read_bytes()==original


async def test_non_16khz_audio_is_rejected_before_copy_or_model_admission(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        encoder = Encoder()
        voices.register_encoder(encoder)
        assert not voices.observe(encoder,AUDIO,(memory_source(now=clock.now),),sample_rate=48000)
        assert encoder.calls==0 and voices.resource_status()['pending']==0


async def test_authenticated_review_api_never_returns_vectors_or_accepts_account_permissions(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        async with ready_browser(bus,version=3) as peer:
            _,identities = await seed(voices,clock)
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
                assert (await client.get('/api/voices')).status_code==401
                body = {'fingerprint_ids':list(identities),'name':'Synthetic reviewed voice','request_id':str(uuid4())}
                assert (await client.post('/api/voices/profiles',headers=dashboard_auth,json={**body,'permission_level':'admin'})).status_code==422
                response = await client.post('/api/voices/profiles',headers=dashboard_auth,json=body)
                assert response.status_code==200
                data = (await client.get('/api/voices/profiles',headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection})).json()
                assert len(data['snapshot']['rows'])==1 and 'embedding' not in json.dumps(data)
                assert 'permission_level' not in response.text and 'name' not in response.json()
                assert (await client.post('/api/admin/users/unknown/enroll',headers=dashboard_auth,content=b'not a WAV')).status_code==409
                assert response.headers['cache-control']=='no-store'
                app.dependency_overrides[require_dashboard_session] = lambda:DashboardPrincipal('reader',frozenset({'home:read'}))
                assert (await client.get('/api/voices',headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection})).status_code==403


async def test_queued_review_rechecks_rotated_credentials_before_committing(tmp_path,dashboard_auth,monkeypatch):
    async with graph(tmp_path) as (_,_,voices,clock):
        _,identities = await seed(voices,clock)
        entered,release = threading.Event(),threading.Event()
        confirm = voices.ledger.confirm
        def held(*args):
            entered.set()
            assert release.wait(3)
            return confirm(*args)
        monkeypatch.setattr(voices.ledger,'confirm',held)
        app = DashboardAPI(voice_registry=voices).app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',headers=dashboard_auth) as client:
            observing = asyncio.create_task(client.post('/api/voices/profiles',json={
                'fingerprint_ids':list(identities),'name':'Synthetic','request_id':str(uuid4())}))
            try:
                await until(entered.is_set)
                monkeypatch.setenv('KZA_DASHBOARD_TOKEN','synthetic-rotated')
                release.set()
                assert (await observing).status_code==403
                assert (await voices.acall(voices.ledger.snapshot))['profiles']==[]
            finally:
                release.set()
                await observing


@pytest.mark.parametrize('parallel',[False,True])
async def test_real_command_processor_never_waits_for_background_voice_model_or_promotes_an_account(tmp_path,parallel):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        entered,release = threading.Event(),threading.Event()
        class Held(Encoder):
            def get_embedding(self,audio):
                entered.set()
                assert release.wait(3)
                return super().get_embedding(audio)
        class STT:
            def transcribe_with_confidence(self,audio,sample_rate):
                return STTResult('Synthetic local command',1.,no_speech_prob=.01,avg_logprob=-.1)
        encoder = Held()
        users = await voices.load_accounts(str(tmp_path/'users.json'))
        processor = CommandProcessor(STT(),speaker_identifier=encoder,user_manager=users,privacy=privacy)
        source = memory_source(now=clock.now)
        try:
            with producer_context('synthetic-room',(source,)):
                result = await processor.process_command(AUDIO,use_parallel=parallel)
            assert result.success and result.user is None and not release.is_set()
            await until(entered.is_set)
            assert voices.resource_status()['pending']==1
        finally:
            release.set()
            await idle(voices)
            await processor.begin_close()
        assert len((await voices.acall(voices.ledger.snapshot))['fingerprints'])==1


async def test_real_ambient_tagger_posts_a_capture_without_reading_legacy_profiles(tmp_path):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        encoder = Encoder()
        def forbidden():
            raise AssertionError('Legacy profile loader used')
        tagger = SpeakerTagger(encoder,forbidden,source_policy=privacy.ledger,voice_registry=voices)
        try:
            source = memory_source(now=clock.now)
            with producer_context('synthetic-room',(source,)):
                assert await tagger.tag(AUDIO)==('unknown',0.)
            await idle(voices)
            assert encoder.calls==1
            assert len((await voices.acall(voices.ledger.snapshot))['fingerprints'])==1
        finally:
            await tagger.begin_close()


async def test_background_queue_stays_bounded_and_owns_its_pcm_copy(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        entered,release = threading.Event(),threading.Event()
        observed = []
        class Held(Encoder):
            def get_embedding(self,audio):
                entered.set()
                assert release.wait(3)
                observed.append(float(audio[0]))
                return super().get_embedding(audio)
        encoder = Held()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now)
        audio = AUDIO.copy()
        capacity = voices.resource_status()['capacity']
        try:
            for _ in range(capacity):
                assert voices.observe(encoder,audio,(source,))
            assert not voices.observe(encoder,audio,(source,))
            assert voices.resource_status()['pending']==capacity
            audio[:]=.75
            await until(entered.is_set)
            assert (await voices.acall(voices.ledger.snapshot))['fingerprints']==[]
            assert not release.is_set()
        finally:
            release.set()
            await idle(voices)
        assert len(observed)==capacity and all(value==float(AUDIO[0]) for value in observed)
        assert voices.resource_status()['pending']==0


@pytest.mark.parametrize('bad',['nan','zero','object','dimensions'])
async def test_invalid_model_vectors_cannot_create_fingerprints(tmp_path,bad):
    async with graph(tmp_path) as (_,_,voices,clock):
        vector = {'nan':np.full(192,np.nan,dtype=np.float32),'zero':np.zeros(192,dtype=np.float32),
            'object':np.array(['private']*192,dtype=object),'dimensions':np.ones((2,192),dtype=np.float32)}[bad]
        with pytest.raises(ValueError,match='invalid_voice_embedding'):
            await voices.acall(voices.ledger.put,'synthetic',vector,1.,(memory_source(now=clock.now),))
        assert (await voices.acall(voices.ledger.snapshot))['fingerprints']==[]


async def test_pending_conversation_fence_does_not_physically_erase_a_review_on_rollback(tmp_path):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        sources,identities = await seed(voices,clock)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic',str(uuid4()),ACTOR)
        privacy.ledger._pending_conversations = frozenset({sources[0].conversation_id})
        assert not (await voices.acall(voices.ledger.snapshot))['profiles']
        await voices.acall(voices.ledger.prune)
        privacy.ledger._pending_conversations = frozenset()
        assert (await voices.acall(voices.ledger.snapshot))['profiles']==[profile]


async def test_orchestrator_audio_entry_only_posts_an_observation_and_never_matches_legacy_users(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        users = await voices.load_accounts(str(tmp_path/'users.json'))
        encoder = Encoder()
        voices.register_encoder(encoder)
        entry = SimpleNamespace(user_manager=users,speaker_id=encoder)
        source = memory_source(now=clock.now)
        assert await MultiUserOrchestrator._identify_speaker(entry,AUDIO,sources=(source,))==(None,None)
        await idle(voices)
        assert len((await voices.acall(voices.ledger.snapshot))['fingerprints'])==1


async def test_expiry_during_centroid_computation_prevents_permanent_confirmation(tmp_path,monkeypatch):
    import src.users.voice_ledger as module
    async with graph(tmp_path) as (_,_,voices,clock):
        _,identities = await seed(voices,clock)
        entered,release = threading.Event(),threading.Event()
        encode = module.vector_bytes
        def held(vector):
            entered.set()
            assert release.wait(3)
            return encode(vector)
        monkeypatch.setattr(module,'vector_bytes',held)
        observing = asyncio.create_task(voices.acall(voices.ledger.confirm,identities,'Synthetic',str(uuid4()),ACTOR))
        try:
            await until(entered.is_set)
            clock.now+=60
            release.set()
            with pytest.raises(LookupError):
                await observing
            assert (await voices.acall(voices.ledger.snapshot))['profiles']==[]
        finally:
            release.set()
            await asyncio.gather(observing,return_exceptions=True)


async def test_restored_voice_rows_reopen_receipts_and_cannot_override_current_tombstones(tmp_path):
    async with graph(tmp_path) as (privacy,owner,voices,clock):
        sources,identities = await seed(voices,clock)
        await voices.acall(voices.ledger.confirm,identities,'Synthetic',str(uuid4()),ACTOR)
        def archived():
            with privacy.ledger._transaction() as db:
                return {table:[tuple(row) for row in db.execute('SELECT * FROM '+table)] for table in (
                    'voice_fingerprints','voice_fingerprint_sources','voice_profiles','voice_profile_sources',
                    'voice_groups','voice_group_members','voice_group_requests',
                    'voice_group_lineage','voice_group_request_lineage')}
        backup = await privacy.acall(archived)
        operation = await owner.request(sources[0].conversation_id,MemoryReviewer('synthetic-admin'))
        await flush(owner)
    with sqlite3.connect(tmp_path/'privacy.db') as db:
        db.execute('PRAGMA foreign_keys=ON')
        for table,rows in backup.items():
            for row in rows:
                db.execute('INSERT OR REPLACE INTO '+table+' VALUES ('+','.join('?' for _ in row)+')',row)
    async with graph(tmp_path) as (privacy,owner,voices,_):
        def local():
            with privacy.ledger._transaction() as db:
                return db.execute("""SELECT s.state FROM erasure_steps s JOIN erasure_targets t ON t.id=s.target_id
                    WHERE s.operation_id=? AND t.participant='voices' AND t.version=1""",(operation['id'],)).fetchone()[0]
        assert await privacy.acall(local)=='pending'
        assert not (await voices.acall(voices.ledger.snapshot))['profiles']
        assert await voices.acall(voices.ledger.pending,sources[0].conversation_id)==0
        await flush(owner)
        assert await privacy.acall(local)=='complete'


async def test_background_similarity_is_only_a_review_suggestion_even_after_manual_confirmation(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        encoder = Encoder()
        voices.register_encoder(encoder)
        for _ in range(3):
            assert voices.observe(encoder,AUDIO,(memory_source(now=clock.now),))
        await idle(voices)
        identities = tuple(row['id'] for row in (await voices.acall(voices.ledger.snapshot))['fingerprints'])
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic name',str(uuid4()),ACTOR)
        source = memory_source(now=clock.now)
        assert voices.observe(encoder,AUDIO,(source,))
        await idle(voices)
        data = await voices.acall(voices.ledger.snapshot)
        row = next(row for row in data['fingerprints'] if row['capture_ids']==[source.capture_id])
        assert row['suggested_profile']==profile['id'] and row['similarity']==pytest.approx(1.)
        assert row['association']=='suggested' and not data['automatic_grouping_accepted']
        users = await voices.load_accounts(str(tmp_path/'users.json'))
        assert users.get_all_users()==[] and users.get_all_embeddings()=={}


async def test_cancelled_registry_observer_does_not_keep_an_eventual_private_result(tmp_path):
    async with graph(tmp_path) as (_,_,voices,_):
        entered,release = threading.Event(),threading.Event()
        references = []
        class Payload:
            pass
        def held():
            entered.set()
            assert release.wait(3)
            payload = Payload()
            references.append(ref(payload))
            return payload
        observing = asyncio.create_task(voices.acall(held))
        try:
            await until(entered.is_set)
            observing.cancel()
            with pytest.raises(asyncio.CancelledError) as retained_error:
                await observing
            release.set()
            await until(lambda:voices.resource_status()['operations']==0)
            assert retained_error.value and references and references[0]() is None
        finally:
            release.set()
