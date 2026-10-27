"""Listening samples retain real capture provenance and cannot become permanent."""

import asyncio
import base64
from contextlib import AsyncExitStack
from io import BytesIO
import sqlite3
import threading
from uuid import uuid4
import wave

import httpx
import numpy as np
import pytest

from src.ambient.speaker_tagger import SpeakerTagger
from src.dashboard.api import DashboardAPI
from src.memory.fact_ledger import MemoryReviewer
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.producer_context import producer_context
from src.pipeline.command_processor import CommandProcessor
from src.stt.whisper_fast import STTResult
from src.users.voice_capture import build_review_capture
from src.users.voice_registry import build_voice_registry
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_live_stream_receipts import flush
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import ACTOR,AUDIO,Encoder,VECTOR,graph,idle
from tests.mocks.ambient_graph import ambient_graph

from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import web_graph

pytestmark = pytest.mark.asyncio


async def observe(voices,clock,*,audio=AUDIO,ttl=60,text='Synthetic source transcript',room='synthetic-room'):
    source = memory_source(now=clock.now,ttl=ttl)
    encoder = Encoder()
    voices.register_encoder(encoder)
    assert voices.observe(encoder,audio,(source,),transcript=text,room_id=room)
    await idle(voices)
    rows = (await voices.acall(voices.ledger.snapshot))['fingerprints']
    row = next(row for row in rows if source.capture_id in row['capture_ids'])
    return source,row['id']


def counts(voices):
    with voices.privacy.ledger._transaction() as db:
        return tuple(db.execute('SELECT COUNT(*),COALESCE(SUM(length(audio_wav)),0),COALESCE(SUM(length(transcript)),0) FROM voice_capture_reviews').fetchone())


async def test_real_background_review_preserves_source_text_room_and_bounded_listening_wav(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        audio = np.full(16000*8,.25,dtype=np.float32)
        source,identity = await observe(voices,clock,audio=audio)
        data = await voices.acall(voices.ledger.review_capture,identity,ACTOR)
        assert data['expires_at']==source.expires_at and data['observed_at']==clock.now
        assert data['transcript']=='Synthetic source transcript' and data['room_id']=='synthetic-room'
        assert data['duration_s']==8 and data['clip_duration_s']==6 and data['clip_truncated']
        assert data['transcript_scope']=='full_segment' and data['audio_mime']=='audio/wav'
        raw = base64.b64decode(data['audio_base64'],validate=True)
        with wave.open(BytesIO(raw),'rb') as reader:
            assert (reader.getnchannels(),reader.getsampwidth(),reader.getframerate(),reader.getnframes())==(1,2,16000,96000)
            assert np.all(np.frombuffer(reader.readframes(96000),dtype='<i2')==8192)
        assert len(raw)<=192044
        snapshot = await voices.acall(voices.ledger.snapshot)
        assert snapshot['fingerprints'][0]['review_available']
        assert 'audio_base64' not in str(snapshot) and 'Synthetic source transcript' not in str(snapshot)


async def test_preview_uses_the_owned_audio_copy_even_when_original_is_reused(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        entered,release = threading.Event(),threading.Event()
        class Held(Encoder):
            def get_embedding(self,audio):
                entered.set()
                assert release.wait(3)
                return super().get_embedding(audio)
        encoder = Held()
        voices.register_encoder(encoder)
        audio = np.full(16000,.5,dtype=np.float32)
        try:
            assert voices.observe(encoder,audio,(memory_source(now=clock.now),))
            await until(entered.is_set)
            audio.fill(0)
        finally:
            release.set()
            await idle(voices)
        [row] = (await voices.acall(voices.ledger.snapshot))['fingerprints']
        data = await voices.acall(voices.ledger.review_capture,row['id'],ACTOR)
        with wave.open(BytesIO(base64.b64decode(data['audio_base64'])),'rb') as reader:
            assert np.all(np.frombuffer(reader.readframes(16000),dtype='<i2')==16384)


async def test_retries_cannot_replace_preview_text_or_renew_original_deadline(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        source = memory_source(now=clock.now,ttl=10)
        first = build_review_capture(AUDIO,'Original synthetic transcript','original-room')
        identity = await voices.acall(voices.ledger.put,'synthetic-model',VECTOR,1.,(source,),first)
        clock.now+=9
        replacement = build_review_capture(AUDIO,'Replacement synthetic transcript','replacement-room')
        assert await voices.acall(voices.ledger.put,'synthetic-model',VECTOR,1.,(source,),replacement)==identity
        data = await voices.acall(voices.ledger.review_capture,identity,ACTOR)
        assert data['transcript']==first.transcript and data['room_id']==first.room_id and data['expires_at']==source.expires_at
        clock.now+=1
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.review_capture,identity,ACTOR)
        await voices.acall(voices.ledger.prune)
        assert await voices.acall(counts,voices)==(0,0,0)


async def test_confirming_a_profile_does_not_permanently_retain_its_listening_samples(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        samples = [await observe(voices,clock,ttl=10) for _ in range(3)]
        identities = tuple(identity for _,identity in samples)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic voice',str(uuid4()),ACTOR)
        assert (await voices.acall(counts,voices))[0]==3
        clock.now+=10
        await voices.acall(voices.ledger.prune)
        assert await voices.acall(counts,voices)==(0,0,0)
        assert (await voices.acall(voices.ledger.snapshot))['profiles']==[profile]
    async with graph(tmp_path) as (_,_,voices,clock):
        clock.now+=48*3600
        assert (await voices.acall(voices.ledger.snapshot))['profiles']==[profile]
        assert await voices.acall(counts,voices)==(0,0,0)
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.review_capture,identities[0],ACTOR)


async def test_source_conversation_erasure_cascades_wave_and_transcript_in_same_transaction(tmp_path):
    async with graph(tmp_path) as (privacy,owner,voices,clock):
        source,identity = await observe(voices,clock)
        operation = await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
        await flush(owner)
        assert await voices.acall(counts,voices)==(0,0,0)
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.review_capture,identity,ACTOR)
        receipt = await privacy.acall(privacy.ledger.erasures.snapshot,operation['id'])
        assert receipt['state']!='complete'


async def test_pending_only_revocation_hides_preview_without_destroying_it_on_rollback(tmp_path):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        source,identity = await observe(voices,clock)
        privacy.ledger._pending_conversations = frozenset({source.conversation_id})
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.review_capture,identity,ACTOR)
        await voices.acall(voices.ledger.prune)
        assert (await voices.acall(counts,voices))[0]==1
        privacy.ledger._pending_conversations = frozenset()
        assert (await voices.acall(voices.ledger.review_capture,identity,ACTOR))['id']==identity


async def test_expiry_during_wave_encoding_cannot_commit_raw_audio_or_text(tmp_path,monkeypatch):
    import src.users.voice_registry as module
    async with graph(tmp_path) as (_,_,voices,clock):
        entered,release = threading.Event(),threading.Event()
        build = module.build_review_capture
        def held(*args):
            entered.set()
            assert release.wait(3)
            return build(*args)
        monkeypatch.setattr(module,'build_review_capture',held)
        encoder = Encoder()
        voices.register_encoder(encoder)
        try:
            assert voices.observe(encoder,AUDIO,(memory_source(now=clock.now,ttl=10),),transcript='Synthetic late text')
            await until(entered.is_set)
            clock.now+=10
        finally:
            release.set()
            await idle(voices)
        assert await voices.acall(counts,voices)==(0,0,0)
        assert not (await voices.acall(voices.ledger.snapshot))['fingerprints']


@pytest.mark.parametrize('parallel',[False,True])
async def test_command_processor_pairs_preview_with_final_transcript_and_actual_producer_room(tmp_path,parallel):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        class STT:
            def transcribe_with_confidence(self,audio,sample_rate):
                return STTResult('Synthetic final command',1.,no_speech_prob=.01,avg_logprob=-.1)
        users = await voices.load_accounts(str(tmp_path/'users.json'))
        processor = CommandProcessor(STT(),speaker_identifier=Encoder(),user_manager=users,privacy=privacy)
        try:
            source = memory_source(now=clock.now)
            with producer_context('actual-synthetic-room',(source,)):
                result = await processor.process_command(AUDIO,use_parallel=parallel)
            await idle(voices)
            [row] = (await voices.acall(voices.ledger.snapshot))['fingerprints']
            data = await voices.acall(voices.ledger.review_capture,row['id'],ACTOR)
            assert data['transcript']==result.text=='Synthetic final command'
            assert data['room_id']=='actual-synthetic-room' and row['capture_ids']==[source.capture_id]
        finally:
            await processor.begin_close()


async def test_ambient_tagger_preserves_review_metadata_without_creating_an_identity(tmp_path):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        tagger = SpeakerTagger(Encoder(),lambda:{},source_policy=privacy.ledger,voice_registry=voices)
        try:
            with producer_context('synthetic-room',(memory_source(now=clock.now),)):
                assert await tagger.tag(AUDIO,transcript='Synthetic ambient phrase',room_id='synthetic-room')==('unknown',0.)
            await idle(voices)
            [row] = (await voices.acall(voices.ledger.snapshot))['fingerprints']
            data = await voices.acall(voices.ledger.review_capture,row['id'],ACTOR)
            assert data['transcript']=='Synthetic ambient phrase' and data['room_id']=='synthetic-room'
        finally:
            await tagger.begin_close()


async def test_real_ambient_transcriber_pairs_stored_capture_id_text_and_room_with_speaker_channel(tmp_path):
    async with ambient_graph(tmp_path) as ambient:
        async with AsyncExitStack() as stack:
            owner = ErasureCoordinator(ambient.privacy)
            stack.push_async_callback(owner.aclose)
            voices = await build_voice_registry(privacy=ambient.privacy,owner=owner,shutdown=stack)
            await ambient.transcriber._tagger.begin_close()
            ambient.transcriber._tagger = SpeakerTagger(Encoder(),lambda:{},
                source_policy=ambient.privacy.ledger,voice_registry=voices)
            segment = ambient.segment()
            segment.audio[:,0] = .25
            segment.audio[:,1] = .75
            await ambient.transcriber._handle_segment('synthetic-a',segment)
            await idle(voices)
            [utterance] = await ambient.store.utterances_between('synthetic-a',segment.t0,segment.t1)
            [row] = (await voices.acall(voices.ledger.snapshot))['fingerprints']
            assert row['capture_ids']==[utterance['capture_id']]
            data = await voices.acall(voices.ledger.review_capture,row['id'],ACTOR)
            assert data['transcript']==utterance['text']=='Synthetic private transcript'
            assert data['room_id']==utterance['room_id']=='synthetic-a'
            assert data['expires_at']==utterance['expires_at']
            with wave.open(BytesIO(base64.b64decode(data['audio_base64'])),'rb') as reader:
                assert np.all(np.frombuffer(reader.readframes(16000),dtype='<i2')==8192)


async def test_existing_fingerprints_do_not_gain_invented_review_material_after_schema_upgrade(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        source = memory_source(now=clock.now)
        identity = await voices.acall(voices.ledger.put,'synthetic-model',VECTOR,1.,(source,))
    with sqlite3.connect(tmp_path/'privacy.db') as db:
        db.execute('DROP TABLE voice_capture_reviews')
    async with graph(tmp_path) as (_,_,voices,_):
        [row] = (await voices.acall(voices.ledger.snapshot))['fingerprints']
        assert row['id']==identity and not row['review_available']
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.review_capture,identity,ACTOR)
        assert await voices.acall(counts,voices)==(0,0,0)


async def test_authenticated_capture_endpoint_applies_original_expiry_and_never_exposes_vectors(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        async with ready_browser(bus,version=3) as peer:
            source,identity = await observe(voices,clock,ttl=10)
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
            url = '/api/voices/captures/'+identity+'/review'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
                assert (await client.get(url)).status_code==401
                response = await client.get(url,headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection})
                assert response.status_code==200 and response.headers['cache-control']=='no-store'
                data = response.json()['snapshot']['rows'][0]
                assert data['expires_at']==source.expires_at and data['audio_base64']
                assert 'embedding' not in response.text
                clock.now+=10
                assert (await client.get(url,headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection})).status_code==404


@pytest.mark.parametrize('change',['expiry','credential','revocation'])
async def test_native_preview_encoding_rechecks_current_authority_without_holding_sqlite(tmp_path,dashboard_auth,monkeypatch,change):
    import src.users.voice_ledger as module
    async with web_graph(tmp_path) as (privacy,owner,voices,clock,bus):
        async with ready_browser(bus,version=3) as peer:
            source,identity = await observe(voices,clock,ttl=10)
            entered,release = threading.Event(),threading.Event()
            encode = module.base64.b64encode
            def held(raw):
                entered.set()
                assert release.wait(3)
                return encode(raw)
            monkeypatch.setattr(module.base64,'b64encode',held)
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                observing = asyncio.create_task(client.get('/api/voices/captures/'+identity+'/review'))
                try:
                    await until(entered.is_set)
                    if change=='expiry':
                        clock.now+=10
                    elif change=='credential':
                        monkeypatch.setenv('KZA_DASHBOARD_TOKEN','synthetic-rotated-credential')
                    else:
                        operation = await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
                        await flush(owner)
                        assert (await privacy.acall(privacy.ledger.erasures.snapshot,operation['id']))['state']!='complete'
                        assert await voices.acall(counts,voices)==(0,0,0)
                        def local_pending():
                            with privacy.ledger._transaction() as db:
                                return db.execute("SELECT pending_count FROM erasure_steps WHERE operation_id=? AND participant='voices' AND target_id IN (SELECT id FROM erasure_targets WHERE version=1)",(operation['id'],)).fetchone()[0]
                        assert await privacy.acall(local_pending)>=1
                    release.set()
                    response = await observing
                    assert response.status_code==(403 if change=='credential' else 404)
                    assert 'audio_base64' not in response.text and 'Synthetic source transcript' not in response.text
                finally:
                    release.set()
                    await asyncio.gather(observing,return_exceptions=True)


@pytest.mark.parametrize('damage',['audio','oversize_audio','text','room'])
async def test_restored_invalid_review_payloads_cannot_cross_the_api_boundary(tmp_path,damage):
    async with graph(tmp_path) as (_,_,voices,clock):
        _,identity = await observe(voices,clock)
        def corrupt():
            with voices.privacy.ledger._transaction() as db:
                statements = {
                    'audio':('UPDATE voice_capture_reviews SET audio_wav=? WHERE fingerprint_id=?',b'not a wave file'*10),
                    'oversize_audio':('UPDATE voice_capture_reviews SET audio_wav=? WHERE fingerprint_id=?',b'x'*192045),
                    'text':('UPDATE voice_capture_reviews SET transcript=? WHERE fingerprint_id=?','x'*8193),
                    'room':('UPDATE voice_capture_reviews SET room_id=? WHERE fingerprint_id=?','x'*129),
                }
                sql,value = statements[damage]
                db.execute(sql,(value,identity))
        await voices.acall(corrupt)
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.review_capture,identity,ACTOR)


async def test_missing_restored_source_index_cannot_hide_a_real_vector_from_pending_counts(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        source,_ = await observe(voices,clock)
        def corrupt():
            with voices.privacy.ledger._transaction() as db:
                db.execute('DELETE FROM voice_fingerprint_sources')
        await voices.acall(corrupt)
        assert await voices.acall(voices.ledger.pending,source.conversation_id)>=1


async def test_orphaned_restored_wave_blocks_every_local_voice_receipt(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        await observe(voices,clock)
    with sqlite3.connect(tmp_path/'privacy.db') as db:
        db.execute('DELETE FROM voice_fingerprints')
    async with graph(tmp_path) as (_,_,voices,_):
        assert (await voices.acall(counts,voices))[0]==1
        assert await voices.acall(voices.ledger.pending,str(uuid4()))>=1
