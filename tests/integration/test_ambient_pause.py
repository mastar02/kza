"""Real pause admission and native drain; historical memory stays readable."""

import asyncio
from contextlib import AsyncExitStack,asynccontextmanager
import threading
from uuid import uuid4

import httpx
import numpy as np
import pytest

from src.ambient.control import AmbientController,build_ambient_control
from src.dashboard.api import DashboardAPI
from src.dashboard.auth import DashboardPrincipal,require_dashboard_session
from src.memory.fact_ledger import MemoryOrigin
from src.users.voice_registry import VoiceRegistry
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.ambient_graph import ambient_graph
from tests.mocks.voice_registry import AUDIO,Encoder

pytestmark = pytest.mark.asyncio
ACTOR = AmbientController('synthetic-admin',lambda:True)


@asynccontextmanager
async def controlled(tmp_path,**kwargs):
    async with ambient_graph(tmp_path,**kwargs) as graph:
        async with AsyncExitStack() as stack:
            control = await build_ambient_control(privacy=graph.privacy,path=graph.path,shutdown=stack)
            graph.clock.now+=.2
            yield graph,control


async def phase(control,expected):
    async with asyncio.timeout(3):
        while control.snapshot()['phase']!=expected:
            await asyncio.sleep(.001)


async def test_pause_seals_only_ambient_origin_and_resume_uses_fresh_admission_without_revoking_history(tmp_path):
    async with controlled(tmp_path) as (g,control):
        room='synthetic-a'
        previous=g.privacy.source_for_capture(str(uuid4()),g.clock.now,room,origin=MemoryOrigin.AMBIENT)
        epoch=g.tap.input_epoch(room)
        control.request(True,0,ACTOR)
        await phase(control,'paused')
        assert epoch.retired and g.tap.buffered_count()==0
        assert g.privacy.sources_available((previous,))
        with pytest.raises(PermissionError):
            g.privacy.source_for_capture(str(uuid4()),g.clock.now,room,origin=MemoryOrigin.AMBIENT)
        command=g.privacy.source_for_capture(str(uuid4()),g.clock.now,room,origin=MemoryOrigin.VOICE)
        assert g.privacy.sources_available((command,))
        control.request(False,control.snapshot()['revision'],ACTOR)
        await phase(control,'running')
        assert g.tap.input_epoch(room) is not epoch
        g.tap.push(room,np.ones((1280,6),dtype=np.float32),ts=g.clock.now+.1,epoch=epoch)
        assert g.tap.buffered_count()==0
        g.clock.now+=.2
        fresh=g.privacy.source_for_capture(str(uuid4()),g.clock.now,room,origin=MemoryOrigin.AMBIENT)
        assert fresh.conversation_id!=previous.conversation_id
        assert g.privacy.sources_available((previous,))


@pytest.mark.parametrize('stage',['stt','profiles','speaker','quality','doa'])
async def test_pause_remains_pending_until_actual_native_stage_exits_and_late_text_cannot_persist(tmp_path,stage):
    async with controlled(tmp_path,stage=stage) as (g,control):
        processing=asyncio.create_task(g.transcriber._handle_segment('synthetic-a',g.segment()))
        try:
            await until(g.gates[stage].entered.is_set)
            control.request(True,0,ACTOR)
            await asyncio.sleep(.02)
            status=control.snapshot()
            assert status['admission_paused'] and status['phase']=='pausing' and status['pending_operations']>0
            with pytest.raises(ValueError,match='busy'):
                control.request(False,status['revision'],ACTOR)
        finally:
            g.gates[stage].release.set()
            await processing
        await phase(control,'paused')
        cursor=await g.store._db.execute('SELECT COUNT(*) FROM utterances')
        assert (await cursor.fetchone())[0]==0
        assert control.snapshot()['pending_operations']==0


async def test_paused_policy_survives_restart_and_has_no_dependency_on_voice_reactivation(tmp_path,dashboard_auth):
    async with controlled(tmp_path) as (_,control):
        control.request(True,0,ACTOR)
        await phase(control,'paused')
        revision=control.snapshot()['revision']
    async with controlled(tmp_path) as (g,control):
        assert control.snapshot()['phase']=='paused' and control.snapshot()['revision']==revision
        app=DashboardAPI(ambient_control=control).app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',headers=dashboard_auth) as client:
            response=await client.post('/api/ambient/control',json={'paused':False,'expected_revision':revision})
            assert response.status_code==202
        await phase(control,'running')
        assert g.gates['stt'].calls==0 and g.gates['vad'].calls==0


async def test_stale_or_unprivileged_controls_cannot_reactivate_or_change_policy(tmp_path,dashboard_auth):
    async with controlled(tmp_path) as (_,control):
        control.request(True,0,ACTOR)
        await phase(control,'paused')
        with pytest.raises(ValueError,match='revision'):
            control.request(False,0,ACTOR)
        with pytest.raises(PermissionError):
            control.request(False,1,AmbientController('synthetic-admin',lambda:False))
        app=DashboardAPI(ambient_control=control).app
        app.dependency_overrides[require_dashboard_session]=lambda:DashboardPrincipal('reader',frozenset({'home:read'}))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',headers=dashboard_auth) as client:
            assert (await client.post('/api/ambient/control',json={'paused':False,'expected_revision':1})).status_code==403
        assert control.snapshot()['phase']=='paused'


async def test_late_credential_revocation_cannot_commit_a_queued_reactivation(tmp_path,monkeypatch):
    async with controlled(tmp_path) as (_,control):
        control.request(True,0,ACTOR)
        await phase(control,'paused')
        entered,release=threading.Event(),threading.Event()
        current=[True]
        original=control._write
        def held(*args):
            entered.set()
            assert release.wait(3)
            return original(*args)
        monkeypatch.setattr(control,'_write',held)
        try:
            control.request(False,1,AmbientController('synthetic-admin',lambda:current[0]))
            await until(entered.is_set)
            current[0]=False
        finally:
            release.set()
        await phase(control,'failed')
        assert control.snapshot()['admission_paused'] and control.snapshot()['persisted_paused']


async def test_pause_includes_actual_background_ambient_voice_observation_but_keeps_voice_lane_open(tmp_path):
    async with ambient_graph(tmp_path) as g:
        async with AsyncExitStack() as stack:
            voices=VoiceRegistry(g.privacy)
            stack.push_async_callback(voices.aclose)
            control=await build_ambient_control(privacy=g.privacy,path=g.path,voices=voices,shutdown=stack)
            g.clock.now+=.2
            entered,release=threading.Event(),threading.Event()
            class Held(Encoder):
                def get_embedding(self,audio):
                    entered.set()
                    assert release.wait(3)
                    return super().get_embedding(audio)
            encoder=Held();voices.register_encoder(encoder)
            source=g.privacy.source_for_capture(str(uuid4()),g.clock.now,'synthetic-a',origin=MemoryOrigin.AMBIENT)
            assert voices.observe(encoder,AUDIO,(source,))
            try:
                await until(entered.is_set)
                control.request(True,0,ACTOR)
                await asyncio.sleep(.01)
                assert control.snapshot()['phase']=='pausing'
                assert not voices.observe(encoder,AUDIO,(source,))
                command=g.privacy.source_for_capture(str(uuid4()),g.clock.now,'synthetic-a',origin=MemoryOrigin.VOICE)
                assert voices.observe(encoder,AUDIO,(command,))
            finally:
                release.set()
            await phase(control,'paused')
            await until(lambda:voices.resource_status()['pending']==0)
            fingerprints=(await voices.acall(voices.ledger.snapshot))['fingerprints']
            assert len(fingerprints)==1 and fingerprints[0]['capture_ids']==[command.capture_id]


async def test_pause_keeps_existing_text_and_original_expiry_while_purge_continues(tmp_path):
    async with controlled(tmp_path) as (g,control):
        await g.transcriber._handle_segment('synthetic-a',g.segment())
        before=await g.store.undistilled_live(require_provenance=True)
        assert len(before)==1
        control.request(True,0,ACTOR)
        await phase(control,'paused')
        assert await g.store.undistilled_live(require_provenance=True)==before
        g.clock.now=before[0]['expires_at']
        await g.store.purge_expired()
        assert await g.store.undistilled_live(require_provenance=True)==[]


@pytest.mark.parametrize('stage',['factory','vad','reset'])
async def test_pause_drains_actual_vad_or_reset_and_discards_preroll_before_resuming(tmp_path,stage):
    async with controlled(tmp_path) as (g,control):
        await g.transcriber.start()
        if stage!='reset':
            g.gates[stage].block()
        data=np.full((1280,6),.2,dtype=np.float32)
        g.tap.push('synthetic-a',data,ts=g.clock.now)
        if stage=='reset':
            await until(lambda:g.gates['vad'].calls>0)
            g.gates['reset'].block()
        try:
            if stage!='reset':
                await until(g.gates[stage].entered.is_set)
            control.request(True,0,ACTOR)
            await until(g.gates[stage].entered.is_set)
            await asyncio.sleep(.01)
            assert control.snapshot()['phase']=='pausing'
            assert control.snapshot()['pending_operations']>0
        finally:
            g.gates[stage].release.set()
        await phase(control,'paused')
        assert not any(room['state_retained'] or room['ready'] for room in g.transcriber.resource_status()['rooms'])
        control.request(False,control.snapshot()['revision'],ACTOR)
        await phase(control,'running')
        g.clock.now+=.2
        previous=g.gates['factory'].calls
        g.tap.push('synthetic-a',data,ts=g.clock.now)
        await until(lambda:g.gates['factory'].calls>previous)


async def test_pause_waits_for_an_old_microphone_branch_but_allows_command_capture(tmp_path,monkeypatch):
    from tests.integration.test_input_callback_retirement import HeldInput,graph,invoke
    from tests.integration.test_shared_input_retirement import ambient
    async with graph(tmp_path,monkeypatch,'multi') as (g,voice):
        async with ambient(g,tmp_path) as path,AsyncExitStack() as stack:
            await path.transcriber.start()
            control=await build_ambient_control(privacy=g.privacy,path=path,shutdown=stack)
            g.clock.now+=.2
            held=HeldInput('clock')
            def clock():
                held.wait('clock')
                return g.clock.now
            monkeypatch.setattr(g.loop,'_microphone_wall_clock',clock)
            pushing=asyncio.create_task(asyncio.to_thread(invoke,g,'multi',held))
            try:
                await until(held.entered.is_set)
                control.request(True,0,ACTOR)
                await asyncio.sleep(.02)
                assert control.snapshot()['phase']=='pausing' and path.tap.ingress_pending()
            finally:
                held.release.set()
                await pushing
            await phase(control,'paused')
            assert path.tap.buffered_count()==0 and voice.resource_status()['queued']==1
            await voice.step()
            assert g.wakes['synthetic-a'].history


async def test_native_archive_exit_is_part_of_pause_even_after_segment_observer_is_cancelled(tmp_path,monkeypatch):
    from src.ambient.audio_archive import AudioArchiver
    async with controlled(tmp_path) as (g,control):
        root=tmp_path/'audio'
        g.store.audio_dir=root
        archiver=AudioArchiver(str(root),enabled=True,privacy=g.privacy,store=g.store,discard_audio=g.store.discard_audio)
        g.transcriber._archiver=archiver
        g.store.audio_inflight=archiver.is_writing
        entered,release=threading.Event(),threading.Event()
        def writing(*args):
            entered.set()
            assert release.wait(3)
            return False
        monkeypatch.setattr(archiver,'_write_sync',writing)
        processing=asyncio.create_task(g.transcriber._handle_segment('synthetic-a',g.segment()))
        try:
            await until(entered.is_set)
            control.request(True,0,ACTOR)
            await asyncio.sleep(.02)
            assert control.snapshot()['phase']=='pausing' and archiver.snapshot_operations()
        finally:
            release.set()
            await processing
        await phase(control,'paused')
        assert not archiver.snapshot_operations()


async def test_paused_distiller_cannot_start_a_new_cycle_and_active_store_callback_remains_owned(tmp_path):
    from src.ambient.distiller import Distiller
    async with controlled(tmp_path) as (g,control):
        await g.transcriber._handle_segment('synthetic-a',g.segment())
        entered,release=asyncio.Event(),asyncio.Event()
        async def chat(prompt):
            return '[{"fact":"Synthetic paused fact","category":"fact","confidence":0.9}]'
        async def save(*args,**kwargs):
            entered.set()
            await release.wait()
            return True
        distiller=Distiller(g.store,chat,save,min_batch=1,privacy=g.privacy)
        g.path.distiller=distiller
        extracting=asyncio.create_task(distiller.distill_once())
        try:
            async with asyncio.timeout(3):
                await entered.wait()
            control.request(True,0,ACTOR)
            await asyncio.sleep(.02)
            assert control.snapshot()['phase']=='pausing'
            assert await distiller.distill_once()==0
        finally:
            release.set()
            await asyncio.gather(extracting,return_exceptions=True)
        await phase(control,'paused')
        assert not distiller.snapshot_operations()


async def test_private_control_http_validates_auth_revision_and_body_before_mutation(tmp_path,dashboard_auth):
    async with controlled(tmp_path) as (_,control):
        app=DashboardAPI(ambient_control=control).app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
            assert (await client.get('/api/ambient/control')).status_code==401
            assert (await client.post('/api/ambient/control',json={'paused':True,'expected_revision':0})).status_code==401
            assert (await client.post('/api/ambient/control',headers=dashboard_auth,json={'paused':True,'expected_revision':0,'mute_total':True})).status_code==422
            assert (await client.post('/api/ambient/control',headers=dashboard_auth,json={'paused':'false','expected_revision':0})).status_code==422
            assert (await client.post('/api/ambient/control',headers=dashboard_auth,json={'paused':True,'expected_revision':0})).status_code==202
            await phase(control,'paused')
            response=await client.get('/api/ambient/control',headers=dashboard_auth)
            assert response.status_code==200 and response.headers['cache-control']=='no-store'
            assert response.json()['phase']=='paused' and not response.json()['total_mute_supported']


async def test_callback_admitted_while_paused_cannot_acquire_an_ambient_branch_during_resume(tmp_path,monkeypatch):
    from tests.integration.test_input_callback_retirement import HeldInput,graph,invoke
    from tests.integration.test_shared_input_retirement import ambient
    class Counted(HeldInput):
        def __array_finalize__(self,parent):
            super().__array_finalize__(parent)
            self.copies=getattr(parent,'copies',None)
        def copy(self,*args,**kwargs):
            self.copies.append(True)
            return super().copy(*args,**kwargs)
    async with graph(tmp_path,monkeypatch,'multi') as (g,voice):
        async with ambient(g,tmp_path) as path,AsyncExitStack() as stack:
            control=await build_ambient_control(privacy=g.privacy,path=path,shutdown=stack)
            g.clock.now+=.2
            control.request(True,0,ACTOR)
            await phase(control,'paused')
            # Exercise the real unseal interleaving: fresh capture epochs have
            # been published but the ambient tap still rejects new branches.
            g.privacy.scopes.resume_origin(MemoryOrigin.AMBIENT)
            held=HeldInput('any').view(Counted);held.copies=[]
            pushing=asyncio.create_task(asyncio.to_thread(invoke,g,'multi',held))
            try:
                await until(held.entered.is_set)
                assert not path.tap.ingress_pending()
                path.transcriber.resume_admission()
            finally:
                held.release.set()
                await pushing
            assert len(held.copies)==1  # Command capture alone owns a copy.
            assert path.tap.buffered_count()==0 and voice.resource_status()['queued']==1


async def test_failed_native_reset_never_reports_paused_or_allows_reactivation_until_repaired(tmp_path,monkeypatch):
    async with controlled(tmp_path) as (g,control):
        original=g.gates['reset'].step
        broken=[True]
        def reset():
            if broken[0]:
                raise RuntimeError('synthetic_reset_failure')
            return original()
        monkeypatch.setattr(g.gates['reset'],'step',reset)
        await g.transcriber.start()
        g.tap.push('synthetic-a',np.full((1280,6),.2,dtype=np.float32),ts=g.clock.now)
        await until(lambda:g.gates['vad'].calls>0)
        control.request(True,0,ACTOR)
        await phase(control,'failed')
        assert control.snapshot()['admission_paused'] and control.snapshot()['persisted_paused']
        assert any(room['state_retained'] for room in g.transcriber.resource_status()['rooms'])
        control.request(False,control.snapshot()['revision'],ACTOR)
        await phase(control,'failed')
        assert control.snapshot()['persisted_paused'] and not g.tap.accepting
        broken[0]=False
        control.request(True,control.snapshot()['revision'],ACTOR)
        await phase(control,'paused')
        assert not any(room['state_retained'] for room in g.transcriber.resource_status()['rooms'])


async def test_close_timeout_keeps_real_native_work_and_admission_sealed_until_actual_exit(tmp_path):
    from src.core.owned_work import WorkCloseTimeout
    async with controlled(tmp_path,stage='stt') as (g,control):
        processing=asyncio.create_task(g.transcriber._handle_segment('synthetic-a',g.segment()))
        try:
            await until(g.gates['stt'].entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await control.aclose()
            assert not control.begin_close().done()
            assert control.snapshot()['pending_operations']>0 and g.store._db is not None
            assert g.privacy.ledger.authority_available and not g.tap.accepting
        finally:
            g.gates['stt'].release.set()
            await processing
            await control.begin_close()
