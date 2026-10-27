"""Real bounded grouping, manual corrections and source-safe group retirement."""

import asyncio
import json
import math
import sqlite3
from uuid import uuid4

import httpx
import numpy as np
import pytest

from src.dashboard.api import DashboardAPI
from src.memory.fact_ledger import MemoryReviewer
from src.users.voice_ledger import VoiceLedger,VoiceReviewer
from tests.integration.test_live_stream_receipts import flush
from tests.mocks.memory_sources import memory_source
from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import ACTOR,graph,web_graph

pytestmark = pytest.mark.asyncio


def vector(angle=0,dimensions=16):
    result = np.zeros(dimensions,dtype=np.float32)
    result[:2] = (math.cos(math.radians(angle)),math.sin(math.radians(angle)))
    return result


async def sample(voices,clock,angle=0,*,ttl=60,duration=3.,model='synthetic',dimensions=16):
    source = memory_source(now=clock.now,ttl=ttl)
    identity = await voices.acall(voices.ledger.put,model,vector(angle,dimensions),duration,(source,))
    return source,identity


async def rows(voices):
    return {row['id']:row for row in (await voices.acall(voices.ledger.snapshot))['fingerprints']}


async def edit(voices,identities,action,*,request=None,revisions=None,actor=ACTOR):
    if revisions is None:
        current = await rows(voices)
        revisions = {current[identity]['group_id']:current[identity]['group_revision'] for identity in identities}
    return await voices.acall(voices.ledger.groups.correct,tuple(identities),revisions,action,request or str(uuid4()),actor)


async def test_similar_eligible_captures_share_a_stable_suggested_group_and_earliest_deadline(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        first,one = await sample(voices,clock,ttl=10)
        initial = (await rows(voices))[one]
        _,two = await sample(voices,clock,ttl=60)
        _,three = await sample(voices,clock,ttl=60)
        current = await rows(voices)
        assert {row['group_id'] for row in current.values()}=={initial['group_id']}
        assert {row['group_revision'] for row in current.values()}=={3}
        assert {row['group_size'] for row in current.values()}=={3}
        assert {row['group_state'] for row in current.values()}=={'suggested'}
        assert all(row['group_similarity']==pytest.approx(1) for row in current.values())
        assert all(row['group_expires_at']==first.expires_at for row in current.values())
        assert current[two]['expires_at']==current[three]['expires_at']==clock.now+60
        assert not (await voices.acall(voices.ledger.snapshot))['automatic_grouping_accepted']


async def test_complete_link_does_not_merge_a_chain_of_near_neighbors(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        _,one = await sample(voices,clock,0)
        _,two = await sample(voices,clock,20)
        _,three = await sample(voices,clock,40)
        current = await rows(voices)
        assert current[one]['group_id']==current[two]['group_id']
        assert current[three]['group_id']!=current[one]['group_id']


async def test_ambiguous_capture_abstains_instead_of_choosing_by_id_or_room(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        await sample(voices,clock,0)
        await sample(voices,clock,40)
        _,ambiguous = await sample(voices,clock,20)
        row = (await rows(voices))[ambiguous]
        assert row['group_state']=='unknown' and row['group_reason']=='ambiguous' and row['group_size']==1


@pytest.mark.parametrize('other',[{'model':'other'},{'dimensions':32},{'duration':1.}])
async def test_incompatible_or_short_audio_cannot_join_a_candidate_group(tmp_path,other):
    async with graph(tmp_path) as (_,_,voices,clock):
        _,one = await sample(voices,clock)
        _,two = await sample(voices,clock,**other)
        current = await rows(voices)
        assert current[one]['group_id']!=current[two]['group_id']
        if other.get('duration'):
            assert current[two]['group_reason']=='short_audio'


async def test_merge_expands_all_selected_groups_and_locks_the_result_against_automatic_changes(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        _,one = await sample(voices,clock,0)
        _,two = await sample(voices,clock,0)
        _,three = await sample(voices,clock,90)
        before = await rows(voices)
        assert await edit(voices,(one,three),'merge')
        current = await rows(voices)
        assert len({row['group_id'] for row in current.values()})==1
        retained = min(before[one]['group_id'],before[three]['group_id'])
        assert current[two]['group_id']==retained and current[two]['group_state']=='manual'
        _,four = await sample(voices,clock,0)
        assert (await rows(voices))[four]['group_id']!=retained


async def test_split_preserves_remainder_id_and_manual_unknown_survives_restart(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        ids = [(await sample(voices,clock))[1] for _ in range(3)]
        before = await rows(voices)
        assert await edit(voices,(ids[0],),'split')
        current = await rows(voices)
        assert current[ids[1]]['group_id']==before[ids[1]]['group_id']
        assert current[ids[1]]['group_state']=='manual'
        assert current[ids[0]]['group_id']!=current[ids[1]]['group_id']
        assert current[ids[0]]['group_reason']=='manual_unknown'
    async with graph(tmp_path) as (_,_,voices,clock):
        assert await rows(voices)==current
        _,extra = await sample(voices,clock)
        assert (await rows(voices))[extra]['group_id'] not in {row['group_id'] for row in current.values()}


async def test_unknown_separates_each_selected_capture_without_naming_or_deleting_audio(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        ids = [(await sample(voices,clock))[1] for _ in range(3)]
        assert await edit(voices,ids[:2],'unknown')
        current = await rows(voices)
        assert len({row['group_id'] for row in current.values()})==3
        assert all(current[identity]['group_reason']=='manual_unknown' for identity in ids[:2])
        assert not (await voices.acall(voices.ledger.snapshot))['profiles']


async def test_corrections_are_idempotent_and_reject_stale_revisions_or_conflicting_retries(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        ids = [(await sample(voices,clock))[1] for _ in range(3)]
        before = await rows(voices)
        revisions = {before[ids[0]]['group_id']:before[ids[0]]['group_revision']}
        request = str(uuid4())
        assert await edit(voices,(ids[0],),'split',request=request,revisions=revisions)
        after = await rows(voices)
        assert await edit(voices,(ids[0],),'split',request=request,revisions=revisions)
        assert await rows(voices)==after
        with pytest.raises(ValueError,match='request_conflict'):
            await edit(voices,(ids[0],),'unknown',request=request,revisions=revisions)
        with pytest.raises(ValueError,match='revision_conflict'):
            await edit(voices,(ids[1],),'split',revisions=revisions)


async def test_group_expiry_never_renews_raw_samples_and_survivors_do_not_rejoin_automatically(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        source,one = await sample(voices,clock,ttl=10)
        other,two = await sample(voices,clock,ttl=60)
        await edit(voices,(one,),'split')
        clock.now = source.expires_at
        await voices.acall(voices.ledger.prune)
        current = await rows(voices)
        assert set(current)=={two} and current[two]['expires_at']==other.expires_at
        assert current[two]['group_state']=='unknown' and current[two]['group_reason']=='prior_group_expired'
        _,three = await sample(voices,clock)
        assert (await rows(voices))[three]['group_id']!=current[two]['group_id']
        def requests():
            with voices.privacy.ledger._transaction() as db:
                return db.execute('SELECT COUNT(*) FROM voice_group_requests').fetchone()[0]
        assert await voices.acall(requests)==0


async def test_partial_page_group_receipt_includes_members_not_visible_on_that_page(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        samples = [await sample(voices,clock) for _ in range(3)]
        ordered = sorted(identity for _,identity in samples)
        hidden = next(source for source,identity in samples if identity==ordered[0])
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                data = (await client.get('/api/voices?cursor='+ordered[1])).json()['snapshot']
                assert len(data['rows'])==1 and data['rows'][0]['group_size']==3
                assert await bus._retention.receipts.pending(hidden.conversation_id)==1


async def test_group_edit_retires_buffered_capture_metadata_but_not_permanent_profile_names(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        samples = [await sample(voices,clock) for _ in range(3)]
        ids = [identity for _,identity in samples]
        profile = await voices.acall(voices.ledger.confirm,tuple(ids),'Synthetic permanent group name',str(uuid4()),ACTOR)
        async with ready_browser(bus,version=3) as peer:
            entered,release = asyncio.Event(),asyncio.Event()
            api = DashboardAPI(voice_registry=voices,event_bus=bus)
            @api.app.middleware('http')
            async def buffer(request,call_next):
                response = await call_next(request)
                if request.url.path=='/api/voices':
                    entered.set()
                    await release.wait()
                return response
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                metadata = (await client.get('/api/voices/profiles')).json()['snapshot']
                viewing = asyncio.create_task(client.get('/api/voices'))
                try:
                    async with asyncio.timeout(3):
                        await entered.wait()
                    other = VoiceLedger(voices.privacy.ledger)
                    current = await rows(voices)
                    revisions = {current[ids[0]]['group_id']:current[ids[0]]['group_revision']}
                    await voices.acall(other.groups.correct,(ids[0],),revisions,'split',str(uuid4()),ACTOR)
                    release.set()
                    assert (await viewing).json()['snapshot'] is None
                    state = next(state for state in bus._retention._states.values() if state.browser.connection==peer.connection)
                    assert state.snapshots[metadata['event_id']].available(voices.privacy.ledger,clock.now)
                    assert (await voices.acall(voices.ledger.snapshot))['profiles']==[profile]
                finally:
                    release.set()
                    await viewing


async def test_erasure_removes_group_links_and_correction_requests_without_losing_independent_raw_samples(tmp_path):
    async with graph(tmp_path) as (privacy,owner,voices,clock):
        samples = [await sample(voices,clock) for _ in range(3)]
        ids = [identity for _,identity in samples]
        await edit(voices,(ids[0],),'split')
        operation = await owner.request(samples[0][0].conversation_id,MemoryReviewer('synthetic-admin'))
        await flush(owner)
        current = await rows(voices)
        assert ids[0] not in current
        assert all(row['group_state']=='unknown' for row in current.values())
        assert await voices.acall(voices.ledger.pending,samples[0][0].conversation_id)==0
        assert (await privacy.acall(privacy.ledger.erasures.snapshot,operation['id']))['state']!='complete'


async def test_native_correction_rechecks_current_actor_and_rolls_back_all_membership_changes(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        ids = [(await sample(voices,clock))[1] for _ in range(2)]
        before = await rows(voices)
        calls = []
        def current():
            calls.append(True)
            return len(calls)<3
        with pytest.raises(PermissionError):
            await edit(voices,(ids[0],),'split',actor=VoiceReviewer('synthetic-admin',current))
        assert await rows(voices)==before


async def test_initial_upgrade_preserves_deadlines_and_never_invents_suggested_groups(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        samples = [await sample(voices,clock) for _ in range(2)]
    with sqlite3.connect(tmp_path/'privacy.db') as db:
        db.execute('DELETE FROM voice_group_members')
        db.execute('DELETE FROM voice_groups')
        db.execute('DELETE FROM voice_group_schema')
    async with graph(tmp_path) as (_,_,voices,_):
        current = await rows(voices)
        assert len({row['group_id'] for row in current.values()})==2
        assert all(row['group_reason']=='unreviewed' and row['group_state']=='unknown' for row in current.values())
        assert all(current[identity]['expires_at']==source.expires_at for source,identity in samples)


async def test_partial_restore_cannot_silently_recreate_missing_membership_or_drop_group_parent_sources(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        samples = [await sample(voices,clock) for _ in range(2)]
        def corrupt():
            with voices.privacy.ledger._transaction() as db:
                db.execute('UPDATE voice_groups SET sources_json=?',(json.dumps([samples[0][0].to_dict()]),))
        await voices.acall(corrupt)
        assert await rows(voices)=={}
        assert await voices.acall(voices.ledger.pending,str(uuid4()))>0
    with sqlite3.connect(tmp_path/'privacy.db') as db:
        db.execute('DELETE FROM voice_group_members')
    async with graph(tmp_path) as (_,_,voices,_):
        assert await rows(voices)=={}
        assert await voices.acall(voices.ledger.pending,str(uuid4()))>0


async def test_grouping_expiry_during_native_comparison_cannot_commit_a_capture(tmp_path,monkeypatch):
    async with graph(tmp_path) as (_,_,voices,clock):
        original = voices.ledger.groups.add
        source = memory_source(now=clock.now,ttl=1)
        def held(*args):
            result = original(*args)
            clock.now = source.expires_at
            return result
        monkeypatch.setattr(voices.ledger.groups,'add',held)
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.put,'synthetic',vector(),3.,(source,))
        def count():
            with voices.privacy.ledger._transaction() as db:
                return tuple(db.execute('SELECT (SELECT COUNT(*) FROM voice_fingerprints),(SELECT COUNT(*) FROM voice_groups)').fetchone())
        assert await voices.acall(count)==(0,0)


async def test_http_corrections_require_auth_exact_revisions_and_reject_permission_or_ttl_fields(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        ids = [(await sample(voices,clock))[1] for _ in range(2)]
        current = await rows(voices)
        revisions = {current[ids[0]]['group_id']:current[ids[0]]['group_revision']}
        app = DashboardAPI(voice_registry=voices,event_bus=bus).app
        body = {'fingerprint_ids':[ids[0]],'expected_revisions':revisions,'action':'split','request_id':str(uuid4())}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
            assert (await client.post('/api/voices/groups/corrections',json=body)).status_code==401
            assert (await client.post('/api/voices/groups/corrections',headers=dashboard_auth,json={**body,'expires_at':None})).status_code==422
            response = await client.post('/api/voices/groups/corrections',headers=dashboard_auth,json=body)
            assert response.status_code==200 and response.json()=={'updated':True}
            assert response.headers['Cache-Control']=='no-store'
            stale = await client.post('/api/voices/groups/corrections',headers=dashboard_auth,json={**body,'request_id':str(uuid4())})
            assert stale.status_code==409


async def test_full_group_cannot_grow_or_merge_past_review_capacity(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        ids = [(await sample(voices,clock))[1] for _ in range(33)]
        current = await rows(voices)
        assert current[ids[0]]['group_size']==32 and current[ids[-1]]['group_size']==1
        with pytest.raises(ValueError,match='merge_capacity'):
            await edit(voices,(ids[0],ids[-1]),'merge')
        assert await rows(voices)==current


async def test_manual_merge_never_discards_sources_to_fit_a_group(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        sources = [tuple(memory_source(now=clock.now) for _ in range(130)) for _ in range(2)]
        ids = [await voices.acall(voices.ledger.put,'synthetic',vector(index*90),3.,page) for index,page in enumerate(sources)]
        current = await rows(voices)
        with pytest.raises(ValueError,match='source_capacity'):
            await edit(voices,ids,'merge')
        assert await rows(voices)==current


async def test_pending_only_revocation_hides_groups_but_does_not_destroy_manual_corrections(tmp_path):
    async with graph(tmp_path) as (privacy,_,voices,clock):
        samples = [await sample(voices,clock) for _ in range(2)]
        await edit(voices,(samples[0][1],),'split')
        before = await rows(voices)
        prior = privacy.ledger._pending_conversations
        privacy.ledger._pending_conversations = prior | {samples[0][0].conversation_id}
        try:
            assert await rows(voices)=={}
            await voices.acall(voices.ledger.prune)
        finally:
            privacy.ledger._pending_conversations = prior
        assert await rows(voices)==before


async def test_partial_group_lineage_cannot_become_complete_when_its_missing_parent_member_is_erased(tmp_path):
    async with graph(tmp_path) as (_,owner,voices,clock):
        first,one = await sample(voices,clock)
        second,_ = await sample(voices,clock)
        group = (await rows(voices))[one]['group_id']
        def partial():
            with voices.privacy.ledger._transaction() as db:
                db.execute('UPDATE voice_groups SET sources_json=?',(json.dumps([first.to_dict()]),))
        await voices.acall(partial)
        await owner.request(second.conversation_id,MemoryReviewer('synthetic-admin'))
        await flush(owner)
        assert await voices.acall(voices.ledger.pending,second.conversation_id)>0
        assert await rows(voices)=={}
        _,extra = await sample(voices,clock)
        visible = await rows(voices)
        assert set(visible)=={extra} and visible[extra]['group_id']!=group
        assert await voices.acall(voices.ledger.pending,second.conversation_id)>0
        clock.now=first.expires_at
        assert await voices.acall(voices.ledger.pending,second.conversation_id)>0
        await voices.acall(voices.ledger.prune)
        assert await voices.acall(voices.ledger.pending,second.conversation_id)==0


async def test_partial_correction_request_lineage_remains_unknown_after_its_missing_parent_is_erased(tmp_path):
    async with graph(tmp_path) as (_,owner,voices,clock):
        first,one = await sample(voices,clock)
        second,_ = await sample(voices,clock)
        request = str(uuid4())
        before = await rows(voices)
        revisions = {before[one]['group_id']:before[one]['group_revision']}
        await edit(voices,(one,),'split',request=request,revisions=revisions)
        def partial():
            with voices.privacy.ledger._transaction() as db:
                db.execute('UPDATE voice_group_requests SET sources_json=?',(json.dumps([first.to_dict()]),))
        await voices.acall(partial)
        with pytest.raises(LookupError):
            await edit(voices,(one,),'split',request=request,revisions=revisions)
        await owner.request(second.conversation_id,MemoryReviewer('synthetic-admin'))
        await flush(owner)
        assert await voices.acall(voices.ledger.pending,second.conversation_id)>0
        clock.now=first.expires_at
        await voices.acall(voices.ledger.prune)
        assert await voices.acall(voices.ledger.pending,second.conversation_id)==0


async def test_missing_group_proofs_are_never_reconstructed_by_idle_cleanup_or_new_observations(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        await sample(voices,clock)
        def partial():
            with voices.privacy.ledger._transaction() as db:
                db.execute('DELETE FROM voice_group_lineage')
        await voices.acall(partial)
        await voices.acall(voices.ledger.prune)
        assert await rows(voices)=={}
        assert await voices.acall(voices.ledger.pending,str(uuid4()))>0
        _,new = await sample(voices,clock)
        assert set(await rows(voices))=={new}
        assert await voices.acall(voices.ledger.pending,str(uuid4()))>0


async def test_credential_rotation_during_proof_write_rolls_back_the_final_group_commit(tmp_path):
    async with graph(tmp_path) as (_,_,voices,clock):
        ids = [(await sample(voices,clock))[1] for _ in range(2)]
        before = await rows(voices)
        current = [True]
        def trace(sql):
            if sql.startswith('INSERT INTO voice_group_request_lineage'):
                current[0] = False
        def tracing(callback):
            with voices.privacy.ledger._transaction() as db:
                db.set_trace_callback(callback)
        await voices.acall(tracing,trace)
        try:
            with pytest.raises(PermissionError):
                await edit(voices,(ids[0],),'split',actor=VoiceReviewer('synthetic-admin',lambda:current[0]))
        finally:
            await voices.acall(tracing,None)
        assert await rows(voices)==before
