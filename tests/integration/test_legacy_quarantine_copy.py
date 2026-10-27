"""Actual private-copy migration keeps unknown legacy facts out of inference."""

import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from src.memory.fact_ledger import FactLedger, MemoryReviewer
from src.memory.legacy_quarantine import stage_legacy_quarantine
import src.memory.legacy_quarantine as migration
from src.memory.privacy_services import PrivacyServices
from src.memory.services import build_memory_services
from tests.mocks.legacy_memory import legacy_memory
from tests.mocks.memory_graph import MemoryClient

pytestmark = pytest.mark.asyncio


def rows(root):
    with sqlite3.connect(root/'candidate.sqlite3') as db:
        return db.execute('SELECT record_id,state,document,metadata_json,vector,encoding,digest FROM legacy_quarantine_records_v1 ORDER BY record_id').fetchall()


async def test_copy_preserves_selected_documents_metadata_and_vectors_without_inventing_sources(tmp_path):
    source = legacy_memory(tmp_path)
    result = await stage_legacy_quarantine(str(source.preview),str(source.output))
    assert (result.snapshot_records,result.selected_records,result.quarantined_records,result.unselected_records,result.vector_records)==(3,2,2,1,2)
    assert result.source_unchanged and not result.replayed
    stored = rows(source.output)
    assert [row[0] for row in stored] == list(source.selected)
    assert all(row[1]=='quarantined' and row[2]=='private legacy marker '+row[0] for row in stored)
    assert all(len(row[4])==12 and row[5]=='FLOAT32' for row in stored)
    for row in stored:
        metadata = json.loads(row[3])
        assert 'legacy_insertion_metadata' in metadata and 'sources' not in metadata
    assert source.output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o077 == 0 for path in source.output.iterdir())
    ledger = FactLedger(str(source.output/'candidate.sqlite3'))
    ledger.initialize()
    try:
        assert ledger.eligible_ids()==ledger.list()==ledger.jobs()==[]
        assert ledger.snapshot()['facts']==ledger.snapshot()['proposals']==[]
        assert ledger.status()['states']=={}
        for identifier in source.selected:
            assert ledger.get(identifier,review=True) is None
            with pytest.raises(LookupError,match='memory_proposal_unavailable'):
                ledger.confirm(identifier,MemoryReviewer('synthetic-admin'))
    finally:
        ledger.close()


async def test_resume_verifies_exact_copy_without_reopening_source_or_renewing_data(tmp_path,monkeypatch):
    source = legacy_memory(tmp_path)
    first = await stage_legacy_quarantine(str(source.preview),str(source.output))
    before = rows(source.output)
    readonly = migration._readonly
    def guarded(path,**kwargs):
        assert path!=source.source
        return readonly(path,**kwargs)
    monkeypatch.setattr(migration,'_readonly',guarded)
    monkeypatch.setattr(migration,'_fingerprints',lambda _: pytest.fail('resume accessed original'))
    second = await stage_legacy_quarantine(None,str(source.output),resume=True)
    assert second.batch_id==first.batch_id and second.replayed
    assert rows(source.output)==before


@pytest.mark.parametrize('invalid', ['duplicate','missing','empty'])
async def test_invalid_selection_cannot_silently_import_another_record(tmp_path,invalid):
    source = legacy_memory(tmp_path)
    value = json.loads(source.preview.read_text())
    value['record_metadata'] = {'duplicate':[{'id':source.selected[0]}]*2,
        'missing':[{'id':'not-in-snapshot'}],'empty':[]}[invalid]
    source.preview.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        await stage_legacy_quarantine(str(source.preview),str(source.output))
    assert not (source.output/'candidate.sqlite3').exists()


@pytest.mark.parametrize('overlap', ['source_child','source_parent','existing'])
async def test_new_copy_never_overwrites_original_or_existing_destination(tmp_path,overlap):
    source = legacy_memory(tmp_path)
    output = {'source_child':source.source.parent/'child','source_parent':tmp_path,'existing':source.output}[overlap]
    if overlap=='existing':
        output.mkdir()
        (output/'marker').write_text('keep')
    before = hashlib.sha256(source.source.read_bytes()).hexdigest()
    with pytest.raises((ValueError,FileExistsError)):
        await stage_legacy_quarantine(str(source.preview),str(output))
    assert hashlib.sha256(source.source.read_bytes()).hexdigest()==before
    if overlap=='existing':
        assert (output/'marker').read_text()=='keep'


@pytest.mark.parametrize('problem', ['snapshot','selection','row_removed','row_changed'])
async def test_retry_rejects_modified_or_removed_quarantine_and_never_recreates_it(tmp_path,problem):
    source = legacy_memory(tmp_path)
    await stage_legacy_quarantine(str(source.preview),str(source.output))
    if problem=='snapshot':
        with (source.output/'legacy.sqlite3').open('ab') as stream:
            stream.write(b'changed')
    elif problem=='selection':
        path = source.output/'manifest.json'
        manifest=json.loads(path.read_text())
        manifest['selected_ids'].append('unselected')
        path.write_text(json.dumps(manifest))
    else:
        with sqlite3.connect(source.output/'candidate.sqlite3') as db:
            if problem=='row_removed':
                db.execute('DELETE FROM legacy_quarantine_records_v1 WHERE record_id=?',(source.selected[0],))
            else:
                db.execute("UPDATE legacy_quarantine_records_v1 SET document='changed' WHERE record_id=?",(source.selected[0],))
    before=rows(source.output)
    with pytest.raises(ValueError):
        await stage_legacy_quarantine(None,str(source.output),resume=True)
    assert rows(source.output)==before


@pytest.mark.parametrize('target', ['candidate.sqlite3','candidate.sqlite3.lock'])
@pytest.mark.parametrize('link', ['symlink','hardlink'])
async def test_resume_does_not_follow_candidate_or_lease_links_into_original(tmp_path,target,link):
    source=legacy_memory(tmp_path)
    await stage_legacy_quarantine(str(source.preview),str(source.output))
    path=source.output/target
    path.unlink()
    if link=='symlink':
        path.symlink_to(source.source)
    else:
        os.link(source.source,path)
    before=hashlib.sha256(source.source.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='legacy_candidate_not_private'):
        await stage_legacy_quarantine(None,str(source.output),resume=True)
    assert hashlib.sha256(source.source.read_bytes()).hexdigest()==before


async def test_failed_import_rolls_back_whole_batch_and_can_resume_same_snapshot(tmp_path,monkeypatch):
    source=legacy_memory(tmp_path)
    record=migration._record
    calls=0
    def fail(*args):
        nonlocal calls
        calls+=1
        if calls==2:
            raise RuntimeError('synthetic import failure')
        return record(*args)
    with monkeypatch.context() as patch:
        patch.setattr(migration,'_record',fail)
        with pytest.raises(RuntimeError,match='synthetic import failure'):
            await stage_legacy_quarantine(str(source.preview),str(source.output))
    with sqlite3.connect(source.output/'candidate.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='legacy_quarantine_records_v1'").fetchone()[0]==0
    resumed=await stage_legacy_quarantine(None,str(source.output),resume=True)
    assert resumed.quarantined_records==2 and not resumed.replayed


async def test_cancelled_import_keeps_writer_lease_until_actual_transaction_finishes(tmp_path,monkeypatch):
    source=legacy_memory(tmp_path)
    entered,release=threading.Event(),threading.Event()
    importing=migration._import
    def held(*args):
        entered.set()
        assert release.wait(2)
        return importing(*args)
    with monkeypatch.context() as patch:
        patch.setattr(migration,'_import',held)
        task=asyncio.create_task(stage_legacy_quarantine(str(source.preview),str(source.output)))
        try:
            async with asyncio.timeout(2):
                while not entered.is_set():
                    await asyncio.sleep(.001)
            task.cancel()
            await asyncio.sleep(.01)
            contender=PrivacyServices(str(source.output/'candidate.sqlite3'))
            try:
                with pytest.raises(RuntimeError,match='privacy_writer_already_active'):
                    await contender.acall(contender.initialize)
            finally:
                await contender.aclose()
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    resumed=await stage_legacy_quarantine(None,str(source.output),resume=True)
    assert resumed.replayed and resumed.quarantined_records==2


async def test_actual_cli_returns_only_counts_and_fixed_errors(tmp_path):
    source=legacy_memory(tmp_path)
    script=Path(__file__).resolve().parents[2]/'scripts/stage_legacy_memory.py'
    process=await asyncio.create_subprocess_exec(sys.executable,str(script),'--preview',str(source.preview),
        '--output-dir',str(source.output),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    stdout,stderr=await asyncio.wait_for(process.communicate(),10)
    assert process.returncode==0,stderr.decode()
    assert json.loads(stdout)['quarantined_records']==2
    assert b'private legacy marker' not in stdout+stderr
    process=await asyncio.create_subprocess_exec(sys.executable,str(script),'--preview',str(source.preview),
        '--output-dir',str(source.output),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    stdout,stderr=await asyncio.wait_for(process.communicate(),10)
    assert process.returncode==1 and not stdout and b'private legacy marker' not in stderr


@pytest.mark.parametrize('problem', ['missing_vector','short_vector','stale_sequence','missing_document'])
async def test_incomplete_legacy_payload_cannot_be_reported_as_a_preserved_record(tmp_path,problem):
    source=legacy_memory(tmp_path)
    with sqlite3.connect(source.source) as db:
        if problem=='missing_document':
            db.execute("DELETE FROM embedding_metadata WHERE key='chroma:document' AND id=(SELECT id FROM embeddings WHERE embedding_id=?)",(source.selected[0],))
        elif problem=='missing_vector':
            db.execute('UPDATE embeddings_queue SET vector=NULL WHERE id=?',(source.selected[0],))
        elif problem=='short_vector':
            db.execute('UPDATE embeddings_queue SET vector=? WHERE id=?',(b'short',source.selected[0]))
        else:
            db.execute('UPDATE embeddings_queue SET seq_id=9999999 WHERE id=?',(source.selected[0],))
    with pytest.raises(ValueError,match='legacy_(vector|document)_unavailable'):
        await stage_legacy_quarantine(str(source.preview),str(source.output))
    with sqlite3.connect(source.output/'candidate.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='legacy_quarantine_records_v1'").fetchone()[0]==0


async def test_quarantined_copy_is_ineligible_in_real_runtime_even_with_a_contaminated_index(tmp_path):
    source=legacy_memory(tmp_path)
    await stage_legacy_quarantine(str(source.preview),str(source.output))
    client=MemoryClient(source.output/'application-index')
    poisoned=client.get_or_create_collection('legacy_memories_policy_v1').collection
    poisoned.upsert(ids=list(source.selected),documents=['private legacy marker']*2,
        embeddings=[[1.,2.,3.]]*2,metadatas=[{'fact_policy':1,'confirmed':True}]*2)
    async with AsyncExitStack() as stack:
        privacy=PrivacyServices(str(source.output/'candidate.sqlite3'))
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        services=await build_memory_services({'memory':{'long_term_collection':'legacy_memories',
            'privacy_db_path':str(source.output/'candidate.sqlite3'),
            'preferences_path':str(source.output/'preferences.json')}},shutdown=stack,privacy=privacy,chroma_client=client)
        assert await services.manager.long_term.asearch_relevant('private legacy marker')==[]
        assert privacy.ledger.snapshot()['facts']==[] and privacy.ledger.snapshot()['proposals']==[]
        assert len(rows(source.output))==2


@pytest.mark.parametrize('kind', ['directory','fifo'])
async def test_source_must_be_a_regular_database_before_any_copy_io(tmp_path,kind):
    source=legacy_memory(tmp_path)
    path=tmp_path/'not-a-database'
    if kind=='directory':
        path.mkdir()
    else:
        os.mkfifo(path)
    preview=json.loads(source.preview.read_text())
    preview['source']['path']=str(path)
    source.preview.write_text(json.dumps(preview))
    with pytest.raises(ValueError,match='legacy_source_not_regular'):
        await stage_legacy_quarantine(str(source.preview),str(source.output))
    assert not source.output.exists()
