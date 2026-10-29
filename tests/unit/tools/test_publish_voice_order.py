"""Rehearse code promotion/rollback against private synthetic persistent state."""

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import marshal
from pathlib import Path
import socket
import sqlite3
import tarfile
import time
from types import SimpleNamespace

import pytest

from tools import publish_voice_order as release


def pyc(source):
    return importlib.util.MAGIC_NUMBER + b'\0' * 12 + marshal.dumps(compile(source, 'synthetic.py', 'exec'))


@pytest.fixture(params=['order','loading'])
def layout(tmp_path,request,monkeypatch):
    root = tmp_path / 'releases'
    previous = root / 'slots/previous/runtime'
    state = previous.parent / 'state'
    previous.mkdir(parents=True)
    for name in ('config', 'data', 'models', 'logs'):
        (state / name).mkdir(parents=True)
    (state / 'config/settings.yaml').write_text('synthetic: true\n')
    (state / '.env').write_text('SYNTHETIC_SETTING=private\n')
    with sqlite3.connect(state / 'data/state.db') as db:
        db.execute('CREATE TABLE locations (root TEXT)')
        db.execute('INSERT INTO locations VALUES (?)', (str((state / 'data').resolve()),))
    old = {'src/__init__.pyc': pyc(''), 'src/main.pyc': pyc('READY=True'),
           'src/users/voice_ledger.pyc': pyc('ORDER="old"'),
           'src/dashboard/voice_api.pyc': pyc('CURSOR=67'),
           'src/dashboard/frontend/obs/src/voice-view.js': b'old order',
           'src/dashboard/frontend/obs/src/voice-channel.js': b'old channel'}
    new = {**old, 'src/users/voice_ledger.pyc': pyc('ORDER="date"'),
           'src/dashboard/voice_api.pyc': pyc('CURSOR=85'),
           'src/dashboard/frontend/obs/src/voice-view.js': b'new order'}
    if request.param=='loading':
        canonical=root/'canonical-state'
        state.rename(canonical)
        state.mkdir()
        for name in release.RESOURCES:
            (state/name).symlink_to(canonical/name)
        with sqlite3.connect(state/'data/state.db') as db:
            db.execute('UPDATE locations SET root=?',(str((state/'data').resolve()),))
        monkeypatch.setattr(release,'RELEASE_TAG','voice-loading')
        monkeypatch.setattr(release,'COMMIT','c'*40)
        monkeypatch.setattr(release,'PREVIOUS_COMMIT','d'*40)
        monkeypatch.setattr(release,'APP_CHANGES',{'src/dashboard/frontend/obs/src/voice-view.js',
                                                 'src/dashboard/frontend/obs/src/voice-channel.js'})
        new={**old,'src/dashboard/frontend/obs/src/voice-view.js':b'visible player',
             'src/dashboard/frontend/obs/src/voice-channel.js':b'loading status'}
    def manifest(commit, files):
        return {'format':'kza-cpython-bytecode-v1','python':'3.13',
                'python_magic':importlib.util.MAGIC_NUMBER.hex(),'git_commit':commit,
                'external_paths':['config','data','models','logs'],
                'files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    for name, data in old.items():
        target = previous / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (previous / 'release.json').write_text(json.dumps(manifest(release.PREVIOUS_COMMIT, old)))
    for name in ('config','data','.env'):
        (previous / name).symlink_to(Path('../state') / name)
    (root / 'current').symlink_to('slots/previous/runtime')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    archive = bundle / 'kza-runtime.tar.gz'
    with tarfile.open(archive,'w:gz') as package:
        for name, data in {**new,'release.json':json.dumps(manifest(release.COMMIT,new)).encode()}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            package.addfile(info,io.BytesIO(data))
    return SimpleNamespace(root=root,previous=previous,state=state,bundle=bundle,archive=archive,
                           sha=hashlib.sha256(archive.read_bytes()).hexdigest(),new=new)


def test_staged_code_keeps_all_canonical_state_locations(layout):
    target, manifest, resources = release.prepare_code_slot(layout.archive,layout.root,layout.previous,
                                                           'candidate',release.COMMIT,layout.sha)
    assert manifest['git_commit']==release.COMMIT
    assert (layout.root/'current').resolve()==layout.previous
    for resource in release.RESOURCES:
        assert (target/'state'/resource).resolve()==(layout.state/resource).resolve()==resources[resource]
    assert (target/'runtime/data').resolve()==(layout.state/'data').resolve()
    with sqlite3.connect(target/'runtime/data/state.db') as db:
        assert db.execute('SELECT root FROM locations').fetchone()[0]==str((layout.state/'data').resolve())
    assert release.verify_runtime(target/'runtime')['git_commit']==release.COMMIT


def test_corrupt_archive_or_modified_runtime_cannot_be_staged(layout):
    with pytest.raises(release.ReleaseError,match='archive_hash'):
        release.prepare_code_slot(layout.archive,layout.root,layout.previous,'bad',release.COMMIT,'0'*64)
    (layout.previous/'src/main.pyc').write_bytes(b'changed')
    with pytest.raises(release.ReleaseError,match='runtime_integrity'):
        release.prepare_code_slot(layout.archive,layout.root,layout.previous,'bad',release.COMMIT,layout.sha)
    assert not (layout.root/'slots/bad').exists()


def test_database_backup_does_not_replace_current_data(layout):
    destination=layout.root/'snapshots'
    assert release.backup_databases(layout.state/'data',destination)==1
    with sqlite3.connect(layout.state/'data/state.db') as db:
        db.execute('INSERT INTO locations VALUES (?)',('new synthetic user data',))
    with sqlite3.connect(destination/'state.db') as db:
        assert db.execute('SELECT COUNT(*) FROM locations').fetchone()[0]==1
    with sqlite3.connect(layout.state/'data/state.db') as db:
        assert db.execute('SELECT COUNT(*) FROM locations').fetchone()[0]==2


def test_override_changes_only_working_directory(layout):
    original=('[Service]\nWorkingDirectory='+str(layout.state)+'\n'
              'Environment=PYTHONPATH=/home/kza/releases/current\n'
              'Environment=CONFIG_PATH='+str(layout.state/'config/settings.yaml')+'\n'
              'StandardOutput=append:'+str(layout.state/'logs/console.log')+'\n').encode()
    selected=layout.root/'slots/candidate/state'
    changed=release.service_override(original,layout.state,selected)
    assert changed.replace(str(selected).encode(),str(layout.state).encode())==original
    with pytest.raises(release.ReleaseError,match='working_directory'):
        release.service_override(original,layout.root/'wrong',selected)


@pytest.mark.parametrize('fail_postcheck',[False,True])
def test_operator_publication_and_automatic_code_rollback_preserve_current_data(layout,monkeypatch,fail_postcheck):
    import dotenv

    override=layout.root.parent/'.config/systemd/user/kza-voice.service.d/release.conf'
    override.parent.mkdir(parents=True)
    original=('[Service]\nWorkingDirectory='+str(layout.state)+'\n'
              'Environment=PYTHONPATH=/home/kza/releases/current\n').encode()
    override.write_bytes(original)
    original_path=Path
    def mapped_path(value):
        value=str(value)
        if value.startswith('/home/kza/'):
            return layout.root.parent/value.removeprefix('/home/kza/')
        return original_path(value)
    monkeypatch.setattr(release,'Path',mapped_path)
    monkeypatch.setattr(release.os,'getuid',lambda:1000)
    monkeypatch.setattr(release,'ARCHIVE_SHA256',layout.sha)
    monkeypatch.setattr(dotenv,'dotenv_values',lambda path:{'KZA_DASHBOARD_TOKEN':'synthetic-release-token'})
    events=[]
    monkeypatch.setattr(release,'_control',lambda action:events.append(action))
    profile={name:[1.] for name in release.noise.STATIC_PARAMS}
    profile[release.noise.PARAMETER]=[.51]
    snapshot={'microphones':[{'room_id':room,'capture':{'state':'receiving',
        'last_callback_at':time.time(),'last_nonzero_at':time.time(),'frames_received':16000}}
        for room in ('hall','escritorio')]}
    monkeypatch.setattr(release,'_hall_profile',lambda runtime:(deepcopy(profile),deepcopy(snapshot)))
    def ready(token,required):
        assert required=={'hall','escritorio'}
        return {'health_status':'unhealthy','receiving_rooms':sorted(required)}
    monkeypatch.setattr(release,'_wait_ready',ready)
    def preserve(expected,runtime):
        assert expected==profile
        events.append('preserve_hall')
    monkeypatch.setattr(release,'_preserve_hall',preserve)
    def check(bundle,token,manifest):
        assert manifest['git_commit']==release.COMMIT
        with sqlite3.connect(layout.state/'data/state.db') as db:
            db.execute('INSERT INTO locations VALUES (?)',('new synthetic user data',))
        if fail_postcheck:
            raise release.ReleaseError('synthetic_postcheck_failure')
        return {'descending':True,'rows_checked':100,'pages_checked':2,'frontend_hash_matches':True}
    monkeypatch.setattr(release,'_https_check',check)
    monkeypatch.setattr(release.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0))
    result=release.publish(layout.bundle)
    assert events[:3]==['stop','daemon-reload','start']
    with sqlite3.connect(layout.state/'data/state.db') as db:
        assert db.execute('SELECT COUNT(*) FROM locations').fetchone()[0]==2
    if fail_postcheck:
        assert result['status']=='previous_release_restored'
        assert (layout.root/'current').resolve()==layout.previous
        assert override.read_bytes()==original
        assert events[-4:]==['stop','daemon-reload','start','preserve_hall']
    else:
        assert result['status']=='active' and result['hall_PP_MIN_NN']==.51
        assert result['database_backups']==1
        assert (layout.root/'current').resolve()!=layout.previous
        assert (layout.root/'current/data').resolve()==(layout.state/'data').resolve()
        assert result['receiving_rooms']==['escritorio','hall']
        restored=release.publish(layout.bundle,rollback=True)
        assert restored['status']=='rolled_back'
        assert (layout.root/'current').resolve()==layout.previous
        assert override.read_bytes()==original
        previous_events=list(events)
        assert release.publish(layout.bundle,rollback=True)['status']=='already_rolled_back'
        assert events==previous_events
    assert release.verify_runtime(layout.previous)['git_commit']==release.PREVIOUS_COMMIT


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong_order',[False,True])
async def test_postcheck_uses_real_authenticated_voice_pagination(tmp_path,dashboard_auth,monkeypatch,wrong_order):
    import asyncio

    import uvicorn

    from src.dashboard.api import DashboardAPI
    from tests.mocks.memory_sources import memory_source
    from tests.mocks.voice_registry import VECTOR,web_graph

    token=dashboard_auth['Authorization'].split(' ',1)[1]
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        for _ in range(70):
            clock.now+=.1
            source=memory_source(now=clock.now,ttl=3600)
            await voices.acall(voices.ledger.put,'synthetic-release',VECTOR.copy(),1.,(source,))
        if wrong_order:
            original=voices.ledger.project
            def reversed_page(*args):
                result=original(*args)
                result.rows.reverse()
                return result
            monkeypatch.setattr(voices.ledger,'project',reversed_page)
        dashboard=DashboardAPI(voice_registry=voices,event_bus=bus,observability_use_mocks=False)
        listener=socket.socket()
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(dashboard.app,log_level='critical',lifespan='off'))
        serving=asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if serving.done():
                        serving.result()
                    await asyncio.sleep(.01)
            asset=Path('src/dashboard/frontend/obs/src/voice-view.js').read_bytes()
            if wrong_order:
                with pytest.raises(release.ReleaseError,match='not_descending'):
                    await release.check_dashboard(token,hashlib.sha256(asset).hexdigest(),base_url=f'http://127.0.0.1:{port}')
            else:
                channel=Path('src/dashboard/frontend/obs/src/voice-channel.js').read_bytes()
                assets={'/obs/src/voice-channel.js':hashlib.sha256(channel).hexdigest()}
                result=await release.check_dashboard(token,hashlib.sha256(asset).hexdigest(),base_url=f'http://127.0.0.1:{port}',assets=assets)
                assert result['descending'] and result['rows_checked']==70 and result['pages_checked']==2
                assert result['unauthenticated_private_api_status']==401
                assert result['frontend_assets_checked']==2
                with pytest.raises(release.ReleaseError,match='frontend_hash'):
                    await release.check_dashboard(token,hashlib.sha256(asset).hexdigest(),base_url=f'http://127.0.0.1:{port}',
                                                  assets={'/obs/src/voice-channel.js':'0'*64})
        finally:
            server.should_exit=True
            async with asyncio.timeout(5):
                await serving
