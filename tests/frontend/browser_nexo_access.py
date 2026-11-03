"""Actual Chromium coverage for NEXO access and exact evaluator UI.

Synthetic credentials remain in process memory. Evidence never serializes a
credential, a request body, an input value, or raw CDP exception details.
"""

import asyncio
from contextlib import AsyncExitStack
from functools import partial
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

import aiohttp
import numpy as np
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.dashboard.live_erasure import build_live_services
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.privacy_services import PrivacyServices
from src.users.user_manager import PermissionLevel
from src.users.voice_registry import build_voice_registry
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_typed_controls import Page
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import AUDIO, Encoder


EVIDENCE = Path(os.environ.get('KZA_NEXO_ACCESS_EVIDENCE', 'data/autonomous-roadmap-20261102/t013/nexo-access-browser'))
INSTRUMENT = r"""(() => {
  window.__emptyCallbacks=[];window.__holdEmpty=false;window.__urls=new Set();
  window.__credentialPosts=[];window.__loseIssueReceipt=false;window.__copyRequests=0;window.__copiedShape=false;
  window.__evaluationReads=0;
  Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async value=>{
    window.__copyRequests++;window.__copiedShape=/^kza_[A-Za-z0-9_-]{43}$/.test(value);
  }}});
  const add=HTMLMediaElement.prototype.addEventListener,remove=HTMLMediaElement.prototype.removeEventListener,wrappers=new WeakMap();
  HTMLMediaElement.prototype.addEventListener=function(type,callback,options){
    if(type==='emptied'){const wrapped=event=>window.__holdEmpty?window.__emptyCallbacks.push(()=>callback(event)):callback(event);wrappers.set(callback,wrapped);return add.call(this,type,wrapped,options);}
    return add.call(this,type,callback,options);
  };
  HTMLMediaElement.prototype.removeEventListener=function(type,callback,options){return remove.call(this,type,wrappers.get(callback)||callback,options);};
  const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
  URL.createObjectURL=blob=>{const value=create(blob);window.__urls.add(value);return value;};
  URL.revokeObjectURL=value=>{window.__urls.delete(value);return revoke(value);};
  const fetch=window.fetch.bind(window);
  window.fetch=async(path,options={})=>{
    if(typeof path==='string' && path.startsWith('/api/evaluation/samples/'))window.__evaluationReads++;
    if(path==='/api/admin/auth/credentials' && options.method==='POST'){
      const command=JSON.parse(options.body);window.__credentialPosts.push(command.request_id);
      const response=await fetch(path,options);
      if(window.__loseIssueReceipt){window.__loseIssueReceipt=false;return new Proxy(response,{get(target,key){
        if(key==='json')return async()=>{let document=await target.json();document=null;throw Error('synthetic lost access receipt');};
        const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
      }});}
      return response;
    }
    return fetch(path,options);
  };
})()"""


def redacted(value):
    return re.sub(r'kza_[A-Za-z0-9_-]{43}', '[redacted]', json.dumps(value, ensure_ascii=False))


async def click_text(page, selector, text):
    await page.evaluate("(()=>{const target=[...document.querySelectorAll(" + json.dumps(selector)
        + ")].find(node=>node.textContent.trim()===" + json.dumps(text)
        + ");if(!target)throw Error('missing named control');target.click()})()")


async def escape(page):
    await page.command('Input.dispatchKeyEvent', {'type': 'keyDown', 'key': 'Escape', 'code': 'Escape', 'windowsVirtualKeyCode': 27})
    await page.command('Input.dispatchKeyEvent', {'type': 'keyUp', 'key': 'Escape', 'code': 'Escape', 'windowsVirtualKeyCode': 27})


async def main():
    os.umask(0o077)
    EVIDENCE.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-nexo-access-root'
    requests, checks = [], []
    phase = 'setup'

    class SafePage(Page):
        async def wait_for(self, expression):
            try:
                return await super().wait_for(expression)
            except AssertionError:
                state = await self.evaluate("({text:document.body.innerText,ready:document.readyState,dialogs:[...document.querySelectorAll('dialog')].map(d=>({id:d.id,open:d.open})),copyRequests:window.__copyRequests,posts:window.__credentialPosts?.length,reads:window.__evaluationReads,urls:window.__urls?.size,emptyCallbacks:window.__emptyCallbacks?.length})")
                (EVIDENCE/'failure.json').write_text(redacted({'phase': phase, 'condition': expression, 'state': state,
                    'requests': requests, 'runtime_error_count': len(self.errors)})+'\n')
                # Inputs containing secrets always remain masked in evidence.
                await self.evaluate("for(const input of document.querySelectorAll('input'))if(/^kza_/.test(input.value))input.type='password'")
                await screenshot(self, EVIDENCE/'failure.png')
                raise AssertionError('Browser condition failed in phase: '+phase) from None

    browser_data = Path.home()/'snap/chromium/common'
    browser_data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kza-nexo-access-', dir=browser_data) as temporary:
        directory = Path(temporary)
        async with AsyncExitStack() as stack:
            users = None
            authority = DashboardAuthority(directory/'auth.db', session_ttl_s=600,
                account_provider=lambda: users.active_accounts() if users is not None else None)
            stack.push_async_callback(authority.aclose)
            await authority.initialize()
            root = authority.resolve('synthetic-nexo-access-root')
            operator = await authority.issue_credential(root, subject='synthetic-access-operator',
                role=DashboardRole.OPERATOR, expires_at=time.time()+600)
            clock = Clock(time.time())
            privacy = PrivacyServices(str(directory/'privacy.db'), clock=clock)
            stack.push_async_callback(privacy.aclose)
            await privacy.acall(privacy.initialize)
            owner = ErasureCoordinator(privacy)
            stack.push_async_callback(owner.aclose)
            bus = await build_live_services({}, privacy=privacy, owner=owner, shutdown=stack)
            voices = await build_voice_registry(privacy=privacy, owner=owner, shutdown=stack)
            users = await voices.load_accounts(str(directory/'legacy-users.json'),
                retirement_proof=lambda identities: authority.users_retired(identities)
                    and voices.ledger.people.user_links_cleared(identities))
            person_a, _ = await voices.acall(users.add_user, 'Synthetic homonym', PermissionLevel.ADMIN)
            person_b, _ = await voices.acall(users.add_user, 'Synthetic second person', PermissionLevel.ADMIN)
            assert person_a is not None and person_b is not None
            assert (await voices.acall(partial(users.update_user, person_b.user_id, name='Synthetic homonym')))[0]
            encoder = Encoder(); voices.register_encoder(encoder)
            for index in range(3):
                source = memory_source(now=clock.now, ttl=600)
                assert voices.observe(encoder, np.tile(AUDIO, 3), (source,),
                    transcript=f'Synthetic evaluator phrase {index}', room_id='Synthetic private room')
                async with asyncio.timeout(5):
                    while voices.resource_status()['pending'] or voices.resource_status()['operations']:
                        await asyncio.sleep(.001)
            identities = [row['id'] for row in (await voices.acall(voices.ledger.snapshot))['fingerprints']]
            assert len(identities) == 3
            dashboard = DashboardAPI(authority=authority, event_bus=bus, voice_registry=voices, user_manager=users)
            @dashboard.app.middleware('http')
            async def record(request, call_next):
                response = await call_next(request)
                requests.append({'method': request.method, 'path': request.url.path, 'status': response.status_code})
                return response
            listener = socket.socket(); listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            browser = None
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        await asyncio.sleep(.02)
                profile = directory/'chromium'
                with (EVIDENCE/'chromium.log').open('w') as output:
                    browser = subprocess.Popen(['chromium', '--headless', '--disable-gpu', '--use-angle=swiftshader',
                        '--enable-unsafe-swiftshader', '--no-first-run', '--disable-background-networking',
                        '--autoplay-policy=no-user-gesture-required', '--remote-debugging-port=0',
                        f'--user-data-dir={profile}', 'about:blank'], stdout=output, stderr=output)
                    async with asyncio.timeout(15):
                        while not (profile/'DevToolsActivePort').exists():
                            assert browser.poll() is None
                            await asyncio.sleep(.05)
                    debugging = int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                    async with aiohttp.ClientSession() as client:
                        async with client.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            target = next(item for item in await response.json() if item['type'] == 'page')
                    async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                        page = SafePage(connection)
                        await page.command('Runtime.enable'); await page.command('Page.enable')
                        await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': INSTRUMENT})
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        await page.command('Page.navigate', {'url': f'http://127.0.0.1:{port}/#accesos'})
                        phase = 'operator login and home credential'
                        await page.wait_for("document.querySelector('#login-error')?.textContent.length>0")
                        await login(page, operator.token, 'Administración')
                        await page.wait_for("!document.querySelector('#access-view').hidden && !document.querySelector('#add-access').disabled")
                        await page.evaluate("document.querySelector('#add-access').click()")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('#access-dialog form select')[1].options.length===3")
                        assert await page.evaluate("[...document.querySelectorAll('#access-dialog form select')[1].options].filter(option=>option.value).every(option=>option.textContent.includes(option.value))")
                        await page.evaluate("(()=>{const form=document.querySelector('#access-dialog form');form.querySelector('input[type=text]').value='Synthetic browser home';form.querySelectorAll('select')[1].value="+json.dumps(person_b.user_id)+";form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("document.querySelector('#access-secret-dialog').open && document.querySelector('#access-secret-dialog input').type==='password'")
                        assert await page.evaluate("window.__copyRequests===0 && !document.querySelector('#token').value")
                        await screenshot(page, EVIDENCE/'access-secret-masked.png')
                        await click_text(page, '#access-secret-dialog button', 'Copiar credencial')
                        await page.wait_for("window.__copyRequests===1 && window.__copiedShape && document.querySelector('.access-secret-status').textContent.includes('copiada')")
                        await escape(page)
                        await page.wait_for("!document.querySelector('#access-secret-dialog').open && !document.querySelector('#access-secret-dialog').children.length")
                        catalog = authority.commands.catalog(root)['rows']
                        home = next(row for row in catalog if row['subject'] == 'Synthetic browser home')
                        assert home['user_id'] == person_b.user_id and home['role'] == 'home'
                        checks.append('home grant uses selected canonical homonym; explicit masked secret copy and Escape retirement')
                        await screenshot(page, EVIDENCE/'access-desktop.png')
                        for width in (390, 320):
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': True})
                            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                            await screenshot(page, EVIDENCE/f'access-{width}.png', full=False)
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        phase = 'modify and revoke'
                        selector = '[data-credential-id='+json.dumps(home['id'])+']'
                        await page.wait_for('!!document.querySelector('+json.dumps(selector)+')')
                        await click_text(page, selector+' button', 'Modificar')
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('#access-dialog form select')[1].options.length===3")
                        await page.evaluate("(()=>{const form=document.querySelector('#access-dialog form');form.querySelectorAll('select')[1].value="+json.dumps(person_a.user_id)+";form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("!document.querySelector('#access-dialog').open && !document.querySelector('#add-access').disabled")
                        changed = next(row for row in authority.commands.catalog(root)['rows'] if row['id'] == home['id'])
                        assert changed['revision'] == home['revision']+1 and changed['user_id'] == person_a.user_id
                        await click_text(page, selector+' button', 'Retirar')
                        await click_text(page, '#access-dialog button', 'Retirar acceso')
                        await page.wait_for('!document.querySelector('+json.dumps(selector)+') && !document.querySelector("#add-access").disabled')
                        checks.append('real CAS modification and revocation')
                        phase = 'operator credential form'
                        await page.evaluate("document.querySelector('#add-access').click()")
                        await page.evaluate("(()=>{const form=document.querySelector('#access-dialog form');form.querySelector('input[type=text]').value='Synthetic browser operator';const role=form.querySelector('select');role.value='operator';role.dispatchEvent(new Event('change'));form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("document.querySelector('#access-secret-dialog').open")
                        await escape(page)
                        await page.wait_for("!document.querySelector('#access-secret-dialog').open && !document.querySelector('#add-access').disabled")
                        checks.append('HOME and OPERATOR submit with irrelevant sample fields disabled')
                        phase = 'lost receipt'
                        await page.evaluate("document.querySelector('#add-access').click()")
                        await page.wait_for("document.querySelector('#access-dialog').open")
                        before = await page.evaluate('window.__credentialPosts.length')
                        await page.evaluate("(()=>{window.__loseIssueReceipt=true;const form=document.querySelector('#access-dialog form');form.querySelector('input[type=text]').value='Synthetic lost receipt';form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("document.querySelector('.access-receipts').textContent.includes('Aplicado por el backend') && !document.querySelector('#add-access').disabled")
                        await click_text(page, '#access-view button', 'Actualizar accesos')
                        await page.wait_for("!document.querySelector('#add-access').disabled")
                        assert await page.evaluate('window.__credentialPosts.length==='+str(before+1)+" && !document.querySelector('#access-secret-dialog').open")
                        assert sum(row['subject'] == 'Synthetic lost receipt' for row in authority.commands.catalog(root)['rows']) == 1
                        checks.append('lost JSON receipt reconciles without second POST or secret')
                        phase = 'voice selection and native retirement before import'
                        await page.evaluate("document.querySelector('#app-nav [data-view=voices]').click()")
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===3")
                        await click_text(page, '#voices-content button', 'Revisar muestras')
                        for identity in identities[:2]:
                            await page.evaluate('document.querySelector('+json.dumps('[data-capture-id="'+identity+'"] input')+').click()')
                        await page.evaluate("document.querySelector('[data-capture-id] button').click()")
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2")
                        await page.evaluate("window.__player=document.querySelector('#voice-dialog audio');window.__player.muted=true;window.__player.play();for(const input of document.querySelectorAll('.voice-evaluation-selection input')){input.checked=true;input.dispatchEvent(new Event('change'))}window.__holdEmpty=true")
                        await click_text(page, '.voice-evaluation-selection button', 'Preparar acceso de evaluación')
                        await page.wait_for('window.__emptyCallbacks.length>0')
                        assert await page.evaluate("document.querySelector('#access-view').hidden && !document.querySelector('#access-dialog').open")
                        await page.evaluate("window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(resolve=>resolve())")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('.access-sample').length===2")
                        imported = await page.evaluate("[...document.querySelectorAll('.access-sample')].map(row=>row.querySelector('input').value)")
                        assert set(imported) == set(identities[:2])
                        await screenshot(page, EVIDENCE/'evaluation-scope-dialog.png')
                        phase = 'issue exact evaluator grant'
                        await page.evaluate("(()=>{const form=document.querySelector('#access-dialog form');form.querySelector('input[type=text]').value='Synthetic evaluator access';const rows=[...form.querySelectorAll('.access-sample')];for(const row of rows){const id=row.querySelector('input').value;for(const mode of row.querySelectorAll('input[type=checkbox]'))mode.checked=id==="+json.dumps(identities[0])+"||mode.value==='audio_preview'}form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("document.querySelector('#access-secret-dialog').open")
                        evaluator_token = await page.evaluate("document.querySelector('#access-secret-dialog input').value")
                        assert isinstance(evaluator_token, str) and evaluator_token.startswith('kza_')
                        granted = next(row for row in authority.commands.catalog(root)['rows'] if row['subject'] == 'Synthetic evaluator access')
                        assert {row['fingerprint_id']: row['modalities'] for row in granted['samples']} == {
                            identities[0]: ['audio_preview', 'transcript_full'], identities[1]: ['audio_preview']}
                        await escape(page)
                        await page.evaluate("document.querySelector('#logout').click()")
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        checks.append('two real captures imported after native emptied; exact per-sample modalities issued')
                        phase = 'evaluator automatic manifest and audio'
                        evaluator_start = len(requests)
                        await login(page, evaluator_token, 'Evaluación temporal')
                        evaluator_token = None
                        await page.wait_for("!document.querySelector('#evaluation-view').hidden && document.querySelector('#evaluation-fingerprint')?.options.length===2 && !document.querySelector('#evaluation-fingerprint').disabled")
                        assert not any(row['path'].startswith(('/api/home', '/api/voices', '/api/admin/users', '/api/conversations')) for row in requests[evaluator_start:])
                        await page.evaluate("(()=>{const select=document.querySelector('#evaluation-fingerprint');select.value="+json.dumps(identities[0])+";select.dispatchEvent(new Event('change'))})()")
                        await page.wait_for("!document.querySelector('#evaluation-modality').disabled && document.querySelector('#evaluation-modality').options.length===2")
                        await page.evaluate("(()=>{const mode=document.querySelector('#evaluation-modality');mode.value='audio_preview';mode.dispatchEvent(new Event('change'))})()")
                        await page.wait_for("!document.querySelector('#evaluation-content button[type=submit]').disabled")
                        await page.evaluate("document.querySelector('#evaluation-content form').requestSubmit()")
                        await page.wait_for("!!document.querySelector('[data-evaluation-sample] audio')")
                        await page.evaluate("window.__player=document.querySelector('[data-evaluation-sample] audio');window.__player.muted=true;window.__player.play()")
                        await page.wait_for('window.__player.readyState>=2 && !window.__player.paused')
                        assert await page.evaluate("!document.querySelector('.evaluation-transcript') && window.__urls.size===1")
                        phase = 'modality change holds real media retirement'
                        await page.evaluate("window.__holdEmpty=true;const mode=document.querySelector('#evaluation-modality');mode.value='transcript_full';mode.dispatchEvent(new Event('change'))")
                        await page.wait_for('window.__emptyCallbacks.length>0')
                        assert await page.evaluate("!document.querySelector('[data-evaluation-sample] audio') && document.querySelector('#evaluation-content button[type=submit]').disabled")
                        await page.evaluate("window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(resolve=>resolve())")
                        await page.wait_for("!document.querySelector('#evaluation-content button[type=submit]').disabled && document.querySelector('#evaluation-modality').value==='transcript_full'")
                        await page.evaluate("document.querySelector('#evaluation-content form').requestSubmit()")
                        await page.wait_for("document.querySelector('.evaluation-transcript')?.textContent.startsWith('Synthetic evaluator phrase')")
                        assert await page.evaluate("!document.querySelector('[data-evaluation-sample] audio') && window.__urls.size===0")
                        checks.append('audio-only payload; modality change drains native audio before separate transcript query')
                        await screenshot(page, EVIDENCE/'evaluation-desktop.png')
                        phase = 'mobile evaluator layout'
                        for width in (390, 320):
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': True})
                            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                            await screenshot(page, EVIDENCE/f'evaluation-{width}.png', full=False)
                        phase = 'visibility retirement and no automatic payload replay'
                        reads_before = await page.evaluate('window.__evaluationReads')
                        await page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'))")
                        await page.wait_for("!document.querySelector('.evaluation-transcript') && !document.querySelector('[data-evaluation-sample] audio') && window.__urls.size===0")
                        await page.evaluate("delete document.hidden;document.dispatchEvent(new Event('visibilitychange'))")
                        await page.wait_for("document.querySelector('#evaluation-fingerprint')?.options.length===2 && !document.querySelector('#evaluation-fingerprint').disabled")
                        assert await page.evaluate('window.__evaluationReads==='+str(reads_before)+" && !document.querySelector('.evaluation-transcript') && !document.querySelector('[data-evaluation-sample] audio')")
                        await page.evaluate("(()=>{const select=document.querySelector('#evaluation-fingerprint');select.value="+json.dumps(identities[1])+";select.dispatchEvent(new Event('change'))})()")
                        await page.wait_for("!document.querySelector('#evaluation-modality').disabled && document.querySelector('#evaluation-modality').options.length===1")
                        assert await page.evaluate("document.querySelector('#evaluation-modality').value==='audio_preview'")
                        phase = 'native revocation during playback'
                        await page.evaluate("document.querySelector('#evaluation-content form').requestSubmit()")
                        await page.wait_for("!!document.querySelector('[data-evaluation-sample] audio')")
                        await page.evaluate("window.__player=document.querySelector('[data-evaluation-sample] audio');window.__player.muted=true;window.__player.play()")
                        await page.wait_for('window.__player.readyState>=2 && !window.__player.paused')
                        await page.evaluate('window.__holdEmpty=true')
                        revoking = asyncio.create_task(authority.revoke_credential(root, granted['id']))
                        await page.wait_for("window.__emptyCallbacks.length>0 && !document.querySelector('[data-evaluation-sample] audio') && window.__urls.size===0")
                        assert await page.evaluate('window.__player.paused && !window.__player.isConnected')
                        await page.evaluate('window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(resolve=>resolve())')
                        assert await asyncio.wait_for(revoking, 5)
                        checks.append('native credential revocation removes playback and Blob while decoder retirement is held')
                        await page.evaluate("document.querySelector('#logout').click()")
                        await page.wait_for("document.querySelector('#login-error').textContent==='La sesión ya no está vigente.' && !document.querySelector('#token').disabled")
                        assert await page.evaluate("!document.querySelector('#evaluation-content').children.length && window.__urls.size===0 && document.querySelector('#token').value===''")
                        assert not any(row['path'].startswith(('/api/home', '/api/voices', '/api/admin/users', '/api/conversations')) for row in requests[evaluator_start:])
                        checks.append('mobile 390/320; hidden view clears payload, show loads metadata only; audio-only second grant; logout clears')
                        await login(page, operator.token, 'Administración')
                        await page.wait_for("document.querySelector('#add-access') && !document.querySelector('#add-access').disabled")
                        assert await page.evaluate("!document.querySelector('#evaluation-content').children.length")
                        checks.append('logout confirms an already revoked session without blocking a new authorized login')
                        assert not page.errors, 'Unexpected browser runtime errors'
                        result = {'passed': True, 'checks': checks, 'request_count': len(requests),
                            'credential_posts': await page.evaluate('window.__credentialPosts.length'),
                            'runtime_error_count': len(page.errors),
                            'screenshots': ['access-secret-masked.png', 'evaluation-390.png', 'evaluation-320.png']}
                        (EVIDENCE/'result.json').write_text(json.dumps(result, indent=2)+'\n')
                        print(json.dumps(result))
            finally:
                if browser is not None and browser.poll() is None:
                    browser.terminate(); await asyncio.to_thread(browser.wait, 10)
                server.should_exit = True
                await serving
                await dashboard.stop()


if __name__ == '__main__':
    asyncio.run(main())
