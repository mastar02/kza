"""Actual NEXO voice UI, individual login, synthetic PCM and native media drain."""

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
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
from src.users.voice_registry import build_voice_registry
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_typed_controls import Page
from tests.mocks.ambient_graph import Clock
from tests.mocks.voice_registry import AUDIO, Encoder
from tests.mocks.memory_sources import memory_source


INSTRUMENT = r"""(() => {
  window.__voiceSent=[];window.__voiceSockets=0;window.__emptyCallbacks=[];window.__holdEmpty=false;window.__urls=new Set();
  const NativeSocket=WebSocket;
  window.WebSocket=class extends NativeSocket { constructor(...args){super(...args);window.__voiceSockets++;}
    send(value){window.__voiceSent.push(JSON.parse(value));return super.send(value);} };
  const add=HTMLMediaElement.prototype.addEventListener,remove=HTMLMediaElement.prototype.removeEventListener,wrappers=new WeakMap();
  HTMLMediaElement.prototype.addEventListener=function(type,callback,options){
    if(type==='emptied'){const wrapped=event=>window.__holdEmpty?window.__emptyCallbacks.push(()=>callback(event)):callback(event);wrappers.set(callback,wrapped);return add.call(this,type,wrapped,options);}
    return add.call(this,type,callback,options);};
  HTMLMediaElement.prototype.removeEventListener=function(type,callback,options){return remove.call(this,type,wrappers.get(callback)||callback,options);};
  const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
  URL.createObjectURL=blob=>{const value=create(blob);window.__urls.add(value);return value;};URL.revokeObjectURL=value=>{window.__urls.delete(value);return revoke(value);};
  const fetch=window.fetch.bind(window);window.__confirmationIds=[];
  window.fetch=async(path,options={})=>{const response=await fetch(path,options);
    if(path==='/api/voices/profiles' && options.method==='POST'){
      window.__confirmationIds.push(JSON.parse(options.body).request_id);
      if(window.__loseConfirmReceipt){window.__loseConfirmReceipt=false;return new Proxy(response,{get(target,key){if(key==='arrayBuffer')return async()=>{await target.arrayBuffer();throw Error('synthetic lost confirmation receipt');};const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});}
    }return response;};
})()"""


async def main() -> None:
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-nexo-voice-root'
    evidence = Path('data/autonomous-roadmap-20261102/t047/voices-browser')
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    class DiagnosticPage(Page):
        async def wait_for(self, expression):
            try:
                await super().wait_for(expression)
            except AssertionError:
                diagnostic = await self.evaluate("({text:document.body.innerText,html:document.querySelector('#voice-dialog')?.innerHTML,ready:document.readyState})")
                (evidence/'failure.json').write_text(json.dumps({'condition':expression, 'document':diagnostic, 'errors':self.errors, 'requests':requests}, indent=2)+'\n')
                await screenshot(self, evidence/'failure.png')
                raise
    browser_data = Path.home() / 'snap/chromium/common'
    browser_data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kza-nexo-voices-', dir=browser_data) as temporary:
        directory = Path(temporary)
        async with AsyncExitStack() as stack:
            authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=180)
            stack.push_async_callback(authority.aclose)
            await authority.initialize()
            root = authority.resolve('synthetic-nexo-voice-root')
            operator = await authority.issue_credential(root, subject='synthetic-voice-operator', role=DashboardRole.OPERATOR, expires_at=time.time()+240)
            home = await authority.issue_credential(root, subject='synthetic-voice-home', role=DashboardRole.HOME, expires_at=time.time()+240)
            clock = Clock(time.time())
            privacy = PrivacyServices(str(directory/'privacy.db'), clock=clock)
            stack.push_async_callback(privacy.aclose)
            await privacy.acall(privacy.initialize)
            owner = ErasureCoordinator(privacy)
            stack.push_async_callback(owner.aclose)
            bus = await build_live_services({}, privacy=privacy, owner=owner, shutdown=stack)
            voices = await build_voice_registry(privacy=privacy, owner=owner, shutdown=stack)
            encoder = Encoder(); voices.register_encoder(encoder)
            for index in range(4):
                source = memory_source(now=clock.now, ttl=180)
                assert voices.observe(encoder, np.tile(AUDIO, 3), (source,), transcript=f'Synthetic NEXO phrase {index}', room_id='Synthetic room')
                async with asyncio.timeout(5):
                    while voices.resource_status()['pending'] or voices.resource_status()['operations']:
                        await asyncio.sleep(.001)
            captures = (await voices.acall(voices.ledger.snapshot))['fingerprints']
            identities = [row['id'] for row in captures]
            dashboard = DashboardAPI(authority=authority, event_bus=bus, voice_registry=voices)
            requests = []
            @dashboard.app.middleware('http')
            async def record(request, call_next):
                response = await call_next(request)
                requests.append({'method':request.method, 'path':request.url.path, 'status':response.status_code})
                return response
            listener = socket.socket(); listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            browser = None
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        await asyncio.sleep(.02)
                profile = directory / 'chromium'
                with (evidence/'chromium.log').open('w') as output:
                    browser = subprocess.Popen(['chromium', '--headless', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-first-run', '--disable-background-networking', '--autoplay-policy=no-user-gesture-required', '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank'], stdout=output, stderr=output)
                    async with asyncio.timeout(15):
                        while not (profile/'DevToolsActivePort').exists():
                            assert browser.poll() is None
                            await asyncio.sleep(.05)
                    debugging = int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                    async with aiohttp.ClientSession() as client:
                        async with client.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            target = next(item for item in await response.json() if item['type']=='page')
                    async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                        page = DiagnosticPage(connection)
                        await page.command('Runtime.enable'); await page.command('Page.enable')
                        await page.command('Page.addScriptToEvaluateOnNewDocument', {'source':INSTRUMENT})
                        await page.command('Emulation.setDeviceMetricsOverride', {'width':1425, 'height':1180, 'deviceScaleFactor':1, 'mobile':False})
                        await page.command('Page.navigate', {'url':f'http://127.0.0.1:{port}/#voces'})
                        await page.wait_for("document.querySelector('#login-error')?.textContent.length>0")
                        await login(page, operator.token, 'Administración')
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===4 && document.querySelector('#voice-connection').textContent.includes('vigente')")
                        assert await page.evaluate("window.__voiceSockets===1 && window.__voiceSent[0].version===3 && !document.querySelector('#voices-content').textContent.includes('ilustrativos')")
                        assert not any(row['path'] in {'/api/conversations','/api/users'} for row in requests)
                        await screenshot(page, evidence/'voices-desktop.png')
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Revisar muestras').click()")
                        await page.evaluate("document.querySelector('[data-capture-id] button').click()")
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2 && document.querySelector('#voice-dialog').textContent.includes('Synthetic NEXO phrase')")
                        await page.evaluate("window.__player=document.querySelector('#voice-dialog audio');window.__player.muted=true;window.__player.play()")
                        assert await page.evaluate("!window.__player.paused && window.__urls.size===1")
                        for identity in identities[:3]:
                            await page.evaluate('document.querySelector('+json.dumps(f'[data-capture-id="{identity}"] input')+').click()')
                        await page.evaluate("(()=>{window.__loseConfirmReceipt=true;const input=document.querySelector('#voice-dialog input[type=text]');input.value='Synthetic NEXO profile';input.dispatchEvent(new Event('input'));document.querySelector('#voice-dialog form').requestSubmit()})()")
                        await page.wait_for("document.querySelector('#voice-dialog .identity-form-error').textContent.includes('Actualizá')")
                        assert len((await voices.acall(voices.ledger.snapshot))['profiles']) == 1
                        await page.evaluate("document.querySelector('#voice-dialog form').requestSubmit()")
                        assert await page.evaluate('window.__confirmationIds.length===1')
                        await page.evaluate("[...document.querySelectorAll('#voice-dialog>button')].find(button=>button.textContent==='Actualizar').click()")
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===4 && !document.querySelector('#voice-dialog .identity-form-error').textContent")
                        for identity in identities[:3]:
                            await page.evaluate('document.querySelector('+json.dumps(f'[data-capture-id="{identity}"] input')+').click()')
                        await page.evaluate("(()=>{const input=document.querySelector('#voice-dialog input[type=text]');input.value='Synthetic NEXO profile';input.dispatchEvent(new Event('input'));document.querySelector('#voice-dialog form').requestSubmit()})()")
                        await page.wait_for('window.__confirmationIds.length===2')
                        assert await page.evaluate('window.__confirmationIds[0]===window.__confirmationIds[1]')
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Synthetic NEXO profile')")
                        assert len((await voices.acall(voices.ledger.snapshot))['profiles']) == 1
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===4")
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        await screenshot(page, evidence/'voices-profile.png')
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Revisar muestras').click()")
                        await page.evaluate("document.querySelector('[data-capture-id] button').click()")
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2")
                        await screenshot(page, evidence/'voices-review.png')
                        await page.evaluate("window.__player=document.querySelector('#voice-dialog audio');window.__holdEmpty=true;document.querySelector('#app-nav [data-view=home]').click();document.querySelector('#app-nav [data-view=voices]').click()")
                        await page.wait_for('window.__emptyCallbacks.length>0')
                        assert await page.evaluate("window.__voiceSockets===1 && !document.querySelector('#voices-content').children.length && !document.querySelector('#voice-dialog').open && !document.querySelector('#voice-dialog').children.length && window.__urls.size===0 && window.__player.paused && !window.__player.hasAttribute('src')")
                        await page.evaluate("window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(callback=>callback())")
                        await page.wait_for("window.__voiceSockets===2 && document.querySelectorAll('[data-capture-id]').length===4")
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Revisar muestras').click();document.querySelector('[data-capture-id] button').click()")
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2")
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        await page.wait_for('window.__urls.size===0')
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Revisar muestras').click();document.querySelector('[data-capture-id] input').click();[...document.querySelectorAll('#voice-dialog button')].find(button=>button.textContent==='Dejar como desconocido').click()")
                        await page.wait_for("document.querySelectorAll('[data-voice-group]').length===2 && document.querySelectorAll('[data-capture-id]').length===4")
                        corrected = (await voices.acall(voices.ledger.snapshot))['fingerprints']
                        assert sum(row['group_reason']=='manual_unknown' for row in corrected) == 1
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        for width in [390, 320]:
                            await page.command('Emulation.setDeviceMetricsOverride', {'width':width, 'height':844, 'deviceScaleFactor':1, 'mobile':True})
                            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                            await screenshot(page, evidence/f'voices-{width}.png', full=False)
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Borrar perfil').click()")
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Aún no hay')")
                        assert not (await voices.acall(voices.ledger.snapshot))['profiles']
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Actualizar').click()")
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===4")
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Revisar muestras').click()")
                        for identity in identities[:3]:
                            await page.evaluate('document.querySelector('+json.dumps(f'[data-capture-id="{identity}"] input')+').click()')
                        await page.evaluate("(()=>{const input=document.querySelector('#voice-dialog input[type=text]');input.value='Synthetic NEXO profile';input.dispatchEvent(new Event('input'));document.querySelector('#voice-dialog form').requestSubmit()})()")
                        await page.wait_for('window.__confirmationIds.length===3')
                        assert await page.evaluate('window.__confirmationIds[2]!==window.__confirmationIds[1]')
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Synthetic NEXO profile')")
                        assert len((await voices.acall(voices.ledger.snapshot))['profiles']) == 1
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        await page.evaluate("document.querySelector('#logout').click()")
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        marker = len(requests)
                        await login(page, home.token, 'Acceso de hogar')
                        assert await page.evaluate("!document.querySelector('#app-nav [data-view=voices]') && !document.querySelector('#voices-content').children.length && !document.querySelector('#voice-dialog').children.length && window.__urls.size===0")
                        assert not any(row['path'].startswith('/api/voices') for row in requests[marker:])
                        assert not page.errors, page.errors
                        result = {'passed':True, 'individual_http_login':True, 'synthetic_pcm_only':True, 'native_media_drain':True, 'voice_control_only':True, 'lost_receipt_single_profile':True, 'explicit_recreation_after_delete':True, 'requests':requests, 'screenshots':['voices-desktop.png','voices-profile.png','voices-review.png','voices-390.png','voices-320.png']}
                        (evidence/'result.json').write_text(json.dumps(result, indent=2)+'\n')
                        print(json.dumps(result))
            finally:
                if browser is not None and browser.poll() is None:
                    browser.terminate(); await asyncio.to_thread(browser.wait, 10)
                server.should_exit = True
                await serving
                await dashboard.stop()


if __name__ == '__main__':
    asyncio.run(main())
