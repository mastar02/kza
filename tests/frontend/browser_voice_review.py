"""Real Chromium voice review, PCM playback and exact retirement acknowledgements."""

import asyncio
import argparse
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
from src.dashboard import system_monitor
from src.dashboard.live_erasure import build_live_services
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.users.voice_registry import build_voice_registry
from tests.frontend.browser_typed_controls import Page
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import AUDIO,Encoder,idle


async def main(*,browser_binary='chromium'):
    """Exercise actual UI events, HTTP/v3 WebSocket and native HTML media teardown."""
    token = 'synthetic-voice-review-browser'
    os.environ['KZA_DASHBOARD_TOKEN'] = token
    browser_data = Path.home()/'snap/chromium/common'
    browser_data.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kza-voice-browser-',dir=browser_data) as work:
        directory = Path(work)
        async with AsyncExitStack() as stack:
            clock = Clock(time.time())
            privacy = PrivacyServices(str(directory/'privacy.db'),clock=clock)
            stack.push_async_callback(privacy.aclose)
            await privacy.acall(privacy.initialize)
            owner = ErasureCoordinator(privacy)
            stack.push_async_callback(owner.aclose)
            bus = await build_live_services({},privacy=privacy,owner=owner,shutdown=stack)
            voices = await build_voice_registry(privacy=privacy,owner=owner,shutdown=stack)
            encoder = Encoder()
            voices.register_encoder(encoder)
            async def sample(text,ttl=60):
                source = memory_source(now=clock.now,ttl=ttl)
                assert voices.observe(encoder,np.tile(AUDIO,3),(source,),transcript=text,room_id='Synthetic listening room')
                await idle(voices)
                rows = (await voices.acall(voices.ledger.snapshot))['fingerprints']
                identity = next(row['id'] for row in rows if source.capture_id in row['capture_ids'])
                return source,identity
            samples = [await sample('Synthetic reviewed phrase '+str(index)) for index in range(3)]
            extra,extra_id = await sample('Synthetic audio retirement phrase')
            dashboard = DashboardAPI(event_bus=bus,voice_registry=voices,observability_use_mocks=False)
            original_gpu,original_services = system_monitor.gpu_snapshot,system_monitor.services_snapshot
            requests = []
            system_monitor.gpu_snapshot,system_monitor.services_snapshot = lambda:None,lambda:None
            @dashboard.app.middleware('http')
            async def cookie(request,call_next):
                response = await call_next(request)
                requests.append({'method':request.method,'path':request.url.path,'status':response.status_code})
                if request.url.path=='/':
                    response.set_cookie('kza_dashboard_token',token,httponly=True,samesite='strict')
                return response
            listener = socket.socket()
            listener.bind(('127.0.0.1',0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(dashboard.app,log_level='warning'))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            browser = None
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        await asyncio.sleep(.02)
                profile = directory/'chromium'
                with (directory/'chromium.log').open('w') as log:
                    browser = subprocess.Popen([browser_binary,'--headless','--disable-gpu','--no-first-run',
                        '--remote-debugging-port=0',f'--user-data-dir={profile}',f'http://127.0.0.1:{port}/'],stdout=log,stderr=log)
                    async with asyncio.timeout(15):
                        while not (profile/'DevToolsActivePort').exists():
                            if browser.poll() is not None:
                                raise RuntimeError((directory/'chromium.log').read_text()[:2000])
                            await asyncio.sleep(.05)
                    debugging = int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            target = next(item for item in await response.json() if item['type']=='page')
                    async with connect(target['webSocketDebuggerUrl'],max_size=None) as connection:
                        page = Page(connection)
                        await page.command('Runtime.enable')
                        await page.command('Network.enable')
                        await page.command('Network.setCookie',{'name':'kza_dashboard_token','value':token,
                            'url':f'http://127.0.0.1:{port}/','httpOnly':True,'sameSite':'Strict'})
                        await page.command('Network.setBlockedURLs',{'urls':['https://unpkg.com/*','https://fonts.googleapis.com/*','https://fonts.gstatic.com/*']})
                        await page.command('Page.navigate',{'url':f'http://127.0.0.1:{port}/obs/'})
                        try:
                            await page.wait_for("document.querySelector('#view-live') && window.__kzaLiveWs?.readyState===1")
                        except AssertionError:
                            print(json.dumps({'diagnostic':'voice_browser_start','requests':requests[-12:],
                                'client':await page.evaluate("({ready:document.readyState,react:typeof React,babel:typeof Babel,login:!!document.querySelector('input[type=password]'),live:!!document.querySelector('#view-live'),socket:window.__kzaLiveWs?.readyState})"),
                                'errors':page.errors}))
                            raise
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(item=>item.textContent.includes('Voces')).click()")
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===4")
                        assert await page.evaluate("document.querySelectorAll('[data-voice-group-id]').length===1")
                        assert not await page.evaluate("document.querySelector('#view-users').textContent.includes('Embedding space')")
                        await page.evaluate('''(() => {
                          window.__voicePlayers=[]; window.__voiceEmptyCallbacks=[]; window.__holdVoiceEmpty=false;
                          window.__voiceUrls=new Set(); const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
                          URL.createObjectURL=b=>{const value=create(b);window.__voiceUrls.add(value);return value;};
                          URL.revokeObjectURL=value=>{window.__voiceUrls.delete(value);return revoke(value);};
                          const add=HTMLMediaElement.prototype.addEventListener,remove=HTMLMediaElement.prototype.removeEventListener;
                          const wrappers=new WeakMap();
                          HTMLMediaElement.prototype.addEventListener=function(type,callback,options) {
                            if(type==='emptied') { const wrapped=event=>window.__holdVoiceEmpty ? window.__voiceEmptyCallbacks.push(()=>callback(event)) : callback(event);
                              wrappers.set(callback,wrapped);return add.call(this,type,wrapped,options); }
                            return add.call(this,type,callback,options);
                          };
                          HTMLMediaElement.prototype.removeEventListener=function(type,callback,options) {return remove.call(this,type,wrappers.get(callback)||callback,options);};
                        })()''')
                        identities = [identity for _,identity in samples]
                        await page.evaluate('''(() => { for(const id of %s) document.querySelector(`[data-capture-id="${id}"] input`).click();
                          const input=document.querySelector('[aria-label="Nombre del perfil"]');input.value='Synthetic browser profile';input.dispatchEvent(new Event('input'));
                          [...document.querySelectorAll('button')].find(button=>button.textContent==='Confirmar perfil').click(); })()'''%json.dumps(identities))
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Synthetic browser profile')")
                        profiles = (await voices.acall(voices.ledger.snapshot))['profiles']
                        assert len(profiles)==1
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===4 && window.__kzaVoiceState.captures.status()==='real'")
                        async def correct(identities,button,group_count):
                            await page.evaluate('''(() => { for(const id of %s) document.querySelector(`[data-capture-id="${id}"] input`).click();
                              [...document.querySelectorAll('button')].find(item=>item.textContent===%s).click(); })()'''%(json.dumps(identities),json.dumps(button)))
                            await page.wait_for("document.querySelectorAll('[data-voice-group-id]').length==="+str(group_count)+" && window.__kzaVoiceState.captures.status()==='real'")
                        await correct([identities[0]],'Separar muestras seleccionadas',2)
                        await correct(identities[:2],'Unir grupos seleccionados',1)
                        await correct([identities[2]],'Dejar como desconocido',2)
                        corrected = {row['id']:row for row in (await voices.acall(voices.ledger.snapshot))['fingerprints']}
                        assert corrected[identities[2]]['group_reason']=='manual_unknown'
                        assert corrected[identities[0]]['group_state']=='manual'
                        assert await page.evaluate("document.body.textContent.includes('Synthetic browser profile')")
                        async def listen(identity,text):
                            try:
                                await page.wait_for('!!document.querySelector('+json.dumps('[data-capture-id="'+identity+'"] button')+')')
                            except AssertionError:
                                print(json.dumps({'diagnostic':'voice_view_capture_read','requests':requests[-12:],
                                    'expected':identity,'client':await page.evaluate("({status:window.__kzaVoiceState.captures.status(),ids:window.__kzaVoiceState.captures.snapshot().map(row=>row.id),dom:[...document.querySelectorAll('[data-capture-id]')].map(row=>({id:row.dataset.captureId,buttons:row.querySelectorAll('button').length})),profiles:window.__kzaVoiceState.profiles.snapshot().length})")}))
                                raise
                            await page.evaluate('document.querySelector('+json.dumps('[data-capture-id="'+identity+'"] button')+').click()')
                            await page.wait_for('document.querySelector(\'[aria-label="Escuchar muestra"]\').textContent.includes('+json.dumps(text)+') && !!document.querySelector("audio")')
                            played = await page.command('Runtime.evaluate',{'expression':"(async()=>{const audio=document.querySelector('audio');audio.muted=true;window.__voicePlayers.push(audio);window.__lastPreviewUrl=audio.src;await audio.play();})()",
                                'awaitPromise':True,'returnByValue':True,'userGesture':True})
                            assert 'exceptionDetails' not in played,played
                            await page.wait_for("document.querySelector('audio')?.readyState>=2")
                            assert await page.evaluate("document.querySelector('[aria-label=\"Escuchar muestra\"]').textContent.includes('Synthetic listening room')")
                            assert await page.evaluate("(()=>{const element=document.querySelector('audio');window.__kzaVoiceState.review.setStatus('real');return document.querySelector('audio')===element;})()")
                        await listen(extra_id,'Synthetic audio retirement phrase')
                        await page.evaluate('window.__holdVoiceEmpty=true')
                        await owner.request(extra.conversation_id,MemoryReviewer('synthetic-admin'))
                        await page.wait_for("!document.body.textContent.includes('Synthetic audio retirement phrase') && !document.querySelector('audio') && window.__voiceUrls.size===0 && window.__voiceEmptyCallbacks.length>0")
                        player_states = await page.evaluate("window.__voicePlayers.map(audio=>({paused:audio.paused,connected:audio.isConnected,hasCurrent:!!audio.currentSrc,ready:audio.readyState,network:audio.networkState,hasSrc:audio.hasAttribute('src'),buffers:audio.buffered.length,emptyDuration:Number.isNaN(audio.duration)}))")
                        if not all(item['paused'] and not item['connected'] and not item['hasSrc'] and item['buffers']==0 and item['emptyDuration'] and item['ready']==0 and item['network']==0 for item in player_states):
                            print(json.dumps({'diagnostic':'voice_media_teardown','players':player_states}))
                        assert all(item['paused'] and not item['connected'] and not item['hasSrc'] and item['buffers']==0 and item['emptyDuration'] and item['ready']==0 and item['network']==0 for item in player_states)
                        assert await page.evaluate("fetch(window.__lastPreviewUrl).then(()=>false,()=>true)")
                        assert await bus._retention.receipts.pending(extra.conversation_id)>0
                        await page.evaluate('window.__holdVoiceEmpty=false;for(const callback of window.__voiceEmptyCallbacks.splice(0))callback()')
                        try:
                            async with asyncio.timeout(3):
                                while await bus._retention.receipts.pending(extra.conversation_id):
                                    await asyncio.sleep(.01)
                        except TimeoutError:
                            print(json.dumps({'diagnostic':'voice_pending_after_empty',
                                'client':await page.evaluate("({pending:window.__kzaVoiceState.review.pendingRetirements(),token:window.__kzaVoiceState.review.deliveryToken(),socket:window.__kzaLiveWs?.readyState,players:window.__voicePlayers.map(audio=>({paused:audio.paused,network:audio.networkState,ready:audio.readyState,provider:audio.srcObject===null,child:!!audio.querySelector('source'),buffers:audio.buffered.length,emptyDuration:Number.isNaN(audio.duration)}))})")}))
                            raise
                        assert await page.evaluate("document.body.textContent.includes('Synthetic browser profile')")
                        clock.now+=48*3600
                        await voices.acall(voices.ledger.prune)
                        await page.evaluate("[...document.querySelectorAll('button')].find(button=>button.textContent==='Actualizar').click()")
                        await page.wait_for("window.__kzaVoiceState.profiles.status()==='real' && window.__kzaVoiceState.captures.status()==='real' && document.body.textContent.includes('Synthetic browser profile') && !document.querySelector('[data-capture-id]')")
                        await page.evaluate("[...document.querySelectorAll('button')].find(button=>button.textContent==='Borrar perfil').click()")
                        await page.wait_for("!document.body.textContent.includes('Synthetic browser profile') && window.__kzaVoiceState.profiles.status()==='real'")
                        assert not (await voices.acall(voices.ledger.snapshot))['profiles']
                        expiring,expiry_id = await sample('Synthetic independent idle expiry',ttl=3)
                        await page.evaluate("[...document.querySelectorAll('button')].find(button=>button.textContent==='Actualizar').click()")
                        await listen(expiry_id,'Synthetic independent idle expiry')
                        await page.wait_for("!document.body.textContent.includes('Synthetic independent idle expiry') && !document.querySelector('audio') && window.__voiceUrls.size===0")
                        assert voices.privacy.sources_available((expiring,))  # Frozen server clock: retirement was client-driven.
                        try:
                            async with asyncio.timeout(3):
                                while await bus._retention.receipts.pending(expiring.conversation_id):
                                    await asyncio.sleep(.01)
                        except TimeoutError:
                            print(json.dumps({'diagnostic':'voice_idle_ack','client':await page.evaluate("({status:window.__kzaVoiceState.review.status(),pending:window.__kzaVoiceState.review.retirementStatus(),captureToken:window.__kzaVoiceState.captures.deliveryToken(),reviewToken:window.__kzaVoiceState.review.deliveryToken(),socket:window.__kzaLiveWs?.readyState})")}))
                            raise
                        leaving,leaving_id = await sample('Synthetic leaving voice view')
                        await page.evaluate("[...document.querySelectorAll('button')].find(button=>button.textContent==='Actualizar').click()")
                        await listen(leaving_id,'Synthetic leaving voice view')
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(item=>item.textContent.includes('Live')).click()")
                        await page.wait_for("document.querySelector('#view-live') && window.__voiceUrls.size===0 && window.__kzaVoiceState.review.snapshot().length===0")
                        async with asyncio.timeout(3):
                            while await bus._retention.receipts.pending(leaving.conversation_id):
                                await asyncio.sleep(.01)
                        _,auth_id = await sample('Synthetic voice session loss')
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(item=>item.textContent.includes('Voces')).click()")
                        await listen(auth_id,'Synthetic voice session loss')
                        os.environ['KZA_DASHBOARD_TOKEN']='synthetic-rotated-voice-token'
                        await page.wait_for("!document.querySelector('#view-users') && !document.body.textContent.includes('Synthetic voice session loss') && window.__voiceUrls.size===0")
                        assert await page.evaluate("window.__voicePlayers.every(audio=>audio.paused && !audio.hasAttribute('src') && audio.readyState===0 && audio.buffered.length===0 && Number.isNaN(audio.duration))")
                        assert page.errors==[]
                        print(json.dumps({'passed':True,'actual_chromium_http_v3_ws_react_and_native_audio':True,
                            'manual_profile_confirmation_and_revision_delete':True,'review_uses_real_transcript_and_room':True,
                            'suggested_group_then_ui_split_merge_and_explicit_unknown':True,
                            'raw_group_corrections_leave_permanent_profile_names_unchanged':True,
                            'permanent_profile_survives_raw_expiry':True,'raw_retirement_waits_for_actual_emptied_callback':True,
                            'urls_cache_dom_and_player_source_retired':True,'idle_expiry_with_frozen_server_clock':True,
                            'navigation_and_session_loss_stop_audio_and_clear_text':True,'external_bootstrap_assets_blocked':True,
                            'javascript_exceptions':len(page.errors),'synthetic_encoder_and_sources':True,
                            'browser_binary':browser_binary,
                            'grouping_calibration_all_browser_copies_or_hardware_acceptance':False}))
            finally:
                if browser is not None:
                    browser.terminate()
                    await asyncio.to_thread(browser.wait,timeout=10)
                server.should_exit=True
                await serving
                system_monitor.gpu_snapshot,system_monitor.services_snapshot = original_gpu,original_services


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser-binary',default='chromium')
    asyncio.run(main(browser_binary=parser.parse_args().browser_binary))
