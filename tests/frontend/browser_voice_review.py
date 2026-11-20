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
from urllib.parse import urlsplit

import aiohttp
import numpy as np
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import (
    DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample, source_digest,
)
from src.dashboard import system_monitor
from src.dashboard.live_erasure import build_live_services
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.users.voice_registry import build_voice_registry
from tests.frontend.browser_support import Page, browser_binary as select_browser, browser_profile_root, evidence_directory, install_login_session, press_enter
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import AUDIO,Encoder


EVIDENCE = evidence_directory('KZA_VOICE_REVIEW_EVIDENCE','data/browser-runs/voice-review')
OBS_REVIEW = '#view-users [aria-label="Escuchar muestra"]'
NEXO_REVIEW = '#voice-dialog [aria-label="Escuchar muestra"]'
EVALUATION = '[data-evaluation-sample]'
CAPTURE_QUERY = '/api/voices/captures/query'
VOICE_FILTERS = '.voice-filters'


class VoicePage(Page):
    """Record voice request method and URL from Chromium without retaining body data."""

    def __init__(self, connection):
        super().__init__(connection)
        self.voice_requests = []

    def on_request(self, request):
        url = urlsplit(request['url'])
        if url.path.startswith('/api/voices/'):
            self.voice_requests.append((request['method'], url))


async def filter_with_keyboard(page, total):
    """Apply a filter with a native key and keep focus while the list updates."""
    checkbox = f'document.querySelector({json.dumps(VOICE_FILTERS + " input[type=checkbox]")})'
    marker = len(page.voice_requests)
    await page.evaluate(f'{checkbox}.focus()')
    for kind in ('rawKeyDown','keyUp'):
        await page.command('Input.dispatchKeyEvent', {'type':kind,'key':' ','code':'Space',
            'windowsVirtualKeyCode':32,'nativeVirtualKeyCode':32})
    await page.wait_for(f'{checkbox}?.checked && document.querySelector(".voice-filter-status")?.textContent==="{total} muestras vigentes."')
    assert await page.evaluate(f'document.activeElement==={checkbox}')
    assert [(method,url.path,url.query) for method,url in page.voice_requests[marker:]] == [('POST',CAPTURE_QUERY,'')]


async def search_voice(page, term):
    """Submit an admin search and check that it sends no transcript or review request."""
    search = f'document.querySelector({json.dumps(VOICE_FILTERS + " input[type=search]")})'
    private_locations = '''JSON.stringify([location.href,history.length,history.state,{...localStorage},{...sessionStorage}])'''
    before = await page.evaluate(private_locations)
    assert await page.evaluate('''[...document.querySelectorAll('.voice-filters input,.voice-filters select')].every(
        input=>input.getAttribute('autocomplete')==='off' && !input.hasAttribute('name') && !input.closest('form'))''')
    marker = len(page.voice_requests)
    await page.evaluate(f'{search}.focus()')
    await page.command('Input.insertText', {'text':term})
    assert len(page.voice_requests) == marker
    await press_enter(page)
    await page.wait_for('document.querySelector(".voice-filter-status")?.textContent==="1 muestra vigente."')
    assert await page.evaluate(private_locations) == before
    assert [(method,url.path,url.query) for method,url in page.voice_requests[marker:]] == [('POST', CAPTURE_QUERY, '')]


async def read_transcript(page, identity, panel, text):
    """Open one transcript and check the real HTTP and DOM contracts."""
    selector = f'[data-capture-id="{identity}"] button'
    marker = len(page.voice_requests)
    await click_button(page, selector,'Leer transcripción')
    target = f'document.querySelector({json.dumps(panel)})'
    await page.wait_for(f'{target}?.textContent.includes({json.dumps(text)}) && !{target}.querySelector("audio")')
    assert [(method,url.path,url.query) for method,url in page.voice_requests[marker:]] == [
        ('GET', f'/api/voices/captures/{identity}/transcript', '')]
    assert not await page.evaluate('''JSON.stringify([location.href,history.state,{...localStorage},{...sessionStorage}]).includes('''
        + json.dumps(text) + ')')


def obs_capture_ready(identity: str) -> str:
    """Require the specific capture and its enabled listen control."""
    buttons = json.dumps(f'#view-users [data-capture-id="{identity}"] button:not(:disabled)')
    return 'window.__kzaVoiceState?.captures.status()==="real" && ' + (
        f'[...document.querySelectorAll({buttons})].some(button=>button.textContent==="Escuchar")')


async def refresh_obs_capture(page, identity, requests):
    """Activate OBS refresh with the keyboard and require the fresh capture to render."""
    assert await page.evaluate('''(() => {
        const button=[...document.querySelectorAll('#view-users .page-head button')]
            .find(node=>node.textContent==='Actualizar');
        const filters=window.__kzaVoiceController?.captureFilters();
        if (!button || button.disabled || !filters || Object.values(filters).some(Boolean)) return false;
        button.focus();
        return document.activeElement===button;
    })()'''), 'OBS refresh must be enabled and use the default filters'
    marker, server_marker = len(page.voice_requests), len(requests)
    await press_enter(page)
    try:
        await page.wait_for(obs_capture_ready(identity))
        observed = [(method,url.path,url.query) for method,url in page.voice_requests[marker:]]
        assert ('GET','/api/voices/profiles','') in observed
        assert ('POST',CAPTURE_QUERY,'') in observed
    except AssertionError:
        print(json.dumps({'diagnostic':'obs_refresh_capture','expected':identity,'requests':requests[server_marker:],
            'client':await page.evaluate('''({
                captures:window.__kzaVoiceState?.captures.status(),
                profiles:window.__kzaVoiceState?.profiles.status(),
                socket:window.__kzaLiveWs?.readyState,
                rendered:[...document.querySelectorAll('#view-users [data-capture-id]')].map(row=>row.dataset.captureId)
            })''')}))
        raise


async def click_button(page, selector, label):
    """Click a visible DOM button through Chromium's input pipeline."""
    point = await page.evaluate('''(() => {
      const button=[...document.querySelectorAll(%s)].find(node=>node.textContent===%s);
      if (!button || button.disabled) return null;
      button.scrollIntoView({block:'center',behavior:'instant'});
      const {x,y,width,height}=button.getBoundingClientRect();
      return {x:x+width/2,y:y+height/2};
    })()''' % (json.dumps(selector),json.dumps(label)))
    assert point is not None,(selector,label)
    await page.command('Input.dispatchMouseEvent',{'type':'mousePressed','button':'left','clickCount':1,**point})
    await page.command('Input.dispatchMouseEvent',{'type':'mouseReleased','button':'left','clickCount':1,**point})


async def instrument_audio(page):
    """Observe native playback and hold selected callbacks at the browser boundary."""
    await page.command('Page.addScriptToEvaluateOnNewDocument',{'source':'''(() => {
      window.__voicePlayers=[]; window.__voiceEmptyCallbacks=[]; window.__holdVoiceEmpty=false;
      window.__voicePlayingCallbacks=[]; window.__holdVoicePlaying=false;
      window.__voiceMeasurement=null; window.__voicePlayCalls=[]; window.__rejectVoicePlay=false;
      window.__voicePendingPlayerIsEmpty=(card, placeholder)=>{
        const audio=card.querySelector('audio');
        if (!placeholder) return !audio;
        return !!audio&&audio.controls&&!audio.hasAttribute('src')&&audio.srcObject===null&&audio.paused&&!audio.autoplay;
      };
      const createElement=document.createElement.bind(document);
      document.createElement=(...args)=>{
        const element=createElement(...args);
        if(String(args[0]).toLowerCase()==='audio' && window.__voiceMeasurement){
          const timing=window.__voiceMeasurement;timing.created=performance.now();
          element.muted=true;
          element.addEventListener('canplay',()=>{timing.ready??=performance.now();});
        }
        return element;
      };
      const fetchReal=window.fetch;
      window.fetch=function(path,...args) {
        const timing=window.__voiceMeasurement, url=new URL(path,location.href);
        if(timing && (url.pathname.endsWith('/review') || url.pathname.startsWith('/api/evaluation/samples/'))){
          const card=document.querySelector(timing.review);
          timing.onRequest={busy:card.getAttribute('aria-busy'),loading:card.textContent.includes(timing.pending),
            empty:window.__voicePendingPlayerIsEmpty(card,timing.placeholder),ms:performance.now()-timing.clicked};
        }
        return fetchReal.call(this,path,...args);
      };
      window.__voiceUrls=new Set(); const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
      URL.createObjectURL=b=>{const value=create(b);window.__voiceUrls.add(value);return value;};
      URL.revokeObjectURL=value=>{window.__voiceUrls.delete(value);return revoke(value);};
      const nativePlay=HTMLMediaElement.prototype.play;
      HTMLMediaElement.prototype.play=function() {
        window.__voicePlayCalls.push({trusted:window.event?.type==='click' && window.event.isTrusted,
          activated:navigator.userActivation.isActive, feedback:this.parentElement?.textContent.includes('Iniciando reproducción…'),
          blocked:this.parentElement?.querySelector('button')?.disabled});
        if(window.__rejectVoicePlay){
          window.__rejectVoicePlay=false;
          return new Promise((_,reject)=>{window.__finishVoiceRejection=()=>reject(new DOMException('Synthetic playback rejection','NotAllowedError'));});
        }
        return nativePlay.call(this);
      };
      const add=HTMLMediaElement.prototype.addEventListener,remove=HTMLMediaElement.prototype.removeEventListener;
      const wrappers=new WeakMap();
      HTMLMediaElement.prototype.addEventListener=function(type,callback,options) {
        if(type==='emptied' || type==='playing') {
          const wrapped=event=>{
            if(type==='emptied' && window.__holdVoiceEmpty) window.__voiceEmptyCallbacks.push(()=>callback(event));
            else if(type==='playing' && window.__holdVoicePlaying) window.__voicePlayingCallbacks.push(()=>callback(event));
            else callback(event);
          };
          wrappers.set(callback,wrapped);return add.call(this,type,wrapped,options);
        }
        return add.call(this,type,callback,options);
      };
      HTMLMediaElement.prototype.removeEventListener=function(type,callback,options) {return remove.call(this,type,wrappers.get(callback)||callback,options);};
    })()'''})


async def start_review(page, selector, label, review, pending):
    """Measure the pending state in the click turn and at the actual HTTP call."""
    feedback = await page.evaluate('''(() => {
      const review=%s,pending=%s,placeholder=%s;
      const control=[...document.querySelectorAll(%s)].find(node=>node.textContent===%s);
      control.scrollIntoView({block:'center',behavior:'instant'});
      window.__voiceMeasurement={clicked:performance.now(),review,pending,placeholder,playCalls:window.__voicePlayCalls.length};
      control.click();
      const card=document.querySelector(review),box=card.getBoundingClientRect();
      return {busy:card.getAttribute('aria-busy'),loading:card.textContent.includes(pending),blocked:control.disabled,
        empty:window.__voicePendingPlayerIsEmpty(card,placeholder),visible:box.top>=0&&box.bottom<=innerHeight,
        ms:performance.now()-window.__voiceMeasurement.clicked};
    })()''' % tuple(map(json.dumps,(review,pending,review==OBS_REVIEW,selector,label))))
    elapsed = feedback.pop('ms')
    assert feedback=={'busy':'true','loading':True,'blocked':True,'empty':True,'visible':True} and elapsed<=100,(elapsed,feedback)
    await page.wait_for('!!window.__voiceMeasurement.onRequest')
    at_request = await page.evaluate('window.__voiceMeasurement.onRequest')
    request_ms = at_request.pop('ms')
    assert at_request=={'busy':'true','loading':True,'empty':True},at_request
    return {'click_to_pending_ms':elapsed,'click_to_http_ms':request_ms}


async def play_review(page, review, *, reject=False):
    """Prove explicit playback, rejection/retry and confirmation by native playing."""
    player = f'document.querySelector({json.dumps(review + " audio")})'
    card = f'document.querySelector({json.dumps(review)})'
    buttons = review + ' button'
    assert await page.evaluate('window.__voicePlayCalls.length===window.__voiceMeasurement.playCalls')
    assert await page.evaluate(f'{player}.paused && !{player}.autoplay && !{player}.controls && !{card}.textContent.includes("Reproduciendo…")')
    before = await page.evaluate('window.__voicePlayCalls.length')
    await page.evaluate(f'window.__voicePlayers.push({player});window.__lastPreviewUrl={player}.src')
    if reject:
        await page.evaluate('window.__rejectVoicePlay=true')
        await click_button(page,buttons,'Reproducir')
        assert await page.evaluate(f'{card}.textContent.includes("Iniciando reproducción…") && {card}.querySelector("button").disabled')
        await page.evaluate('window.__finishVoiceRejection();window.__finishVoiceRejection=null')
        await page.wait_for(f'{card}.textContent.includes("No se pudo reproducir el audio. Tocá Reproducir para reintentar.")')
        assert await page.evaluate(f'{player}.paused && !{card}.querySelector("button").disabled')

    await page.evaluate('window.__holdVoicePlaying=true')
    await click_button(page,buttons,'Reproducir')
    await page.wait_for('window.__voicePlayingCallbacks.length>0')
    assert await page.evaluate(f'''(() => {{
      {player}.dispatchEvent(new Event('canplay'));
      return {card}.textContent.includes('Iniciando reproducción…') && !{card}.textContent.includes('Reproduciendo…');
    }})()''')
    await page.evaluate('window.__holdVoicePlaying=false;for(const callback of window.__voicePlayingCallbacks.splice(0))callback()')
    await page.wait_for(f'{card}.textContent.includes("Reproduciendo…") && !{player}.paused && {player}.readyState>=2')
    calls = await page.evaluate(f'window.__voicePlayCalls.slice({before})')
    assert calls==[{'trusted':True,'activated':True,'feedback':True,'blocked':True}]*(2 if reject else 1),calls


async def nexo_readers(page, base, operator, evaluator, identity):
    """Exercise the two NEXO entry points with real scoped login and HTTP."""
    await install_login_session(page,base,operator)
    await page.command('Page.navigate',{'url':base+'/?t043=voices#voces'})
    await page.wait_for("[...document.querySelectorAll('#voices-content button')].some(button=>button.textContent==='Revisar muestras')")
    await click_button(page,'#voices-content button','Revisar muestras')
    listen = f'#voice-dialog [data-capture-id="{identity}"] button'
    await page.wait_for(f'!!document.querySelector({json.dumps(listen)}) && !document.querySelector({json.dumps(listen)}).disabled')
    voices_timing = await start_review(page,listen,'Escuchar',NEXO_REVIEW,'Cargando audio…')
    assert await page.evaluate("[...document.querySelectorAll('#voice-dialog > button')].find(button=>button.textContent==='Actualizar').disabled")
    await page.wait_for(f'!!document.querySelector({json.dumps(NEXO_REVIEW + " audio")})')
    await play_review(page,NEXO_REVIEW,reject=True)
    await click_button(page,'#voice-dialog > button','Cerrar')
    await page.wait_for("!document.querySelector('#voice-dialog audio') && window.__voiceUrls.size===0")

    await install_login_session(page,base,evaluator)
    await page.command('Page.navigate',{'url':base+'/?t043=evaluator#evaluacion'})
    submit = '#evaluation-content button[type=submit]'
    await page.wait_for(f'!!document.querySelector({json.dumps(submit)}) && !document.querySelector({json.dumps(submit)}).disabled')
    evaluation_timing = await start_review(page,submit,'Consultar muestra',EVALUATION,'Consultando la modalidad elegida…')
    await page.wait_for(f'!!document.querySelector({json.dumps(EVALUATION + " audio")})')
    await play_review(page,EVALUATION,reject=True)
    await page.evaluate("window.dispatchEvent(new PageTransitionEvent('pagehide'))")
    await page.wait_for("!document.querySelector('[data-evaluation-sample] audio') && window.__voiceUrls.size===0")
    return {'nexo_voices':voices_timing,'nexo_evaluation':evaluation_timing}


async def main(*,browser_binary=None):
    """Exercise actual UI events, HTTP/v3 WebSocket and native HTML media teardown."""
    token = 'synthetic-voice-review-browser'
    os.environ['KZA_DASHBOARD_TOKEN'] = token
    EVIDENCE.mkdir(mode=0o700,parents=True,exist_ok=True)
    browser_binary = browser_binary or select_browser()
    browser_data = browser_profile_root()
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
                # The locator is a test-only native read. Let the real UI and
                # model owners finish before consuming a registry I/O slot;
                # keep the sampler's original three-second observation budget.
                async with asyncio.timeout(3):
                    while voices.resource_status()['pending'] or voices.resource_status()['operations']:
                        await asyncio.sleep(.001)
                rows = (await voices.acall(voices.ledger.snapshot))['fingerprints']
                identity = next(row['id'] for row in rows if source.capture_id in row['capture_ids'])
                return source,identity
            samples = [await sample('Synthetic reviewed phrase '+str(index)) for index in range(3)]
            nexo_source,nexo_id = samples[0]
            authority = DashboardAuthority(directory/'auth.db',session_ttl_s=600)
            stack.push_async_callback(authority.aclose)
            await authority.initialize()
            root = authority.resolve(token)
            operator = await authority.issue_credential(root,subject='synthetic-nexo-reviewer',
                role=DashboardRole.OPERATOR,expires_at=clock.now+120)
            approved = EvaluationSample(nexo_id,frozenset({source_digest(nexo_source.to_dict())}),
                frozenset({EvaluationModality.AUDIO_PREVIEW}),clock.now+60)
            evaluator = await authority.issue_credential(root,subject='synthetic-nexo-evaluator',
                role=DashboardRole.EVALUATOR,expires_at=clock.now+120,samples=(approved,))
            dashboard = DashboardAPI(authority=authority,event_bus=bus,voice_registry=voices,observability_use_mocks=False)
            original_gpu,original_services = system_monitor.gpu_snapshot,system_monitor.services_snapshot
            requests = []
            system_monitor.gpu_snapshot,system_monitor.services_snapshot = lambda:None,lambda:None
            @dashboard.app.middleware('http')
            async def cookie(request,call_next):
                path = request.url.path
                if ((path.startswith('/api/voices/captures/') and path.endswith('/review'))
                        or path.startswith('/api/evaluation/samples/')):
                    await asyncio.sleep(.25)  # Expose loading feedback before the real response.
                response = await call_next(request)
                requests.append({'method':request.method,'path':request.url.path,'status':response.status_code})
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
                with (EVIDENCE/'chromium.log').open('w') as log:
                    browser = subprocess.Popen([browser_binary,'--headless','--disable-gpu','--no-first-run',
                        '--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
                    async with asyncio.timeout(15):
                        while not (profile/'DevToolsActivePort').exists():
                            if browser.poll() is not None:
                                raise RuntimeError((EVIDENCE/'chromium.log').read_text()[:2000])
                            await asyncio.sleep(.05)
                    debugging = int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            target = next(item for item in await response.json() if item['type']=='page')
                    async with connect(target['webSocketDebuggerUrl'],max_size=None) as connection:
                        page = VoicePage(connection)
                        await page.command('Runtime.enable')
                        await page.command('Page.enable')
                        await instrument_audio(page)
                        await page.command('Network.enable')
                        await page.command('Network.setBlockedURLs',{'urls':['https://unpkg.com/*','https://fonts.googleapis.com/*','https://fonts.gstatic.com/*']})
                        base = f'http://127.0.0.1:{port}'
                        nexo_timings = await nexo_readers(page,base,operator.token,evaluator.token,nexo_id)
                        # Keep the OBS retirement receipt checks isolated from the NEXO sessions.
                        extra,extra_id = await sample('Synthetic audio retirement phrase')
                        await install_login_session(page,base,token)
                        await page.command('Page.navigate',{'url':base+'/obs/'})
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
                        identities = [identity for _,identity in samples]
                        assert await page.evaluate('''(() => { for(const id of %s) document.querySelector(`[data-capture-id="${id}"] input`).click();
                          const input=document.querySelector('[aria-label="Nombre del perfil"]');input.value='Synthetic browser profile';input.dispatchEvent(new Event('input'));
                          const confirm=[...document.querySelectorAll('button')].find(button=>button.textContent==='Confirmar perfil');
                          confirm.click();confirm.click();
                          return confirm.disabled && document.querySelector('[aria-label="Muestras de voz"]').textContent.includes('Aplicando cambios…');
                        })()'''%json.dumps(identities))
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Synthetic browser profile')")
                        profiles = (await voices.acall(voices.ledger.snapshot))['profiles']
                        assert len(profiles)==1
                        assert sum(item['path']=='/api/voices/profiles' and item['method']=='POST' for item in requests)==1
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
                        assert await page.evaluate("[...document.querySelectorAll('[data-capture-id]')].every(row=>row.textContent.includes('Synthetic listening room'))")
                        await filter_with_keyboard(page,4)
                        await search_voice(page,'retirement phrase')
                        await read_transcript(page,extra_id,OBS_REVIEW,'Synthetic audio retirement phrase')
                        await click_button(page,VOICE_FILTERS+' button','Quitar filtros')
                        await page.wait_for("document.querySelectorAll('[data-capture-id]').length===4")
                        assert await page.evaluate(f'document.querySelector({json.dumps(OBS_REVIEW)})?.textContent.includes("Synthetic audio retirement phrase") && !document.querySelector("audio")')
                        loading_timings = []
                        async def listen(identity,text):
                            try:
                                await page.wait_for(obs_capture_ready(identity))
                            except AssertionError:
                                print(json.dumps({'diagnostic':'voice_view_capture_read','requests':requests[-12:],
                                    'expected':identity,'client':await page.evaluate("({status:window.__kzaVoiceState.captures.status(),ids:window.__kzaVoiceState.captures.snapshot().map(row=>row.id),dom:[...document.querySelectorAll('[data-capture-id]')].map(row=>({id:row.dataset.captureId,buttons:row.querySelectorAll('button').length})),profiles:window.__kzaVoiceState.profiles.snapshot().length})")}))
                                raise
                            control = f'#view-users [data-capture-id="{identity}"] button'
                            review_requests = sum(item['path'].endswith('/review') for item in requests)
                            timing = await start_review(page,control,'Escuchar',OBS_REVIEW,'Cargando audio…')
                            assert await page.evaluate("""(() => {
                              const buttons=[...document.querySelectorAll('[data-capture-id] button')];
                              const other=buttons.find(button=>button.textContent==='Escuchar');
                              other?.click();
                              const refresh=[...document.querySelectorAll('#view-users button')].find(button=>button.textContent==='Actualizar');
                              return refresh.disabled && buttons.length>0 && buttons.every(button=>button.disabled);
                            })()""")
                            await page.wait_for('document.querySelector('+json.dumps(OBS_REVIEW)+').textContent.includes('+json.dumps(text)+') && !!document.querySelector("audio")')
                            assert sum(item['path'].endswith('/review') for item in requests)==review_requests+1
                            try:
                                async with asyncio.timeout(2):
                                    while not await page.evaluate("document.querySelector('audio')?.readyState>=3 && Number.isFinite(window.__voiceMeasurement?.ready)"):
                                        await asyncio.sleep(.02)
                            except TimeoutError:
                                print(json.dumps({'diagnostic':'voice_audio_not_prepared',
                                    'media':await page.evaluate("(()=>{const audio=document.querySelector('audio');return {preload:audio?.preload,ready:audio?.readyState,network:audio?.networkState,durationKnown:Number.isFinite(audio?.duration),timing:window.__voiceMeasurement};})()") }))
                                raise
                            timing.update(await page.evaluate("(()=>{const m=window.__voiceMeasurement;return {click_to_player_ms:m.created-m.clicked,player_to_ready_ms:m.ready-m.created,click_to_ready_ms:m.ready-m.clicked};})()"))
                            loading_timings.append(timing)
                            await play_review(page,OBS_REVIEW,reject=identity==extra_id)
                            assert await page.evaluate("document.querySelector('[aria-label=\"Escuchar muestra\"]').textContent.includes('Synthetic listening room')")
                            assert await page.evaluate("(()=>{const element=document.querySelector('audio');window.__kzaVoiceState.review.setStatus('real');return document.querySelector('audio')===element;})()")
                            review_requests = sum(item['path'].endswith('/review') for item in requests)
                            assert await page.evaluate('''(() => {const audio=document.querySelector('audio');
                              document.querySelector(%s).click();return document.querySelector('audio')===audio&&!audio.paused;
                            })()'''%json.dumps('[data-capture-id="'+identity+'"] button'))
                            assert sum(item['path'].endswith('/review') for item in requests)==review_requests
                        await listen(extra_id,'Synthetic audio retirement phrase')
                        same_sample = len(page.voice_requests)
                        await click_button(page,f'[data-capture-id="{extra_id}"] button','Leer transcripción')
                        assert await page.evaluate("document.querySelector('audio')?.src===window.__lastPreviewUrl")
                        assert len(page.voice_requests)==same_sample
                        await click_button(page,OBS_REVIEW+' button','Pausar')
                        await page.wait_for("document.querySelector('audio').paused && document.querySelector('audio').parentElement.querySelector('button').textContent==='Reproducir'")
                        await page.evaluate('window.__rejectVoicePlay=true')
                        await click_button(page,OBS_REVIEW+' button','Reproducir')
                        await page.evaluate("window.__oldVoiceStatus=document.querySelector('audio').parentElement.querySelector('[role=status]')")
                        await page.evaluate('window.__holdVoiceEmpty=true')
                        await owner.request(extra.conversation_id,MemoryReviewer('synthetic-admin'))
                        await page.wait_for("!document.body.textContent.includes('Synthetic audio retirement phrase') && !document.querySelector('audio') && window.__voiceUrls.size===0 && window.__voiceEmptyCallbacks.length>0")
                        player_states = await page.evaluate("window.__voicePlayers.map(audio=>({paused:audio.paused,connected:audio.isConnected,hasCurrent:!!audio.currentSrc,ready:audio.readyState,network:audio.networkState,hasSrc:audio.hasAttribute('src'),buffers:audio.buffered.length,emptyDuration:Number.isNaN(audio.duration)}))")
                        if not all(item['paused'] and not item['connected'] and not item['hasSrc'] and item['buffers']==0 and item['emptyDuration'] and item['ready']==0 and item['network']==0 for item in player_states):
                            print(json.dumps({'diagnostic':'voice_media_teardown','players':player_states}))
                        assert all(item['paused'] and not item['connected'] and not item['hasSrc'] and item['buffers']==0 and item['emptyDuration'] and item['ready']==0 and item['network']==0 for item in player_states)
                        assert await page.evaluate("fetch(window.__lastPreviewUrl).then(()=>false,()=>true)")
                        assert await bus._retention.receipts.pending(extra.conversation_id)>0
                        await page.evaluate('window.__finishVoiceRejection();window.__finishVoiceRejection=null')
                        assert await page.evaluate("window.__oldVoiceStatus.textContent==='Iniciando reproducción…' && !document.body.textContent.includes('Tocá Reproducir para reintentar.')")
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
                        await page.wait_for("!document.body.textContent.includes('Synthetic browser profile') && window.__kzaVoiceState.profiles.status()==='real' && window.__kzaVoiceState.captures.status()==='real'")
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
                        # Release the frozen server deadline after proving client-only expiry.
                        # A room-options refresh must not readmit the audio fixture during the text case.
                        clock.now = expiring.expires_at
                        await voices.acall(voices.ledger.prune)
                        expiring_text,expiring_text_id = await sample('Synthetic independent text expiry',ttl=3)
                        await refresh_obs_capture(page,expiring_text_id,requests)
                        await read_transcript(page,expiring_text_id,OBS_REVIEW,'Synthetic independent text expiry')
                        await page.wait_for("!document.body.textContent.includes('Synthetic independent text expiry')")
                        assert voices.privacy.sources_available((expiring_text,))
                        clock.now = expiring_text.expires_at
                        await voices.acall(voices.ledger.prune)
                        retired_text,retired_text_id = await sample('Synthetic erased text')
                        await refresh_obs_capture(page,retired_text_id,requests)
                        await read_transcript(page,retired_text_id,OBS_REVIEW,'Synthetic erased text')
                        await owner.request(retired_text.conversation_id,MemoryReviewer('synthetic-admin'))
                        await page.wait_for("!document.body.textContent.includes('Synthetic erased text')")
                        leaving,leaving_id = await sample('Synthetic leaving voice view')
                        await refresh_obs_capture(page,leaving_id,requests)
                        await listen(leaving_id,'Synthetic leaving voice view')
                        await page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'))")
                        assert await page.evaluate("!document.querySelector('audio') && window.__voiceUrls.size===0 && !document.body.textContent.includes('Synthetic leaving voice view')")
                        await page.evaluate("delete document.hidden;document.dispatchEvent(new Event('visibilitychange'))")
                        await listen(leaving_id,'Synthetic leaving voice view')
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(item=>item.textContent.includes('Live')).click()")
                        await page.wait_for("document.querySelector('#view-live') && window.__voiceUrls.size===0 && window.__kzaVoiceState.review.snapshot().length===0")
                        async with asyncio.timeout(3):
                            while await bus._retention.receipts.pending(leaving.conversation_id):
                                await asyncio.sleep(.01)
                        _,auth_id = await sample('Synthetic voice session loss')
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(item=>item.textContent.includes('Voces')).click()")
                        # Entering Voces starts its own refresh. listen() waits for the resulting enabled row.
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
                            'selected_audio_ready_before_play':True,'audio_loading_timings':loading_timings,
                            'immediate_visible_feedback_before_delayed_http':True,'trusted_reproducir_playing_and_rejection_retry':True,'reselect_keeps_current_playing_sample':True,
                            'visibility_clears_obs_and_old_rejection_is_ignored':True,'nexo_timings':nexo_timings,
                            'filtered_captures_and_explicit_transcript':True,'transcript_expiry_and_retirement':True,
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
    parser.add_argument('--browser-binary',default=None)
    asyncio.run(main(browser_binary=parser.parse_args().browser_binary))
    (EVIDENCE/'browser-result.json').write_text(json.dumps({'passed':True,'synthetic_fixture':True})+'\n')
