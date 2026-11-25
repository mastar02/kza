"""Actual NEXO voice UI, individual login, synthetic PCM and native media drain."""

import asyncio
from contextlib import AsyncExitStack
from functools import partial
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from unittest.mock import patch
from uuid import uuid4

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
from src.users.voice_ledger import VoiceReviewer
from src.users.voice_registry import build_voice_registry
from tests.frontend.browser_nexo_css_parse import check_stylesheets
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_support import (
    FixtureRows, assert_nexo_copy, choose_option,
    assert_frame_consistent,
    Page, audit_current_accessibility, audit_modal_focus, audit_modal_return, audit_status, audit_view, browser_binary,
    browser_profile_root, press_escape,
    evidence_directory,
)
from tests.frontend.browser_voice_review import (
    CAPTURE_QUERY, NEXO_REVIEW, VOICE_FILTERS, VoicePage, click_button, filter_with_keyboard, read_transcript, search_voice,
)
from tests.mocks.ambient_graph import Clock
from tests.mocks.voice_registry import AUDIO, Encoder
from tests.mocks.memory_sources import memory_source


INSTRUMENT = r"""(() => {
  window.__voiceSent=[];window.__voiceSockets=0;window.__socketPaths=[];window.__requestPaths=[];window.__emptyCallbacks=[];window.__holdEmpty=false;window.__urls=new Set();
  window.__voiceBfcache=false;window.addEventListener('pageshow',event=>{window.__voiceBfcache=event.persisted;});
  const NativeSocket=WebSocket;
  window.WebSocket=class extends NativeSocket { constructor(...args){super(...args);window.__voiceSockets++;window.__socketPaths.push(new URL(args[0]).pathname);}
    send(value){window.__voiceSent.push(JSON.parse(value));return super.send(value);} };
  const add=HTMLMediaElement.prototype.addEventListener,remove=HTMLMediaElement.prototype.removeEventListener,wrappers=new WeakMap();
  HTMLMediaElement.prototype.addEventListener=function(type,callback,options){
    if(type==='emptied'){const wrapped=event=>window.__holdEmpty?window.__emptyCallbacks.push(()=>callback(event)):callback(event);wrappers.set(callback,wrapped);return add.call(this,type,wrapped,options);}
    return add.call(this,type,callback,options);};
  HTMLMediaElement.prototype.removeEventListener=function(type,callback,options){return remove.call(this,type,wrappers.get(callback)||callback,options);};
  const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
  // Casa creates its own texture URLs when a household session opens after logout.
  URL.createObjectURL=blob=>{
    const value=create(blob);
    if (!blob.type.startsWith('image/')) window.__urls.add(value);
    return value;
  };
  URL.revokeObjectURL=value=>{window.__urls.delete(value);return revoke(value);};
  const fetch=window.fetch.bind(window);window.__confirmationIds=[];window.__personWrites=[];window.__holdPeople=false;window.__peopleCallbacks=[];window.__holdPersonProfiles=false;window.__personProfileCallbacks=[];
  window.fetch=async(path,options={})=>{
    window.__requestPaths.push(new URL(path,location.href).pathname);
    const response=await fetch(path,options);
    if(path==='/api/admin/auth/whoami' && response.ok && window.__withoutPermissions){
      const identity=await response.json();
      identity.permissions=identity.permissions.filter(permission=>!window.__withoutPermissions.includes(permission));
      return new Response(JSON.stringify(identity),{headers:{'Content-Type':'application/json'}});
    }
    if(path==='/api/admin/users' && !options.method)return new Proxy(response,{get(target,key){if(key==='json')return async()=>{const value=await target.json();if(window.__holdPeople)await new Promise(resolve=>window.__peopleCallbacks.push(resolve));if(window.__peopleState==='error')throw Error('synthetic unavailable people');return window.__peopleState==='empty'?[]:value;};const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});
    if(window.__voiceReadState && typeof path==='string' && path.startsWith('/api/voices/profiles?')
      && !new URL(path,location.href).searchParams.has('person_id')){
      if(window.__voiceReadState==='hold')await new Promise(resolve=>{window.__releaseVoiceRead=resolve;});
      if(window.__voiceReadState==='error')return new Response(null,{status:503});
    }
    if(typeof path==='string' && path.startsWith('/api/voices/profiles?') && new URL(path,location.href).searchParams.has('person_id'))return new Proxy(response,{get(target,key){if(key==='json')return async()=>{const value=await target.json();if(window.__holdPersonProfiles)await new Promise(resolve=>window.__personProfileCallbacks.push(resolve));return value;};const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});
    if(options.method==='PUT' && path.endsWith('/person'))window.__personWrites.push({body:JSON.parse(options.body),status:response.status});
    if(path==='/api/voices/profiles' && options.method==='POST'){
      window.__confirmationIds.push(JSON.parse(options.body).request_id);
      if(window.__loseConfirmReceipt){window.__loseConfirmReceipt=false;return new Proxy(response,{get(target,key){if(key==='arrayBuffer')return async()=>{await target.arrayBuffer();throw Error('synthetic lost confirmation receipt');};const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});}
    }return response;};
})()"""


def person_link(voices,profile_id):
    with voices.ledger.ledger._transaction() as db:
        return tuple(db.execute('SELECT user_id,revision FROM voice_profile_people WHERE profile_id=?',(profile_id,)).fetchone())


def remaining_person_links(voices,identity):
    with voices.ledger.ledger._transaction() as db:
        return db.execute('SELECT COUNT(*) FROM voice_profile_people WHERE user_id=?',(identity,)).fetchone()[0]


async def editable_person(page):
    await page.wait_for("document.querySelector('#voice-person-edit') && !document.querySelector('#voice-person-edit').disabled")


async def choose_person(page, label):
    await editable_person(page)
    await page.evaluate("document.querySelector('#voice-person-edit').click()")
    await page.wait_for("document.querySelector('#voice-link-dialog').open")
    await choose_option(page, '#voice-person-choice', label or 'Sin persona asignada · desvincular')
    await page.evaluate("document.querySelector('#voice-link-dialog form').requestSubmit()")


async def capture_personas_matrix(page: Page, evidence: Path, state: str, *, audit_text: bool = True,
                                  dialog: str | None = None) -> None:
    target = evidence / 'matrix'
    target.mkdir(mode=0o700, exist_ok=True)

    async def theme(value: str) -> None:
        await page.evaluate('''(() => {
            const selector = document.querySelector('#theme-slot select');
            selector.value = ''' + json.dumps(value) + ''';
            selector.dispatchEvent(new Event('change'));
        })()''')

    previous = await page.evaluate("document.querySelector('#theme-slot select').value")
    await page.command('Emulation.setEmulatedMedia', {'features': [{'name': 'prefers-reduced-motion', 'value': 'reduce'}]})
    try:
        for value in ('claro', 'oscuro'):
            await theme(value)
            for width in (1440, 768, 390):
                await page.command('Emulation.setDeviceMetricsOverride',
                    {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': width <= 390})
                assert await page.evaluate(f"document.documentElement.dataset.theme === {json.dumps(value)} && innerWidth === {width}")
                modal = await page.evaluate("!!document.querySelector('dialog[open]')")
                await screenshot(page, target / f'{state}-{value}-{width}.png', full=not modal)
                if dialog and width in {390, 1440}:
                    await audit_current_accessibility(page, evidence, state, dialog)
                if state in ('people-data', 'access-person-empty', 'voices-data'):
                    view = '#voices-view' if state == 'voices-data' else '#people-view'
                    await assert_frame_consistent(page, view, value)
                if not audit_text:
                    continue
                failures = await page.evaluate('''(() => {
                    const panel = document.querySelector('dialog[open], #people-view:not([hidden]), #voices-view:not([hidden])');
                    const color = value => value.match(/[0-9.]+/g).map(Number);
                    const linear = v => v / 255 <= .04045 ? v / 3294.6 : ((v / 255 + .055) / 1.055) ** 2.4;
                    const luminance = ([r, g, b]) => .2126 * linear(r) + .7152 * linear(g) + .0722 * linear(b);
                    const visible = element => element.checkVisibility({visibilityProperty: true});
                    const content = document.querySelector('dialog[open]') || panel;
                    const failures = [];
                    for (const element of content.querySelectorAll('h1,h2,h3,h4,p,span,strong,button,label')) {
                        if (!visible(element) || element.closest(':disabled,[aria-disabled=true]') ||
                            ![...element.childNodes].some(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim())) continue;
                        const style = getComputedStyle(element);
                        let background, current = element;
                        while (current && !background) {
                            const value = getComputedStyle(current).backgroundColor;
                            if (value.startsWith('rgb(')) background = color(value);
                            current = current.parentElement;
                        }
                        const pair = [luminance(color(style.color)), luminance(background)].sort((a, b) => a - b);
                        const ratio = (pair[1] + .05) / (pair[0] + .05);
                        const size = parseFloat(style.fontSize);
                        const large = size >= 24 || size >= 18.67 && Number(style.fontWeight) >= 700;
                        if (ratio < (large ? 3 : 4.5) || size < 14) failures.push([element.textContent.trim(), ratio, size]);
                    }
                    return failures;
                })()''')
                assert not failures, f'{state} {value} {width}: contrast or text size: {failures}'
    finally:
        await theme(previous)
        await page.command('Emulation.setEmulatedMedia', {'features': []})
        await page.command('Emulation.setDeviceMetricsOverride',
            {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})


async def assert_person_scope(page: VoicePage, *, voices: bool, access: bool) -> None:
    # The permission response can omit either capability although People still opens.
    await page.wait_for("!!document.querySelector('#people-grid [data-person][aria-pressed=true]')")
    if voices:
        await page.wait_for("window.__socketPaths.includes('/ws/voices') && window.__requestPaths.includes('/api/voices/profiles')")
    if access:
        await page.wait_for("window.__requestPaths.includes('/api/admin/auth/credentials')")
    result = await page.evaluate('''({
        voice_dom: !!document.querySelector('#person-detail .person-voice-details'),
        voice_request: window.__requestPaths.some(path => path.startsWith('/api/voices')),
        voice_socket: window.__socketPaths.includes('/ws/voices'),
        access_dom: !!document.querySelector('#person-detail [data-person-access]'),
        access_request: window.__requestPaths.some(path => path.startsWith('/api/admin/auth/credentials')),
    })''')
    expected = {'voice_dom': voices, 'voice_request': voices, 'voice_socket': voices,
                'access_dom': access, 'access_request': access}
    assert result == expected, f'People capability leaked or was withheld: {result} != {expected}'


async def refresh_voices(page):
    await page.evaluate("[...document.querySelectorAll('#voices-content .page-head button')].find(button=>button.textContent==='Actualizar').click()")


async def open_samples(page, count):
    """Open the dialog and wait for its actionable sample rows."""
    await click_button(page,'#voices-content .page-head button','Revisar muestras')
    await page.wait_for(f'''document.querySelector('#voice-dialog')?.open &&
        document.querySelectorAll('#voice-dialog .review-sample input:not(:disabled)').length==={count}''')


async def audit_filter_history(page, capture_rows, selected):
    """Exercise same-document history and a cross-document browser restore."""
    open_button = '#voices-content .page-head button'
    await click_button(page,open_button,'Revisar muestras')
    await page.wait_for("document.querySelectorAll('.review-sample').length===4")
    await filter_with_keyboard(page,4)
    await search_voice(page,'phrase 2')
    await read_transcript(page, selected, NEXO_REVIEW, 'Synthetic NEXO phrase 2',
                          selector=capture_rows.selector(selected, 'button'))
    await page.evaluate("document.querySelector('#app-nav [data-view=home]').click()")
    await page.wait_for("location.hash==='#hogar' && !document.querySelector('#voice-dialog')?.open")
    await page.evaluate('history.back()')
    await page.wait_for("location.hash==='#voces' && document.querySelectorAll('.review-sample').length===4")
    await page.evaluate('history.forward()')
    await page.wait_for("location.hash==='#hogar' && !document.querySelector('#voice-dialog')?.open")
    await page.evaluate('history.back()')
    await page.wait_for("location.hash==='#voces' && document.querySelectorAll('.review-sample').length===4")
    await click_button(page,open_button,'Revisar muestras')
    reset = '''document.querySelector('.voice-filters input[type=search]')?.value==='' &&
        document.querySelector('.voice-filters input[type=checkbox]')?.checked===false &&
        document.querySelector('.voice-filter-status')?.textContent==='4 muestras vigentes.' &&
        !document.querySelector('.nexo-voice-review')?.textContent.includes('Synthetic NEXO phrase 2')'''
    assert await page.evaluate(reset)
    await search_voice(page,'phrase 2')
    await read_transcript(page, selected, NEXO_REVIEW, 'Synthetic NEXO phrase 2',
                          selector=capture_rows.selector(selected, 'button'))
    await page.command('Page.navigate',{'url':'about:blank'})
    await page.wait_for("location.href==='about:blank'")
    await page.evaluate('history.back()')
    await page.wait_for("location.hash==='#voces' && document.querySelectorAll('.review-sample').length===4")
    await click_button(page,open_button,'Revisar muestras')
    assert await page.evaluate(reset)
    result = await page.evaluate('''({persisted:window.__voiceBfcache===true,
        notRestoredReasons:performance.getEntriesByType('navigation')[0]?.notRestoredReasons?.toJSON?.() ?? null})''')
    await click_button(page,'#voice-dialog > button','Cerrar')
    return result


async def main() -> None:
    await asyncio.to_thread(check_stylesheets)
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-nexo-voice-root'
    evidence = evidence_directory('KZA_NEXO_VOICES_EVIDENCE', 'data/autonomous-roadmap-20261102/t047/voices-browser-runs')
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    class DiagnosticPage(VoicePage):
        async def wait_for(self, expression):
            try:
                await super().wait_for(expression)
            except AssertionError:
                diagnostic = await self.evaluate("({text:document.body.innerText,html:document.querySelector('#voice-dialog')?.innerHTML,ready:document.readyState})")
                state = await self.voice_diagnostic()
                (evidence/'failure.json').write_text(json.dumps({'condition':expression, 'document':diagnostic,
                    'voice':state, 'requests':requests}, indent=2)+'\n')
                if state['client']['visibility'] == 'visible':
                    try:
                        await screenshot(self, evidence/'failure.png')
                    except (AssertionError, RuntimeError, TimeoutError) as error:
                        print(f'Failure screenshot unavailable: {error}')
                raise
    binary = browser_binary()
    browser_data = browser_profile_root()
    with tempfile.TemporaryDirectory(prefix='kza-nexo-voices-', dir=browser_data) as temporary:
        directory = Path(temporary)
        async with AsyncExitStack() as stack:
            users = None
            authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=600,
                account_provider=lambda: users.active_accounts() if users is not None else None)
            stack.push_async_callback(authority.aclose)
            await authority.initialize()
            root = authority.resolve('synthetic-nexo-voice-root')
            operator = await authority.issue_credential(root, subject='synthetic-voice-operator', role=DashboardRole.OPERATOR, expires_at=time.time()+600)
            home = await authority.issue_credential(root, subject='synthetic-voice-home', role=DashboardRole.HOME, expires_at=time.time()+600)
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
            person_a,_ = await voices.acall(users.add_user,'Synthetic homonym',PermissionLevel.ADMIN)
            person_b,_ = await voices.acall(users.add_user,'Synthetic other person',PermissionLevel.ADMIN)
            assert person_a is not None and person_b is not None
            homonym_b = 'SYNTHÉTIC  HOMONYM'
            renamed,_ = await voices.acall(partial(users.update_user,person_b.user_id,name=homonym_b))
            assert renamed
            encoder = Encoder(); voices.register_encoder(encoder)
            for index in range(4):
                source = memory_source(now=clock.now, ttl=180)
                assert voices.observe(encoder, np.tile(AUDIO, 3 + index), (source,), transcript=f'Synthetic NEXO phrase {index}',
                    room_id={1: 'living', 2: 'living', 3: 'kitchen'}.get(index))
                async with asyncio.timeout(5):
                    while voices.resource_status()['pending'] or voices.resource_status()['operations']:
                        await asyncio.sleep(.001)
            captures = (await voices.acall(voices.ledger.snapshot))['fingerprints']
            identities = [row['id'] for row in captures]
            capture_by_seconds = {row['duration_s']: row['id'] for row in captures}
            dashboard = DashboardAPI(authority=authority, event_bus=bus, voice_registry=voices, user_manager=users)
            requests, query_rooms = [], []

            @dashboard.app.middleware('http')
            async def record(request, call_next):
                if request.method == 'POST' and request.url.path == CAPTURE_QUERY:
                    query_rooms.append((await request.json()).get('room'))
                response = await call_next(request)
                requests.append({'method':request.method, 'path':request.url.path, 'query':str(request.url.query), 'status':response.status_code})
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
                    browser = subprocess.Popen([binary, '--headless', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-first-run', '--disable-background-networking', '--autoplay-policy=no-user-gesture-required', '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank'], stdout=output, stderr=output)
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
                        capture_rows = FixtureRows('#voice-dialog .review-sample', '.row-meta')
                        labels = [(capture['id'], f"{capture['duration_s']:.1f} s · ") for capture in captures]
                        await capture_rows.bind_many(page, labels, prefix=True)
                        await page.prepare_desktop()
                        await page.command('Network.enable')
                        await page.command('Page.addScriptToEvaluateOnNewDocument', {'source':INSTRUMENT})
                        await page.command('Page.navigate', {'url':f'http://127.0.0.1:{port}/#voces'})
                        await page.wait_for("document.querySelector('#login-error')?.textContent.length>0")
                        await page.evaluate("window.__voiceReadState='hold'")
                        await login(page, operator.token, 'Administración')
                        await page.wait_for("window.__releaseVoiceRead && document.querySelector('#voice-profile-list').textContent.includes('Consultando')")
                        await capture_personas_matrix(page, evidence, 'voices-loading')
                        await page.evaluate("window.__voiceReadState='error';window.__releaseVoiceRead()")
                        await page.wait_for("document.querySelector('#voice-profile-list').textContent.includes('No se pudo consultar')")
                        await capture_personas_matrix(page, evidence, 'voices-unavailable')
                        await page.evaluate("window.__voiceReadState=null")
                        await refresh_voices(page)
                        await page.wait_for("document.querySelectorAll('.review-sample').length===4 && document.querySelector('#voice-connection').textContent.includes('vigente')")
                        assert await page.evaluate("window.__voiceSockets===1 && window.__voiceSent[0].version===3 && !document.querySelector('#voices-content').textContent.includes('ilustrativos')")
                        assert not any(row['path'] in {'/api/conversations','/api/users'} for row in requests)
                        await screenshot(page, evidence/'voices-desktop.png')
                        await capture_personas_matrix(page, evidence, 'voices-data')
                        await open_samples(page,4)
                        await capture_personas_matrix(page, evidence, 'voices-review-dialog')
                        for width in (320,390,768,1280,1440):
                            await page.command('Emulation.setDeviceMetricsOverride',{'width':width,'height':844,
                                'deviceScaleFactor':1,'mobile':width<=390})
                            assert await page.evaluate('''(() => {
                                const dialog=document.querySelector('#voice-dialog');
                                return dialog.scrollWidth<=dialog.clientWidth+1 &&
                                    [...dialog.querySelectorAll('.voice-filters input:not([type=checkbox]),.voice-filters select')]
                                        .every(control=>parseFloat(getComputedStyle(control).fontSize)>=16) &&
                                    [...dialog.querySelectorAll('.review-sample input[type=checkbox]')].every(control=>{
                                        const target=control.getBoundingClientRect();
                                        return target.width>=44 && target.height>=44;
                                    });
                            })()''')
                        await page.command('Emulation.setDeviceMetricsOverride',{'width':1425,'height':1180,'deviceScaleFactor':1,'mobile':False})
                        async def room_for_sample(seconds: float) -> str:
                            selector = json.dumps(capture_rows.selector(capture_by_seconds[seconds], '.row-meta'))
                            return await page.evaluate(f"document.querySelector({selector}).textContent.split(' · ').at(-1)")

                        living_label = await room_for_sample(4.0)
                        kitchen_label = await room_for_sample(6.0)
                        assert {living_label, kitchen_label} == {'Habitación 1', 'Habitación 2'}
                        assert await room_for_sample(5.0) == living_label
                        assert await room_for_sample(3.0) == 'Sin habitación registrada'
                        await audit_view(page, evidence, 'voces-editor', (('#voice-dialog input[type=text]', None),),
                                         screenshot, widths=(1440, 768, 390))
                        room_filter = '#voice-dialog .voice-filters select'
                        for label, room, count in ((living_label, 'living', 2), (kitchen_label, 'kitchen', 1)):
                            marker = len(query_rooms)
                            await choose_option(page, room_filter, label)
                            await page.wait_for(f"document.querySelectorAll('.review-sample').length === {count}")
                            assert query_rooms[marker:] == [{'kind': 'room', 'id': room}]
                        await click_button(page, VOICE_FILTERS + ' button', 'Quitar filtros')
                        await page.wait_for("document.querySelectorAll('.review-sample').length === 4")
                        await filter_with_keyboard(page,4)
                        await search_voice(page,'phrase 1')
                        sample_a = capture_by_seconds[4.0]
                        await read_transcript(page, sample_a, NEXO_REVIEW, 'Synthetic NEXO phrase 1',
                                              selector=capture_rows.selector(sample_a, 'button'))
                        assert await page.evaluate(
                            f"document.querySelector('.nexo-voice-review .row-meta').textContent === {json.dumps(living_label)}")
                        await click_button(page,VOICE_FILTERS+' button','Quitar filtros')
                        await page.wait_for("document.querySelectorAll('.review-sample').length===4")
                        assert await page.evaluate("document.querySelector('.nexo-voice-review')?.textContent.includes('Synthetic NEXO phrase 1')")
                        controls_a = capture_rows.selector(sample_a, 'button')
                        marker = len(page.voice_requests)
                        await click_button(page,controls_a,'Escuchar')
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2 && document.querySelector('#voice-dialog').textContent.includes('Synthetic NEXO phrase')")
                        assert [(method,url.path,url.query) for method,url in page.voice_requests[marker:]] == [
                            ('GET',f'/api/voices/captures/{sample_a}/review','')]
                        await page.evaluate("window.__sampleA=document.querySelector('#voice-dialog audio')")
                        marker = len(page.voice_requests)
                        await click_button(page,controls_a,'Leer transcripción')
                        assert await page.evaluate("document.querySelector('#voice-dialog audio')===window.__sampleA && window.__urls.size===1")
                        assert len(page.voice_requests)==marker
                        sample_b = next(identity for identity in identities if identity!=sample_a)
                        await read_transcript(page, sample_b, NEXO_REVIEW, 'Synthetic NEXO phrase',
                                              selector=capture_rows.selector(sample_b, 'button'))
                        assert await page.evaluate("!window.__sampleA.isConnected && window.__urls.size===0")
                        await click_button(page,controls_a,'Escuchar')
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2")
                        await page.evaluate("window.__player=document.querySelector('#voice-dialog audio');window.__player.muted=true;window.__player.play()")
                        assert await page.evaluate("!window.__player.paused && window.__urls.size===1")
                        for identity in identities[:3]:
                            await page.evaluate(f"document.querySelector({json.dumps(capture_rows.selector(identity, 'input'))}).click()")
                        await page.evaluate("(()=>{window.__loseConfirmReceipt=true;const input=document.querySelector('#voice-dialog input[type=text]');input.value='Synthetic NEXO profile';input.dispatchEvent(new Event('input'));document.querySelector('#voice-dialog form').requestSubmit()})()")
                        await page.wait_for("document.querySelector('#voice-dialog .identity-form-error').textContent.includes('Actualizá')")
                        assert len((await voices.acall(voices.ledger.snapshot))['profiles']) == 1
                        await page.evaluate("document.querySelector('#voice-dialog form').requestSubmit()")
                        assert await page.evaluate('window.__confirmationIds.length===1')
                        await page.evaluate("[...document.querySelectorAll('#voice-dialog>button')].find(button=>button.textContent==='Actualizar').click()")
                        await page.wait_for("document.querySelectorAll('.review-sample').length===4 && !document.querySelector('#voice-dialog .identity-form-error').textContent")
                        for identity in identities[:3]:
                            await page.evaluate(f"document.querySelector({json.dumps(capture_rows.selector(identity, 'input'))}).click()")
                        await page.evaluate("(()=>{const input=document.querySelector('#voice-dialog input[type=text]');input.value='Synthetic NEXO profile';input.dispatchEvent(new Event('input'));document.querySelector('#voice-dialog form').requestSubmit()})()")
                        await page.wait_for('window.__confirmationIds.length===2')
                        assert await page.evaluate('window.__confirmationIds[0]===window.__confirmationIds[1]')
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Synthetic NEXO profile')")
                        assert len((await voices.acall(voices.ledger.snapshot))['profiles']) == 1
                        await page.wait_for("document.querySelectorAll('.review-sample').length===4")
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        await editable_person(page)
                        profile_id = (await voices.acall(voices.ledger.snapshot))['profiles'][0]['id']
                        await choose_person(page,f"{'Synthetic homonym'} · Referencia: {person_a.user_id}")
                        await editable_person(page)
                        assert await voices.acall(person_link,voices,profile_id) == (person_a.user_id,1)
                        assert await page.evaluate("document.querySelector('#voice-linked-person').textContent==='Synthetic homonym'")
                        await screenshot(page,evidence/'voices-person-linked.png')
                        await page.evaluate("(()=>{const trigger=document.querySelector('#voice-person-edit');trigger.focus();trigger.click()})()")
                        await page.wait_for("document.querySelector('#voice-link-dialog').open")
                        link_focus = await audit_modal_focus(page, '#voice-link-dialog')
                        await screenshot(page,evidence/'voices-person-dialog.png')
                        await assert_nexo_copy(page, '#voice-link-dialog')
                        reference = await page.evaluate(r'''(() => {
                            const options = [...document.querySelector('#voice-person-choice').options]
                                .filter(option => /^\d+$/.test(option.value));
                            const hint = document.querySelector('#voice-person-reference');
                            return { options: options.map(option => option.textContent),
                                hint: hint.textContent, secondary: hint.classList.contains('row-meta') };
                        })()''')
                        assert set(reference['options']) == {
                            f'Synthetic homonym · Referencia: {person_a.user_id}',
                            f'{homonym_b} · Referencia: {person_b.user_id}',
                        }, 'Both spellings of a normalized homonym must show their own reference'
                        assert reference['secondary'] and 'Hay 2 personas con este nombre' in reference['hint']
                        assert person_a.user_id in reference['hint']
                        await capture_personas_matrix(page, evidence, 'voices-link-homonyms')
                        await audit_view(page, evidence, 'voces-vincular', (('#voice-person-choice', None),),
                                         screenshot, widths=(1440, 390))
                        await press_escape(page)
                        await page.wait_for("!document.querySelector('#voice-link-dialog').open")
                        await audit_modal_return(page, evidence, 'voces-persona', link_focus,
                                                 dialog='#voice-link-dialog', opener='#voice-person-edit')
                        await choose_person(page,None)
                        await page.wait_for("document.querySelector('#voice-linked-person')?.textContent==='Sin persona asignada'")
                        assert await voices.acall(person_link,voices,profile_id) == (None,2)
                        await choose_person(page,f"{homonym_b} · Referencia: {person_b.user_id}")
                        await editable_person(page)
                        assert await voices.acall(person_link,voices,profile_id) == (person_b.user_id,3)
                        original_set = voices.ledger.people.set_person
                        def raced_set(identity,user_id,profile_revision,link_revision,actor):
                            original_set(identity,person_a.user_id,profile_revision,link_revision,actor)
                            return original_set(identity,user_id,profile_revision,link_revision,actor)
                        with patch.object(voices.ledger.people,'set_person',raced_set):
                            await choose_person(page,None)
                            await page.wait_for("document.querySelector('#voices-content .identity-form-error').textContent.includes('Actualizá')")
                        assert await page.evaluate("window.__personWrites.length===4 && window.__personWrites.at(-1).status===409")
                        await audit_status(page, evidence, 'voces-persona', '409', '#voices-content .identity-form-error')
                        assert await voices.acall(person_link,voices,profile_id) == (person_a.user_id,4)
                        await refresh_voices(page); await editable_person(page)
                        with patch.object(voices.ledger.people,'set_person',side_effect=RuntimeError('synthetic_unavailable')):
                            await choose_person(page,f"{homonym_b} · Referencia: {person_b.user_id}")
                            await page.wait_for("document.querySelector('#voices-content .identity-form-error').textContent.includes('Actualizá')")
                        assert await page.evaluate("window.__personWrites.length===5 && window.__personWrites.at(-1).status===503")
                        await audit_status(page, evidence, 'voces-persona', '503', '#voices-content .identity-form-error')
                        assert await voices.acall(person_link,voices,profile_id) == (person_a.user_id,4)
                        await refresh_voices(page); await editable_person(page)
                        await page.evaluate('window.__holdPeople=true')
                        await refresh_voices(page)
                        await page.wait_for('window.__peopleCallbacks.length===1')
                        changed,_ = await voices.acall(partial(users.update_user,person_a.user_id,name='Synthetic updated person'))
                        assert changed
                        await refresh_voices(page)
                        await page.wait_for("document.querySelector('#voice-person-load')?.textContent==='Consultar personas' && !document.querySelector('#voice-person-load').disabled")
                        await page.evaluate('window.__holdPeople=false;window.__peopleCallbacks.splice(0).forEach(resolve=>resolve())')
                        assert await page.evaluate("!document.querySelector('#voices-content').textContent.includes('Synthetic homonym')")
                        await page.evaluate("document.querySelector('#voice-person-load').click()")
                        await page.wait_for("document.querySelector('#voice-linked-person')?.textContent==='Synthetic updated person'")
                        await page.evaluate("document.querySelector('#voice-person-edit').click()")
                        await assert_nexo_copy(page, '#voice-link-dialog')
                        assert await page.evaluate(r'''(() => {
                            const choices = [...document.querySelector('#voice-person-choice').options].filter(option => /^\d+$/.test(option.value));
                            return choices.length === 2 && choices.every(option =>
                                !option.textContent.includes('Referencia:') && !/\buser_[a-f0-9]{32}\b/.test(option.textContent))
                                && !document.querySelector('#voice-person-reference').textContent;
                        })()'''), 'Distinct names must not show references'
                        await page.evaluate("document.querySelector('#voice-link-dialog>button').click()")
                        before_profile = (await voices.acall(voices.ledger.snapshot))['profiles'][0]
                        async with aiohttp.ClientSession() as client:
                            async with client.delete(f'http://127.0.0.1:{port}/api/admin/users/{person_a.user_id}',headers={'Authorization':'Bearer synthetic-nexo-voice-root'}) as response:
                                assert response.status == 200
                                await response.read()
                        await refresh_voices(page)
                        await page.wait_for("document.querySelector('#voice-linked-person')?.textContent==='Sin persona asignada'")
                        assert await voices.acall(person_link,voices,profile_id) == (None,5)
                        after_profile = (await voices.acall(voices.ledger.snapshot))['profiles'][0]
                        assert all(after_profile[key] == before_profile[key] for key in ['id','revision','name'])
                        assert users.get_user(person_a.user_id) is None
                        assert users.get_user(person_b.user_id).permission_level is PermissionLevel.ADMIN
                        await editable_person(page)
                        await screenshot(page, evidence/'voices-profile.png')
                        await capture_personas_matrix(page, evidence, 'voices-profile-data')
                        await open_samples(page,4)
                        await page.evaluate("document.querySelector('.review-sample button').click()")
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2")
                        await screenshot(page, evidence/'voices-review.png')
                        await capture_personas_matrix(page, evidence, 'voices-review')
                        await page.evaluate("window.__player=document.querySelector('#voice-dialog audio');window.__holdEmpty=true;document.querySelector('#rail-nav [data-section=home]').click();document.querySelector('#rail-nav [data-section=people]').click()")
                        await page.wait_for('window.__emptyCallbacks.length>0')
                        assert await page.evaluate("window.__voiceSockets===1 && !document.querySelector('#voices-content').children.length && !document.querySelector('#voice-dialog').open && !document.querySelector('#voice-dialog').children.length && window.__urls.size===0 && window.__player.paused && !window.__player.hasAttribute('src')")
                        await page.evaluate("window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(callback=>callback())")
                        await page.wait_for("window.__voiceSockets===2 && document.querySelectorAll('.review-sample').length===4")
                        await open_samples(page,4)
                        await page.evaluate("document.querySelector('#voice-dialog .review-sample button').click()")
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2")
                        mark_capture_queries = len(page.voice_requests)
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        await page.wait_for('window.__urls.size===0')
                        await open_samples(page,4)
                        assert not any(method=='POST' and url.path==CAPTURE_QUERY
                            for method,url in page.voice_requests[mark_capture_queries:])
                        await page.evaluate("document.querySelector('#voice-dialog .review-sample input').click();[...document.querySelectorAll('#voice-dialog button')].find(button=>button.textContent==='Dejar como desconocido').click()")
                        await page.wait_for("document.querySelectorAll('[data-voice-group]').length===2 && document.querySelectorAll('.review-sample').length===4")
                        corrected = (await voices.acall(voices.ledger.snapshot))['fingerprints']
                        assert sum(row['group_reason']=='manual_unknown' for row in corrected) == 1
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        audits = {'voces': await audit_view(page, evidence, 'voces', (
                            ('#section-views [data-view=voices]', None), ('#voices-content button', 'Revisar muestras'),
                            ('#voices-content button', 'Borrar perfil'), ('#voices-content button', 'Actualizar'), ('#logout', None)), screenshot)}
                        for width in [390, 320]:
                            await page.command('Emulation.setDeviceMetricsOverride', {'width':width, 'height':844, 'deviceScaleFactor':1, 'mobile':True})
                            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                            await screenshot(page, evidence/f'voices-{width}.png', full=False)
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Borrar perfil').click()")
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Aún no hay')")
                        assert not (await voices.acall(voices.ledger.snapshot))['profiles']
                        await page.evaluate("[...document.querySelectorAll('#voices-content button')].find(button=>button.textContent==='Actualizar').click()")
                        await page.wait_for("document.querySelectorAll('.review-sample').length===4")
                        await open_samples(page,4)
                        for identity in identities[:3]:
                            await page.evaluate(f"document.querySelector({json.dumps(capture_rows.selector(identity, 'input'))}).click()")
                        await page.evaluate("(()=>{const input=document.querySelector('#voice-dialog input[type=text]');input.value='Synthetic NEXO profile';input.dispatchEvent(new Event('input'));document.querySelector('#voice-dialog form').requestSubmit()})()")
                        await page.wait_for('window.__confirmationIds.length===3')
                        assert await page.evaluate('window.__confirmationIds[2]!==window.__confirmationIds[1]')
                        await page.wait_for("document.querySelector('[aria-label=\"Perfiles confirmados\"]').textContent.includes('Synthetic NEXO profile')")
                        assert len((await voices.acall(voices.ledger.snapshot))['profiles']) == 1
                        await page.evaluate("document.querySelector('#voice-dialog>button').click()")
                        await editable_person(page)
                        inverse_id = (await voices.acall(voices.ledger.snapshot))['profiles'][0]['id']
                        await choose_person(page,homonym_b)
                        await editable_person(page)
                        third,_ = await voices.acall(users.add_user,'Synthetic third person',PermissionLevel.ADMIN)
                        assert third is not None
                        setup_actor=VoiceReviewer.from_dashboard(root.principal,
                            lambda:root.current(frozenset({'voices:review','admin:manage'})),person_authority=authority)
                        def seed_inverse_pages():
                            for index in range(50):
                                profile=voices.ledger.confirm(tuple(identities[:3]),f'Synthetic inverse {index}',str(uuid4()),setup_actor)
                                voices.ledger.people.set_person(profile['id'],person_b.user_id,1,0,setup_actor)
                        await voices.acall(seed_inverse_pages)
                        inverse_marker = len(requests)
                        await page.command('Emulation.setDeviceMetricsOverride', {'width':1425,'height':1180,'deviceScaleFactor':1,'mobile':False})
                        await page.evaluate("document.querySelector('#section-views [data-view=people]').click()")
                        await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===2")
                        inverse_person = f"""[...document.querySelectorAll('#people-grid [data-person]')]
                            .find(card => card.querySelector('.row-title').textContent === {json.dumps(homonym_b)})"""
                        no_profiles_person = """[...document.querySelectorAll('#people-grid [data-person]')]
                            .find(card => card.querySelector('.row-title').textContent === 'Synthetic third person')"""
                        await page.evaluate(f"({inverse_person}).click()")
                        await page.wait_for("document.querySelectorAll('#person-detail [data-person-voice]').length===50 && document.querySelector('#person-detail').textContent.includes('50 en esta página')")
                        first_page=await page.evaluate("[...document.querySelectorAll('#person-detail [data-person-voice]')].map(row=>row.querySelector('.row-title').textContent)")
                        await page.evaluate("document.querySelector('#person-detail [data-voice-next]').click()")
                        await page.wait_for("document.querySelectorAll('#person-detail [data-person-voice]').length===1 && document.querySelector('#person-detail').textContent.includes('1 en esta página')")
                        second_page=await page.evaluate("[...document.querySelectorAll('#person-detail [data-person-voice]')].map(row=>row.querySelector('.row-title').textContent)")
                        assert len(first_page) == 50 and len(second_page) == 1
                        assert set(first_page + second_page) == {'Synthetic NEXO profile', *(f'Synthetic inverse {i}' for i in range(50))}
                        await page.evaluate("document.querySelector('#person-detail [data-voice-refresh]').click()")
                        await page.wait_for("document.querySelectorAll('#person-detail [data-person-voice]').length===50")
                        assert not any(row['path'] in {'/api/voices',CAPTURE_QUERY} or row['path'].endswith(('/review','/transcript')) for row in requests[inverse_marker:])
                        filtered_reads=[row for row in requests[inverse_marker:] if row['path']=='/api/voices/profiles']
                        assert filtered_reads and all('person_id=' in row['query'] and 'include_person=true' in row['query'] for row in filtered_reads)
                        await screenshot(page,evidence/'person-voices-desktop.png')
                        await capture_personas_matrix(page, evidence, 'people-data')
                        audits['personas-voces'] = await audit_view(page, evidence, 'personas-voces', (
                            ('#section-views [data-view=people]', None), ('#people-grid [data-person]', None),
                            ('#person-detail [data-voice-refresh]', None), ('#logout', None)), screenshot)
                        for width in [390,320]:
                            await page.command('Emulation.setDeviceMetricsOverride', {'width':width,'height':844,'deviceScaleFactor':1,'mobile':True})
                            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                            await page.evaluate("document.querySelector('.person-voice-details').scrollIntoView({block:'center'})")
                            await screenshot(page,evidence/f'person-voices-{width}.png',full=False,scroll_top=False)
                            assert await page.evaluate("(()=>{const rect=document.querySelector('.person-voice-details').getBoundingClientRect();return rect.top>=0 && rect.top<innerHeight && rect.bottom<=innerHeight})()")
                        await page.command('Emulation.setDeviceMetricsOverride', {'width':1425,'height':1180,'deviceScaleFactor':1,'mobile':False})
                        await page.evaluate("window.__holdPersonProfiles=true;document.querySelector('#person-detail [data-voice-refresh]').click()")
                        await page.wait_for('window.__personProfileCallbacks.length===1')
                        prior_sockets=await page.evaluate('window.__voiceSockets')
                        inverse_transition = len(requests)
                        await page.evaluate(f"""({no_profiles_person}).click(); ({inverse_person}).click();
                            document.querySelector('#section-views [data-view=voices]').click()""")
                        assert await page.evaluate('window.__voiceSockets==='+str(prior_sockets)+" && !document.querySelector('#person-detail') && !document.querySelector('#voices-content').children.length")
                        await page.evaluate('window.__holdPersonProfiles=false;window.__personProfileCallbacks.splice(0).forEach(resolve=>resolve())')
                        await editable_person(page)
                        await page.wait_for("document.querySelectorAll('.review-sample').length===4")
                        assert await page.evaluate('window.__voiceSockets==='+str(prior_sockets+1))
                        inverse_resumed = len(requests)
                        await page.evaluate("document.querySelector('#section-views [data-view=people]').click()")
                        await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===2")
                        await page.evaluate(f"({no_profiles_person}).click()")
                        await page.wait_for("document.querySelector('#person-detail').textContent.includes('Sin perfiles en esta consulta.')")
                        await capture_personas_matrix(page, evidence, 'people-voice-empty')
                        with patch.object(voices.ledger,'project',side_effect=RuntimeError('synthetic_profile_unavailable')):
                            await page.evaluate("document.querySelector('#person-detail [data-voice-refresh]').click()")
                            await page.wait_for("document.querySelector('#person-detail').textContent.includes('No se pudieron consultar')")
                            assert await page.evaluate("!document.querySelector('#person-detail').textContent.includes('0 en esta página')")
                            await capture_personas_matrix(page, evidence, 'people-voice-unavailable')
                        await page.evaluate("document.querySelector('#person-detail [data-voice-refresh]').click()")
                        await page.wait_for("document.querySelector('#person-detail').textContent.includes('Sin perfiles en esta consulta.')")
                        await page.evaluate(f"({inverse_person}).click()")
                        await page.wait_for("document.querySelectorAll('#person-detail [data-person-voice]').length===50")
                        await page.evaluate("document.querySelector('#person-detail .identity-text-link').click();document.querySelector('#person-form [name=permission_level]').value='child';document.querySelector('#person-form').requestSubmit()")
                        await page.wait_for("document.querySelector('#person-detail .person-role-info')?.textContent.includes('Niño/a') && document.querySelectorAll('#person-detail [data-person-voice]').length===50")
                        assert users.get_user(person_b.user_id).permission_level is PermissionLevel.CHILD
                        assert await voices.acall(person_link,voices,inverse_id)==(person_b.user_id,1)
                        await page.evaluate("document.querySelector('#person-detail .identity-danger').click()")
                        await capture_personas_matrix(page, evidence, 'people-remove-dialog')
                        await page.evaluate("document.querySelector('#confirm-person-removal').click()")
                        await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===1")
                        assert users.get_user(person_b.user_id) is None
                        assert await voices.acall(person_link,voices,inverse_id)==(None,2)
                        assert len((await voices.acall(voices.ledger.snapshot))['profiles'])==51
                        assert await voices.acall(remaining_person_links,voices,person_b.user_id)==0
                        inverse_requests = requests[inverse_marker:inverse_transition] + requests[inverse_resumed:]
                        assert all(row['method']=='GET' and row['path']=='/api/voices/profiles'
                            for row in inverse_requests if row['path'].startswith('/api/voices'))
                        await page.evaluate("document.querySelector('#section-views [data-view=voices]').click()")
                        await editable_person(page)
                        filter_history = await audit_filter_history(page, capture_rows, capture_by_seconds[5.0])
                        await page.evaluate('window.__holdPeople=true')
                        await refresh_voices(page)
                        await page.wait_for('window.__peopleCallbacks.length===1')
                        await page.evaluate("document.querySelector('#logout').click()")
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        marker = len(requests)
                        assert await page.evaluate("document.querySelector('#token').disabled")
                        await page.evaluate('window.__holdPeople=false;window.__peopleCallbacks.splice(0).forEach(resolve=>resolve())')
                        await page.wait_for("!document.querySelector('#token').disabled")
                        await login(page, home.token, 'Acceso de hogar')
                        assert await page.evaluate("!document.querySelector('#app-nav [data-view=voices], #rail-nav [data-section=people]') && !document.querySelector('#voices-content').children.length && !document.querySelector('#voice-dialog').children.length && window.__urls.size===0")
                        assert not any(row['path'].startswith('/api/voices') for row in requests[marker:])
                        assert not await page.evaluate("document.querySelector('.voice-filters input[type=search]')")
                        assert await page.evaluate("!document.querySelector('#voice-link-dialog').children.length && !document.querySelector('#voices-content').children.length")
                        assert not page.errors, page.errors
                        # A household credential is also denied by direct route.
                        await page.evaluate("window.__requestPaths=[];window.__socketPaths=[]")
                        await page.evaluate('''history.pushState({}, '', '#personas');
                            dispatchEvent(new PopStateEvent('popstate'));''')
                        assert await page.evaluate('''!document.querySelector('#people-view').children.length &&
                            !window.__requestPaths.some(path => ['/api/admin/users', '/api/admin/auth/credentials', '/api/voices/profiles'].includes(path)) &&
                            !window.__socketPaths.includes('/ws/voices')''')
                        denied = [
                            (['auth:manage'], True, False),
                            (['voices:read'], False, True),
                            (['voices:review'], False, True),
                            (['auth:manage', 'voices:read', 'voices:review'], False, False),
                        ]
                        for removed, can_see_voices, can_see_access in denied:
                            await page.evaluate("document.querySelector('#logout').click()")
                            await page.wait_for("!document.querySelector('#login-panel').hidden && !document.querySelector('#token').disabled")
                            await page.evaluate(
                                "window.__requestPaths=[];window.__socketPaths=[];window.__withoutPermissions="
                                + json.dumps(removed))
                            await login(page, operator.token, 'Administración')
                            if not await page.evaluate("!document.querySelector('#people-view').hidden"):
                                await page.evaluate("document.querySelector('#rail-nav [data-section=people]').click()")
                            await assert_person_scope(page, voices=can_see_voices, access=can_see_access)
                        await page.evaluate("window.__holdPeople=true;document.querySelector('#refresh-people').click()")
                        await page.wait_for("window.__peopleCallbacks.length===1 && document.querySelector('#people-status').textContent.includes('Consultando')")
                        await capture_personas_matrix(page, evidence, 'people-loading')
                        await page.evaluate("window.__peopleState='empty';window.__holdPeople=false;window.__peopleCallbacks.splice(0).forEach(resolve=>resolve())")
                        await page.wait_for("document.querySelector('#people-grid').textContent.includes('Todavía no hay')")
                        await capture_personas_matrix(page, evidence, 'people-empty')
                        await page.evaluate("window.__peopleState='error';document.querySelector('#refresh-people').click()")
                        await page.wait_for("document.querySelector('#people-status').classList.contains('error-message')")
                        await capture_personas_matrix(page, evidence, 'people-unavailable')
                        await page.evaluate("document.querySelector('#logout').click()")
                        await page.wait_for("!document.querySelector('#login-panel').hidden && !document.querySelector('#token').disabled")
                        await page.evaluate("history.replaceState({}, '', '#personas');window.__requestPaths=[];window.__socketPaths=[];window.__withoutPermissions=['admin:manage']")
                        await login(page, operator.token, 'Administración')
                        assert await page.evaluate('''!document.querySelector('#people-view').children.length &&
                            !window.__requestPaths.some(path => ['/api/admin/users', '/api/admin/auth/credentials', '/api/voices/profiles'].includes(path)) &&
                            !window.__socketPaths.includes('/ws/voices')''')
                        result = {'passed':True, 'individual_http_login':True, 'synthetic_pcm_only':True, 'native_media_drain':True, 'voice_control_only':True, 'lost_receipt_single_profile':True, 'explicit_recreation_after_delete':True, 'manual_person_link':True, 'homonyms_by_canonical_id':True, 'person_scope_permission_matrix':True, 'visual_matrix':sorted(path.name for path in (evidence/'matrix').glob('*.png')), 'independent_acoustic_revision':True, 'conflict_unavailable_no_retry':True, 'old_catalogue_new_token_rejected':True, 'account_delete_unlinks':True, 'people_decoder_across_logout':True, 'inverse_person_profiles':True, 'inverse_scope_drain':True, 'inverse_no_audio_or_mutations':True, 'inverse_native_pagination_51':True, 'account_level_preserves_voice_link':True, 'inverse_empty_unavailable_distinct':True, 'requests':requests, 'screenshots':['voices-desktop.png','voices-profile.png','voices-person-linked.png','voices-person-dialog.png','voices-review.png','voices-390.png','voices-320.png','person-voices-desktop.png','person-voices-390.png','person-voices-320.png'],
                                  'reflow_widths':{view:sorted(audit['widths'],key=int) for view,audit in audits.items()},
                                  'filter_widths':[320,390,768,1280,1440], 'filter_history':filter_history,
                                  'explicit_transcript_read_audio_replacement':True,'a11y_evidence':[f'a11y-{view}.json' for view in audits]}
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
