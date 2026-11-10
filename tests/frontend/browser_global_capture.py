"""Actual factory, authenticated React and Chromium synthetic-media retirement."""
from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import aiohttp
import pytest
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard import system_monitor
from src.main import build_application
from src.wakeword.detector import WakeWordDetector
from tests.frontend.browser_support import browser_binary
from tests.frontend.browser_typed_controls import Page
from tests.integration.test_capability_application import configuration, leaves
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_wake_graph_retirement import configured
from tests.mocks.capture_models import capture_factories, capture_stt
from tests.mocks.task_transport import TaskTransport


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(os.environ.get('KZA_CAPTURE_BROWSER_EVIDENCE', str(
    ROOT / f'data/autonomous-roadmap-20261102/t023/c/browser-react/run-{time.time_ns()}')))
WATCHED = (
    'src/main.py', 'src/application.py', 'src/memory/capture_control.py',
    'src/memory/capture_native.py', 'src/memory/capture_recorders.py',
    'src/dashboard/capture_ingress.py', 'src/dashboard/capture_browser_api.py',
    'src/dashboard/capture_api.py', 'src/dashboard/auth.py', 'src/dashboard/admin.py',
    'src/dashboard/frontend/obs/src/views/privacy.jsx',
    'src/dashboard/frontend/obs/src/views/admin.jsx',
    'src/dashboard/frontend/obs/src/capture-recorder.js',
    'tests/mocks/capture_models.py', 'tests/frontend/browser_global_capture.py',
)
INSTRUMENT = r"""
(() => {
  const witness=window.__captureWitness={streams:[],contexts:[],events:[],fetches:[],holdAck:true,ackPending:false,acksCompleted:0};
  const get=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia=async options=>{
    witness.events.push(['gum-start',performance.now()]);
    try {const stream=await get(options);witness.streams.push(stream);witness.events.push(['gum-return',performance.now()]);return stream;}
    catch(error){witness.events.push(['gum-error',String(error.name),performance.now()]);throw error;}
  };
  const Audio=window.AudioContext;
  window.AudioContext=function(...args){const context=new Audio(...args);witness.contexts.push(context);return context;};
  window.AudioContext.prototype=Audio.prototype;
  let api;
  const existing=window.KzaCaptureRecorder;
  Object.defineProperty(window,'KzaCaptureRecorder',{configurable:true,get(){return api;},set(value){
    const create=value.createModalOwner;value.createModalOwner=options=>{const owner=create(options);witness.owner=owner;return owner;};api=value;
  }});
  if(existing)window.KzaCaptureRecorder=existing;
  const fetch=window.fetch.bind(window);
  window.fetch=async(input,options)=>{
    const path=new URL(typeof input==='string'?input:input.url,location.href).pathname;
    if(path==='/api/capture/recorders/retire'){
      if(witness.holdAck){witness.ackPending=true;await new Promise(resolve=>{witness.releaseAck=()=>{witness.holdAck=false;resolve();};});}
      const response=await fetch(input,options);witness.fetches.push(['retire',response.status]);if(response.ok)witness.acksCompleted++;return response;
    }
    const response=await fetch(input,options);
    if(path.startsWith('/api/capture/recorders/'))witness.fetches.push([path.split('/').at(-1),response.status]);
    return response;
  };
})();
"""


def hashes() -> dict:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in WATCHED}


async def screenshot(page: Page, name: str) -> str:
    image = await page.command('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
    payload = base64.b64decode(image['data'])
    (EVIDENCE / name).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


async def click_button(page: Page, text: str) -> None:
    """Use a trusted actual input event, including the media user gesture."""
    await page.command('Page.bringToFront')
    point = await page.evaluate("(async () => {const button=[...document.querySelectorAll('button')].find(row=>row.textContent.includes("
        + json.dumps(text) + "));if(!button||button.disabled)return null;button.scrollIntoView({block:'center',behavior:'instant'});await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);const rect=button.getBoundingClientRect();const x=rect.x+rect.width/2,y=rect.y+rect.height/2,hit=document.elementFromPoint(x,y);return {x,y,inside:x>=0&&x<innerWidth&&y>=0&&y<innerHeight,hit:hit===button||button.contains(hit),enabled:!button.disabled};})()")
    assert point is not None and point['inside'] and point['hit'] and point['enabled'], (text, point)
    coordinates = {'x': point['x'], 'y': point['y']}
    await page.command('Input.dispatchMouseEvent', {**coordinates, 'type': 'mousePressed', 'button': 'left', 'clickCount': 1})
    await page.command('Input.dispatchMouseEvent', {**coordinates, 'type': 'mouseReleased', 'button': 'left', 'clickCount': 1})


async def open_page(client, debug: int, target_id: str) -> tuple:
    async with client.get(f'http://127.0.0.1:{debug}/json/list') as response:
        target = next(row for row in await response.json() if row['id'] == target_id)
    connection = await connect(target['webSocketDebuggerUrl'], max_size=None)
    page = Page(connection)
    await page.command('Runtime.enable')
    await page.command('Network.enable')
    await page.command('Network.setBlockedURLs', {'urls': [
        'https://unpkg.com/*', 'https://fonts.googleapis.com/*', 'https://fonts.gstatic.com/*']})
    return page, connection


def _browser_binary() -> str:
    """Select an owned headless browser without changing the Linux default."""
    return browser_binary('KZA_CAPTURE_BROWSER_BINARY', 'capture_browser_binary_missing')


async def exercise(receipt: dict) -> None:
    binary = _browser_binary()
    receipt.update(browser_binary=binary, browser_platform=sys.platform)
    browser_root = None
    if sys.platform != 'darwin':
        browser_root = Path.home() / 'snap/chromium/common'
        browser_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kza-global-capture-browser-', dir=browser_root) as directory:
        work = Path(directory)
        receipt['browser_profile_root'] = str(work.parent)
        patch = pytest.MonkeyPatch()
        browser = None
        serving = None
        pages = []
        connections = []
        browser_log = work / 'chromium.log'
        external_attempts = []
        try:
            token = 'synthetic-global-capture-browser'
            patch.setenv('KZA_DASHBOARD_TOKEN', token)
            providers = capture_factories(leaves.__wrapped__(patch, work))

            class WakeBackend:
                def reset(self):
                    return None

            def wake_leaf(*args, **kwargs):
                detector = WakeWordDetector(*args, **kwargs)
                detector._oww_model = WakeBackend()
                detector._loaded = True
                detector._wake_has_backend = True
                return detector

            providers = replace(providers, wake_detector=wake_leaf)
            import src.ambient.transcriber as transcriber
            patch.setattr(transcriber, '_build_ambient_stt_engine', lambda *args: capture_stt())
            patch.setattr(system_monitor, 'gpu_snapshot', lambda: None)
            patch.setattr(system_monitor, 'services_snapshot', lambda: None)
            connect_original, resolve_original = socket.socket.connect, socket.getaddrinfo

            def local(host):
                return host in (None, 'localhost', '127.0.0.1', '::1', b'localhost', b'127.0.0.1')

            def connect_local(sock, address):
                if isinstance(address, tuple) and not local(address[0]):
                    external_attempts.append('connect')
                    raise RuntimeError('external_network_denied')
                return connect_original(sock, address)

            def resolve_local(host, *args, **kwargs):
                if not local(host):
                    external_attempts.append('resolve')
                    raise RuntimeError('external_network_denied')
                return resolve_original(host, *args, **kwargs)

            patch.setattr(socket.socket, 'connect', connect_local)
            patch.setattr(socket, 'getaddrinfo', resolve_local)
            async with TaskTransport().serve() as home_url, AsyncExitStack() as stack:
                config, _ = configuration(work, home_url)
                config['rooms'] = configured(work, home_url, rooms=True)['rooms']
                config['ambient'] = {'enabled': True, 'shadow_mode': True,
                    'db_path': str(work / 'ambient.sqlite'),
                    'keep_audio': {'enabled': False, 'dir': str(work / 'audio')},
                    'quality': {'enabled': False}}
                graph = await build_application(config, shutdown=stack, factories=providers)
                receipt['stage'] = 'factory_returned'
                print(json.dumps({'stage': receipt['stage'],
                                  'capture': graph.capture_control.snapshot()}), flush=True)
                assert graph.capture_control.snapshot()['supported'] is True
                assert graph.capture_control.ambient_control is not None
                receipt['actual_factory_and_control'] = True
                listener = socket.socket()
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
                base = f'http://127.0.0.1:{port}'
                server = uvicorn.Server(uvicorn.Config(graph.dashboard.app, log_level='warning'))
                serving = asyncio.create_task(server.serve(sockets=[listener]))
                await until(lambda: server.started)
                try:
                    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as client:
                        async with client.post(base + '/api/admin/auth/login', json={'token': token}) as response:
                            assert response.status == 200
                            cookie = response.cookies['kza_dashboard_token'].value
                            assert cookie != token
                        async with client.post(base + '/api/admin/users', json={
                                'name': 'Browser synthetic', 'permission_level': 'adult'}) as response:
                            assert response.status in (200, 201), await response.text()
                        profile = work / 'chromium'
                        command = [binary, '--headless', '--disable-gpu', '--no-first-run',
                            '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
                            '--mute-audio', '--autoplay-policy=no-user-gesture-required',
                            '--proxy-server=http://127.0.0.1:9', '--proxy-bypass-list=localhost;127.0.0.1;[::1]',
                            '--remote-debugging-port=0', f'--user-data-dir={profile}']
                        with browser_log.open('w') as log:
                            browser = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                            async with asyncio.timeout(20):
                                while not (profile / 'DevToolsActivePort').exists():
                                    if browser.poll() is not None:
                                        raise RuntimeError(browser_log.read_text()[:2000])
                                    await asyncio.sleep(.05)
                            debug = int((profile / 'DevToolsActivePort').read_text().splitlines()[0])
                            async with client.get(f'http://127.0.0.1:{debug}/json/list') as response:
                                target = next(row for row in await response.json() if row['type'] == 'page')
                            admin, connection = await open_page(client, debug, target['id'])
                            pages.append(admin)
                            connections.append(connection)
                            await admin.command('Network.setCookie', {'name': 'kza_dashboard_token',
                                'value': cookie, 'url': base + '/', 'httpOnly': True, 'sameSite': 'Strict'})
                            await admin.command('Emulation.setDeviceMetricsOverride', {
                                'width': 1280, 'height': 900, 'deviceScaleFactor': 1, 'mobile': False})
                            await admin.command('Page.navigate', {'url': base + '/obs/'})
                            await admin.wait_for("document.querySelector('#view-live') && window.__kzaLiveWs?.readyState===1")
                            await admin.evaluate(INSTRUMENT)
                            assert await admin.evaluate("window.__captureWitness!==undefined && typeof window.KzaCaptureRecorder?.createModalOwner==='function'")
                            second = await admin.command('Target.createTarget', {'url': 'about:blank'})
                            privacy, connection = await open_page(client, debug, second['targetId'])
                            pages.append(privacy)
                            connections.append(connection)
                            await privacy.command('Page.navigate', {'url': base + '/obs/'})
                            await privacy.wait_for("document.querySelector('#view-live') && window.__kzaLiveWs?.readyState===1")
                            await privacy.command('Page.bringToFront')
                            await privacy.evaluate("[...document.querySelectorAll('.nav-item')].find(row=>row.textContent.includes('Privacidad')).click()")
                            await privacy.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Entrada de audio habilitada')")
                            await privacy.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental habilitada')")
                            await click_button(privacy, 'Pausar captura ambiental')
                            await privacy.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental pausada')")
                            assert graph.capture_control.capture_allowed is True
                            await admin.command('Page.bringToFront')
                            await admin.evaluate("[...document.querySelectorAll('.nav-item')].find(row=>row.textContent.includes('Admin')).click()")
                            await admin.wait_for("document.querySelector('h1')?.textContent==='Admin' && [...document.querySelectorAll('tr')].some(row=>row.textContent.includes('Browser synthetic'))")
                            await admin.evaluate("[...document.querySelectorAll('tr')].find(row=>row.textContent.includes('Browser synthetic')).querySelectorAll('button')[0].click()")
                            await admin.wait_for("[...document.querySelectorAll('button')].some(row=>row.textContent.includes('grabar muestra 1'))")
                            await click_button(admin, 'grabar muestra 1')
                            await admin.wait_for("window.__captureWitness?.streams.length===1 && window.__captureWitness.contexts.length===1 && window.__captureWitness.contexts[0].state==='running'")
                            await admin.wait_for("window.__captureWitness?.owner?.sampleCount()===1")
                            await click_button(admin, 'grabar muestra 2')
                            await admin.wait_for("window.__captureWitness.streams.length===2 && window.__captureWitness.streams.at(-1).getTracks().some(track=>track.readyState==='live')")
                            assert graph.capture_control.recorders.pending() > 0
                            await privacy.command('Page.bringToFront')
                            await click_button(privacy, 'Pausar captura global')
                            await admin.wait_for("window.__captureWitness.ackPending && window.__captureWitness.owner.sampleCount()===0 && window.__captureWitness.streams.every(stream=>stream.getTracks().every(track=>track.readyState==='ended')) && window.__captureWitness.contexts.every(context=>context.state==='closed')")
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Hay grabaciones del navegador pendientes de retiro')")
                            pending = graph.capture_control.snapshot()
                            assert pending['phase'] == 'muting' and pending['pending_browser_recorders'] > 0
                            assert pending['total_mute_confirmed'] is False
                            assert await privacy.evaluate("[...document.querySelectorAll('button')].find(row=>row.textContent==='Reactivar captura global').disabled")
                            receipt['pending_until_browser_stop_ack'] = True
                            await admin.evaluate('window.__captureWitness.releaseAck()')
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Captura global pausada y confirmada')")
                            assert graph.capture_control.snapshot()['total_mute_confirmed'] is True
                            assert await admin.evaluate('window.__captureWitness.acksCompleted===1 && window.__captureWitness.owner.current()===false && window.__captureWitness.owner.sampleCount()===0')
                            receipt['native_fake_device_tracks_and_contexts_closed'] = True
                            receipt['modal_samples_cleared_and_real_retire_HTTP'] = True
                            receipt['desktop_png_sha256'] = await screenshot(privacy, 'desktop.png')
                            await privacy.command('Emulation.setDeviceMetricsOverride', {
                                'width': 390, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True})
                            await privacy.evaluate('window.scrollTo(0,0)')
                            assert await privacy.evaluate("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Captura global pausada y confirmada')")
                            receipt['mobile_png_sha256'] = await screenshot(privacy, 'mobile.png')
                            receipt['mobile_observation_only'] = True
                            await privacy.command('Emulation.setDeviceMetricsOverride', {
                                'width': 1280, 'height': 900, 'deviceScaleFactor': 1, 'mobile': False})
                            await click_button(privacy, 'Reactivar captura global')
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Entrada de audio habilitada')")
                            assert graph.capture_control.capture_allowed is True
                            assert graph.capture_control.ambient_control.snapshot()['persisted_paused'] is True
                            assert graph.capture_control.ambient_control.snapshot()['phase'] == 'paused'
                            await privacy.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental pausada')")
                            receipt['authenticated_nonvoice_resume_preserves_ambient_preference'] = True
                            await click_button(privacy, 'Reactivar captura ambiental')
                            await privacy.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental habilitada')")
                            assert graph.capture_control.capture_allowed is True
                            native = await privacy.evaluate("""(async()=>{
                              const base='/api/capture/native/calibration/';
                              const post=(action,body,credentials='same-origin')=>fetch(base+action,{method:'POST',credentials,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
                              const registration=await post('register',{});if(!registration.ok)throw Error('native reservation unavailable');
                              const reserved=await registration.json();window.__calibrationRetirementToken=reserved.token;
                              const activation=await post('activate',{token:reserved.token});if(!activation.ok)throw Error('native activation unavailable');
                              const active=await activation.json();return {kind:active.kind,state:active.state,capture_allowed:active.capture_allowed};
                            })()""")
                            assert native == {'kind': 'calibration', 'state': 'active', 'capture_allowed': True}
                            training = await privacy.evaluate("""(async()=>{
                              const base='/api/capture/native/wakeword-training/';
                              const post=(action,body)=>fetch(base+action,{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
                              const registration=await post('register',{});if(!registration.ok)throw Error('training reservation unavailable');
                              const reserved=await registration.json();window.__wakewordTrainingRetirementToken=reserved.token;
                              const activation=await post('activate',{token:reserved.token});if(!activation.ok)throw Error('training activation unavailable');
                              const active=await activation.json();return {kind:active.kind,state:active.state,capture_allowed:active.capture_allowed};
                            })()""")
                            assert training == {'kind': 'wakeword_training', 'state': 'active', 'capture_allowed': True}
                            diagnostics = await privacy.evaluate("""(async()=>{
                              const base='/api/capture/native/wakeword-diagnostics/';
                              const post=(action,body)=>fetch(base+action,{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
                              const registration=await post('register',{});if(!registration.ok)throw Error('diagnostics reservation unavailable');
                              const reserved=await registration.json();window.__wakewordDiagnosticsRetirementToken=reserved.token;
                              const activation=await post('activate',{token:reserved.token});if(!activation.ok)throw Error('diagnostics activation unavailable');
                              const active=await activation.json();return {kind:active.kind,state:active.state,capture_allowed:active.capture_allowed};
                            })()""")
                            assert diagnostics == {'kind': 'wakeword_diagnostics', 'state': 'active', 'capture_allowed': True}
                            zone_setup = await privacy.evaluate("""(async()=>{
                              const base='/api/capture/native/zone-setup/';
                              const post=(action,body)=>fetch(base+action,{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
                              const registration=await post('register',{});if(!registration.ok)throw Error('zone setup reservation unavailable');
                              const reserved=await registration.json();window.__zoneSetupRetirementToken=reserved.token;
                              const activation=await post('activate',{token:reserved.token});if(!activation.ok)throw Error('zone setup activation unavailable');
                              const active=await activation.json();return {kind:active.kind,state:active.state,capture_allowed:active.capture_allowed};
                            })()""")
                            assert zone_setup == {'kind': 'zone_setup', 'state': 'active', 'capture_allowed': True}
                            manual_voice = await privacy.evaluate("""(async()=>{
                              const base='/api/capture/native/manual-voice-command/';
                              const post=(action,body)=>fetch(base+action,{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
                              const registration=await post('register',{});if(!registration.ok)throw Error('manual voice reservation unavailable');
                              const reserved=await registration.json();window.__manualVoiceRetirementToken=reserved.token;
                              const activation=await post('activate',{token:reserved.token});if(!activation.ok)throw Error('manual voice activation unavailable');
                              const active=await activation.json();return {kind:active.kind,state:active.state,capture_allowed:active.capture_allowed};
                            })()""")
                            assert manual_voice == {'kind': 'manual_voice_command', 'state': 'active', 'capture_allowed': True}
                            before_native_mute = graph.capture_control.snapshot()
                            assert before_native_mute['pending_native_clients'] == 5
                            assert before_native_mute['pending_browser_recorders'] == 0
                            await click_button(privacy, 'Pausar captura global')
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Hay capturas de herramientas pendientes de retiro')")
                            native_pending = graph.capture_control.snapshot()
                            assert native_pending['pending_native_clients'] == 5
                            assert native_pending['pending_browser_recorders'] == 0
                            assert native_pending['total_mute_confirmed'] is False
                            assert await privacy.evaluate("[...document.querySelectorAll('button')].find(row=>row.textContent==='Reactivar captura global').disabled")
                            receipt['five_native_kinds_separate_pending_count_and_DOM'] = True
                            receipt['five_native_kinds_pending_snapshot'] = native_pending
                            retired = await privacy.evaluate("""(async()=>{
                              const response=await fetch('/api/capture/native/calibration/retire',{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:window.__calibrationRetirementToken})});
                              const value=await response.json();if(response.ok&&value.retired===true)window.__calibrationRetirementToken=null;
                              return {status:response.status,retired:value.retired};
                            })()""")
                            assert retired == {'status': 200, 'retired': True}
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Hay capturas de herramientas pendientes de retiro')")
                            partial_native = graph.capture_control.snapshot()
                            assert partial_native['pending_native_clients'] == 4
                            assert partial_native['pending_browser_recorders'] == 0
                            assert partial_native['total_mute_confirmed'] is False
                            assert await privacy.evaluate("[...document.querySelectorAll('button')].find(row=>row.textContent==='Reactivar captura global').disabled")
                            receipt['four_native_kinds_separate_pending_count_and_DOM'] = True
                            receipt['four_native_kinds_pending_snapshot'] = partial_native
                            receipt['after_calibration_retirement_snapshot'] = partial_native
                            receipt['one_native_retirement_does_not_confirm_other'] = True
                            training_retired = await privacy.evaluate("""(async()=>{
                              const response=await fetch('/api/capture/native/wakeword-training/retire',{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:window.__wakewordTrainingRetirementToken})});
                              const value=await response.json();if(response.ok&&value.retired===true)window.__wakewordTrainingRetirementToken=null;
                              return {status:response.status,retired:value.retired};
                            })()""")
                            assert training_retired == {'status': 200, 'retired': True}
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Hay capturas de herramientas pendientes de retiro')")
                            last_native = graph.capture_control.snapshot()
                            assert last_native['pending_native_clients'] == 3
                            assert last_native['pending_browser_recorders'] == 0
                            assert last_native['total_mute_confirmed'] is False
                            assert await privacy.evaluate("[...document.querySelectorAll('button')].find(row=>row.textContent==='Reactivar captura global').disabled")
                            receipt['three_native_kinds_separate_pending_count_and_DOM'] = True
                            receipt['three_native_kinds_pending_snapshot'] = last_native
                            receipt['after_training_retirement_snapshot'] = last_native
                            receipt['two_retirements_cannot_confirm_pending_diagnostics'] = True
                            diagnostics_retired = await privacy.evaluate("""(async()=>{
                              const response=await fetch('/api/capture/native/wakeword-diagnostics/retire',{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:window.__wakewordDiagnosticsRetirementToken})});
                              const value=await response.json();if(response.ok&&value.retired===true)window.__wakewordDiagnosticsRetirementToken=null;
                              return {status:response.status,retired:value.retired};
                            })()""")
                            assert diagnostics_retired == {'status': 200, 'retired': True}
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Hay capturas de herramientas pendientes de retiro')")
                            zone_pending = graph.capture_control.snapshot()
                            assert zone_pending['pending_native_clients'] == 2
                            assert zone_pending['pending_browser_recorders'] == 0
                            assert zone_pending['total_mute_confirmed'] is False
                            assert await privacy.evaluate("[...document.querySelectorAll('button')].find(row=>row.textContent==='Reactivar captura global').disabled")
                            receipt['two_native_kinds_separate_pending_count_and_DOM'] = True
                            receipt['two_native_kinds_pending_snapshot'] = zone_pending
                            receipt['after_diagnostics_retirement_snapshot'] = zone_pending
                            receipt['three_retirements_cannot_confirm_pending_zone_setup'] = True
                            zone_retired = await privacy.evaluate("""(async()=>{
                              const response=await fetch('/api/capture/native/zone-setup/retire',{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:window.__zoneSetupRetirementToken})});
                              const value=await response.json();if(response.ok&&value.retired===true)window.__zoneSetupRetirementToken=null;
                              return {status:response.status,retired:value.retired};
                            })()""")
                            assert zone_retired == {'status': 200, 'retired': True}
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Hay capturas de herramientas pendientes de retiro')")
                            manual_pending = graph.capture_control.snapshot()
                            assert manual_pending['pending_native_clients'] == 1
                            assert manual_pending['pending_browser_recorders'] == 0
                            assert manual_pending['total_mute_confirmed'] is False
                            assert await privacy.evaluate("[...document.querySelectorAll('button')].find(row=>row.textContent==='Reactivar captura global').disabled")
                            receipt['manual_voice_pending_snapshot'] = manual_pending
                            manual_retired = await privacy.evaluate("""(async()=>{
                              const response=await fetch('/api/capture/native/manual-voice-command/retire',{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:window.__manualVoiceRetirementToken})});
                              const value=await response.json();if(response.ok&&value.retired===true)window.__manualVoiceRetirementToken=null;
                              return {status:response.status,retired:value.retired};
                            })()""")
                            assert manual_retired == {'status': 200, 'retired': True}
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Captura global pausada y confirmada')")
                            native_muted = graph.capture_control.snapshot()
                            assert native_muted['pending_native_clients'] == native_muted['pending_browser_recorders'] == 0
                            assert native_muted['total_mute_confirmed'] is True
                            await click_button(privacy, 'Reactivar captura global')
                            await privacy.wait_for("document.querySelector('section[aria-label=\"Captura global\"]')?.textContent.includes('Entrada de audio habilitada')")
                            assert graph.capture_control.capture_allowed is True
                            receipt['native_calibration_token_only_HTTP_retire_and_UI_resume'] = True
                            receipt['native_calibration_HTTP_fixture_only_no_CLI_or_hardware'] = True
                            receipt['wakeword_training_token_only_HTTP_retire_and_UI_resume'] = True
                            receipt['both_native_kinds_HTTP_fixture_only_no_CLI_or_hardware'] = True
                            receipt['wakeword_diagnostics_token_only_HTTP_retire_and_UI_resume'] = True
                            receipt['three_native_kinds_HTTP_fixture_only_no_CLI_or_hardware'] = True
                            receipt['zone_setup_token_only_HTTP_retire_and_UI_resume'] = True
                            receipt['four_native_kinds_HTTP_fixture_only_no_CLI_or_hardware'] = True
                            receipt['manual_voice_token_only_HTTP_retire_and_UI_resume'] = True
                            receipt['five_native_kinds_HTTP_fixture_only_no_CLI_or_hardware'] = True
                            assert all(page.errors == [] for page in pages), [page.errors for page in pages]
                            assert external_attempts == []
                            assert not any(name in sys.modules for name in (
                                'onnxruntime', 'kokoro', 'speechbrain', 'faster_whisper', 'openwakeword'))
                            receipt.update(passed=True, javascript_exceptions=0,
                                browser_fake_device_only=True, external_attempts=external_attempts,
                                physical_or_all_criteria_acceptance=False)
                except BaseException:
                    receipt['capture_at_failure'] = graph.capture_control.snapshot()
                    receipt['ambient_at_failure'] = graph.capture_control.ambient_control.snapshot()
                    receipt['browser_failure_witnesses'] = []
                    receipt['javascript_errors_at_failure'] = [page.errors for page in pages]
                    for index, page in enumerate(pages):
                        try:
                            witness = await page.evaluate("(() => {const w=window.__captureWitness;return {view:document.querySelector('h1')?.textContent,events:w?.events,fetches:w?.fetches,current:w?.owner?.current(),samples:w?.owner?.sampleCount(),tracks:w?.streams.map(stream=>stream.getTracks().map(track=>track.readyState)),contexts:w?.contexts.map(context=>context.state),ackPending:w?.ackPending};})()")
                            receipt['browser_failure_witnesses'].append(witness)
                            await screenshot(page, f'failure-{index}.png')
                        except Exception:
                            receipt['browser_failure_witnesses'].append({'unavailable': True})
                    raise
                finally:
                    for page in pages:
                        try:
                            await page.evaluate("""(async()=>{
                              for(const [key,path] of [['__calibrationRetirementToken','/api/capture/native/calibration/retire'],['__wakewordTrainingRetirementToken','/api/capture/native/wakeword-training/retire'],['__wakewordDiagnosticsRetirementToken','/api/capture/native/wakeword-diagnostics/retire'],['__zoneSetupRetirementToken','/api/capture/native/zone-setup/retire'],['__manualVoiceRetirementToken','/api/capture/native/manual-voice-command/retire']]){
                                const token=window[key];if(!token)continue;
                                const response=await fetch(path,{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
                                if(response.ok)window[key]=null;
                              }
                            })()""")
                        except Exception:
                            pass
                    for page in pages[:1]:
                        try:
                            await page.evaluate('window.__captureWitness?.releaseAck?.();window.KzaCaptureRecorder?.stopAll()')
                        except Exception:
                            pass
                    for connection in connections:
                        await connection.close()
                    if browser is not None:
                        try:
                            os.killpg(browser.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            await asyncio.to_thread(browser.wait, timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(browser.pid, signal.SIGKILL)
                            await asyncio.to_thread(browser.wait, timeout=5)
                        receipt['chromium_wait_returned'] = True
                    server.should_exit = True
                    await serving
                    receipt['actual_HTTP_server_returned'] = True
        finally:
            if browser_log.exists():
                shutil.copyfile(browser_log, EVIDENCE / 'chromium.log')
            patch.undo()


async def main() -> None:
    """Preserve first results and actual process/HTTP cleanup independently."""
    EVIDENCE.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt = {'passed': False, 'source_pre': hashes(), 'started_at': time.time()}
    try:
        async with asyncio.timeout(120):
            await exercise(receipt)
    finally:
        receipt['source_post'] = hashes()
        receipt['source_unchanged'] = receipt['source_pre'] == receipt['source_post']
        receipt['finished_at'] = time.time()
        (EVIDENCE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
